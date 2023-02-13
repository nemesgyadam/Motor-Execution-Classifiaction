import sys
import hashlib
import pygsheets


def gen_participant_id():
    print('\nType in participant information for id generation')
    print('Use lowercase, ascii only, no leading zeros in date\n')
    name = input("Name: ")
    print("Born in..")
    year =  input("\tyear:  ")
    month = input("\tmonth: ")
    day =   input("\tday:   ")

    succ = year.isdigit() and month.isdigit() and day.isdigit() and ' ' in name
    if not succ:
        print('ERROR: Parts of date are not int or no space in name detected!', file=sys.stderr)
        return None

    to_hash = f'{name}{int(year)}{int(month)}{int(day)}'.replace(' ','').lower()
    hashed = hashlib.md5(to_hash.encode())
    return hashed.hexdigest()[:8]


if __name__ == '__main__':

    # get participant id
    participant_id = None
    cant_get_it_right = True

    while cant_get_it_right:
        participant_id = gen_participant_id()
        cant_get_it_right = participant_id is None or input('Filled correctly (y/n): ').lower() != 'y'
    
    # session id generator
    key_path = 'keys\\experiment-377414-94e458f24082.json'
    gc = pygsheets.authorize(service_account_file=key_path)
    sheet = gc.open('Experiments')
    experiment_sheet = sheet[0]
    number_of_occurrences = len(experiment_sheet.find(participant_id))
    
    print(f'Participant ID: {participant_id}\n')
    print(f'Session ID:     S{number_of_occurrences + 1:03d}')
