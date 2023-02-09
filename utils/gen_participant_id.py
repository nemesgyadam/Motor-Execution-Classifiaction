import sys
import hashlib


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
    
    print(f'Copy participant ID to the Experiments table: {participant_id}\n')
    # input('Press Enter when done')
