import os
import time
import argparse
import numpy as np
from pathlib import Path


from config.arms_offline import config
from utils.stim_utils import Stimulus

clear = lambda: os.system("cls")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("Device", type=str, help="Unicorn or Arc")
    parser.add_argument("Subject", help="Name of the Subject")
    parser.add_argument(
        "--n_samples", default=1, help="Number of samples per class to record"
    )
    return parser.parse_args(args)


def generate_order(n_classes, n_samples_per_class=1):
    """
    Generate a random order of tasks.
    Return list of task indexes.
    """
    lists = []
    for i in range(n_classes):
        tmp = np.empty([n_samples_per_class])
        tmp.fill(i)
        lists.append(tmp)
    order = np.vstack(lists).ravel().astype(np.int32)
    np.random.shuffle(order)
    return order


def get_available_session_name(subject_dir):
    """
    Get the next available session name.
    """
    i = 1
    while os.path.exists(os.path.join(subject_dir, f"session_{str(i).zfill(2)}")):
        i += 1
    session = str(i).zfill(2)
    return "session_" + session


def prepare_folder(subject, log=True):
    """
    Prepare the folder structure for the data.
    """
    subject_dir = os.path.join(config.data_path, subject)
    session = get_available_session_name(subject_dir)
    res_dir = os.path.join(subject_dir, session)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    if log:
        print()
        print(f"Subject [{subject}] {session} prepared!")
        print()
    return res_dir


def init_device(device):
    """
    Initialize the device based on the given argument.
    Note: UnicornPy cant be imported without licence.
          Therefore we have to import it here.
    """
    if device == "Unicorn":
        from utils.Unicorn import UnicornWrapper

        return UnicornWrapper()
    elif device == "Arc":
        from utils.Arc import ArcWrapper

        return ArcWrapper()
    else:
        raise ValueError("Device not supported")


def collect_data(device, config, n_samples_per_class=1):
    """
    The complete data collection process.
    Generate tasks in random order.
    Start recording.
    Instruct the users.
    -------------------------
    Trigger codes:
    0: No trigger
    10: End of sample
    1:  CLass 1 start trigger
    2:  CLass 2 start trigger
    3:  CLass 3 start trigger
    -------------------------
    ...
    """
    tasks = generate_order(len(config.classes), n_samples_per_class)
    stim = Stimulus(config)
    device.start_session()

    print("Franky says RELAX!")
    stim.show("Relax")
    time.sleep(config.relax_time)

    for i, task in enumerate(tasks):
        clear()

        if config.fixation_length > 0:
            print("Stand By! ({}/{})".format(i + 1, len(tasks)))
            stim.show("Fixation")
            time.sleep(config.fixation_length)

        print(config.commands[task])
        stim.show(config.classes[task])
        #TODO random wait
        if config.erp_length > 0:  # Wait out the ERP response
            time.sleep(config.erp_length)
        device.trigger(task + 1)
        time.sleep(config.sample_length)

        device.trigger(10)

        if (
            config.rest_length > 0 and i != len(tasks) - 1
        ):  # No Rest period after last trial
            print("Rest period.")
            stim.show("Blank")
            time.sleep(config.rest_length)

    time.sleep(1)
    result = device.get_session_data()
    device.stop()
    stim.stop()
    return result


def save_results(results, res_dir):
    """
    Save the results to the given directory.
    In npy format.
    TODO: Add support for other formats.
    """
    np.save(os.path.join(res_dir, "data.npy"), results)
    print(f"Data saved to {res_dir}")


def main(args=None):
    """
    Executes the main program.
    Based on the given arguments and config file.
    """
    args = parse_args(args)
    res_dir = prepare_folder(args.Subject)
    clear()
    device = init_device(args.Device)

    results = collect_data(device, config, int(args.n_samples))

    save_results(results, res_dir)


if __name__ == "__main__":
    main()
