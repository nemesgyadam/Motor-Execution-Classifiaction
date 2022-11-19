import os
import time
import argparse
import numpy as np
from pathlib import Path


from config.arms_offline import config

clear = lambda: os.system("cls")
data_root = "data/"


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("Device", type=str, help="device or Arc")
    parser.add_argument("Subject", help="Name of the Subject")
    parser.add_argument(
        "--n_samples", default=1, help="Number of samples per class to record"
    )
    return parser.parse_args(args)


def generate_order(n_classes, n_samples_per_class=1):
    lists = []
    for i in range(n_classes):
        tmp = np.empty([n_samples_per_class])
        tmp.fill(i)
        lists.append(tmp)
    order = np.vstack(lists).ravel().astype(np.int32)
    np.random.shuffle(order)
    return order


def collect_data(device, sample_length, n_samples_per_class=1):
    classes = config.classes
    results = [[] for i in range(len(classes))]
    tasks = generate_order(len(classes), n_samples_per_class)

    device.listen()
    print("Franky says RELAX!")
    time.sleep(5)

    for i, task in enumerate(tasks):
        clear()

        print("Stand By! ({}/{})".format(i + 1, len(tasks)))
        time.sleep(1)
        print(config.commands[task])
        time.sleep(2)
        data = device.get_latest_data(sample_length * device.sample_rate)
        results[task].append(data)
    device.stop()
    return results, classes


def get_session(subject_dir):
    i = 1
    while os.path.exists(os.path.join(subject_dir, f"session_{str(i).zfill(2)}")):
        i += 1
    session = str(i).zfill(2)
    return "session_" + session


def main(args=None):
    args = parse_args(args)
    subject_dir = os.path.join(data_root, args.Subject)
    session = get_session(subject_dir)
    res_dir = os.path.join(subject_dir, session)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    print()
    print(f"Subject [{args.Subject}] {session} started!")
    print()
    if args.Device == "Unicorn":
        from utils.Unicorn import UnicornWrapper
        device = ArcWrapper()

    elif args.Device == "Arc":
        from utils.Arc import ArcWrapper
        device = ArcWrapper()

    else:
        raise ValueError("Device not supported")

    results, classes = collect_data(
        device, int(config.sample_length), int(args.n_samples)
    )

    # Save results to file
    print(f"Saving data to {res_dir}")
    i = 0
    for i, result in enumerate(results):
        result = np.asarray(result)
        np.save(os.path.join(res_dir, classes[i]), result)
    i += 1


if __name__ == "__main__":
    main()
