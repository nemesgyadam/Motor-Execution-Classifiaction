import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.Unicorn import UnicornWrapper
from config.arms_offline import config

clear = lambda: os.system("cls")
data_root = "data/"


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("Subject", help="Name of the Subject")
    parser.add_argument("--n_samples", default=1, help="Number of samples to record")
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


def collect_data(
    Unicorn, sample_length,  n_samples_per_class=1
):
    classes = config.classes
    results = [[] for i in range(len(classes))]
    tasks = generate_order(len(classes), n_samples_per_class)

    time.sleep(2)

    for i, task in enumerate(tasks):
        clear()

        print("Stand By! ({}/{})".format(i + 1, len(tasks)))
        time.sleep(1)
        print(config.commands[task])

        data = Unicorn.get_data(sample_length)
        results[task].append(data)
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
    Unicorn = UnicornWrapper(frame_length=25)

    results, classes = collect_data(Unicorn, int(config.sample_length))

    # Save results to file
    print(f"Saving data to {res_dir}")
    i = 0
    for i, result in enumerate(tqdm(results)):
        result = np.asarray(result)
        np.save(os.path.join(res_dir,classes[i]), result)
    i += 1
    Unicorn.stop()


if __name__ == "__main__":
    main()
