""""
Sample script to read data from Unicorn.
"""

import sys

sys.path.append("./")

import argparse

from utils.Unicorn import UnicornWrapper


def parse_args(args):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    Unicorn = UnicornWrapper()

    # Read 2 seconds of data.
    data = Unicorn.get_data(2*Unicorn.sample_rate)
    print(data.shape)

    Unicorn.stop()


if __name__ == "__main__":
    main()
