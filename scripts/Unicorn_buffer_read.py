""""
Sample script to read data from Unicorn,
using the buffer.
"""

import sys

sys.path.append("./")


import argparse
from utils.Unicorn import UnicornWrapper


def parse_args(args):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)

#TODO update
def main(args=None):
    args = parse_args(args)
    Unicorn = UnicornWrapper()

    Unicorn.listen()

    last_two_secs = Unicorn.get_latest_data(500)  # 2 seconds of data 250Hz
    print(last_two_secs.shape)

    Unicorn.stop()


if __name__ == "__main__":
    main()
