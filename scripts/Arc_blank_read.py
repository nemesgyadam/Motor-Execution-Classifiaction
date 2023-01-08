""""
Sample script to read data from Arc.
"""

import sys
import time

sys.path.append("./")

import numpy as np
import argparse

from utils.Arc import ArcWrapper


def parse_args(args):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    Arc = ArcWrapper()
    #Arc.listen()

    Arc.continous_listen()
    time.sleep(1)
    for i in range(40):
        time.sleep(0.1)
        Arc.trigger(i)
    data = Arc.session_buffer[:,:2000]
    print(data.shape)
    np.savetxt("test.csv", data, delimiter=",")
    quit()




    time.sleep(2)

    # Read 2 seconds of data.
    data = Arc.get_latest_data(2 * Arc.sample_rate)
    print()
    print(f"Recieved data shape: {data.shape}")
    print()
    Arc.stop()
    # Arc.board.insert_marker(1.) Doesn't work


if __name__ == "__main__":
    main()
