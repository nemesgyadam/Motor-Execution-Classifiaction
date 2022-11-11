import argparse

from utils.Unicorn import UnicornWrapper


def parse_args(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument("model", help="model")
    # parser.add_argument("--target", default = '', help= "Device to control:(None, ros-continous, ros-step, keyboard)")
    return parser.parse_args(args)


def main(args=None):
    # Specifications for the data acquisition.
    # -------------------------------------------------------------------------------------
    args = parse_args(args)
    Unicorn = UnicornWrapper()

    Unicorn.listen()

    last_two_secs = Unicorn.data_buffer[:, -500:]
    print(last_two_secs.shape)

    Unicorn.stop()


if __name__ == "__main__":
    main()
