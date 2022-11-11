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
    Unicorn = UnicornWrapper(frame_length=25)

    # Read 2 seconds of data.
    data = Unicorn.get_data(2)
    print(data.shape)

    Unicorn.stop()


if __name__ == "__main__":
    main()
