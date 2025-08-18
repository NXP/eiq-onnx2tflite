#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import argparse
import pathlib


def get_args():
    parser = argparse.ArgumentParser(description="Overwrite package version")
    parser.add_argument("--version", type=str, required=True, help="New version")
    return parser.parse_args()


def main():
    args = get_args()
    new_version = args.version

    file_paths = [
        pathlib.Path(__file__).parent.parent / "onnx2tflite" / "__init__.py",
    ]

    for file in file_paths:
        with open(file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("__version__"):
                    lines[i] = f'__version__ = "{new_version}"\n'
                    break

        with open(file, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
