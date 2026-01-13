#
# Copyright 2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import argparse
import numpy as np
from pathlib import Path


from onnx2quant.qdq_quantization import RandomDataCalibrationDataReader, InputSpec


def _parse_input_spec(spec: str):
    """Parse model input specification string in format: name=input_0:shape=(1,3,224,224):type=float32"""
    fields = {}
    for seg in spec.strip().split(":"):
        if "=" not in seg:
            raise ValueError(f"Invalid segment '{seg}': expected key=value")
        k, v = seg.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Empty key or value in segment '{seg}'")
        fields[k] = v

    for required in ("name", "shape", "type"):
            if required not in fields:
                raise ValueError(f"Missing required key '{required}'")

    shape = tuple(map(int, fields["shape"].strip("()").split(",")))

    try:
        dtype = np.dtype(fields["type"])
    except TypeError:
        raise ValueError(f"Invalid data type '{fields['type']}'")
        
    min_str = fields.get("min", None)
    max_str = fields.get("max", None)
    if (min_str is None) ^ (max_str is None):
            raise ValueError("If 'min' is set, 'max' must also be set (and vice versa).")
    min_val = dtype.type(min_str) if min_str is not None else None
    max_val = dtype.type(max_str) if max_str is not None else None
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"'min' ({min_val}) cannot be greater than 'max' ({max_val}).")
    
    return fields["name"], InputSpec(
        shape=shape,
        type=dtype,
        min=min_val,
        max=max_val
    )


def _create_output_dirs(dir_names: list[str], output_dir_base: Path) -> None:
    for input_name in dir_names:
        out_dir = output_dir_base / input_name
        out_dir.mkdir(parents=True, exist_ok=True)


def _save_samples(data_reader: RandomDataCalibrationDataReader, output_dir_base: Path) -> None:
        """Save generated synthetic data samples as .npy files organized by input name."""
        for i, d in enumerate(data_reader.data):
            for k, v in d.items():
                np.save(output_dir_base / k / f"{i}.npy", v)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate one or more dataset folders from specifications."
        "This script creates dataset directories and populates them based on provided specs. "
        "Each spec defines the tensor properties (name, shape, dtype, optional value range) "
        "and the target folder name where the generated samples will be stored."
    )
    
    parser.add_argument(
        "-i", "--input-spec",
        dest="input_spec",
        type=str,
        action="extend",
        nargs="+",
        required=True,
        help=(
            "One or more model input specifications."
            "  'name=<input_name>:shape=(<shape>):type=<dtype>[:min=<min>:max=<max>]'\n"
            "Details:"
            "  - shape: tuple in parentheses, e.g. (1,3,224,224). Use integers only."
            "  - type: supported dtypes are: float32, int64, int8, bool_."
            "  - min/max: (optional) numeric range constraints. If provided, both must be specified;"
            " omit both to skip range validation."
        )
    )

    parser.add_argument("-n", "--num-samples", dest="num_samples",
                        type=int, action="store", nargs="?",
                        default=10,
                        help="Numbers of samples to generate for the synthetic dataset."
                             "Default is 10.")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Path to the resulting dataset. (default: 'synthetic_dataset')")
    parser.add_argument("-z", "--zip", action='store_true', required=False,
                        help="Create a zip archive of the dataset. (default: False)")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")

    try:
        parsed_inputs = dict([_parse_input_spec(x) for x in args.input_spec])

        data_reader = RandomDataCalibrationDataReader(parsed_inputs, num_samples=args.num_samples)

        output_dir_base = Path(args.output or "synthetic_dataset")

        _create_output_dirs(data_reader.data[0].keys(), output_dir_base)
        _save_samples(data_reader, output_dir_base)

        if args.zip:
            import shutil
            shutil.make_archive(str(output_dir_base), 'zip', output_dir_base, ".")
            shutil.rmtree(output_dir_base)
    except Exception as e:
        print(e)