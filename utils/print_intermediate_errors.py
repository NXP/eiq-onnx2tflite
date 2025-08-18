#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import argparse

import numpy as np

from executors import OnnxExecutor, TFLiteExecutor
from model_surgeon import intermediate_tensors_as_outputs
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.converter.builder import optimizer
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type


def main():
    parser = argparse.ArgumentParser(
        prog="print_intermediate_errors",
        description="""
                Print the error of the output of each operator before and after conversion.
            """,
        usage="python print_intermediate_errors.py input_file.onnx"
    )
    parser.add_argument("onnx_file")

    model_path = parser.parse_args().onnx_file

    model = intermediate_tensors_as_outputs(model_path, ".*")

    # Optimizations could cause issues. For example Conv -> Relu would be fused together and the Conv output would not
    #  be accessible in the TFLite model.
    cc = ConversionConfig()
    cc.optimization_whitelist = [
        optimizer.Optimization.REMOVE_UNUSED_TENSORS,
        optimizer.Optimization.KEEP_ONE_EMPTY_BUFFER
    ]
    converted_tflite = bytes(convert.convert_model(model, cc))

    inputs = [
        {'shape': [dim.dim_value for dim in tensor.type.tensor_type.shape.dim],
         'type': to_numpy_type(tensor.type.tensor_type.elem_type)} for
        tensor in model.graph.input]
    input_data = {i: np.arange(np.prod(tensor['shape'])).reshape(tensor['shape']).astype(tensor['type']) for i, tensor
                  in enumerate(inputs)}

    onnx_executor = OnnxExecutor(model.SerializeToString(), save_model=True)
    output_onnx = onnx_executor.inference(input_data)

    tflite_executor = TFLiteExecutor(model_content=converted_tflite, save_model=True)
    output_tflite = tflite_executor.inference(input_data)

    cyan = "\033[96m"
    green = '\033[92m'
    yellow = "\033[93m"
    bold = "\033[1m"
    end_format = "\033[0m"

    non_matched_tensors_count = 0

    for output_name, tflite_out in output_tflite.items():
        onnx_output_name = output_name.removesuffix("_channels_first")
        if onnx_output_name in output_onnx:
            o_value = output_onnx[onnx_output_name]
            t_value = tflite_out

            if o_value.dtype in [np.uint8, np.int8]:
                o_value = o_value.astype(np.int16)
                t_value = t_value.astype(np.int16)

            elif o_value.dtype not in [np.float32, np.float64]:
                raise NotImplementedError(f"Output type {o_value.dtype} is not yet supported by the script.")

            if np.issubdtype(o_value.dtype, np.integer):
                max_err = f"{np.abs(o_value - t_value).max()}"
            else:
                max_err = f"{np.abs(o_value - t_value).max():.6f}"

            tflite_name = "" if output_name == onnx_output_name else f" (TFLite: '{output_name}')"

            print(f"{cyan + onnx_output_name + end_format} {green + str(o_value.shape) + end_format}{tflite_name}\n"
                  f"\t└─ Max error = {yellow + bold + max_err + end_format}")
        else:
            non_matched_tensors_count += 1

    if non_matched_tensors_count != 0:
        print(f"Unable to match {non_matched_tensors_count} TFLite tensors.")


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print('FAILED')
        print(e)
