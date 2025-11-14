#
# Copyright 2023-2025 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
from typing import Dict, Union

import numpy
import numpy as np
import onnx
import onnxruntime
from onnx.reference import ReferenceEvaluator

from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.model_shape_inference import ModelShapeInference

# If executed on i.MX platform, there is no tensorflow module. And typically the intention is to use the tflite python
# interpreter available in tflite_runtime
try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite
from onnx2tflite.src import logger
from onnx2tflite.src.converter import convert


class OnnxExecutor:

    def __init__(self, model_path_or_bytes: Union[str, bytes, os.PathLike],
                 save_model=False, saved_model_name="model.onnx", skip_shape_inference=False,
                 reference_evaluation=False):
        """
        Construct OnnxExecutor used to quickly run inference on ONNX model.

        :param model_path_or_bytes: Path to file or byte representation of executed ONNX model.
        :param save_model: If true and model was provided through "model_content",
            model is saved to storage with name "saved_model_name".
        :param saved_model_name: Model name used when model stored to storage. Default
            value is "model.onnx".
        :param skip_shape_inference: If true, shape inference is skipped.
        :param reference_evaluation: If true, model is evaluated with 'ReferenceEvaluator' from
            ONNX repository. This evaluator doesn't support operators from other domains (for
            example 'com.microsoft').
        """
        # TODO 'ReferenceEvaluator' has much better support in ONNX 1.16.0 for quantized models, because there
        # are some fixes related to De/QuantizeLinear. Inspect more after upgrade.

        if save_model and isinstance(model_path_or_bytes, bytes):
            model = onnx.load_model_from_string(model_path_or_bytes)
            if not skip_shape_inference:
                model = ModelShapeInference.infer_shapes(model)

            onnx.save_model(model, saved_model_name)

        if isinstance(model_path_or_bytes, bytes):
            model = onnx.load_model_from_string(model_path_or_bytes)
        else:
            model = onnx.load_model(model_path_or_bytes)

        if reference_evaluation:
            self.sess = ReferenceEvaluator(model)
        else:
            def has_only_x64_u8u8_compatible_ops(model) -> bool:
                # Check if model contains only ops that support U8U8 matrix multiplication.
                quant_precision_non_compat_ops = ["QLinearConv", "QGemm", "QLinearMatMul"]

                for node in model.graph.node:
                    if node.op_type in quant_precision_non_compat_ops:
                        return False

                return True

            try:
                sess_options = onnxruntime.SessionOptions()
                if has_only_x64_u8u8_compatible_ops(model):
                    sess_options.add_session_config_entry("session.x64quantprecision", "1")
                    logger.w(f"OnnxExecutor: Running in mode using U8U8 matrix multiplication instructions.")

                self.sess = onnxruntime.InferenceSession(model.SerializeToString(), sess_options)
            except Exception as e:
                # Something has failed, try without session options.
                logger.w(f"OnnxExecutor: Failed to create inference session. Reason: {e}. Trying default session.")
                self.sess = onnxruntime.InferenceSession(model.SerializeToString())

    def _get_input_name(self, idx=0):
        if isinstance(self.sess, ReferenceEvaluator):
            return self.sess.input_names[idx]
        else:
            return self.sess.get_inputs()[idx].name

    def _get_output_names(self):
        if isinstance(self.sess, ReferenceEvaluator):
            return self.sess.output_names
        else:
            return list(map(lambda x: x.name, self.sess.get_outputs()))

    def inference(self, input_data: Union[numpy.ndarray, Dict[int, numpy.ndarray]]) \
            -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        if isinstance(input_data, numpy.ndarray):
            input_feed = {self._get_input_name(0): input_data}
        elif isinstance(input_data, Dict):
            input_feed = {}
            for (index, data) in input_data.items():
                input_feed[self._get_input_name(index)] = data
        else:
            raise AttributeError(f"Input data in unexpected format {type(input_data)}")

        output_data = self.sess.run(None, input_feed)

        # Flatten output if there is only one value in output dictionary
        if len(output_data) == 1:
            return np.asarray(output_data[0])
        else:
            return {output_name: output_data[idx] for idx, output_name in enumerate(self._get_output_names())}


class TFLiteExecutor:
    _interpreter: tflite.Interpreter

    def __init__(self, model_path: str = None, model_content=None,
                 save_model=False, saved_model_name="model.tflite", delegate_path=None, num_threads=None,
                 op_resolver_type=tflite.experimental.OpResolverType.AUTO):
        """
        Construct TFLiteExecutor used to quickly run inference on TFLite model.
        Exactly one of "model_path" and "model_content" must be specified.

        :param model_path: Path to executed TFLite model.
        :param model_content: Path to byte representation of TFLite model.
        :param save_model: If true and model was provided through "model_content",
            model is saved to storage with name "saved_model_name".
        :param saved_model_name: Model name used when model stored to storage. Default
            value is "model.tflite".
        :param delegate_path: External delegate to be used for the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is None.
        :param num_threads: number of threads to be used by the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is None.
        :param op_resolver_type: Op kernels to be used by the TFLiteExecutor, see
            https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter for details. Default value is
            tflite.experimental.OpResolverType.AUTO.
        """
        assert model_path is not None or model_content is not None
        assert model_path is None or model_content is None

        if delegate_path is not None:
            delegate = [tflite.load_delegate(delegate_path)]
        else:
            delegate = None

        if save_model:
            with open(saved_model_name, "wb") as f:
                f.write(model_content)

        if model_path is not None:
            self._interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=delegate,
                                                   num_threads=num_threads,
                                                   experimental_op_resolver_type=op_resolver_type)
        else:
            self._interpreter = tflite.Interpreter(model_content=model_content, experimental_delegates=delegate,
                                                   num_threads=num_threads,
                                                   experimental_op_resolver_type=op_resolver_type)

        self._interpreter.allocate_tensors()

    def inference(self, input_data: Union[numpy.ndarray, Dict[int, numpy.ndarray]]) \
            -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        if isinstance(input_data, numpy.ndarray):
            self._interpreter.set_tensor(input_details[0]['index'], input_data)
        elif isinstance(input_data, Dict):
            if len(input_data) != len(input_details):
                logger.w(f"Number of model inputs: '{len(input_details)}', and provided input data: '{len(input_data)}'"
                         f" is not the same. Using first {len(input_details)} inputs tensors.")
            for index in range(len(input_details)):
                self._interpreter.set_tensor(input_details[index]['index'], input_data[index])

        self._interpreter.invoke()

        output_data = {}

        for output_detail in output_details:
            output_data[output_detail['name']] = self._interpreter.get_tensor(output_detail['index'])

        # Flatten output if there is only one value in output dictionary
        if len(output_data) == 1:
            return np.asarray(next(iter(output_data.values())))
        else:
            return output_data

    def get_output_details(self, index):
        return self._interpreter.get_output_details()[index]


def compare_output_arrays(tfl_output: np.ndarray, onnx_output: np.ndarray, output_name: str, rtol: float = 1.e-5,
                          atol: float = 1.e-8):
    """ Assert that the provided numpy arrays are equal.

    :param tfl_output: Numpy array holding the output of the TFLite model.
    :param onnx_output: Numpy array holding the output of the ONNX model.
    :param output_name: Common name of the above arrays.
    :param rtol: Relative tolerance.
    :param atol: Absolute tolerance.
    """
    tfl_output = tfl_output.astype(np.float32)
    onnx_output = onnx_output.astype(np.float32)

    if tfl_output.dtype != np.bool_ and tfl_output.size != 0:
        logger.i(f"Maximum output difference of the `{output_name}`tensor: {np.max(np.abs(tfl_output - onnx_output))}. "
                 f"(atol={atol}, rtol={rtol})")

    assert tfl_output.shape == onnx_output.shape, "Output shapes don't match!"

    assert np.allclose(tfl_output, onnx_output, rtol=rtol, atol=atol,
                       equal_nan=True), f"Output values of the `{output_name}` tensor don't match!"


def _get_output_tensor_for_name(name: str, model: onnx.ModelProto) -> onnx.ValueInfoProto:
    for output in model.graph.output:
        if output.name == name:
            return output

    logger.e(logger.Code.INTERNAL_ERROR, '_get_output_for_name(): failed to find output.')


def _assert_array_shape_equals_output_shape(output: onnx.ValueInfoProto, array: np.ndarray, verify_output_shape: bool):
    if not verify_output_shape:
        return

    def _shape_from_value_info(vi: onnx.ValueInfoProto) -> list[int]:
        return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]

    assert _shape_from_value_info(output) == list(array.shape), \
        f'Output tensor `{output.name}` has a different shape than its corresponding output data. ' \
        f'{_shape_from_value_info(output)} != {list(array.shape)}'


def convert_run_compare(onnx_model: onnx.ModelProto, input_data, check_model=True, rtol=1.e-5, atol=1.e-8,
                        save_models=False, input_data_tflite=None, verify_output_shape=True,
                        conversion_config: ConversionConfig = ConversionConfig(),
                        tflite_op_resolver_type=tflite.experimental.OpResolverType.AUTO,
                        reference_onnx_evaluation=False) -> (TFLiteExecutor, OnnxExecutor):
    if check_model:
        onnx.checker.check_model(onnx_model)

    tfl_model = convert.convert_model(onnx_model, conversion_config)

    onnx_executor = OnnxExecutor(onnx_model.SerializeToString(), save_model=save_models,
                                 skip_shape_inference=conversion_config.skip_shape_inference,
                                 reference_evaluation=reference_onnx_evaluation)
    onnx_output = onnx_executor.inference(input_data)

    tflite_input_data = input_data if input_data_tflite is None else input_data_tflite
    tflite_executor = TFLiteExecutor(model_content=bytes(tfl_model), save_model=save_models,
                                     op_resolver_type=tflite_op_resolver_type)
    tflite_output = tflite_executor.inference(tflite_input_data)

    if not conversion_config.skip_shape_inference:
        onnx_model = ModelShapeInference.infer_shapes(onnx_model)

    if isinstance(tflite_output, dict) and isinstance(onnx_output, dict):
        if len(set(tflite_output.keys()).symmetric_difference(set(onnx_output.keys()))) == 0:
            # Both TFLite and ONNX output dictionaries have the same keys.
            for output_name, tflite_out in tflite_output.items():
                compare_output_arrays(tflite_out, onnx_output[output_name], output_name, rtol, atol)
                _assert_array_shape_equals_output_shape(_get_output_tensor_for_name(output_name, onnx_model),
                                                        onnx_output[output_name], verify_output_shape)

        else:
            logger.e(logger.Code.INTERNAL_ERROR, "Original ONNX and converted TFLite models have different outputs.")

    elif isinstance(tflite_output, np.ndarray) and isinstance(onnx_output, np.ndarray):
        compare_output_arrays(tflite_output, onnx_output, 'main output', rtol, atol)
        _assert_array_shape_equals_output_shape(onnx_model.graph.output[0], onnx_output, verify_output_shape)

    else:
        # This can happen for example, if the TFLite model does not have some outputs, which are in the ONNX model.
        logger.e(logger.Code.NOT_IMPLEMENTED, "Original ONNX and converted TFLite models have different number of "
                                              "outputs. Testing is not implemented for this case.")

    return tflite_executor, onnx_executor
