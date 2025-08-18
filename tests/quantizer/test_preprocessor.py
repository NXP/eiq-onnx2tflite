#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx
from onnx import TensorProto

from onnx2quant.qdq_quantization import QDQQuantizer, RandomDataCalibrationDataReader
from onnx2quant.quantization_config import QuantizationConfig
from onnx2tflite.src import logger
from tests import executors
from tests.executors import OnnxExecutor


def test_preprocessor__replace_div_with_mul(intermediate_tflite_model_provider):
    shape = [42]

    np.random.seed(23)
    static_value = (np.random.random(shape) + 2.).astype('float32')

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Add', ['x1', 'x2'], ['x3']),
            onnx.helper.make_node('Div', ['x3', 'static_value'], ['x4']),
            onnx.helper.make_node('Add', ['x4', 'x4'], ['y']),
        ],
        'Quantizer test',
        [
            onnx.helper.make_tensor_value_info('x1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('x2', TensorProto.FLOAT, shape)
        ],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
        [onnx.helper.make_tensor('static_value', TensorProto.FLOAT, shape, static_value)]
    )
    onnx_model = onnx.helper.make_model(graph)

    no_preprocessing_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    no_preprocessing_config.replace_div_with_mul = False

    preprocessing_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    preprocessing_config.replace_div_with_mul = True

    onnx_model_with_div = QDQQuantizer().quantize_model(onnx_model, no_preprocessing_config)
    onnx_model_with_mul = QDQQuantizer().quantize_model(onnx_model, preprocessing_config)

    # Make sure the option to disable this preprocessing step was printed.
    assert '--no-replace-div-with-mul' in logger.conversion_log.get_logs()['qdq_quantizer'][0]['message']

    # Make sure the node got replaced.
    assert onnx_model_with_div.graph.node[7].op_type == 'Div'  # Div model has the `Div` node.
    assert onnx_model_with_mul.graph.node[8].op_type == 'Mul'  # Mul model has the `Mul` node.
    assert 'static_value_0' in onnx_model_with_mul.graph.node[8].input[1]

    data = {
        0: np.random.random(shape).astype('float32'),
        1: np.random.random(shape).astype('float32')
    }

    # Make sure the pre-processed model can be converted and inferred correctly.
    executors.convert_run_compare(onnx_model_with_mul, data)

    # Make sure the model with `Div` and the model with `Mul` produce the same output.
    output_with_div = OnnxExecutor(onnx_model_with_div.SerializeToString()).inference(data)
    output_with_mul = OnnxExecutor(onnx_model_with_mul.SerializeToString()).inference(data)
    assert np.allclose(output_with_div, output_with_mul, atol=0.007)


def test_preprocessor__replace_constant_with_static_tensor(intermediate_tflite_model_provider):
    shape = [42]

    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Constant', [], ['x1'],
                                  value=onnx.helper.make_tensor('', TensorProto.FLOAT, [1], [0.123])),
            onnx.helper.make_node('Mul', ['x', 'x1'], ['y']),
        ],
        'Quantizer test',
        [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())],
    )
    onnx_model = onnx.helper.make_model(graph)

    preprocessing_config = QuantizationConfig(RandomDataCalibrationDataReader.from_onnx_model(onnx_model))
    preprocessing_config.replace_constant_with_static_tensor = True

    onnx_model = QDQQuantizer().quantize_model(onnx_model, preprocessing_config)

    # Make sure there is not `Constant` in the model.
    constant_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Constant']
    assert len(constant_nodes) == 0

    np.random.seed(23)
    data = np.random.random(shape).astype('float32')

    # Make sure the pre-processed model can be converted and inferred correctly.
    executors.convert_run_compare(onnx_model, data)
