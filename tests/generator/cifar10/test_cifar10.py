#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#

import os.path
import pathlib

import flatbuffers
import numpy as np

import onnx2tflite.src.tflite_generator.builtin_options.conv_2d_options as Conv2D
import onnx2tflite.src.tflite_generator.builtin_options.fully_connected_options as FullyConnected
import onnx2tflite.src.tflite_generator.builtin_options.max_pool_2d_options as MaxPool2D
import onnx2tflite.src.tflite_generator.builtin_options.softmax_options as Softmax
import onnx2tflite.src.tflite_generator.tflite_model as tflite_model
from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.Padding import Padding
from onnx2tflite.lib.tflite.TensorType import TensorType
from tests.data_loader import DataLoader
from tests.executors import TFLiteExecutor


def build_operators():
    return tflite_model.Operators([
        tflite_model.Operator(tflite_model.OperatorInputs([1, 2, 3]), tflite_model.OperatorOutputs([4]),
                              Conv2D.Conv2D(stride_h=1, stride_w=1)),
        tflite_model.Operator(tflite_model.OperatorInputs([4]), tflite_model.OperatorOutputs([5]),
                              MaxPool2D.MaxPool2D(Padding.VALID, 2, 2, 2, 2), 2),
        tflite_model.Operator(tflite_model.OperatorInputs([5, 6, 7]), tflite_model.OperatorOutputs([8]),
                              Conv2D.Conv2D(stride_w=1, stride_h=1)),
        tflite_model.Operator(tflite_model.OperatorInputs([8]), tflite_model.OperatorOutputs([9]),
                              MaxPool2D.MaxPool2D(Padding.VALID, 2, 2, 2, 2), 2),
        tflite_model.Operator(tflite_model.OperatorInputs([9, 10, 11]), tflite_model.OperatorOutputs([12]),
                              Conv2D.Conv2D(stride_w=1, stride_h=1)),
        tflite_model.Operator(tflite_model.OperatorInputs([12]), tflite_model.OperatorOutputs([13]),
                              MaxPool2D.MaxPool2D(Padding.VALID, 2, 2), 2),
        tflite_model.Operator(tflite_model.OperatorInputs([13, 14, 15]), tflite_model.OperatorOutputs([16]),
                              FullyConnected.FullyConnected(), 1),
        tflite_model.Operator(tflite_model.OperatorInputs([16]), tflite_model.OperatorOutputs([0]),
                              Softmax.Softmax(1.0), 3),
    ])


def build_tensors():
    tensors = tflite_model.Tensors()

    # Output
    output_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([0.99609375]),
                                         tflite_model.Scale([0.00390625]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 10]), "CifarNet/Predictions/Reshape_1",
                                       0, TensorType.UINT8, output_q))

    # Input
    input_q = tflite_model.Quantization(tflite_model.Min([-1.0078740119934082]), tflite_model.Max([1.0]),
                                        tflite_model.Scale([0.007874015718698502]),
                                        tflite_model.ZeroPoint([128]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 32, 32, 3]), "input",
                                       1, TensorType.UINT8, input_q))

    # Conv1
    conv1_w_q = tflite_model.Quantization(tflite_model.Min([-1.6849952936172485]),
                                          tflite_model.Max([1.2710195779800415]),
                                          tflite_model.Scale([0.01163785345852375]), tflite_model.ZeroPoint([146]))
    tensors.append(
        tflite_model.Tensor(tflite_model.Shape([32, 5, 5, 3]), "CifarNet/conv1/weights_quant/FakeQuantWithMinMaxVars",
                            2, TensorType.UINT8, conv1_w_q))

    conv1_b_q = tflite_model.Quantization(scale=tflite_model.Scale([0.00009163664071820676]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([32]), "CifarNet/conv1/Conv2D_bias",
                                       3, TensorType.INT32, conv1_b_q))

    conv1_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([23.805988311767578]),
                                            tflite_model.Scale([0.09335681796073914]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 32, 32, 32]), "CifarNet/conv1/Relu",
                                       4, TensorType.UINT8, conv1_out_q))

    # MaxPool1
    max_pool1_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([23.805988311767578]),
                                                tflite_model.Scale([0.09335681796073914]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 16, 16, 32]), "CifarNet/pool1/MaxPool",
                                       5, TensorType.UINT8, max_pool1_out_q))

    # Conv2
    conv2_w_q = tflite_model.Quantization(tflite_model.Min([-0.8235113024711609]),
                                          tflite_model.Max([0.7808409929275513]),
                                          tflite_model.Scale([0.006316347513347864]), tflite_model.ZeroPoint([131]))
    tensors.append(
        tflite_model.Tensor(tflite_model.Shape([32, 5, 5, 32]), "CifarNet/conv2/weights_quant/FakeQuantWithMinMaxVars",
                            6, TensorType.UINT8, conv2_w_q))

    conv2_b_q = tflite_model.Quantization(scale=tflite_model.Scale([0.0005896741058677435]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([32]), "CifarNet/conv2/Conv2D_bias",
                                       7, TensorType.INT32, conv2_b_q))

    conv2_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([21.17963981628418]),
                                            tflite_model.Scale([0.08305741101503372]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 16, 16, 32]), "CifarNet/conv2/Relu",
                                       8, TensorType.UINT8, conv2_out_q))

    # MaxPool2
    max_pool1_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([21.17963981628418]),
                                                tflite_model.Scale([0.08305741101503372]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 8, 8, 32]), "CifarNet/pool2/MaxPool",
                                       9, TensorType.UINT8, max_pool1_out_q))

    # Conv3
    conv3_w_q = tflite_model.Quantization(tflite_model.Min([-0.490180641412735]),
                                          tflite_model.Max([0.4940822720527649]),
                                          tflite_model.Scale([0.003875050926581025]), tflite_model.ZeroPoint([127]))
    tensors.append(
        tflite_model.Tensor(tflite_model.Shape([64, 5, 5, 32]), "CifarNet/conv3/weights_quant/FakeQuantWithMinMaxVars",
                            10, TensorType.UINT8, conv3_w_q))

    conv3_b_q = tflite_model.Quantization(scale=tflite_model.Scale([0.0003218516940250993]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([64]), "CifarNet/conv3/Conv2D_bias",
                                       11, TensorType.INT32, conv3_b_q))

    conv3_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([26.186586380004883]),
                                            tflite_model.Scale([0.10269249230623245]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 8, 8, 64]), "CifarNet/conv3/Relu",
                                       12, TensorType.UINT8, conv3_out_q))

    # MaxPool3
    max_pool1_out_q = tflite_model.Quantization(tflite_model.Min([0.0]), tflite_model.Max([26.186586380004883]),
                                                tflite_model.Scale([0.10269249230623245]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 4, 4, 64]), "CifarNet/pool3/MaxPool",
                                       13, TensorType.UINT8, max_pool1_out_q))

    # FullyConnected
    fc_w_q = tflite_model.Quantization(tflite_model.Min([-0.25385990738868713]),
                                       tflite_model.Max([0.38874608278274536]),
                                       tflite_model.Scale([0.002529944758862257]), tflite_model.ZeroPoint([101]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([10, 1024]),
                                       "CifarNet/logits/weights_quant/FakeQuantWithMinMaxVars/transpose",
                                       14, TensorType.UINT8, fc_w_q))

    fc_b_q = tflite_model.Quantization(scale=tflite_model.Scale([0.00025980634381994605]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([10]), "CifarNet/logits/MatMul_bias",
                                       15, TensorType.INT32, fc_b_q))

    fc_out_q = tflite_model.Quantization(tflite_model.Min([-13.407316207885742]), tflite_model.Max([22.58074378967285]),
                                         tflite_model.Scale([0.14112964272499084]), tflite_model.ZeroPoint([95]))
    tensors.append(tflite_model.Tensor(tflite_model.Shape([1, 10]), "CifarNet/logits/BiasAdd",
                                       16, TensorType.UINT8, fc_out_q))

    return tensors


def build_buffers():
    buffers_dir = os.path.join(pathlib.Path(__file__).parent, "buffers")
    buffers = tflite_model.Buffers()

    # Output
    buffers.append(tflite_model.Buffer())

    # Input
    buffers.append(tflite_model.Buffer())

    # Conv1
    conv_w = np.load(os.path.join(buffers_dir, "conv1-weights")).flatten()
    buffers.append(tflite_model.Buffer(conv_w, TensorType.UINT8))
    conv_b = np.load(os.path.join(buffers_dir, "conv1-bias")).flatten()
    buffers.append(tflite_model.Buffer(conv_b))
    buffers.append(tflite_model.Buffer())

    # MaxPool1
    buffers.append(tflite_model.Buffer())

    # Conv2
    conv_w = np.load(os.path.join(buffers_dir, "conv2-weights")).flatten()
    buffers.append(tflite_model.Buffer(conv_w, TensorType.UINT8))
    conv_b = np.load(os.path.join(buffers_dir, "conv2-bias")).flatten()
    buffers.append(tflite_model.Buffer(conv_b))
    buffers.append(tflite_model.Buffer())

    # MaxPool2
    buffers.append(tflite_model.Buffer())

    # Conv3
    conv_w = np.load(os.path.join(buffers_dir, "conv3-weights")).flatten()
    buffers.append(tflite_model.Buffer(conv_w, TensorType.UINT8))
    conv_b = np.load(os.path.join(buffers_dir, "conv3-bias")).flatten()
    buffers.append(tflite_model.Buffer(conv_b))
    buffers.append(tflite_model.Buffer())

    # MaxPool3
    buffers.append(tflite_model.Buffer())

    # Fully Connected
    fc_w = np.load(os.path.join(buffers_dir, "fc-weights")).flatten()
    buffers.append(tflite_model.Buffer(fc_w, TensorType.UINT8))
    fc_b = np.load(os.path.join(buffers_dir, "fc-bias")).flatten()
    buffers.append(tflite_model.Buffer(fc_b))
    buffers.append(tflite_model.Buffer())

    return buffers


def build_model():
    """ Generate the 'cifar10_model.tflite' """
    operator_codes = tflite_model.OperatorCodes([
        tflite_model.OperatorCode(BuiltinOperator.CONV_2D),
        tflite_model.OperatorCode(BuiltinOperator.FULLY_CONNECTED),
        tflite_model.OperatorCode(BuiltinOperator.MAX_POOL_2D),
        tflite_model.OperatorCode(BuiltinOperator.SOFTMAX),
    ])

    sub_graphs = tflite_model.SubGraphs([
        tflite_model.SubGraph(
            tflite_model.SubGraphInputs([1]),
            tflite_model.SubGraphOutputs([0]),
            build_tensors(),
            build_operators()
        )
    ])

    return tflite_model.Model(3, "TOCO Converted.", build_buffers(), operator_codes, sub_graphs)


def test_cifar10_output_of_manually_created_model_is_equal_to_original(cifar10_tflite):
    input_data = DataLoader.load_image(os.path.join(cifar10_tflite["inputs_dir"], "airplane1.png"))

    original_model_executor = TFLiteExecutor(cifar10_tflite["model_path"])
    reference_output = original_model_executor.inference(input_data)

    builder = flatbuffers.Builder(3400)
    model = build_model()
    model.gen_tflite(builder)

    executor = TFLiteExecutor(model_content=bytes(builder.Output()))
    generated_output = executor.inference(input_data)

    assert np.equal(reference_output, generated_output).all()
