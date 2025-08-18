#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import numpy as np
import onnx.helper
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from tests import executors


def test_simple_fusing__consuming_model_input(intermediate_tflite_model_provider):
    """
                  │    float32
               ┌──▼───┐
               │ Cast │                                │    float32
               └──┬───┘                             ┌──▼───┐
                  │    double                       │ Cast │
               ┌──▼───┐                             └──┬───┘
               │ Cast │             ─────►             │    int32
               └──┬───┘                             ┌──▼──┐
                  │    int32                        │ Mul │
               ┌──▼──┐                              └──┬──┘
               │ Mul │                                 │
               └──┬──┘
                  │
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Cast', ['x'], ['x1'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.INT32),
                onnx.helper.make_node('Mul', ['x2', 'x2'], ['y'])
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.CAST, BuiltinOperator.MUL
    ])


def test_simple_fusing__producing_model_output(intermediate_tflite_model_provider):
    """
                  │   float32
               ┌──▼──┐
               │ Mul │                                 │    float32
               └──┬──┘                              ┌──▼──┐
                  │   double                        │ Mul │
               ┌──▼───┐                             └──┬──┘
               │ Cast │             ─────►             │    int32
               └──┬───┘                             ┌──▼───┐
                  │   int32                         │ Cast │
               ┌──▼───┐                             └──┬───┘
               │ Cast │                                │
               └──┬───┘
                  │
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y'], to=TensorProto.INT32)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.INT32, ())]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.CAST
    ])


def test_forked_fusing(intermediate_tflite_model_provider):
    """
            │
         ┌──▼──┐
         │ Mul │
         └──┬──┘
            │   float32                                 │
         ┌──▼───┐                                    ┌──▼──┐
         │ Cast │                                    │ Mul │
         └──┬───┘                                    └──┬──┘
          ┌─┴──────────┐   double      ─────►        ┌──┴─────────┐   float32
       ┌──▼───┐     ┌──▼───┐                      ┌──▼───┐     ┌──▼───┐
       │ Cast │     │ Cast │                      │ Cast │     │ Cast │
       └──┬───┘     └──┬───┘                      └──┬───┘     └──┬───┘
          │   int64    │   int32                     │   int64    │   int32
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y1'], to=TensorProto.INT64),
                onnx.helper.make_node('Cast', ['x2'], ['y2'], to=TensorProto.INT32)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor_value_info('y1', TensorProto.INT64, ()),
                onnx.helper.make_tensor_value_info('y2', TensorProto.INT32, ())
            ]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.CAST, BuiltinOperator.CAST
    ])


def test_impossible_fusing(intermediate_tflite_model_provider):
    """
            │
         ┌──▼──┐
         │ Mul │
         └──┬──┘
            │   float32
         ┌──▼───┐
         │ Cast │
         └──┬───┘
          ┌─┴──────────┬──────────────┐   double
       ┌──▼───┐     ┌──▼───┐          ▼
       │ Cast │     │ Cast │     model output
       └──┬───┘     └──┬───┘
          │   int64    │   int32
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y1'], to=TensorProto.INT64),
                onnx.helper.make_node('Cast', ['x2'], ['y2'], to=TensorProto.INT32)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor_value_info('y1', TensorProto.INT64, ()),
                onnx.helper.make_tensor_value_info('y2', TensorProto.INT32, ()),
                onnx.helper.make_tensor_value_info('x2', TensorProto.DOUBLE, ())
            ]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.CAST, BuiltinOperator.CAST, BuiltinOperator.CAST
    ])


def test_simple_complete_removal__consuming_model_input(intermediate_tflite_model_provider):
    """
                  │    float32
               ┌──▼───┐
               │ Cast │
               └──┬───┘
                  │    double                          │    float32
               ┌──▼───┐                             ┌──▼──┐
               │ Cast │             ─────►          │ Mul │
               └──┬───┘                             └──┬──┘
                  │    float32                         │
               ┌──▼──┐
               │ Mul │
               └──┬──┘
                  │
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Cast', ['x'], ['x1'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.FLOAT),
                onnx.helper.make_node('Mul', ['x2', 'x2'], ['y'])
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL
    ])


def test_simple_complete_removal__producing_model_output(intermediate_tflite_model_provider):
    """
                      │
                   ┌──▼──┐
                   │ Mul │
                   └──┬──┘
                      │    float32                         │    float32
                   ┌──▼───┐                             ┌──▼──┐
                   │ Cast │             ─────►          │ Mul │
                   └──┬───┘                             └──┬──┘
                      │    double                          │
                   ┌──▼───┐
                   │ Cast │
                   └──┬───┘
                      │   float32
        """
    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y'], to=TensorProto.FLOAT)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, ())]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL
    ])


def test_forked_partial_removal(intermediate_tflite_model_provider):
    """
            │
         ┌──▼──┐
         │ Mul │
         └──┬──┘
            │   float32                                 │
         ┌──▼───┐                                    ┌──▼──┐
         │ Cast │                                    │ Mul │
         └──┬───┘                                    └──┬──┘
          ┌─┴──────────┐   double      ─────►        ┌──┴─────────┐   float32
       ┌──▼───┐     ┌──▼───┐                                   ┌──▼───┐
       │ Cast │     │ Cast │                                   │ Cast │
       └──┬───┘     └──┬───┘                                   └──┬───┘
          │  float32   │   int32                                  │   int32
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y1'], to=TensorProto.FLOAT),
                onnx.helper.make_node('Cast', ['x2'], ['y2'], to=TensorProto.INT32)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, ()),
                onnx.helper.make_tensor_value_info('y2', TensorProto.INT32, ())
            ]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.CAST
    ])


def test_forked_partial_removal__multiple_model_outputs(intermediate_tflite_model_provider):
    """
            │
         ┌──▼──┐
         │ Mul │
         └──┬──┘
            │   float32                                 │
         ┌──▼───┐                                    ┌──▼──┐
         │ Cast │                                    │ Mul │
         └──┬───┘                                    └──┬──┘
          ┌─┴──────────┐   double      ─────►        ┌──┴─────────┐   float32
       ┌──▼───┐     ┌──▼───┐                                   ┌──▼───┐
       │ Cast │     │ Cast │                                   │ Cast │
       └──┬───┘     └──┬───┘                                   └──┬───┘
          │  float32   │   float32                                │   float32


        The 2nd `Cast` cannot be skipped, because its input and output tensors are both model outputs. If the op was
         removed, its input and output would be combined into 1 tensor, which would have to represent 2 model outputs
         with 2 different names, which is not possible.
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Mul', ['x', 'x'], ['x1']),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x2'], ['y1'], to=TensorProto.FLOAT),
                onnx.helper.make_node('Cast', ['x2'], ['y2'], to=TensorProto.FLOAT)
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, ()),
                onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, ())
            ]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.CAST
    ])


def test_forked_complete_removal(intermediate_tflite_model_provider):
    """
            │  float32
         ┌──▼───┐
         │ Cast │                                       ┌─────┴──────┐  float32
         └──┬───┘                                    ┌──▼──┐      ┌──▼──┐
          ┌─┴──────────┐  double       ─────►        │ Mul │      │ Mul │
       ┌──▼───┐     ┌──▼───┐                         └──┬──┘      └──┬──┘
       │ Cast │     │ Cast │                            │            │
       └──┬───┘     └──┬───┘
          │  float32   │  float32
       ┌──▼──┐      ┌──▼──┐
       │ Mul │      │ Mul │
       └──┬──┘      └──┬──┘
          │            │
    """

    shape = [42]
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node('Cast', ['x'], ['x1'], to=TensorProto.DOUBLE),
                onnx.helper.make_node('Cast', ['x1'], ['x2'], to=TensorProto.FLOAT),
                onnx.helper.make_node('Cast', ['x1'], ['x3'], to=TensorProto.FLOAT),
                onnx.helper.make_node('Mul', ['x2', 'x2'], ['y1']),
                onnx.helper.make_node('Mul', ['x3', 'x3'], ['y2'])
            ],
            'Cast pruning test',
            [onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
            [
                onnx.helper.make_tensor_value_info('y1', TensorProto.FLOAT, ()),
                onnx.helper.make_tensor_value_info('y2', TensorProto.FLOAT, ())
            ]
        )
    )

    data = np.random.random(shape).astype(np.float32)
    executors.convert_run_compare(onnx_model, data)

    intermediate_tflite_model_provider.assert_converted_model_has_operators([
        BuiltinOperator.MUL, BuiltinOperator.MUL
    ])
