#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import os
import pathlib

import onnx
import pytest
from onnx import TensorProto

from onnx2tflite.lib.tflite.BuiltinOperator import BuiltinOperator
from onnx2tflite.lib.tflite.TensorType import TensorType
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.converter.builder.model_builder import ModelBuilder
from onnx2tflite.src.converter.convert import convert_model
from onnx2tflite.src.converter.tensor_utils import tensor_has_data
from onnx2tflite.src.tflite_optimizer.operator_rules import HasFusedActivationFunction, NoFusedActivationFunction
from onnx2tflite.src.tflite_optimizer.optimizer import Optimization
from onnx2tflite.src.tflite_optimizer.pattern_matcher import MultipleSameOps, OneOf, Op, PatternMatcher
from onnx2tflite.src.tflite_optimizer.tensor_rules import TensorConsumedOnlyBy, TensorHasData, TensorHasOneConsumer, \
    TensorHasType, TensorIsQuantized

_ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent.joinpath("artifacts")


@pytest.fixture
def _builder_for_custom_model(mocker, intermediate_tflite_model_provider) -> ModelBuilder:
    """
                     │
                  ┌──▼──┐
                  │ Mul │
                  └──┬──┘
           ┌─────────┼─────────┐
        ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
        │ Add │   │ Add │   │ Add │
        └──┬──┘   └──┬──┘   └──┬──┘
        ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
        │ Mul │   │ Mul │   │ Mul ◄───── right static input
        └──┬──┘   └──┬──┘   └──┬──┘
                          ┌────┴────┐
                       ┌──▼──┐   ┌──▼──┐
                       │ Add │   │ Add │
                       └──┬──┘   └──┬──┘
    """
    shape = [2, 4, 6, 8]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node('Mul', ['x', 'x'], ['mul1_out']),

            onnx.helper.make_node('Add', ['mul1_out', 'b1'], ['add1_out']),
            onnx.helper.make_node('Add', ['mul1_out', 'b2'], ['add2_out']),
            onnx.helper.make_node('Add', ['mul1_out', 'b3'], ['add3_out']),

            onnx.helper.make_node('Mul', ['add1_out', 'add1_out'], ['o1']),
            onnx.helper.make_node('Mul', ['add2_out', 'add2_out'], ['o2']),
            onnx.helper.make_node('Mul', ['add3_out', 'static'], ['mul2_out']),

            onnx.helper.make_node('Add', ['mul2_out', 'b1'], ['o3']),
            onnx.helper.make_node('Add', ['mul2_out', 'b2'], ['o4']),

        ],
        'PatternMatching test',
        [
            onnx.helper.make_tensor_value_info('x', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('b1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('b2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('b3', TensorProto.FLOAT, shape)
        ],
        [
            onnx.helper.make_tensor_value_info('o1', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('o2', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('o3', TensorProto.FLOAT, shape),
            onnx.helper.make_tensor_value_info('o4', TensorProto.FLOAT, shape),
        ],
        [
            onnx.helper.make_tensor('static', TensorProto.FLOAT, [1], [2]),
        ]
    )
    onnx_model = onnx.helper.make_model(graph)

    spy = mocker.spy(convert, 'build_conversion_context')
    convert_model(onnx_model)
    builder = spy.spy_return.tflite_builder

    # Make sure the TFLite model is what we expect it to be.
    expected_operators = [
        BuiltinOperator.MUL,
        BuiltinOperator.ADD,
        BuiltinOperator.ADD,
        BuiltinOperator.ADD,
        BuiltinOperator.MUL,
        BuiltinOperator.MUL,
        BuiltinOperator.MUL,
        BuiltinOperator.ADD,
        BuiltinOperator.ADD
    ]
    intermediate_tflite_model_provider.assert_converted_model_has_operators(expected_operators)

    return builder


@pytest.fixture
def _builder_for_alexnet_model(mocker, intermediate_tflite_model_provider) -> ModelBuilder:
    # Use Alexnet for testing.
    model_path = os.path.join(_ARTIFACTS_DIR, "downloaded", "bvlcalexnet-12", "model.onnx")

    spy = mocker.spy(convert, 'build_conversion_context')
    config = ConversionConfig()
    config.optimization_blacklist = [Optimization.MOVE_ACTIVATION_BEFORE_CONCAT]
    convert_model(model_path, conversion_config=config)
    builder = spy.spy_return.tflite_builder

    # Make sure the TFLite model is what we expect it to be.
    expected_operators = [
        BuiltinOperator.TRANSPOSE,
        BuiltinOperator.CONV_2D,  # + fused Relu
        BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.SPLIT,
        BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,  # In parallel
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.RELU,
        BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.CONV_2D,  # + fused Relu
        BuiltinOperator.SPLIT,
        BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,  # In parallel
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.RELU,
        BuiltinOperator.SPLIT,
        BuiltinOperator.CONV_2D, BuiltinOperator.CONV_2D,  # In parallel
        BuiltinOperator.CONCATENATION,
        BuiltinOperator.RELU,
        BuiltinOperator.MAX_POOL_2D,
        BuiltinOperator.RESHAPE,
        BuiltinOperator.FULLY_CONNECTED,  # + fused Relu
        BuiltinOperator.FULLY_CONNECTED,  # + fused Relu
        BuiltinOperator.FULLY_CONNECTED,
        BuiltinOperator.SOFTMAX,
    ]
    intermediate_tflite_model_provider.assert_converted_model_has_operators(expected_operators)

    return builder


def test_sequence_of_two_ops__simple(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'pool5_1'
        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'w' in tensor_map.keys() and tensor_map['w'].name == 'fc6_w_00'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'
        assert 'z' in tensor_map.keys() and tensor_map['z'].name == 'fc6_3'


def test_sequence_of_two_ops__nameless_tensors(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], [None], ['y']),
            Op(['FullyConnected'], ['y', None, 'b'], [None])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'


def test_sequence_of_two_ops__none_io(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], outputs=['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'w' in tensor_map.keys() and tensor_map['w'].name == 'fc6_w_00'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'


def test_sequence_of_two_ops__ellipsis__end__representing_two_tensors(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', ...], ['z'])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'pool5_1'
        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'z' in tensor_map.keys() and tensor_map['z'].name == 'fc6_3'


def test_sequence_of_two_ops__ellipsis__end__representing_no_tensors(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y', ...]),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'pool5_1'
        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'w' in tensor_map.keys() and tensor_map['w'].name == 'fc6_w_00'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'
        assert 'z' in tensor_map.keys() and tensor_map['z'].name == 'fc6_3'


def test_sequence_of_two_ops__ellipsis__middle(_builder_for_alexnet_model):
    with pytest.raises(logger.Error) as e:
        PatternMatcher(
            _builder_for_alexnet_model,
            [
                Op(['Reshape'], ['x'], ['y']),
                Op(['FullyConnected'], ['y', ..., 'b'], ['z'])
            ])
    assert e.value.error_code == logger.Code.INTERNAL_ERROR
    assert '`...` can only be used' in e.value.msg


def test_sequence_of_two_ops__operator_rule__match(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'], [HasFusedActivationFunction()])
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'pool5_1'
        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'w' in tensor_map.keys() and tensor_map['w'].name == 'fc6_w_00'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'
        assert 'z' in tensor_map.keys() and tensor_map['z'].name == 'fc6_3'


def test_sequence_of_two_ops__operator_rule__no_match(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'], [NoFusedActivationFunction()])
        ])

    for _, _, _, _ in matcher.match_patterns():
        # Shouldn't find any matches.
        raise Exception


def test_sequence_of_two_ops__tensor_rule__match(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'])
        ],
        [
            TensorHasOneConsumer('y'),
            TensorHasType('b', TensorType.FLOAT32),
        ])

    for [reshape, fc], tensor_map, _, _ in matcher.match_patterns():
        assert reshape.builtin_options.operator_type == BuiltinOperator.RESHAPE
        assert fc.builtin_options.operator_type == BuiltinOperator.FULLY_CONNECTED

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'pool5_1'
        assert 'y' in tensor_map.keys() and tensor_map['y'].name == 'OC2_DUMMY_0'
        assert 'w' in tensor_map.keys() and tensor_map['w'].name == 'fc6_w_00'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'fc6_b_0'
        assert 'z' in tensor_map.keys() and tensor_map['z'].name == 'fc6_3'


def test_sequence_of_two_ops__tensor_rule__match_2(_builder_for_custom_model):
    # There are 2 instances which almost match, except for the static tensor. These are found before the
    #  one we want. Make sure only the correct one is matched.
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Add'], outputs=['x']),
            Op(['Mul'], ['x', 'b'])
        ],
        [
            TensorHasData('b')
        ])

    for [add, mul], tensor_map, _, _ in matcher.match_patterns():
        assert add.builtin_options.operator_type == BuiltinOperator.ADD
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL

        assert 'x' in tensor_map.keys() and tensor_map['x'].name == 'add3_out'
        assert 'b' in tensor_map.keys() and tensor_map['b'].name == 'static'


def test_sequence_of_two_ops__tensor_rule__no_match(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Reshape'], ['x'], ['y']),
            Op(['FullyConnected'], ['y', 'w', 'b'], ['z'])
        ],
        [
            TensorIsQuantized('w'),
        ])

    for _, _, _, _ in matcher.match_patterns():
        # Shouldn't find any matches.
        raise Exception


def test_forked_graph__using_op_block_only__simple(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], ['axis', 'x'], ['y1', 'y2']),
            Op(['Conv2D'], ['y1', 'w1', 'b1'], ['z1']),
            Op(['Conv2D'], ['y2', 'w2', 'b2'], ['z2']),
            Op(['Concatenation'], ['z1', 'z2'], ['out'])
        ])

    num_occurrences = 0
    for [split, conv1, conv2, concat], tensor_map, _, _ in matcher.match_patterns():
        num_occurrences += 1
        assert split.builtin_options.operator_type == BuiltinOperator.SPLIT
        assert conv1.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert conv2.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert concat.builtin_options.operator_type == BuiltinOperator.CONCATENATION

        assert 'axis' in tensor_map.keys()
        assert 'x' in tensor_map.keys()
        assert 'y1' in tensor_map.keys()
        assert 'y2' in tensor_map.keys()
        assert 'w1' in tensor_map.keys()
        assert 'b1' in tensor_map.keys()
        assert 'z1' in tensor_map.keys()
        assert 'w2' in tensor_map.keys()
        assert 'b2' in tensor_map.keys()
        assert 'z2' in tensor_map.keys()
        assert 'out' in tensor_map.keys()

        if num_occurrences == 1:  # Only check the tensor names the first time.
            assert tensor_map['axis'].name == 'split_dim_'
            assert tensor_map['x'].name == 'pool1_1'
            assert tensor_map['y1'].name == 'pool1_1_group_0'
            assert tensor_map['y2'].name == 'pool1_1_group_1'
            assert tensor_map['w1'].name == 'conv2_w_0_group_0'
            assert tensor_map['b1'].name == 'conv2_b_0_group_0'
            assert tensor_map['z1'].name == 'conv2_1_group_0'
            assert tensor_map['w2'].name == 'conv2_w_0_group_1'
            assert tensor_map['b2'].name == 'conv2_b_0_group_1'
            assert tensor_map['z2'].name == 'conv2_1_group_0_group_1'
            assert tensor_map['out'].name == 'conv2_1'

    # This pattern appears in total 3 times in the model.
    assert num_occurrences == 3


def test_forked_graph__using_op_block_only__more_complex(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], ['axis', None], ['y1', 'y2']),
            Op(['Conv2D'], ['y1', ...], ['z1']),
            Op(['Conv2D'], ['y2', None, 'b2']),
            Op(['Concatenation'], ['z1', 'z2'], [...])
        ])

    num_occurrences = 0
    for [split, conv1, conv2, concat], tensor_map, _, _ in matcher.match_patterns():
        num_occurrences += 1
        assert split.builtin_options.operator_type == BuiltinOperator.SPLIT
        assert conv1.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert conv2.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert concat.builtin_options.operator_type == BuiltinOperator.CONCATENATION

        if num_occurrences == 1:  # Only check the tensor names the first time.
            assert tensor_map['axis'].name == 'split_dim_'
            assert tensor_map['y1'].name == 'pool1_1_group_0'
            assert tensor_map['y2'].name == 'pool1_1_group_1'
            assert tensor_map['z1'].name == 'conv2_1_group_0'
            assert tensor_map['b2'].name == 'conv2_b_0_group_1'
            assert tensor_map['z2'].name == 'conv2_1_group_0_group_1'

    # This pattern appears in total 3 times in the model.
    assert num_occurrences == 3


def test_forked_graph__using_op_block_only__tensor_rule(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], ['axis', None], ['y1', 'y2']),
            Op(['Conv2D'], ['y1', ...], ['z1']),
            Op(['Conv2D'], ['y2', None, 'b2']),
            Op(['Concatenation'], ['z1', 'z2'], ['z']),
            Op(['Relu'], ['z'], ['out']),
        ], [
            TensorConsumedOnlyBy('z', 'MaxPool2D')  # This rule cause only 1 instance of the pattern to match.
        ])

    for [split, conv1, conv2, concat], tensor_map, _, _ in matcher.match_patterns():
        assert split.builtin_options.operator_type == BuiltinOperator.SPLIT
        assert conv1.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert conv2.builtin_options.operator_type == BuiltinOperator.CONV_2D
        assert concat.builtin_options.operator_type == BuiltinOperator.CONCATENATION

        assert tensor_map['axis'].name == 'split_dim_1'
        assert tensor_map['y1'].name == 'conv4_2_group_0'
        assert tensor_map['y2'].name == 'conv4_2_group_1'
        assert tensor_map['z1'].name == 'conv5_1_group_0'
        assert tensor_map['b2'].name == 'conv5_b_0_group_1'
        assert tensor_map['z2'].name == 'conv5_1_group_0_group_1'
        assert tensor_map['z'].name == 'conv5_1'
        assert tensor_map['out'].name == 'conv5_2'


def test_multiple_same_ops__simple(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Mul'], outputs=['x']),
            MultipleSameOps(['Add'], ['x', ...], ['y']),
        ])

    match_count = 0
    for [mul, add_ops], tensor_map, _, _ in matcher.match_patterns():
        match_count += 1
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL
        assert type(add_ops) == list

        if match_count == 1:
            assert len(add_ops) == 3
        elif match_count == 2:
            assert len(add_ops) == 2

        assert all(add_op.builtin_options.operator_type == BuiltinOperator.ADD for add_op in add_ops)

    # The model contains 2 instances of this pattern.
    assert match_count == 2


def test_multiple_same_ops__tensor_rules(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Mul'], outputs=['x']),
            MultipleSameOps(['Add'], ['x', ...], ['y']),
        ],
        [
            TensorConsumedOnlyBy('y', 'Mul')
        ])

    for [mul, add_ops], tensor_map, _, _ in matcher.match_patterns():
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL
        assert type(add_ops) == list

        # Only the first instance of the pattern satisfies the tensor rule.
        assert len(add_ops) == 3

        assert all(add_op.builtin_options.operator_type == BuiltinOperator.ADD for add_op in add_ops)


def test_invalid_pattern__op_not_first(_builder_for_custom_model):
    with pytest.raises(logger.Error) as e:
        PatternMatcher(
            _builder_for_custom_model,
            [
                MultipleSameOps(['Add'], ['x', ...], ['y'])
            ])
    assert e.value.error_code == logger.Code.INTERNAL_ERROR
    assert 'The first block must be an `Op`' in e.value.msg


def test_invalid_pattern__consuming_tensor_set(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Mul'], outputs=['x']),
            MultipleSameOps(['Add'], ['x', ...], ['y']),
            Op(['Mul'], ['y'])
        ])
    with pytest.raises(logger.Error) as e:
        for _, _, _, _ in matcher.match_patterns():
            pass

    assert e.value.error_code == logger.Code.INTERNAL_ERROR
    assert 'consuming a set of tensors `y` is not yet supported.' in e.value.msg


def test_one_of__different_op_type(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Mul'], outputs=['x']),
            OneOf([
                Op(['Mul'], ['x', ...], ['y']),  # `Mul` after `Mul` is not in the model.
                Op(['Add'], ['x', ...], ['y'])  # This `Add` will be matched.
            ])
        ])

    for [mul, second_op], _, _, _ in matcher.match_patterns():
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL
        assert second_op.builtin_options.operator_type == BuiltinOperator.ADD


def test_one_of__all_match(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Mul'], outputs=['x']),
            OneOf([
                # ALl these ops are the same. Make sure only 1 match is yielded per `Mul` operator.
                Op(['Add'], ['x', ...], ['y']),
                Op(['Add'], ['x', ...], ['y']),
                Op(['Add'], ['x', ...], ['y'])
            ])
        ])

    cnt = 0
    for [mul, add], _, _, _ in matcher.match_patterns():
        cnt += 1
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL
        assert add.builtin_options.operator_type == BuiltinOperator.ADD

    assert cnt == 2  # `Mul` into `Add` appears 2 times in the model (the `Mul` must be unique).


def test_one_of__tensor_rules(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(['Add'], outputs=['x']),
            OneOf([
                Op(['Mul'], ['b', 'x'], ['y']),
                Op(['Mul'], ['x', 'b'], ['y'])  # This one will be matched.
            ])
        ],
        [
            TensorHasData('b')
        ])

    for [add, mul], tensor_map, _, _ in matcher.match_patterns():
        assert add.builtin_options.operator_type == BuiltinOperator.ADD
        assert mul.builtin_options.operator_type == BuiltinOperator.MUL
        assert tensor_has_data(tensor_map['b'])


def test_op__mathing_inputs_from_the_back__last_input(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['FullyConnected'], [..., 'b'], ['x']),
            Op(['FullyConnected'], ['x', ...], op_rules=[HasFusedActivationFunction()]),  # This ensures a single match.
        ])

    found = False
    for _, tensor_map, _, _ in matcher.match_patterns():
        assert 'b' in tensor_map.keys()
        assert tensor_map['b'].name == 'fc6_b_0'  # The bias we want to match.
        found = True

    assert found


def test_op__mathing_inputs_from_the_back__second_to_last_input(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['FullyConnected'], [..., 'w', None], ['x']),
            Op(['FullyConnected'], ['x', ...], op_rules=[HasFusedActivationFunction()]),  # This ensures a single match.
        ])

    found = False
    for _, tensor_map, _, _ in matcher.match_patterns():
        assert 'w' in tensor_map.keys()
        assert tensor_map['w'].name == 'fc6_w_00'  # The weights we want to match.
        found = True

    assert found


def test_op__mathing_outputs_from_the_back__last_output(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], outputs=[..., 'x']),
            Op(['Conv2D'], ['x', ...], ['y']),
            Op(['Concatenation'], [..., 'y']),
        ])

    count = 0
    expected_names = [
        'pool1_1_group_1',
        'conv3_2_group_1',
        'conv4_2_group_1'
    ]
    for _, tensor_map, _, _ in matcher.match_patterns():
        assert 'x' in tensor_map.keys()
        assert tensor_map['x'].name == expected_names[count]
        count += 1

    # This pattern appears 3 times in the model.
    assert count == 3


def test_op__mathing_outputs_from_the_back__second_to_last_output(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], outputs=[..., 'x', None]),
            Op(['Conv2D'], ['x', ...], ['y']),
            Op(['Concatenation'], ['y', ...]),
        ])

    count = 0
    expected_names = [
        'pool1_1_group_0',
        'conv3_2_group_0',
        'conv4_2_group_0'
    ]
    for _, tensor_map, _, _ in matcher.match_patterns():
        assert 'x' in tensor_map.keys()
        assert tensor_map['x'].name == expected_names[count]
        count += 1

    # This pattern appears 3 times in the model.
    assert count == 3


def test_multiple_same_ops__mathing_inputs_from_the_back(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Split'], outputs=[None, 'x']),
            Op(['Conv2D'], ['x', ...], ['y']),
            MultipleSameOps(['Concatenation'], [..., 'y']),
        ])

    count = 0
    expected_names = [
        'conv2_1_group_0_group_1',
        'conv4_1_group_0_group_1',
        'conv5_1_group_0_group_1'
    ]
    for [_, _, concat_ops], tensor_map, _, _ in matcher.match_patterns():
        assert len(concat_ops) == 1  # Just 1 `Concatenation` was matched with the `MultipleSameOps`.
        assert 'y' in tensor_map.keys()
        assert tensor_map['y'].name == expected_names[count]
        count += 1

    assert count == 3  # The pattern appears 3 times in the model.


def test_multiple_same_ops__mathing_outputs_from_the_back(_builder_for_alexnet_model):
    matcher = PatternMatcher(
        _builder_for_alexnet_model,
        [
            Op(['Relu'], outputs=['x']),
            MultipleSameOps(['Split'], [..., 'x', ...], [..., 'y']),
        ])

    count = 0
    for [_, split_ops], tensor_map, _, _ in matcher.match_patterns():
        assert len(split_ops) == 1  # Just 1 `Split` was matched with the `MultipleSameOps`.
        assert 'y' in tensor_map.keys()
        assert len(tensor_map['y']) == 1
        assert tensor_map['y'][0].name == 'conv4_2_group_1'
        count += 1

    assert count == 1  # The pattern appears just once in the model.


def test_multiple_same_ops__none_ops(_builder_for_custom_model):
    with pytest.raises(logger.Error) as e:
        PatternMatcher(
            _builder_for_custom_model,
            [
                Op(['Mul'], outputs=['x']),
                MultipleSameOps(None, ['x', ...], [..., 'y']),
            ])
    assert e.value.error_code == logger.Code.INTERNAL_ERROR
    assert "`MultipleSameOps` doesn't support `ops=None`" in e.value.msg


def test_op__none_ops(_builder_for_custom_model):
    matcher = PatternMatcher(
        _builder_for_custom_model,
        [
            Op(inputs=[..., 'x'])
        ], [
            TensorHasData('x')
        ])

    count = 0
    for [op], tensor_map, _, _ in matcher.match_patterns():
        count += 1
        assert op.builtin_options.operator_type == BuiltinOperator.MUL  # Only 1 `Mul` matches.

    assert count == 1  # Only 1 operator matches the pattern.
