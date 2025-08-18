#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import math
from typing import Tuple

import numpy as np
import onnx
import onnx.shape_inference
import pytest
from onnx import TensorProto

from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter import convert
from onnx2tflite.src.onnx_parser.meta.types import to_numpy_type
from tests import executors


@pytest.mark.parametrize("a_shape, b_shape, c_shape, alpha, beta", [
    pytest.param((8, 4), (4, 16), (8, 16), 1.0, 1.0, id="a=1.0 b=1.0"),
    pytest.param((8, 4), (4, 16), (8, 16), 1.0, 0.0, id="a=1.0 b=0.0"),
    pytest.param((8, 4), (4, 16), (8, 16), 0.33, 1.0, id="a=0.33 b=1.0"),
    pytest.param((8, 4), (4, 16), (8, 16), 1.0, 0.66, id="a=1.0 b=0.66"),
    pytest.param((8, 4), (4, 16), (8, 16), 0.5, 0.5, id="a=0.5 b=0.5"),
    pytest.param((8, 4), (4, 16), (8, 16), 16.75, 32.25, id="a=16.75 b=32.25"),
])
def test_gemm_whole_with_scalars(a_shape: Tuple[int],
                                 b_shape: Tuple[int],
                                 c_shape: Tuple[int],
                                 alpha: float,
                                 beta: float):
    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        2: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, c_shape, alpha, beta, transA, transB", [
    pytest.param((4, 8), (4, 16), (8, 16), 1.0, 1.0, 1, 0, id="non-scaled transA"),
    pytest.param((4, 8), (4, 16), (8, 16), 0.5, 2.0, 1, 0, id="scaled transA"),
    pytest.param((8, 4), (16, 4), (8, 16), 1.0, 1.0, 0, 1, id="non-scaled transB"),
    pytest.param((8, 4), (16, 4), (8, 16), 0.5, 2.0, 0, 1, id="scaled transB"),
    pytest.param((4, 8), (16, 4), (8, 16), 1.0, 1.0, 1, 1, id="non-scaled transA transB"),
    pytest.param((4, 8), (16, 4), (8, 16), 0.5, 2.0, 1, 1, id="scaled transA transB"),
])
def test_gemm_whole_with_valid_transposes(a_shape: Tuple[int],
                                          b_shape: Tuple[int],
                                          c_shape: Tuple[int],
                                          alpha: float,
                                          beta: float,
                                          transA: int,
                                          transB: int):
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        alpha=alpha,
        beta=beta,
        transA=transA,
        transB=transB)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        2: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, c_shape, transA, transB", [
    pytest.param((4, 8), (4, 16), (8, 16), 1, 0, id="transA"),
    pytest.param((8, 4), (16, 4), (8, 16), 0, 1, id="transB"),
    pytest.param((4, 8), (16, 4), (8, 16), 1, 1, id="transA+transB"),
])
def test_gemm_biased__second_input_static(
        a_shape: Tuple[int], b_shape: Tuple[int], c_shape: Tuple[int], transA: int, transB: int):
    b_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT))

    node = onnx.helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        transA=transA,
        transB=transB)

    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("B", TensorProto.FLOAT, b_shape, b_data)]
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, c_shape, transA, transB", [
    pytest.param((4, 8), (4, 16), (8, 16), 1, 0, id="transA"),
    pytest.param((8, 4), (16, 4), (8, 16), 0, 1, id="transB"),
    pytest.param((4, 8), (16, 4), (8, 16), 1, 1, id="transA+transB"),
])
def test_gemm_biased__first_input_static(
        a_shape: Tuple[int], b_shape: Tuple[int], c_shape: Tuple[int], transA: int, transB: int):
    a_data = np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT))

    node = onnx.helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        transA=transA,
        transB=transB)

    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
        initializer=[onnx.helper.make_tensor("A", TensorProto.FLOAT, a_shape, a_data)]
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("a_shape, b_shape, transA, transB", [
    pytest.param((4, 8), (4, 16), 1, 0, id="transA"),
    pytest.param((8, 4), (16, 4), 0, 1, id="transB"),
    pytest.param((4, 8), (16, 4), 1, 1, id="transA+transB"),
])
def test_gemm_both_inputs_static(a_shape: Tuple[int], b_shape: Tuple[int], transA: int, transB: int):
    a_data = np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT))
    b_data = np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT))

    node = onnx.helper.make_node(
        "Gemm",
        inputs=["A", "B"],
        outputs=["Y"],
        transA=transA,
        transB=transB)

    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
        initializer=[
            onnx.helper.make_tensor("A", TensorProto.FLOAT, a_shape, a_data),
            onnx.helper.make_tensor("B", TensorProto.FLOAT, b_shape, b_data),
        ]
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    executors.convert_run_compare(model, {})


@pytest.mark.parametrize("input_type", [
    pytest.param(TensorProto.FLOAT16, id="f16"),
    pytest.param(TensorProto.DOUBLE, id="f64"),
    pytest.param(TensorProto.INT32, id="i32"),
    pytest.param(TensorProto.INT64, id="i64"),
    pytest.param(TensorProto.UINT32, id="u32"),
    pytest.param(TensorProto.UINT64, id="u64"),
])
def test_gemm_invalid_types(input_type: int):
    N, K, M = 8, 10, 16
    a_shape, b_shape, c_shape = (N, K), (K, M), (N, M)

    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"])
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", input_type, a_shape),
            onnx.helper.make_tensor_value_info("B", input_type, b_shape),
            onnx.helper.make_tensor_value_info("C", input_type, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", input_type, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    with pytest.raises(logger.Error):
        convert.convert_model(model)
    assert logger.conversion_log.get_node_error_code(0) == logger.Code.CONVERSION_IMPOSSIBLE


def test_gemm_zero_scalars():
    N, K, M = 8, 10, 16
    a_shape, b_shape, c_shape = (N, K), (K, M), (N, M)

    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=0.0, beta=0.0)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        2: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("alpha", (1.0, 0.5, 2.0), ids=lambda x: f"alpha-{x}")
def test_gemm_unbiased(alpha: float):
    N, K, M = 8, 10, 16
    a_shape, b_shape = (N, K), (K, M)

    node = onnx.helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=alpha)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)


@pytest.mark.parametrize("alpha", (1.0, 0.5, 2.0), ids=lambda x: f"alpha-{x}")
def test_gemm_canceled_bias_from_beta(alpha: float):
    N, K, M = 8, 10, 16
    a_shape, b_shape, c_shape = (N, K), (K, M), (N, M)

    node_add = onnx.helper.make_node("Add", ["P", "Q"], ["C"])
    node_gemm = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=0.0)
    graph = onnx.helper.make_graph(
        [node_add, node_gemm],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("P", TensorProto.FLOAT, c_shape),
            onnx.helper.make_tensor_value_info("Q", TensorProto.FLOAT, c_shape),
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        2: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        3: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }

    cc = ConversionConfig({"allow_inputs_stripping": False})
    executors.convert_run_compare(model, input_data, conversion_config=cc)


@pytest.mark.parametrize("a_shape, b_shape, c_shape, alpha, beta", [
    pytest.param((8, 4), (4, 16), (1,), 1.0, 1.0, id="non-scaled bias(1)"),
    pytest.param((8, 4), (4, 16), (1,), 0.5, 2.0, id="scaled bias(1)"),
    pytest.param((8, 4), (4, 16), (8, 1), 1.0, 1.0, id="non-scaled bias(8,1)"),
    pytest.param((8, 4), (4, 16), (8, 1), 0.5, 2.0, id="scaled bias(8,1)"),
    pytest.param((8, 4), (4, 16), (1, 16), 1.0, 1.0, id="non-scaled bias(1,16)"),
    pytest.param((8, 4), (4, 16), (1, 16), 0.5, 2.0, id="scaled bias(1,16)"),
])
def test_gemm_non_matrix_bias(a_shape: Tuple[int],
                              b_shape: Tuple[int],
                              c_shape: Tuple[int],
                              alpha: float,
                              beta: float):
    node = onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta)
    graph = onnx.helper.make_graph(
        [node],
        "graph-gemm",
        [
            onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
            onnx.helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape),
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, ())],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="onnx-gemm",
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)
    input_data = {
        0: np.random.random(math.prod(a_shape)).reshape(a_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        1: np.random.random(math.prod(b_shape)).reshape(b_shape).astype(to_numpy_type(TensorProto.FLOAT)),
        2: np.random.random(math.prod(c_shape)).reshape(c_shape).astype(to_numpy_type(TensorProto.FLOAT)),
    }
    executors.convert_run_compare(model, input_data)
