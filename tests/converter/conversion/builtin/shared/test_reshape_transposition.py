#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import pytest

from onnx2tflite.src.converter.node_converters.shared import reshape_transposition
from onnx2tflite.src.converter.node_converters.shared.reshape_transposition import SingleUnitaryDimensionChangeType


@pytest.mark.parametrize("shape_a,shape_b", [
    pytest.param((2, 1), (1,)),
    pytest.param((1, 1, 1), (2, 1)),
    pytest.param((2, 3, 4), (2, 2, 3, 4)),
    pytest.param((2, 3, 4), (2, 1, 1, 3, 4)),
    pytest.param((2, 1, 1, 1, 2), (2, 1, 1, 1)),
])
def test_is_single_unitary_changes_invalid(shape_a, shape_b):
    change_details = reshape_transposition._single_unitary_dimension_change(shape_a, shape_b)

    assert change_details is None


@pytest.mark.parametrize("shape_a,shape_b,result_index,result_changed_type", [
    pytest.param((1, 1, 1), (1, 1), 0, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((1, 1, 1), (1, 1, 1, 1), 0, SingleUnitaryDimensionChangeType.UNSQUEEZE),
    pytest.param((2, 3, 4), (2, 3, 4, 1), 3, SingleUnitaryDimensionChangeType.UNSQUEEZE),
    pytest.param((2, 3, 4), (2, 1, 3, 4), 1, SingleUnitaryDimensionChangeType.UNSQUEEZE),
    pytest.param([2, 3, 4, 1], [2, 3, 4], 3, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((2, 3, 4, 1), (2, 3, 4), 3, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((1, 2, 3, 4, 5), (2, 3, 4, 5), 0, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((2, 1, 3, 4, 5), (2, 3, 4, 5), 1, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((2, 3, 1, 4, 5), (2, 3, 4, 5), 2, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((2, 3, 4, 1, 5), (2, 3, 4, 5), 3, SingleUnitaryDimensionChangeType.SQUEEZE),
    pytest.param((2, 3, 4, 5, 1), (2, 3, 4, 5), 4, SingleUnitaryDimensionChangeType.SQUEEZE),
])
def test_is_single_unitary_changes_valid(shape_a, shape_b, result_index, result_changed_type):
    changed_index, changed_type = reshape_transposition._single_unitary_dimension_change(shape_a, shape_b)

    assert changed_index == result_index
    assert changed_type == result_changed_type
