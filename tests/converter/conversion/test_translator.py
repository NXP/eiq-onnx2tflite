#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import pytest

from onnx2tflite.src import logger
from onnx2tflite.src.converter.conversion import translator


@pytest.mark.parametrize("rank,input_axis,output_axis", [
    (1, 0, 0),
    (2, 1, 1),
    (3, 1, 2),
    (3, 2, 1),
    (4, 1, 3),  # N[C]HW -> NHW[C]
    (4, 2, 1),  # NC[H]W -> N[H]WC
    (4, 3, 2),  # NCH[W] -> NH[W]C
    (5, 3, 2),  # NCH[W]D -> NH[W]DC
])
def test_channel_last_to_first(rank, input_axis, output_axis):
    perm = translator.create_channels_last_to_channels_first_permutation(rank)
    assert perm[input_axis] == output_axis


@pytest.mark.parametrize("rank,input_axis,output_axis", [
    (1, 0, 0),
    (2, 0, 0),
    (2, 1, 1),
    (4, 1, 2),  # N[H]WC -> NC[H]W
    (4, 2, 3),  # NH[W]C -> NCH[W]
    (4, 3, 1),  # NHW[C] -> N[C]HW
    (5, 1, 2),  # N[H]WCD -> NC[H]WD
    (5, 4, 1),  # NHWD[C] -> N[C]HWD
])
def test_channel_first_to_last(rank, input_axis, output_axis):
    perm = translator.create_channels_first_to_channels_last_permutation(rank)
    assert perm[input_axis] == output_axis


@pytest.mark.parametrize("perm1, perm2, expected_output", [
    ([0, 3, 1, 2], [0, 2, 3, 1], True),
    ([0, 2, 3, 1], [0, 3, 1, 2], True),
    ([0, 2, 3, 4, 1], [0, 4, 1, 2, 3], True),
    ([2, 1, 0], [2, 1, 0], True),
    ([2, 0, 1], [1, 2, 0], True),

    ([2, 1, 0], [1, 2, 0], False),
    ([1, 3, 0, 2], [1, 2, 0, 3], False),
    ([1, 3, 4, 0, 2], [1, 4, 2, 0, 3], False),

    ([1, 3, 4, 0, 2], [1, 4, 2, 0], None),
    ([1, 3, 4], [1, 4, 2, 0], None),
    ([1, 3, 4], [], None),
])
def test_permutations_are_inverse(perm1, perm2, expected_output):
    if expected_output is None:
        # expecting error
        with pytest.raises(logger.Error) as e:
            translator.permutations_are_inverse(perm1, perm2)

        assert e.type == logger.Error

    else:
        assert translator.permutations_are_inverse(perm1, perm2) == expected_output


@pytest.mark.parametrize("perm1, perm2, expected_output", [
    ([0, 3, 1, 2], [0, 2, 3, 1], [0, 1, 2, 3]),
    ([0, 2, 3, 1], [0, 3, 1, 2], [0, 1, 2, 3]),
    ([0, 2, 3, 4, 1], [0, 4, 1, 2, 3], [0, 1, 2, 3, 4]),
    ([2, 1, 0], [2, 1, 0], [0, 1, 2]),
    ([2, 0, 1], [1, 2, 0], [0, 1, 2]),

    ([2, 1, 0], [1, 2, 0], [1, 0, 2]),
    ([1, 3, 0, 2], [1, 2, 0, 3], [3, 0, 1, 2]),
    ([1, 3, 4, 0, 2], [1, 4, 2, 0, 3], [3, 2, 4, 1, 0]),

    ([1, 3, 4, 0, 2], [1, 4, 2, 0], None),
    ([1, 3, 4], [1, 4, 2, 0], None),
    ([1, 3, 4], [], None),
])
def test_combine_permutations(perm1, perm2, expected_output):
    if expected_output is None:
        # expecting error
        with pytest.raises(logger.Error) as e:
            translator.combine_permutations(perm1, perm2)

        assert e.type == logger.Error

    else:
        assert translator.combine_permutations(perm1, perm2) == expected_output
