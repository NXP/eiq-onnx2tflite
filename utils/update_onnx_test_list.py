#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
"""
    update_onnx_test_list

    Script to generate the list of test from onnx/backend/test/data, to include the new tests when upgrading the ONNX
    version.
"""

import os
import pathlib
import re

TEST_FOLDER_BASE = os.path.join(pathlib.Path(__file__).parent.parent, "thirdparty", "onnx", "onnx", "backend", "test", "data")
TEST_GROUPS = ["node", "pytorch-converted", "pytorch-operator", "simple"]

def _to_tuple(s):
    """
    Create a tuple from string spliting to sequence of numbers and literals for proper sorting. E.g. 'test_add_uint16'
    became ('test_add_uint', 16). Hence the sorting will sort the strings lexicographically but sequence of
    digits as numbers.
    Hack: We want the underscore (_) be lexicographically sooner than alphanumerical character, so we replace the _ with
    space. E.g. to properly group tests, like this:
        "test_castlike_FLOAT_to_FLOAT8E4M3FN"
        "test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded"
        "test_castlike_FLOAT_to_FLOAT8E4M3FNUZ"
    """
    tmp = re.sub(r'(\d+)', r' \1 ', s) # Add spaces around sequence of digits
    tmp = tmp.split()                                  # and split the string
    tmp = [int(s) if s.isnumeric() else s.replace("_", " ") for s in tmp]
    tmp = tuple(tmp)
    return tmp

def _tuple_to_str(t):
    """
    Join tuples created by _to_tuple back to string and recontruct back the underscore.
    """
    return ''.join(map(str, t)).replace(" ", "_")


def sort_tests_names(l):
    """
    The function sorts the list lexicographically, but the digits inside the string are sorted as numbers.
    E.g. [add_test_uint8, add_test_uint16]: Pure alphabetical sort would make the *_uint16 precede the uint8,
    as '1' < '8'.
    """
    list_of_tuples = sorted([_to_tuple(i) for i in l])
    return [_tuple_to_str(t) for t in list_of_tuples]

if __name__ == "__main__":
    tests_to_add = {}
    for group in TEST_GROUPS:
        tests_to_add[group] = [f.name for f in os.scandir(os.path.join(TEST_FOLDER_BASE, group))]

    with open("tmp_tests.py","w") as f:
        for group in TEST_GROUPS:
            f.write(f'{group.upper()} = {{\n')
            for test in sort_tests_names(tests_to_add[group]):
                f.write(f'    # "{test}": {{}},\n')
            f.write('}\n\n')


    print("Finished")




