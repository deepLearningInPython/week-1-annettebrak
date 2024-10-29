import numpy as np
from tasks import step, ReLu


def test_relu_default_cutoff():
    array = np.array([-5, 0, 3, -2, 4])
    expected_output = np.array([0, 0, 3, 0, 4])
    np.testing.assert_array_equal(ReLu(array.copy()), expected_output, "Failed on default cutoff")

def test_relu_custom_cutoff():
    array = np.array([-5, 0, 3, -2, 4])
    expected_output = np.array([2, 2, 3, 2, 4])
    np.testing.assert_array_equal(ReLu(array.copy(), 2), expected_output, "Failed on custom cutoff")

def test_relu_empty_array():
    array = np.array([])
    expected_output = np.array([])
    np.testing.assert_array_equal(ReLu(array.copy()), expected_output, "Failed on empty array")


