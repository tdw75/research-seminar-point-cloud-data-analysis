import numpy as np
import pytest

from data.processing import parse_dataset


@pytest.mark.parametrize("name, exp_num", [("train", 3), ("test", 2)])
def test_parse_dataset(name, exp_num):
    points, labels = parse_dataset("files/modelnet10", 32, name)
    expected_labels = np.array(["bathtub"] * exp_num + ["toilet"] * exp_num)
    assert points.shape == (exp_num * 2, 32, 3)
    assert np.array_equal(labels, expected_labels)
