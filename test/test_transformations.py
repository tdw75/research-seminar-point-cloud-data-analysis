import numpy as np
import pytest

from transformations import (
    centre_cloud,
    flip_cloud,
    translate_cloud_random,
    flip_cloud_random,
    scale_cloud_random,
)


class TestTranslate:

    cloud = np.array([[2, 3, 4], [3, 3, 4], [2, 4, 4], [3, 4, 4]])

    def test_translate_cloud_random(self):  # check valid output
        cloud_transformed = translate_cloud_random(self.cloud)
        assert cloud_transformed.shape == self.cloud.shape

    def test_centre_cloud(self):
        cloud_translated = centre_cloud(self.cloud)
        cloud_exp = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]]
        )
        assert np.array_equal(cloud_translated.mean(axis=0), np.array([0, 0, 0]))
        assert np.array_equal(cloud_translated, cloud_exp)


class TestFlip:

    cloud = np.array([[2, 2, 0], [2, 4, 0], [4, 2, 0], [8, 8, 0]])
    cloud_centre = np.mean(cloud, axis=0)

    @pytest.mark.parametrize(
        "axes, cloud_exp",
        [
            ((True, False, False), [[6, 2, 0], [6, 4, 0], [4, 2, 0], [0, 8, 0]]),  # x
            (
                (True, True, False),
                [[6, 6, 0], [6, 4, 0], [4, 6, 0], [0, 0, 0]],
            ),  # x & y
        ],
    )
    def test_flip_cloud(self, axes, cloud_exp):
        cloud_flipped = flip_cloud(self.cloud, axes)
        centre_exp = np.array([4, 4, 0])
        assert np.array_equal(self.cloud_centre, centre_exp)
        assert np.array_equal(cloud_flipped, np.array(cloud_exp))

    def test_flip_cloud_random(self):
        cloud_flipped = flip_cloud_random(self.cloud)
        assert cloud_flipped.shape == self.cloud.shape


class TestScale:

    cloud = np.array([[2, 3, 4], [3, 3, 4], [2, 4, 4], [3, 4, 4]])

    def test_scale_cloud_random(self):
        cloud_scaled = scale_cloud_random(self.cloud)
        assert cloud_scaled.shape == self.cloud.shape
