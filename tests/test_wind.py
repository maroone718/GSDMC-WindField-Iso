"""Tests for gsdmc.wind – wind-speed computation."""

import numpy as np
import pytest

from gsdmc.wind import compute_wind_speed, wind_speed_percentile


class TestComputeWindSpeed:
    def test_3_4_5_triangle(self):
        """Classic Pythagorean triple: sqrt(3² + 4²) = 5."""
        u = np.array([3.0])
        v = np.array([4.0])
        speed = compute_wind_speed(u, v)
        np.testing.assert_allclose(speed, [5.0])

    def test_3d_pythagorean(self):
        """sqrt(1² + 2² + 2²) = 3."""
        u = np.array([1.0])
        v = np.array([2.0])
        w = np.array([2.0])
        speed = compute_wind_speed(u, v, w)
        np.testing.assert_allclose(speed, [3.0])

    def test_zero_wind(self):
        speed = compute_wind_speed(np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(speed, np.zeros(5))

    def test_2d_field(self):
        """Works for 2-D arrays."""
        u = np.ones((4, 3))
        v = np.ones((4, 3))
        speed = compute_wind_speed(u, v)
        expected = np.full((4, 3), np.sqrt(2.0))
        np.testing.assert_allclose(speed, expected)

    def test_3d_field_no_w(self):
        shape = (5, 6, 7)
        u = np.random.default_rng(0).standard_normal(shape)
        v = np.random.default_rng(1).standard_normal(shape)
        speed = compute_wind_speed(u, v)
        expected = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(speed, expected)

    def test_3d_field_with_w(self):
        shape = (5, 6, 7)
        rng = np.random.default_rng(42)
        u, v, w = rng.standard_normal(shape), rng.standard_normal(shape), rng.standard_normal(shape)
        speed = compute_wind_speed(u, v, w)
        expected = np.sqrt(u**2 + v**2 + w**2)
        np.testing.assert_allclose(speed, expected)

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        u = rng.standard_normal((10, 10, 10))
        v = rng.standard_normal((10, 10, 10))
        w = rng.standard_normal((10, 10, 10))
        speed = compute_wind_speed(u, v, w)
        assert np.all(speed >= 0.0)

    def test_broadcast_scalar(self):
        """Scalar inputs are accepted."""
        speed = compute_wind_speed(0.0, 0.0, 0.0)
        assert float(speed) == pytest.approx(0.0)

    def test_output_dtype_float64(self):
        u = np.array([1.0], dtype=np.float32)
        v = np.array([1.0], dtype=np.float32)
        speed = compute_wind_speed(u, v)
        assert speed.dtype == np.float64


class TestWindSpeedPercentile:
    def test_median(self):
        speed = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        p50 = wind_speed_percentile(speed, 50)
        assert p50 == pytest.approx(5.5)

    def test_with_mask(self):
        speed = np.array([1.0, 2.0, 3.0, 100.0])
        mask = np.array([True, True, True, False])
        p100 = wind_speed_percentile(speed, 100, mask=mask)
        assert p100 == pytest.approx(3.0)

    def test_no_finite_raises(self):
        speed = np.array([np.nan, np.inf])
        with pytest.raises(ValueError):
            wind_speed_percentile(speed, 50)

    def test_3d_input(self):
        rng = np.random.default_rng(0)
        speed = rng.uniform(0, 30, (4, 5, 6))
        p = wind_speed_percentile(speed, 90)
        assert 0.0 <= p <= 30.0
