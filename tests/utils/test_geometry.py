"""Tests for the concrete vector math in engine.utils.geometry."""

from __future__ import annotations

import math
import unittest

from engine.types.geometry import Point3D, Vector3D
from engine.utils.geometry import (
    angle_between,
    cross_product,
    distance,
    dot_product,
    heading_angle_degrees,
    is_finite_point,
    magnitude,
    vector_between,
)


class VectorPrimitiveTests(unittest.TestCase):
    def test_vector_between(self) -> None:
        v = vector_between(Point3D(1.0, 2.0, 3.0), Point3D(4.0, 6.0, 3.0))
        self.assertEqual(v, Vector3D(x=3.0, y=4.0, z=0.0))

    def test_magnitude_3_4_5_triangle(self) -> None:
        self.assertAlmostEqual(magnitude(Vector3D(x=3.0, y=4.0, z=0.0)), 5.0)

    def test_dot_product_orthogonal_is_zero(self) -> None:
        self.assertAlmostEqual(dot_product(Vector3D(1, 0, 0), Vector3D(0, 1, 0)), 0.0)

    def test_dot_product_parallel(self) -> None:
        self.assertAlmostEqual(dot_product(Vector3D(2, 0, 0), Vector3D(3, 0, 0)), 6.0)

    def test_cross_product_unit_axes(self) -> None:
        result = cross_product(Vector3D(1, 0, 0), Vector3D(0, 1, 0))
        self.assertEqual(result, Vector3D(x=0.0, y=0.0, z=1.0))

    def test_distance(self) -> None:
        self.assertAlmostEqual(distance(Point3D(0, 0, 0), Point3D(3, 4, 0)), 5.0)

    def test_is_finite_point_true_for_normal_point(self) -> None:
        self.assertTrue(is_finite_point(Point3D(1.0, 2.0, 3.0)))

    def test_is_finite_point_false_for_nan(self) -> None:
        self.assertFalse(is_finite_point(Point3D(math.nan, 0.0, 0.0)))

    def test_is_finite_point_false_for_inf(self) -> None:
        self.assertFalse(is_finite_point(Point3D(math.inf, 0.0, 0.0)))


class AngleBetweenTests(unittest.TestCase):
    def test_perpendicular_vectors_is_90_degrees(self) -> None:
        self.assertAlmostEqual(angle_between(Vector3D(1, 0, 0), Vector3D(0, 1, 0)), 90.0)

    def test_parallel_vectors_is_0_degrees(self) -> None:
        self.assertAlmostEqual(angle_between(Vector3D(2, 0, 0), Vector3D(5, 0, 0)), 0.0)

    def test_opposite_vectors_is_180_degrees(self) -> None:
        self.assertAlmostEqual(angle_between(Vector3D(1, 0, 0), Vector3D(-1, 0, 0)), 180.0)

    def test_45_degree_angle(self) -> None:
        self.assertAlmostEqual(angle_between(Vector3D(1, 0, 0), Vector3D(1, 1, 0)), 45.0)

    def test_zero_length_vector_returns_zero_not_nan(self) -> None:
        self.assertEqual(angle_between(Vector3D(0, 0, 0), Vector3D(1, 0, 0)), 0.0)

    def test_both_zero_length_returns_zero(self) -> None:
        self.assertEqual(angle_between(Vector3D(0, 0, 0), Vector3D(0, 0, 0)), 0.0)


class HeadingAngleTests(unittest.TestCase):
    def test_positive_x_is_zero(self) -> None:
        self.assertAlmostEqual(heading_angle_degrees(Vector3D(1, 0, 0)), 0.0)

    def test_positive_z_is_90(self) -> None:
        self.assertAlmostEqual(heading_angle_degrees(Vector3D(0, 0, 1)), 90.0)

    def test_negative_x_is_180(self) -> None:
        self.assertAlmostEqual(abs(heading_angle_degrees(Vector3D(-1, 0, 0))), 180.0)

    def test_negative_z_is_negative_90(self) -> None:
        self.assertAlmostEqual(heading_angle_degrees(Vector3D(0, 0, -1)), -90.0)

    def test_y_component_is_ignored(self) -> None:
        self.assertAlmostEqual(heading_angle_degrees(Vector3D(1, 999.0, 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
