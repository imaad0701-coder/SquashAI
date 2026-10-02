"""Floor-plane homography between video pixels and court metres, from
manually clicked court landmarks (tools/calibrate_court.py ->
calibrations/<clip>.json).

What it is and isn't:
  - A plane-to-plane projective map of the *floor only*, fitted by the
    normalized Direct Linear Transform (Hartley normalization + SVD), pure
    numpy. Needs >= 4 clicked floor points, not all collinear.
  - No lens model. Phone footage is often wide-angle; uncorrected radial
    distortion shows up as error that varies across the image -- which is
    exactly what leave_one_out() is for. CameraIntrinsics
    (engine.calibration.interfaces) exists for a later distortion step if
    that error turns out to need it.
  - Sits beside the existing CalibrationProvider protocol rather than
    implementing it: that protocol is built around camera intrinsics, which
    a homography doesn't have.
  - One camera pose per calibration. If the camera moves during a clip,
    the map is wrong for the moved part.

Feet: a foot's floor position is its foot_index (toe) landmark -- the 15-point
set has no heel, and the ankle sits ~8 cm above the floor, which a floor
homography would push further back in court space than the foot really is.
A toe on a lifted foot is also off the floor; per-frame positions are
approximate by construction. Positions outside the polygon of clicked points
are flagged `extrapolated`: on footage filmed from behind the back wall the
clicked points cover the front of the court, so anything nearer the camera
is outside the calibrated region.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_PRESENCE, MIN_LANDMARK_VISIBILITY
from engine.calibration.court_geometry import COURT_POINTS
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName

# Relative singular-value floor below which a point set is treated as
# collinear (degenerate). Structural, not tuned: it rejects exact or
# near-exact collinearity, which makes the fit undefined, nothing more.
_COLLINEARITY_RTOL = 1e-6


class CalibrationError(ValueError):
    pass


def _normalizing_transform(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    if mean_dist == 0:
        raise CalibrationError("all points coincide")
    s = math.sqrt(2) / mean_dist
    return np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]])


def _check_not_collinear(points: np.ndarray, label: str) -> None:
    sv = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if sv[0] == 0 or sv[1] / sv[0] < _COLLINEARITY_RTOL:
        raise CalibrationError(f"{label} points are collinear; a floor homography needs points spanning an area")


def _has_general_position_quad(points: np.ndarray) -> bool:
    """True if some 4 of the points have no 3 on one line -- the condition
    for a unique homography. Checked on the *court* points, whose
    coordinates are exact by definition: three clicks on one court line are
    never exactly collinear in pixels (click noise), so a pixel-side or
    rank-based test sees an ill-conditioned system rather than a degenerate
    one and lets a meaningless fit through."""
    scale = float(np.ptp(points, axis=0).max()) or 1.0
    tol = 1e-9 * scale * scale

    def collinear(a, b, c) -> bool:
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= tol

    for quad in itertools.combinations(points, 4):
        if not any(collinear(*tri) for tri in itertools.combinations(quad, 3)):
            return True
    return False


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """H with dst ~ H @ src (homogeneous), normalized DLT."""
    t_src, t_dst = _normalizing_transform(src), _normalizing_transform(dst)
    s = (t_src @ np.c_[src, np.ones(len(src))].T).T
    d = (t_dst @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _u, sv, vt = np.linalg.svd(np.asarray(rows))
    # A unique homography needs the 2n x 9 system to have rank 8 (a 1-D null
    # space). Fewer -- e.g. 4 points with 3 on one line -- leaves a family of
    # solutions and SVD just returns an arbitrary member of it, so refuse.
    if len(sv) < 8 or sv[7] / sv[0] < _COLLINEARITY_RTOL:
        raise CalibrationError("degenerate point configuration (e.g. 3 of 4 points on one line): no unique homography")
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    return h / h[2, 2]


def _apply(h: np.ndarray, x: float, y: float) -> tuple[tuple[float, float], float]:
    p = h @ np.array([x, y, 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2])), float(p[2])


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


@dataclass(frozen=True)
class CourtHomography:
    image_to_court: np.ndarray  # 3x3
    court_to_image: np.ndarray  # 3x3
    calibrated_region: tuple[tuple[float, float], ...]  # convex hull of the clicked points, court metres
    _court_side_sign: float  # sign of projective w (image->court) for pixels below the floor's horizon
    _image_side_sign: float  # sign of projective w (court->image) for floor points in front of the camera

    def pixel_to_court(self, x: float, y: float) -> tuple[float, float] | None:
        """None for pixels on or beyond the floor's horizon line, where the
        floor plane doesn't exist in the image (sky/walls above the horizon)."""
        (cx, cy), w = _apply(self.image_to_court, x, y)
        if w == 0 or math.copysign(1.0, w) != self._court_side_sign:
            return None
        return cx, cy

    def court_to_pixel(self, x: float, y: float) -> tuple[float, float] | None:
        (px, py), w = _apply(self.court_to_image, x, y)
        if w == 0 or math.copysign(1.0, w) != self._image_side_sign:
            return None  # behind the camera
        return px, py

    def is_extrapolated(self, court_xy: tuple[float, float]) -> bool:
        return not _point_in_polygon(court_xy[0], court_xy[1], self.calibrated_region)


def fit_homography(image_points: Sequence[tuple[float, float]],
                   court_points: Sequence[tuple[float, float]]) -> CourtHomography:
    if len(image_points) != len(court_points):
        raise CalibrationError("image and court point counts differ")
    if len(image_points) < 4:
        raise CalibrationError(f"need >= 4 points, got {len(image_points)}")
    img = np.asarray(image_points, dtype=float)
    crt = np.asarray(court_points, dtype=float)
    _check_not_collinear(img, "image")
    _check_not_collinear(crt, "court")
    if not _has_general_position_quad(crt):
        raise CalibrationError(
            "degenerate point set: no 4 points with 3-not-on-one-line (e.g. 3 points along one wall plus 1 other); "
            "add a point off those lines")
    h_ic = _dlt(img, crt)
    h_ci = np.linalg.inv(h_ic)
    h_ci = h_ci / h_ci[2, 2]
    court_sign = math.copysign(1.0, _apply(h_ic, *img[0])[1])
    image_sign = math.copysign(1.0, _apply(h_ci, *crt[0])[1])
    hull = tuple(_convex_hull([tuple(p) for p in crt.tolist()]))
    return CourtHomography(image_to_court=h_ic, court_to_image=h_ci, calibrated_region=hull,
                           _court_side_sign=court_sign, _image_side_sign=image_sign)


@dataclass(frozen=True)
class LeaveOneOutResult:
    name: str
    court_xy: tuple[float, float]
    clicked_px: tuple[float, float]
    predicted_px: tuple[float, float] | None  # held-out court point projected through the fit without it
    pixel_error: float | None
    predicted_court_xy: tuple[float, float] | None  # held-out click mapped to court through the fit without it
    court_error_m: float | None
    extrapolated: bool  # held-out point lies outside the remaining points' region


def leave_one_out(names: Sequence[str], image_points: Sequence[tuple[float, float]],
                  court_points: Sequence[tuple[float, float]]) -> list[LeaveOneOutResult]:
    """Fit without each point in turn and measure how far that point lands
    from where it was clicked (pixels) and from where it really is (metres).
    Needs >= 5 points. A held-out point whose removal leaves a degenerate set
    reports None errors rather than a number from an undefined fit."""
    if len(names) < 5:
        raise CalibrationError("leave-one-out needs >= 5 points (4 remain for each fit)")
    results = []
    for i, name in enumerate(names):
        rest_img = [p for j, p in enumerate(image_points) if j != i]
        rest_crt = [p for j, p in enumerate(court_points) if j != i]
        try:
            h = fit_homography(rest_img, rest_crt)
        except CalibrationError:
            results.append(LeaveOneOutResult(name, tuple(court_points[i]), tuple(image_points[i]),
                                             None, None, None, None, True))
            continue
        pred_px = h.court_to_pixel(*court_points[i])
        pred_court = h.pixel_to_court(*image_points[i])
        results.append(LeaveOneOutResult(
            name=name, court_xy=tuple(court_points[i]), clicked_px=tuple(image_points[i]),
            predicted_px=pred_px,
            pixel_error=None if pred_px is None else math.dist(pred_px, image_points[i]),
            predicted_court_xy=pred_court,
            court_error_m=None if pred_court is None else math.dist(pred_court, court_points[i]),
            extrapolated=h.is_extrapolated(tuple(court_points[i])),
        ))
    return results


@dataclass(frozen=True)
class CourtCalibration:
    clip_id: str
    resolution: tuple[int, int]
    rotation_degrees: int
    point_names: tuple[str, ...]
    homography: CourtHomography
    image_points: tuple[tuple[float, float], ...] = ()  # the clicks, same order as point_names

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CourtCalibration:
        points = [p for p in data["points"] if p.get("pixel") is not None]
        unknown = [p["name"] for p in points if p["name"] not in COURT_POINTS]
        if unknown:
            raise CalibrationError(f"unknown court point name(s): {unknown}")
        names = tuple(p["name"] for p in points)
        pixels = tuple(tuple(p["pixel"]) for p in points)
        h = fit_homography(list(pixels), [COURT_POINTS[n] for n in names])
        return cls(clip_id=data["clip_id"], resolution=tuple(data["resolution"]),
                   rotation_degrees=int(data.get("rotation_degrees", 0)), point_names=names, homography=h,
                   image_points=pixels)


@dataclass(frozen=True)
class CalibrationErrorEstimate:
    error_m: float
    basis: str  # how it was derived, in plain words, for display next to the number
    nearest_point: str
    nearest_distance_m: float


# Inverse-distance-weighting exponent: power 2 is the conventional default
# (Shepard), so nearby leave-one-out points dominate; not tuned.
_IDW_POWER = 2.0


def calibration_error_at(court_xy: tuple[float, float],
                         loo: Sequence[LeaveOneOutResult]) -> CalibrationErrorEstimate | None:
    """The calibration's own error near a court position, interpolated from
    its leave-one-out results by inverse-distance weighting.

    Leave-one-out error at a clicked point is how far that point lands (in
    court metres) when the fit doesn't use it -- the calibration's measured
    error *at that spot*. Interpolating between those spots gives a local
    estimate: small near tightly clustered points, large near points the
    others predict poorly (e.g. far-off front-wall corners).

    Returns None when there's nothing to interpolate from (no computable
    leave-one-out errors). Positions outside the clicked region are
    extrapolated; there the estimate leans on the nearest points' errors and
    may understate the true error -- callers keep the extrapolated flag."""
    usable = [r for r in loo if r.court_error_m is not None]
    if not usable:
        return None
    dists = [math.dist(court_xy, r.court_xy) for r in usable]
    nearest = min(range(len(usable)), key=lambda i: dists[i])
    if dists[nearest] < 1e-9:
        return CalibrationErrorEstimate(usable[nearest].court_error_m, f"leave-one-out error at {usable[nearest].name}",
                                        usable[nearest].name, 0.0)
    weights = [1.0 / d ** _IDW_POWER for d in dists]
    error = sum(w * r.court_error_m for w, r in zip(weights, usable)) / sum(weights)
    return CalibrationErrorEstimate(
        error_m=error,
        basis=f"inverse-distance interpolation of {len(usable)} leave-one-out errors "
              f"(nearest: {usable[nearest].name}, {dists[nearest]:.2f} m away)",
        nearest_point=usable[nearest].name, nearest_distance_m=dists[nearest],
    )


def calibration_leave_one_out(calibration: CourtCalibration) -> list[LeaveOneOutResult] | None:
    """None when the calibration has too few points for leave-one-out (< 5):
    its error can't be estimated, and nothing should pretend otherwise."""
    if len(calibration.point_names) < 5 or len(calibration.image_points) != len(calibration.point_names):
        return None
    return leave_one_out(calibration.point_names, calibration.image_points,
                         [COURT_POINTS[n] for n in calibration.point_names])


@dataclass(frozen=True)
class FootCourtPosition:
    frame_index: int
    left_foot: tuple[float, float] | None
    right_foot: tuple[float, float] | None
    stance_midpoint: tuple[float, float] | None  # only when both feet are mapped
    extrapolated: bool  # any mapped point lies outside the calibrated region


_FEET = (PoseLandmarkName.LEFT_FOOT_INDEX, PoseLandmarkName.RIGHT_FOOT_INDEX)


def foot_court_positions(frames: Sequence[LandmarkFrame], calibration: CourtCalibration) -> list[FootCourtPosition]:
    """Per frame: each foot's toe landmark mapped to court metres, when the
    landmark passes the same visibility/presence gate the angle calculators
    use; None otherwise (never interpolated)."""
    h = calibration.homography
    out = []
    for frame in frames:
        mapped = []
        for name in _FEET:
            lm = frame.pose_landmarks.get(name)
            if lm is None or lm.visibility < MIN_LANDMARK_VISIBILITY or lm.presence < MIN_LANDMARK_PRESENCE:
                mapped.append(None)
            else:
                mapped.append(h.pixel_to_court(lm.position.x, lm.position.y))
        left, right = mapped
        mid = None if left is None or right is None else ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        extrapolated = any(p is not None and h.is_extrapolated(p) for p in (left, right))
        out.append(FootCourtPosition(frame.timing.frame_index, left, right, mid, extrapolated))
    return out
