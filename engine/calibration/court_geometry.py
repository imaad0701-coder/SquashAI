"""Squash singles court floor geometry, in metres, for floor-plane calibration.

Coordinate frame (court space):
  - origin at the front-left floor corner, standing on court facing the
    front wall
  - x across the court, left side wall (0) to right side wall (COURT_WIDTH_M)
  - y from the front wall (0) back to the back wall (COURT_LENGTH_M)

Dimensions follow the World Squash Federation singles court specification:
floor 9.75 m x 6.40 m, short line 5.44 m from the front wall to the line's
rear edge, service boxes 1.60 m square measured to the outer edges of their
lines, lines 50 mm wide. Named points sit on line *centres* (what you click
on video), so line-edge conventions shift them by 25 mm -- far below what a
click on match footage can resolve, and stated here so nobody mistakes the
third decimal for precision.

Only floor points are defined: a floor homography maps the floor plane
only. Wall lines (front-wall out line, tin, service line) are off that plane
and cannot be placed with it.
"""

from __future__ import annotations

from typing import Final

COURT_LENGTH_M: Final[float] = 9.75
COURT_WIDTH_M: Final[float] = 6.40
LINE_WIDTH_M: Final[float] = 0.05
SHORT_LINE_REAR_EDGE_M: Final[float] = 5.44
SERVICE_BOX_SIZE_M: Final[float] = 1.60

_HALF_LINE = LINE_WIDTH_M / 2
SHORT_LINE_Y: Final[float] = SHORT_LINE_REAR_EDGE_M - _HALF_LINE
HALF_COURT_X: Final[float] = COURT_WIDTH_M / 2
BOX_INNER_LEFT_X: Final[float] = SERVICE_BOX_SIZE_M - _HALF_LINE
BOX_INNER_RIGHT_X: Final[float] = COURT_WIDTH_M - SERVICE_BOX_SIZE_M + _HALF_LINE
BOX_BACK_Y: Final[float] = SHORT_LINE_REAR_EDGE_M + SERVICE_BOX_SIZE_M - _HALF_LINE

# Clickable floor landmarks: where floor lines meet each other or a wall.
COURT_POINTS: Final[dict[str, tuple[float, float]]] = {
    "front_left_corner": (0.0, 0.0),
    "front_right_corner": (COURT_WIDTH_M, 0.0),
    "short_line_left_wall": (0.0, SHORT_LINE_Y),
    "left_box_front_inner": (BOX_INNER_LEFT_X, SHORT_LINE_Y),
    "t_junction": (HALF_COURT_X, SHORT_LINE_Y),
    "right_box_front_inner": (BOX_INNER_RIGHT_X, SHORT_LINE_Y),
    "short_line_right_wall": (COURT_WIDTH_M, SHORT_LINE_Y),
    "left_box_back_wall": (0.0, BOX_BACK_Y),
    "left_box_back_inner": (BOX_INNER_LEFT_X, BOX_BACK_Y),
    "right_box_back_inner": (BOX_INNER_RIGHT_X, BOX_BACK_Y),
    "right_box_back_wall": (COURT_WIDTH_M, BOX_BACK_Y),
    "back_left_corner": (0.0, COURT_LENGTH_M),
    "half_court_line_back_wall": (HALF_COURT_X, COURT_LENGTH_M),
    "back_right_corner": (COURT_WIDTH_M, COURT_LENGTH_M),
}

# Floor line segments, for drawing the court back onto video.
COURT_LINES: Final[dict[str, tuple[tuple[float, float], tuple[float, float]]]] = {
    "front_wall_base": ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),
    "left_wall_base": ((0.0, 0.0), (0.0, COURT_LENGTH_M)),
    "right_wall_base": ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),
    "back_wall_base": ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),
    "short_line": ((0.0, SHORT_LINE_Y), (COURT_WIDTH_M, SHORT_LINE_Y)),
    "half_court_line": ((HALF_COURT_X, SHORT_LINE_Y), (HALF_COURT_X, COURT_LENGTH_M)),
    "left_box_inner": ((BOX_INNER_LEFT_X, SHORT_LINE_Y), (BOX_INNER_LEFT_X, BOX_BACK_Y)),
    "left_box_back": ((0.0, BOX_BACK_Y), (BOX_INNER_LEFT_X, BOX_BACK_Y)),
    "right_box_inner": ((BOX_INNER_RIGHT_X, SHORT_LINE_Y), (BOX_INNER_RIGHT_X, BOX_BACK_Y)),
    "right_box_back": ((BOX_INNER_RIGHT_X, BOX_BACK_Y), (COURT_WIDTH_M, BOX_BACK_Y)),
}
