"""Plot generation for the validation harness. Purely presentational: reads
debug_report values already produced by harness.py and draws them. No
engine logic, no new measurements."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.types.landmarks import PoseLandmarkName

_JOINTS = ("knee", "hip", "elbow", "shoulder", "ankle")
_MIDLINE_KEYS = ("trunk_inclination", "pelvis_rotation", "shoulder_rotation")
_KINEMATIC_METRICS = ("velocity", "acceleration", "jerk")


def _timestamps_seconds(frames) -> list[float]:
    return [frame.timing.timestamp_ms / 1000.0 for frame in frames]


def plot_joint_angles(debug_report: dict, output_dir: str) -> str:
    frames = debug_report["landmark_frames"]
    t = _timestamps_seconds(frames)
    angle_measurements = debug_report["angle_measurements"]

    fig, axes = plt.subplots(len(_JOINTS), 1, figsize=(10, 2.6 * len(_JOINTS)), sharex=True)
    for ax, joint in zip(axes, _JOINTS):
        for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
            measurements = angle_measurements.get(f"{joint}_{side}", ())
            ys = [m.angle_degrees if m.is_valid else float("nan") for m in measurements]
            ax.plot(t, ys, label=side, color=color, linewidth=1)
        ax.set_ylabel(f"{joint}\n(deg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Joint angles over time (gaps = invalid/unresolved frames)")
    fig.tight_layout()
    path = os.path.join(output_dir, "joint_angles.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_midline_angles(debug_report: dict, output_dir: str) -> str:
    frames = debug_report["landmark_frames"]
    t = _timestamps_seconds(frames)
    angle_measurements = debug_report["angle_measurements"]

    fig, axes = plt.subplots(len(_MIDLINE_KEYS), 1, figsize=(10, 2.6 * len(_MIDLINE_KEYS)), sharex=True)
    for ax, key in zip(axes, _MIDLINE_KEYS):
        measurements = angle_measurements.get(key, ())
        ys = [m.angle_degrees if m.is_valid else float("nan") for m in measurements]
        ax.plot(t, ys, color="tab:green", linewidth=1)
        ax.set_ylabel(f"{key}\n(deg)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Midline rotation / inclination over time")
    fig.tight_layout()
    path = os.path.join(output_dir, "midline_angles.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_wrist_kinematics(debug_report: dict, output_dir: str) -> str:
    frames = debug_report["landmark_frames"]
    t = _timestamps_seconds(frames)
    kinematics = debug_report["kinematics"]

    fig, axes = plt.subplots(len(_KINEMATIC_METRICS), 1, figsize=(10, 3 * len(_KINEMATIC_METRICS)), sharex=True)
    for ax, metric in zip(axes, _KINEMATIC_METRICS):
        for wrist, color in (("left_wrist", "tab:blue"), ("right_wrist", "tab:orange")):
            samples = kinematics.get(wrist, {}).get(metric, ())
            ys = []
            for sample in samples:
                value = getattr(sample, metric, None)
                if not sample.is_valid or value is None:
                    ys.append(float("nan"))
                else:
                    ys.append((value.x**2 + value.y**2 + value.z**2) ** 0.5)
            ax.plot(t, ys, label=wrist, color=color, linewidth=1)
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Wrist kinematics magnitude (pixels/s, /s^2, /s^3 -- uncalibrated units)")
    fig.tight_layout()
    path = os.path.join(output_dir, "wrist_kinematics.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_posture(debug_report: dict, output_dir: str) -> str:
    frames = debug_report["landmark_frames"]
    t = _timestamps_seconds(frames)
    posture = debug_report["posture"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    com_ys = [
        m.height_ratio if m.is_valid and m.height_ratio is not None else float("nan")
        for m in posture["center_of_mass"]
    ]
    axes[0].plot(t, com_ys, color="tab:purple", linewidth=1)
    axes[0].set_ylabel("COM height_ratio\n(0=ankle, 1=head)")
    axes[0].grid(alpha=0.3)

    wt_ys = [
        m.right_foot_ratio if m.is_valid and m.right_foot_ratio is not None else float("nan")
        for m in posture["weight_transfer"]
    ]
    axes[1].plot(t, wt_ys, color="tab:red", linewidth=1)
    axes[1].axhline(0.0, color="gray", linewidth=0.5)
    axes[1].axhline(1.0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("weight transfer\n(0=left foot, 1=right foot)")
    axes[1].grid(alpha=0.3)

    hs_ys = [
        m.normalized_displacement if m.is_valid and m.normalized_displacement is not None else float("nan")
        for m in posture["head_stability"]
    ]
    axes[2].plot(t, hs_ys, color="tab:brown", linewidth=1)
    axes[2].set_ylabel("head stability\n(displacement / shoulder width)")
    axes[2].grid(alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Posture measurements over time")
    fig.tight_layout()
    path = os.path.join(output_dir, "posture.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confidence_heatmap(debug_report: dict, output_dir: str) -> str:
    frames = debug_report["landmark_frames"]
    names = list(PoseLandmarkName)
    matrix = np.full((len(names), len(frames)), np.nan)
    for col, frame in enumerate(frames):
        for row, name in enumerate(names):
            landmark = frame.pose_landmarks.get(name)
            if landmark is not None:
                matrix[row, col] = landmark.visibility

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="black")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([name.value for name in names], fontsize=8)
    ax.set_xlabel("frame index")
    ax.set_title("Per-landmark visibility over time (black = not tracked / occluded beyond recovery budget)")
    fig.colorbar(im, ax=ax, label="visibility")
    fig.tight_layout()
    path = os.path.join(output_dir, "landmark_confidence_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_plots(debug_report: dict, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    generators = (
        lambda: plot_joint_angles(debug_report, output_dir),
        lambda: plot_midline_angles(debug_report, output_dir),
        lambda: plot_wrist_kinematics(debug_report, output_dir),
        lambda: plot_posture(debug_report, output_dir),
        lambda: plot_confidence_heatmap(debug_report, output_dir),
    )
    paths = []
    for generate in generators:
        try:
            paths.append(generate())
        except Exception as exc:  # a plot failure shouldn't take down the rest of the validation run
            print(f"Warning: a plot failed to generate ({exc})")
    return paths
