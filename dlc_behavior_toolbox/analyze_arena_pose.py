""" 
A script to analyze mouse behavior in an open field arena tracked by DeepLabCut (DLC). 
We expect tracked DLC bodyparts of the following categories:

1. Arena (floor) corners:

A----------------B
|                |
|      >o<      ,|
|       O      -o|
|       |       `|
|                |
|                |
C----------------D

Thus, the corner edge points are counted clockwise starting from the top left corner (A).
Please stick to this convention and the corner names "A", "B", "C", and "D".

2. Mouse points: 

   >O<  ⟵ nose, ear_left, ear_right, head_center
    O
   OOO  ⟵ center (AT LEAST)
    O    
    |   ⟵ tailbase
    L

We expect that at least one body part of the mouse (usually, the body center) is tracked 
by DLC. Additionally, you can also track other body parts, such as the nose, ears, and 
tailbase. Your are free to define the names. 

3. LED (optional):

   ,
  -o  ⟵ led 1, led 2, ...
   `
OPTIONALLY, you can also track LED lights, which, e.g., indicate 2P miniscope imaging, 
optogenetic stimulation, a behavioral stimulation.


author: Fabrizio Musacchio
date:   Aug 5, 2025
"""
# %% IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Ellipse
from itertools import cycle
import matplotlib as mpl
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

import cv2

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATH AND PARAMETERS (ADJUST HERE)
# set your data and results paths here:
#DATA_PATH = "/Users/husker/Workspace/Pinky Camp/OF/"
#RESULTS_PATH = "/Users/husker/Workspace/Pinky Camp/OF/DLC_analysis/"
DATA_PATH = "/Users/husker/Workspace/Henrike DLC/NOR/Trial 2/"
DATA_PATH = "/Users/husker/Workspace/Henrike DLC/cFC/recall/"
#DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/OF DLC/"
# DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/NOR DLC/Hab/"
# DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/NOR DLC/Trial 2/"
# DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/SOR DLC/Hab/"
#DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/SOR DLC/Trial 1/"
# DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/EMP DLC/"
#DATA_PATH = "/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/cFC DLC/recall/"
RESULTS_PATH = os.path.join(os.path.dirname(DATA_PATH), "DLC_analysis")
os.makedirs(RESULTS_PATH, exist_ok=True)
# DATA_PATH = "/Users/husker/Workspace/Denise DLC/cFC 2025/"
# RESULTS_PATH = "/Users/husker/Workspace/Denise DLC/cFC 2025/DLC_analysis/"

# define frame rate and time step:
frame_rate = 30  # fps
time_step = 1 / frame_rate

# arena size (assumed to be square):
arena_size = 49  # cm; adjust this to the size of your arena in cm
arena_size = 25 # our cFC box

# define the size of a pixel (if available):
spatial_unit="cm"

# define whether to cut tracking data after mouse_first_track_delay seconds:
cut_tracking = 360 # False: no cut; otherwise, define number of seconds to define total tracking duration
                   # (rest will be cut)
#cut_tracking = 300 # NOR Hab Plastic, SOR all
cut_tracking = 306 # cFC Plastic
#cut_tracking = 420 # OF Plastic
#cut_tracking = 600 # NOR Trial 1+2 Plastic
mouse_first_track_delay = 2 # define the delay after which tracking starts (in seconds)
                            # i.e., after 'mouse_first_track_delay' seconds the mouse was first
                            # detected, we cut the tracking data to start from this point onward
mouse_first_track_delay = 5
# define likelihood threshold for valid points:
likelihood_threshold = 0.9 # this likelihood refers to the DLC assigned likelihood
                           # for each bodypart; it is a value between 0 and 1, where
                           # 1 means "very likely" and 0 means "not likely at all"
                           # "likely" roughly means "reliable"; by adjusting this 
                           # threshold, you can filter out low-confidence points.
likelihood_threshold = 0.3
# define a switch whether to only use data where the LED light is on:
use_LED_light = False  # if True, only use data where the LED light is on
                     # if False, use all data regardless of the LED light status

# define whether to plot the tracked body part trajectory as a scatter plot (True) or as a line (False):
heat_plot_scatter = True

# define a threshold for movement detection:
movement_threshold = 0.5  # px/frame; note, if you set pixel_size to 1, this is in px/s
                         # if you set pixel_size to a value other than 1, this is in spatial_unit/s
                         # this threshold is used to determine whether a body part is moving or not
                         # if the velocity is above this threshold, the body part is considered to be moving
                         # if the velocity is below this threshold, the body part is considered to be not moving

ylim = 60 # set to a value, e.g., 1000, for fixed scaling
ylim_smoothed = 30 # set to a value, e.g., 1000, for fixed scaling in smoothed velocity plots

use_filtered_data = True # True if you want to use DLC's filtered output (if available) or 
                         # False for the unfiltered one

# for a default OF experiment, define the distance of the border-/center-region border:
border_margin = 10.0  # in spatial_unit (e.g., 10 for OF/50x50 cm box or 5 for cFC/25x25 cm box)

# freezing detection parameters: 
freeze_speed_threshold = movement_threshold  # set to movement_threshold or any other values
freeze_min_duration_s = 0.5  # 1.0 or 2.0 s or any other limit according your protocol
freeze_smooth_win = int(round(0.25 * frame_rate))  # (optional, 250 ms smoothing)
freeze_gap_merge_max_s = 0.25 # the maximum duration of a gap to be merged (in seconds) between freezing bouts

# define subject's bodyparts to analyze:
# keys are your own labels, values are the DLC column bodypart names
mouse_center_point_DLC_bp = {
    'center point': 'centerpoint'} 
""" mouse_center_point_DLC_bp = {
    'center point': 'center', # never change the KEY of this line, just its VALUE
    'tailbase': 'tailbase',   # all subsequent KEYS can be determined by yourself
    'ear_L': 'ear_L',
    'ear_R': 'ear_R',
    'headholder': 'headholder',
} """
mouse_center_point_DLC_bp = {  # Plastik
    'center point': 'centerpoint',
    'nose': 'nose',
    'headcenter': 'headcenter',
    # 'tailbase': 'tailbase',  
    # 'ear_L': 'ear_L',
    # 'ear_R': 'ear_R'
    }
mouse_center_point_DLC_bp = {  # Henrike OF
    'center point': 'centerpoint',
    'nose': 'nose',
    'tailbase': 'tail',
    }
mouse_center_point_DLC_bp = {  # Henrike NOR
    'center point': 'Center Point',
    'nose': 'Nose',
    'tailbase': 'Tail Base',
    'headcenter': 'Head Center',
    }
mouse_center_point_DLC_bp = {  # Henrike cFC
    'center point': 'center point',
    'ear_L': 'ear1',
    'ear_R': 'ear2',
    'tailbase': 'tail base',
    'nose': 'nose',
    'headcenter': 'head center',
    }

""" Multi-animal tracking:
in case of a multi-animal DLC tracking, provide the name of the actual subject. At the
moment, we can not process multiple animals.
"""
exp_mouse_key = "exp_mouse"
# exp_mouse_key = "exp_mpuse" # for EPM


# set arena mode: "rectangle" (default OF/NOR/SOR/cFC) or "epm" (Elevated Plus Maze)
arena_mode = "rectangle"
# arena_mode = "epm"   # set to "epm" for EPM datasets
epm_bp_list = ["headcenter", "centerpoint", "tailbase"] # list of bodyparts to analyze in EPM mode
                                                        # IMPORTANT: These are DLC bodypart NAMES, not your own labels!
# EPM only: how to combine the bp-wise ROI masks into a single "mouse in ROI" mask. Options:
#   "all"  -> all bodyparts from epm_bp_list must be in the same ROI (strict)
#   "kofn" -> at least epm_k_of_n bodyparts must be in the same ROI (more robust)
epm_aggregate_mode = "all"
epm_k_of_n = 2  # used only if epm_aggregate_mode == "kofn"
# EPM: add a synthetic "multi bodypart" entry to the analysis list
if arena_mode.lower().strip() == "epm":
    # This creates an extra row in the measurements table.
    # We will treat it specially in MAIN PROCESSING.
    if "EPM_multi_bp" not in mouse_center_point_DLC_bp:
        mouse_center_point_DLC_bp["EPM_multi_bp"] = epm_bp_list[0]  # placeholder bp name for plotting if needed


# define rectangle arena corners (adjust here, if necessary):
arena_corners_DLC_bp = {
    'top left corner':      'A',
    'top right corner':     'B',
    'bottom left corner':   'C',
    'bottom right corner':  'D'} # adjust the names to the DLC body parts that represent the corners of the arena;


# EPM marker definitions:
# outer arm corners: A...H (clockwise starting at upper arm's top-left; adjust if necessary):
EPM_outer_markers = ["A", "B", "C", "D", "E", "F", "G", "H"]
# center square corners: I..L (clockwise starting at top-left; adjust if necessary):
EPM_center_markers = ["I", "J", "K", "L"]
# override arena corners if in EPM mode:
if arena_mode.lower().strip() == "epm":
    # We use all outer markers A...H to define the transform reference (min-area rectangle):
    arena_corners_DLC_bp = {f"outer_{bp}": bp for bp in EPM_outer_markers}

LED_lights_DLC_bp = {
    'LED light': 'LED_2P'} # adjust 'led' to the name of the DLC body part that represents the LED light;

# define objects (adjust here, if any; otherwise, comment out):
object_zone_margin = 4.0  # in spatial_unit (e.g., cm). Expands the object shape outward.
objects_DLC_bp_list = [
    {
        "name": "object_A",
        "shape": "polygon",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top left corner": "O1_A",
        "top right corner": "O1_B",
        "bottom left corner": "O1_C",
        "bottom right corner": "O1_D",
    },
    {
        "name": "object_B",
        "shape": "polygon",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top left corner": "O2_A",
        "top right corner": "O2_B",
        "bottom left corner": "O2_C",
        "bottom right corner": "O2_D",
    },
    # {
    #     "name": "object_B",
    #     "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
    #     "top point": "O2_A",
    #     "right point": "O2_B",
    #     "bottom point": "O2_C",
    #     "left point": "O2_D",
    # },
]
objects_DLC_bp_list = [
    {
        "name": "object_A",
        "shape": "polygon",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top left corner": "O1_A",
        "top right corner": "O1_B",
        "bottom left corner": "O1_C",
        "bottom right corner": "O1_D",
    },
    # {
    #     "name": "object_C",
    #     "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
    #     "top point": "O3_A",
    #     "right point": "O3_B",
    #     "bottom point": "O3_C",
    #     "left point": "O3_D",
    # },
    {
        "name": "object_D",
        "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top point": "O4_A",
        "right point": "O4_B",
        "bottom point": "O4_C",
        "left point": "O4_D",
    },
]
# SOR Plastik:
objects_DLC_bp_list = [
    {
        "name": "object_A", # empty left
        "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top point": "O1_A",
        "right point": "O1_B",
        "bottom point": "O1_C",
        "left point": "O1_4",
    },
    {
        "name": "object_C", # empty right / new right
        "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top point": "O2_A",
        "right point": "O2_B",
        "bottom point": "O2_C",
        "left point": "O2_D",
    },
    # {
    #     "name": "object_D", # new left
    #     "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
    #     "top point": "O3_A",
    #     "right point": "O3_B",
    #     "bottom point": "O3_C",
    #     "left point": "O3_D",
    # },
]
# NOR Henrike:
objects_DLC_bp_list = [
    {
        "name": "object_A",
        "shape": "polygon",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top left corner": "O1_A",
        "top right corner": "O1_B",
        "bottom left corner": "O1_C",
        "bottom right corner": "O1_D",
    },
    # {
    #     "name": "object_B",
    #     "shape": "polygon",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
    #     "top left corner": "O2_A",
    #     "top right corner": "O2_B",
    #     "bottom left corner": "O2_C",
    #     "bottom right corner": "O2_D",
    # },
    {
        "name": "object_B", # empty right / new right
        "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
        "top point": "O2_A",
        "right point": "O2_B",
        "bottom point": "O2_C",
        "left point": "O2_D",
    },
    # {
    #     "name": "object_D", # new left
    #     "shape": "ellipse",  # accepted: polygon (for "boxes") or ellipse (for "cylinders")
    #     "top point": "O3_A",
    #     "right point": "O3_B",
    #     "bottom point": "O3_C",
    #     "left point": "O3_D",
    # },
]
objects_DLC_bp_list = None # for OF and EPM and cFC only


# in EPM mode, treat arms + center as "objects" (polygons) for zone metrics.
# note: object_zone_margin should usually be 0 for EPM zones!
if arena_mode.lower().strip() == "epm":
    object_zone_margin = 0.0
    # replace / override objects_DLC_bp_list with EPM zones:
    objects_DLC_bp_list = [
        {"name": "center",   "shape": "polygon",
         "top left corner": "I", "top right corner": "J", "bottom right corner": "K", "bottom left corner": "L"},
        {"name": "arm_up",   "shape": "polygon",
         "top left corner": "A", "top right corner": "B", "bottom right corner": "J", "bottom left corner": "I"},
        {"name": "arm_right","shape": "polygon",
         "top left corner": "J", "top right corner": "C", "bottom right corner": "D", "bottom left corner": "K"},
        {"name": "arm_down", "shape": "polygon",
         "top left corner": "L", "top right corner": "K", "bottom right corner": "F", "bottom left corner": "E"},
        {"name": "arm_left", "shape": "polygon",
         "top left corner": "G", "top right corner": "I", "bottom right corner": "L", "bottom left corner": "H"}]
# %% FUNCTIONS
# define a function to order points counterclockwise (e.g., for creating polygons):
def _order_points_ccw(pts: np.ndarray) -> np.ndarray:
    """
    Order points counterclockwise around centroid.
    This prevents self-intersecting quadrilateral plots if corner labels are swapped.
    """
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]

# define a function to plot arena corners and chosen subject bodypart:
def plot_arena_corners_and_mouse(df_cleaned,
                                 arena_corners_DLC_bp,
                                 mouse_bp_name: str,
                                 curr_filename_clean,
                                 curr_results_path,
                                 objects_DLC_bp_list=None,
                                 plot_object_shapes: bool = True):
    """
    Calculate arena corners and plot mouse center points and arena corners.
    Optionally also compute and plot mean object marker positions (raw coordinates).

    Returns
    -------
    arena_corners_raw : dict
        corner_name -> (x, y)
    objects_raw : list[dict]
        List of objects with mean marker coordinates in raw space.
        Empty list if objects_DLC_bp_list is None or markers missing.
    """
    # ---- arena corners ----
    arena_corners_raw = {}
    for corner_name, corner_bp in arena_corners_DLC_bp.items():
        # corner_x = df_cleaned[(corner_bp, 'x')].mean()
        # corner_y = df_cleaned[(corner_bp, 'y')].mean()
        corner_x = df_cleaned[(corner_bp, 'x')].median()
        corner_y = df_cleaned[(corner_bp, 'y')].median()
        arena_corners_raw[corner_name] = (float(corner_x), float(corner_y))

    # ---- objects (raw) ----
    objects_raw = []
    if objects_DLC_bp_list:
        for obj in objects_DLC_bp_list:
            name = obj.get("name", "object")
            shape = obj.get("shape", "").lower().strip()

            marker_means = {}
            missing_any = False

            for k, bp in obj.items():
                if k in {"name", "shape"}:
                    continue
                if (bp, "x") not in df_cleaned.columns or (bp, "y") not in df_cleaned.columns:
                    missing_any = True
                    break
                # mx = df_cleaned[(bp, "x")].mean()
                # my = df_cleaned[(bp, "y")].mean()
                mx = df_cleaned[(bp, "x")].median()
                my = df_cleaned[(bp, "y")].median()
                if not np.isfinite(mx) or not np.isfinite(my):
                    missing_any = True
                    break
                marker_means[k] = (float(mx), float(my))

            if missing_any or len(marker_means) == 0:
                continue

            geom = None
            if plot_object_shapes:
                if shape == "polygon":
                    needed = {"top left corner", "top right corner", "bottom left corner", "bottom right corner"}
                    if needed.issubset(marker_means.keys()):
                        tl = marker_means["top left corner"]
                        tr = marker_means["top right corner"]
                        bl = marker_means["bottom left corner"]
                        br = marker_means["bottom right corner"]

                        pts = np.array([tl, tr, br, bl], dtype=float)
                        pts = _order_points_ccw(pts)
                        geom = {"shape": "polygon", "xy": pts}

                elif shape == "ellipse":
                    needed = {"top point", "right point", "bottom point", "left point"}
                    if needed.issubset(marker_means.keys()):
                        # Ellipse aus vier Extrempunkten (axis-aligned)
                        top = marker_means["top point"]
                        right = marker_means["right point"]
                        bottom = marker_means["bottom point"]
                        left = marker_means["left point"]

                        cx = float(0.5 * (left[0] + right[0]))
                        cy = float(0.5 * (top[1] + bottom[1]))
                        a  = float(0.5 * abs(right[0] - left[0]))   # semi-axis x
                        b  = float(0.5 * abs(bottom[1] - top[1]))   # semi-axis y

                        geom = {"shape": "ellipse", "cx": cx, "cy": cy, "a": a, "b": b}


            objects_raw.append({
                "name": name,
                "shape": shape,
                "markers": marker_means,
                "geom": geom})

    # ---- plot ----
    plt.figure(figsize=(14, 8))
    plt.scatter(df_cleaned[(mouse_bp_name, "x")],
                df_cleaned[(mouse_bp_name, "y")],
                s=10, alpha=0.5, label=f"mouse points ({mouse_bp_name})")

    for corner_name, (corner_x, corner_y) in arena_corners_raw.items():
        plt.scatter(corner_x, corner_y, s=100, label=f"arena {corner_name}",
                    edgecolor="black", alpha=0.5)

    # plot object marker means and optional shapes:
    if objects_raw:
        for obj in objects_raw:
            name = obj["name"]

            for mk, (mx, my) in obj["markers"].items():
                plt.scatter(mx, my, s=80, alpha=0.8, edgecolor="black",
                            label=f"{name} {mk}")

            if plot_object_shapes and (obj["geom"] is not None):
                g = obj["geom"]

                if g["shape"] == "polygon":
                    xy = g["xy"]
                    xy_closed = np.vstack([xy, xy[0]])
                    plt.plot(xy_closed[:, 0], xy_closed[:, 1], linewidth=2, alpha=0.9)
                    plt.text(float(xy[:, 0].mean()), float(xy[:, 1].mean()), name, fontsize=11)

                elif g["shape"] == "ellipse":
                    plt.gca().add_patch(
                        Ellipse((g["cx"], g["cy"]),
                                width=2.0 * g["a"], height=2.0 * g["b"],
                                fill=False, linewidth=2, alpha=0.9))
                    plt.text(g["cx"] + g["a"], g["cy"] + g["b"], name, fontsize=11)


    plt.title(f"raw mouse center points (before transformation)\n{curr_filename_clean}")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_tracks_raw.pdf"), dpi=300)
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_tracks_raw.png"),
                dpi=300, transparent=True)
    plt.close()

    return arena_corners_raw, objects_raw

# define a function for checking window stability:
def window_is_stable(was_in: np.ndarray,
                     start_i: int,
                     win_frames: int,
                     max_false_frac: float,
                     max_false_streak: int) -> bool:
    """Check whether the segment [start_i, start_i+win_frames) is stable enough."""
    seg = was_in[start_i : start_i + win_frames]
    if seg.size < win_frames:
        return False

    false_frac = 1.0 - (seg.sum() / float(win_frames))
    if false_frac > max_false_frac:
        return False

    longest_false = 0
    curr = 0
    for v in seg:
        if not v:
            curr += 1
            if curr > longest_false:
                longest_false = curr
        else:
            curr = 0

    if longest_false > max_false_streak:
        return False

    return True

# define a function for extracting freezing parameters:
def extract_bouts(mask: np.ndarray,
                  frame_rate: float,
                  min_duration_s: float = 1.0,
                  freeze_gap_merge_max_s: float = 0.25):
    """
    Extract bouts (continuous True segments) from a boolean mask, with optional
    tolerance for short False gaps (merged into surrounding True).

    Parameters
    ----------
    mask : np.ndarray
        Boolean array (e.g. is_freeze per frame).
    frame_rate : float
        Frames per second.
    min_duration_s : float
        Minimum duration for a bout to count (seconds).
    freeze_gap_merge_max_s : float
        Merge/ignore False gaps shorter or equal to this duration (seconds).
        Use 0 to disable (strict consecutive bouts).

    Returns
    -------
    bouts : list of dict
        Each dict has {'start_frame','end_frame','start_time_s','end_time_s','duration_s'}.
    summary : dict
        {'total_freezing_time_in_s','num_freeze_bouts','mean_freeze_bout_s','median_freeze_bout_s'}
    """
    mask = np.asarray(mask, dtype=bool).copy()
    n = mask.size
    if n == 0:
        return [], {
            'total_freezing_time_in_s': 0.0,
            'num_freeze_bouts': 0,
            'mean_freeze_bout_s': np.nan,
            'median_freeze_bout_s': np.nan,
        }

    # 1) kurze False-Lücken optional "auffüllen"
    gap_frames = int(round(freeze_gap_merge_max_s * frame_rate))
    if gap_frames > 0:
        val = mask
        diff = np.diff(val.astype(int))
        # False-Run = Übergang True->False (start) bis False->True (end)
        false_starts = np.where(diff == -1)[0] + 1
        false_ends   = np.where(diff ==  1)[0] + 1
        if not val[0]:
            false_starts = np.insert(false_starts, 0, 0)
        if not val[-1]:
            false_ends = np.append(false_ends, n)
        for s, e in zip(false_starts, false_ends):
            if (e - s) <= gap_frames:
                val[s:e] = True
        mask = val

    # 2) Bouts aus dem (ggf. gefüllten) Maskensignal holen
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, n)

    bouts = []
    for s, e in zip(starts, ends):
        dur = (e - s) / frame_rate
        if dur >= min_duration_s:
            bouts.append({
                'start_frame': s,
                'end_frame': e - 1,
                'start_time_s': s / frame_rate,
                'end_time_s': (e - 1) / frame_rate,
                'duration_s': dur
            })

    if bouts:
        durations = [b['duration_s'] for b in bouts]
        summary = {
            'total_freezing_time_in_s': float(np.sum(durations)),
            'num_freeze_bouts': len(durations),
            'mean_freeze_bout_s': float(np.mean(durations)),
            'median_freeze_bout_s': float(np.median(durations)),
        }
    else:
        summary = {
            'total_freezing_time_in_s': 0.0,
            'num_freeze_bouts': 0,
            'mean_freeze_bout_s': np.nan,
            'median_freeze_bout_s': np.nan,
        }

    return bouts, summary

def extract_bouts_strict(mask: np.ndarray, frame_rate: float, min_duration_s: float = 1.0):
    """
    Extract bouts (continuous True segments) from a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array (e.g. is_freeze per frame).
    frame_rate : float
        Frames per second.
    min_duration_s : float
        Minimum duration for a bout to count (seconds).

    Returns
    -------
    bouts : list of dict
        Each dict has {'start_frame', 'end_frame', 'start_time_s', 'end_time_s', 'duration_s'}.
    summary : dict
        Metrics: total_freezing_time, num_bouts, mean_bout, median_bout.
    """
    bouts = []
    min_len = int(round(min_duration_s * frame_rate))
    mask = mask.astype(bool)

    # detect edges
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1

    # handle open bout at start or end
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    # collect bouts
    for s, e in zip(starts, ends):
        dur = (e - s) / frame_rate
        if dur >= min_duration_s:
            bouts.append({
                'start_frame': s,
                'end_frame': e-1,
                'start_time_s': s / frame_rate,
                'end_time_s': (e-1) / frame_rate,
                'duration_s': dur
            })

    # summary
    if bouts:
        durations = [b['duration_s'] for b in bouts]
        summary = {
            'total_freezing_time_in_s': np.sum(durations),
            'num_freeze_bouts': len(durations),
            'mean_freeze_bout_s': float(np.mean(durations)),
            'median_freeze_bout_s': float(np.median(durations)),
        }
    else:
        summary = {
            'total_freezing_time_in_s': 0.0,
            'num_freeze_bouts': 0,
            'mean_freeze_bout_s': np.nan,
            'median_freeze_bout_s': np.nan,
        }

    return bouts, summary
# %% FUNCTIONS (NEW FOR OBJECT ZONES ANALYSIS)
def _pt_mean_xy(df: pd.DataFrame, bp: str) -> tuple[float, float]:
    """Return mean x,y for a bodypart across frames (ignoring NaNs)."""
    x = df[(bp, "x")].mean()
    y = df[(bp, "y")].mean()
    return float(x), float(y)

def _transform_point(M: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    """Perspective-transform a single point."""
    arr = np.array([[list(xy)]], dtype=np.float32)  # shape (1,1,2)
    tr = cv2.perspectiveTransform(arr, M)[0][0]
    return float(tr[0]), float(tr[1])

def transform_objects_raw(objects_raw: list[dict], M: np.ndarray) -> list[dict]:
    """
    Apply perspective transform to objects_raw marker means.
    Returns a new list (does not mutate input).
    """
    if not objects_raw:
        return []

    out = []
    for obj in objects_raw:
        markers_tr = {}
        for mk, xy in obj["markers"].items():
            markers_tr[mk] = _transform_point(M, xy)

        out.append({
            "name": obj.get("name", "object"),
            "shape": obj.get("shape", "").lower().strip(),
            "markers": markers_tr,
        })
    return out

def _polygon_contains_points_DELET_LATER(poly_xy: np.ndarray, 
                             x: np.ndarray, 
                             y: np.ndarray) -> np.ndarray:
    """
    Point-in-polygon test using OpenCV.
    poly_xy: (N,2) float array
    x,y: arrays of same length
    Returns boolean array.
    """
    poly = poly_xy.astype(np.float32).reshape((-1, 1, 2))
    pts = np.vstack([x, y]).T.astype(np.float32).reshape((-1, 1, 2))

    out = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        # p has shape (1,2) here
        px = float(p[0, 0])
        py = float(p[0, 1])
        out[i] = cv2.pointPolygonTest(poly, (px, py), False) >= 0
    return out

def _polygon_contains_points(poly_xy: np.ndarray, 
                             x: np.ndarray, 
                             y: np.ndarray) -> np.ndarray:
    """ 
    Point-in-polygon test using matplotlib.path.Path.
    poly_xy: (N,2) float array
    x,y: arrays of same length
    Returns boolean array.
    """
    poly_xy = np.asarray(poly_xy, dtype=float)
    pts = np.column_stack([x, y]).astype(float)
    return Path(poly_xy).contains_points(pts)

def _ellipse_params_from_markers(markers: dict) -> tuple[float, float, float, float]:
    top    = markers["top point"]
    right  = markers["right point"]
    bottom = markers["bottom point"]
    left   = markers["left point"]

    cx = 0.5 * (left[0] + right[0])
    cy = 0.5 * (top[1] + bottom[1])
    a  = 0.5 * abs(right[0] - left[0])   # semi-axis in x
    b  = 0.5 * abs(bottom[1] - top[1])   # semi-axis in y
    return float(cx), float(cy), float(a), float(b)

def _ellipse_contains_points(cx: float, cy: float, a: float, b: float,
                            x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # avoid division by zero
    if a <= 0 or b <= 0:
        return np.zeros_like(x, dtype=bool)
    dx = (x - cx) / a
    dy = (y - cy) / b
    return (dx * dx + dy * dy) <= 1.0

def _scale_polygon_about_centroid(xy: np.ndarray, margin: float) -> np.ndarray:
    """
    Approximate polygon offset by scaling about centroid.
    margin is in same units as xy coordinates.
    We compute a scale factor s = 1 + margin / r_mean where r_mean is mean radius from centroid.
    """
    xy = np.asarray(xy, dtype=float)
    c = xy.mean(axis=0)
    r = np.sqrt(((xy - c) ** 2).sum(axis=1))
    r_mean = float(np.nanmean(r))
    if not np.isfinite(r_mean) or r_mean <= 1e-9:
        return xy.copy()
    s = 1.0 + float(margin) / r_mean
    return c + s * (xy - c)

def build_objects_geometry_from_raw(objects_raw_transformed: list[dict],
                                   object_zone_margin: float = 0.0,
                                   arena_size: float | None = None) -> list[dict]:
    """
    Build object geometries (inner + zone) from transformed objects_raw marker means.
    Only accepts shapes: 'polygon' and 'ellipse'.
    """
    if not objects_raw_transformed:
        return []

    objects_geom: list[dict] = []

    for obj in objects_raw_transformed:
        name = obj.get("name", "object")
        shape = obj.get("shape", "").lower().strip()
        markers = obj.get("markers", {})

        if shape == "polygon":
            needed = {"top left corner", "top right corner", "bottom left corner", "bottom right corner"}
            if not needed.issubset(markers.keys()):
                continue

            pts = np.array([
                markers["top left corner"],
                markers["top right corner"],
                markers["bottom right corner"],
                markers["bottom left corner"],
            ], dtype=float)

            inner_poly = _order_points_ccw(pts)
            zone_poly = _scale_polygon_about_centroid(inner_poly, float(object_zone_margin))

            if arena_size is not None:
                inner_poly = inner_poly.copy()
                zone_poly = zone_poly.copy()
                inner_poly[:, 0] = np.clip(inner_poly[:, 0], 0.0, arena_size)
                inner_poly[:, 1] = np.clip(inner_poly[:, 1], 0.0, arena_size)
                zone_poly[:, 0] = np.clip(zone_poly[:, 0], 0.0, arena_size)
                zone_poly[:, 1] = np.clip(zone_poly[:, 1], 0.0, arena_size)

            objects_geom.append({
                "name": name,
                "shape": "polygon",
                "markers": markers,
                "inner": {"xy": inner_poly},
                "zone":  {"xy": zone_poly},
            })

        elif shape == "ellipse":
            needed = {"top point", "right point", "bottom point", "left point"}
            if not needed.issubset(markers.keys()):
                continue

            cx, cy, a, b = _ellipse_params_from_markers(markers)

            az = float(a) + float(object_zone_margin)
            bz = float(b) + float(object_zone_margin)

            if arena_size is not None:
                cx = float(np.clip(cx, 0.0, arena_size))
                cy = float(np.clip(cy, 0.0, arena_size))
                az = max(0.0, float(az))
                bz = max(0.0, float(bz))

            objects_geom.append({
                "name": name,
                "shape": "ellipse",
                "markers": markers,
                "inner": {"cx": float(cx), "cy": float(cy), "a": float(a),  "b": float(b)},
                "zone":  {"cx": float(cx), "cy": float(cy), "a": float(az), "b": float(bz)},
            })

        else:
            continue

    return objects_geom

def _count_crossings(mask: np.ndarray) -> int:
    """
    Count boundary crossings between False and True for a boolean mask.
    If mask includes NaNs handled upstream, mask must be clean boolean.
    """
    if mask.size <= 1:
        return 0
    return int(np.sum(np.diff(mask.astype(int)) != 0))

def _count_entries(mask: np.ndarray) -> int:
    """
    Count entry events: transitions from False -> True.
    mask must be a clean boolean array.
    """
    if mask.size <= 1:
        return 0
    m = mask.astype(int)
    return int(np.sum((m[1:] == 1) & (m[:-1] == 0)))

def compute_object_zone_metrics(df_points: pd.DataFrame,
                                bp_name: str,
                                frame_rate: float,
                                objects_geom: list[dict],
                                valid_mask: np.ndarray | None = None,
                                prefix: str = "") -> dict:
    x = df_points[(bp_name, "x")].to_numpy(dtype=float)
    y = df_points[(bp_name, "y")].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    valid = finite if valid_mask is None else (finite & valid_mask.astype(bool))

    out = {}

    for obj in objects_geom:
        name = obj["name"]
        shape = obj["shape"]

        if shape == "polygon":
            inn_xy = obj["inner"]["xy"]
            zon_xy = obj["zone"]["xy"]
            inside_object = _polygon_contains_points(inn_xy, x, y)
            inside_zone   = _polygon_contains_points(zon_xy, x, y)

        elif shape == "ellipse":
            inn = obj["inner"]
            zon = obj["zone"]
            inside_object = _ellipse_contains_points(inn["cx"], inn["cy"], inn["a"], inn["b"], x, y)
            inside_zone   = _ellipse_contains_points(zon["cx"], zon["cy"], zon["a"], zon["b"], x, y)

        else:
            continue

        inside_object = inside_object & valid
        inside_zone   = inside_zone & valid

        valid_idx = np.flatnonzero(valid)
        if valid_idx.size <= 1:
            zone_crossings = 0
            object_crossings = 0
            time_in_zone = 0.0
            time_on_object = 0.0
            zone_entries = 0
            object_entries = 0
        else:
            z_mask = inside_zone[valid_idx]
            o_mask = inside_object[valid_idx]
            zone_crossings = _count_crossings(z_mask)
            object_crossings = _count_crossings(o_mask)
            
            zone_entries     = _count_entries(z_mask)
            object_entries   = _count_entries(o_mask)
            
            time_in_zone = float(z_mask.sum() / frame_rate)
            time_on_object = float(o_mask.sum() / frame_rate)

        key_base = f"{name}"
        out[f"{key_base} time_in_zone_s {prefix}".strip()] = time_in_zone
        out[f"{key_base} time_on_object_s {prefix}".strip()] = time_on_object
        out[f"{key_base} zone_crossings {prefix}".strip()] = zone_crossings
        out[f"{key_base} object_crossings {prefix}".strip()] = object_crossings
        out[f"{key_base} zone_entries {prefix}".strip()] = zone_entries
        out[f"{key_base} object_entries {prefix}".strip()] = object_entries

    return out

def plot_objects_on_arena(objects_geom: list[dict],
                          arena_size: float,
                          curr_filename_clean: str,
                          curr_results_path: str,
                          object_zone_margin: float,
                          spatial_unit: str = "cm"):
    """Optional: plot object inner shapes + zones for sanity check."""
    plt.figure(figsize=(10, 10))

    # arena boundary
    rect_arena = plt.Rectangle((0, 0), arena_size, arena_size,
                               linewidth=2, edgecolor="blue",
                               facecolor="none", linestyle="--", alpha=0.8,
                               label=f"arena boundary ({arena_size} {spatial_unit})")
    plt.gca().add_patch(rect_arena)

    # objects
    inner_labeled = False
    zone_labeled = False
    for obj in objects_geom:
        name = obj["name"]
        shape = obj.get("shape", "")

        if shape == "polygon":
            inn_xy = obj["inner"]["xy"]
            zon_xy = obj["zone"]["xy"]

            zon_closed = np.vstack([zon_xy, zon_xy[0]])
            plt.plot(zon_closed[:, 0], zon_closed[:, 1],
                     linewidth=2, color="orange", alpha=0.9,
                     label=("object zone" if not zone_labeled else None))
            zone_labeled = True

            inn_closed = np.vstack([inn_xy, inn_xy[0]])
            plt.plot(inn_closed[:, 0], inn_closed[:, 1],
                     linewidth=2, color="black", alpha=0.9,
                     label=("object inner" if not inner_labeled else None))
            inner_labeled = True

            c = inn_xy.mean(axis=0)
            plt.text(float(c[0]), float(c[1]), name, fontsize=12)

        elif shape == "ellipse":
            inn = obj["inner"]
            zon = obj["zone"]

            plt.gca().add_patch(
                Ellipse((zon["cx"], zon["cy"]),
                        width=2.0 * zon["a"], height=2.0 * zon["b"],
                        fill=False, edgecolor="orange", linewidth=2,
                        alpha=0.9, label=("object zone" if not zone_labeled else None))
            )
            zone_labeled = True

            plt.gca().add_patch(
                Ellipse((inn["cx"], inn["cy"]),
                        width=2.0 * inn["a"], height=2.0 * inn["b"],
                        fill=False, edgecolor="black", linewidth=2,
                        alpha=0.9, label=("object inner" if not inner_labeled else None))
            )
            inner_labeled = True

            plt.text(float(inn["cx"] + inn["a"]), float(inn["cy"] + inn["b"]), name, fontsize=12)

        else:
            continue

    plt.title(f"object geometries (inner + zone margin={object_zone_margin} {spatial_unit})\n{curr_filename_clean}")
    plt.xlabel(f"x ({spatial_unit})")
    plt.ylabel(f"y ({spatial_unit})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-arena_size * 0.05, arena_size * 1.05)
    plt.ylim(-arena_size * 0.05, arena_size * 1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_objects_geometry.pdf"), dpi=300)
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_objects_geometry.png"), dpi=300, transparent=True)
    plt.close()

def _collect_object_bodyparts(objects_DLC_bp_list: list[dict]) -> set[str]:
    bps = set()
    for obj in objects_DLC_bp_list:
        for k, v in obj.items():
            if k in {"shape", "name"}:
                continue
            bps.add(v)
    return bps

# this function can be REMOVE later: 
def overlay_objectsOLD(ax,
                    objects_geom: list[dict],
                    show_labels: bool = True,
                    inner_lw: float = 2.0,
                    zone_lw: float = 2.0,
                    inner_alpha: float = 0.95,
                    zone_alpha: float = 0.90):
    if not objects_geom:
        return

    inner_labeled = False
    zone_labeled = False

    for obj in objects_geom:
        name = obj.get("name", "object")
        shape = obj.get("shape", None)

        if shape == "polygon":
            inn_xy = obj["inner"]["xy"]
            zon_xy = obj["zone"]["xy"]

            zon_closed = np.vstack([zon_xy, zon_xy[0]])
            ax.plot(zon_closed[:, 0], zon_closed[:, 1],
                    linewidth=zone_lw, alpha=zone_alpha,
                    label=("object zone" if not zone_labeled else None))
            zone_labeled = True

            inn_closed = np.vstack([inn_xy, inn_xy[0]])
            ax.plot(inn_closed[:, 0], inn_closed[:, 1],
                    linewidth=inner_lw, alpha=inner_alpha,
                    label=("object inner" if not inner_labeled else None))
            inner_labeled = True

            if show_labels:
                c = inn_xy.mean(axis=0)
                ax.text(float(c[0]), float(c[1]), name, fontsize=11)

        elif shape == "ellipse":
            inn = obj["inner"]
            zon = obj["zone"]

            ax.add_patch(
                Ellipse((zon["cx"], zon["cy"]),
                        width=2.0 * zon["a"], height=2.0 * zon["b"],
                        fill=False, linewidth=zone_lw, alpha=zone_alpha,
                        label=("object zone" if not zone_labeled else None))
            )
            zone_labeled = True

            ax.add_patch(
                Ellipse((inn["cx"], inn["cy"]),
                        width=2.0 * inn["a"], height=2.0 * inn["b"],
                        fill=False, linewidth=inner_lw, alpha=inner_alpha,
                        label=("object inner" if not inner_labeled else None))
            )
            inner_labeled = True

            if show_labels:
                ax.text(inn["cx"] + inn["a"], inn["cy"] + inn["b"], name, fontsize=11)

        else:
            continue

def overlay_objects(ax,
                    objects_geom: list[dict],
                    show_labels: bool = True,
                    inner_lw: float = 2.0,
                    zone_lw: float = 2.0,
                    inner_alpha: float = 0.95,
                    zone_alpha: float = 0.90,
                    zone_ls: str = "--",
                    inner_ls: str = "-"):

    if not objects_geom:
        return

    # robust color cycle
    color_cycle = cycle(
        mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    used_labels = set()

    def _maybe_label(lbl):
        if lbl in used_labels:
            return None
        used_labels.add(lbl)
        return lbl

    for obj in objects_geom:
        name = obj.get("name", "object")
        shape = obj.get("shape", None)

        color = next(color_cycle)

        if shape == "polygon":
            inn_xy = np.asarray(obj["inner"]["xy"], dtype=float)
            zon_xy = np.asarray(obj["zone"]["xy"], dtype=float)

            inn_closed = np.vstack([inn_xy, inn_xy[0]])
            zon_closed = np.vstack([zon_xy, zon_xy[0]])

            ax.plot(
                zon_closed[:, 0], zon_closed[:, 1],
                linestyle=zone_ls, linewidth=zone_lw,
                alpha=zone_alpha, color=color,
                label=_maybe_label(f"{name} (zone)")
            )

            ax.plot(
                inn_closed[:, 0], inn_closed[:, 1],
                linestyle=inner_ls, linewidth=inner_lw,
                alpha=inner_alpha, color=color,
                label=_maybe_label(f"{name} (inner)")
            )

            if show_labels:
                c = inn_xy.mean(axis=0)
                ax.text(c[0], c[1], name, fontsize=11, color=color)

        elif shape == "ellipse":
            inn = obj["inner"]
            zon = obj["zone"]

            e_zone = Ellipse(
                (zon["cx"], zon["cy"]),
                width=2 * zon["a"], height=2 * zon["b"],
                fill=False, linewidth=zone_lw,
                alpha=zone_alpha, edgecolor=color,
                label=_maybe_label(f"{name} (zone)")
            )
            e_zone.set_linestyle(zone_ls)
            ax.add_patch(e_zone)

            e_inn = Ellipse(
                (inn["cx"], inn["cy"]),
                width=2 * inn["a"], height=2 * inn["b"],
                fill=False, linewidth=inner_lw,
                alpha=inner_alpha, edgecolor=color,
                label=_maybe_label(f"{name} (inner)")
            )
            e_inn.set_linestyle(inner_ls)
            ax.add_patch(e_inn)

            if show_labels:
                ax.text(
                    inn["cx"] + inn["a"],
                    inn["cy"] + inn["b"],
                    name, fontsize=11, color=color
                )

# helper-function for determining line intersection (needed for EPM virtual arena corners estimation):
def _line_intersection(p1, p2, p3, p4, eps=1e-9):
    """
    Intersection of the infinite lines through (p1,p2) and (p3,p4).
    Returns (x,y). Raises if lines are (near) parallel.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p4 = np.asarray(p4, dtype=float)

    r = p2 - p1
    s = p4 - p3

    # Solve p1 + t r = p3 + u s  ->  t r - u s = (p3 - p1)
    A = np.array([[r[0], -s[0]],
                  [r[1], -s[1]]], dtype=float)
    b = (p3 - p1).astype(float)

    det = np.linalg.det(A)
    if abs(det) < eps:
        raise ValueError("Lines are parallel or ill-conditioned for intersection.")

    t, u = np.linalg.solve(A, b)
    x = p1 + t * r
    return float(x[0]), float(x[1])

def _as_np(p):
    return np.asarray(p, dtype=np.float64)

def _cross2(a, b):
    # 2D cross product (scalar): a_x b_y - a_y b_x
    return float(a[0]*b[1] - a[1]*b[0])

def _line_from_points(p, q):
    p = _as_np(p)
    q = _as_np(q)
    d = q - p
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("Degenerate line: points too close or identical.")
    d = d / n
    return p, d  # point-direction form

def _line_through_point_parallel_to(p_through, p_dir, q_dir):
    # direction from (p_dir -> q_dir), line passes through p_through
    _, d = _line_from_points(p_dir, q_dir)
    p0 = _as_np(p_through)
    return p0, d

def _intersection_point(line1, line2):
    # line: (p, d) with p + t d
    p, r = line1
    q, s = line2
    p = _as_np(p); r = _as_np(r)
    q = _as_np(q); s = _as_np(s)

    rxs = _cross2(r, s)
    if abs(rxs) < 1e-12:
        raise ValueError("Lines are parallel or nearly parallel; cannot intersect robustly.")

    t = _cross2((q - p), s) / rxs
    return p + t * r

def _pick_top_point(A, B, y_down=True):
    A = _as_np(A); B = _as_np(B)
    # if y_down: smaller y = higher (top)
    if y_down:
        return A if A[1] < B[1] else B
    else:
        return A if A[1] > B[1] else B

def _pick_bottom_point(E, F, y_down=True):
    E = _as_np(E); F = _as_np(F)
    if y_down:
        return E if E[1] > F[1] else F
    else:
        return E if E[1] < F[1] else F

def _pick_left_point(G, H):
    G = _as_np(G); H = _as_np(H)
    return G if G[0] < H[0] else H

def _pick_right_point(C, D):
    C = _as_np(C); D = _as_np(D)
    return C if C[0] > D[0] else D

def epm_virtual_corners_shifted_long_lines(arena_corners_raw, y_down=True):
    """
    Implements exactly:
    * top line: parallel to H-C, shifted to pass through top(A,B)
    * bottom line: parallel to G-D, shifted to pass through bottom(E,F)
    * left line: parallel to A-F, shifted to pass through left(G,H)
    * right line: parallel to B-E, shifted to pass through right(C,D)
    Returns corners as float32 array ordered [TL, TR, BL, BR].
    """

    A = arena_corners_raw["outer_A"]
    B = arena_corners_raw["outer_B"]
    C = arena_corners_raw["outer_C"]
    D = arena_corners_raw["outer_D"]
    E = arena_corners_raw["outer_E"]
    F = arena_corners_raw["outer_F"]
    G = arena_corners_raw["outer_G"]
    H = arena_corners_raw["outer_H"]

    # Anchor points (your "shift so it goes through ...")
    P_top = _pick_top_point(A, B, y_down=y_down)         # through A or B
    P_bottom = _pick_bottom_point(E, F, y_down=y_down)   # through E or F
    P_left = _pick_left_point(G, H)                      # through G or H
    P_right = _pick_right_point(C, D)                    # through C or D

    # Direction-defining long lines (your H-C, G-D, A-F, B-E)
    top_line = _line_through_point_parallel_to(P_top, H, C)       # parallel to HC
    bottom_line = _line_through_point_parallel_to(P_bottom, G, D) # parallel to GD
    left_line = _line_through_point_parallel_to(P_left, A, F)     # parallel to AF
    right_line = _line_through_point_parallel_to(P_right, B, E)   # parallel to BE

    # Intersections
    TL = _intersection_point(top_line, left_line)
    TR = _intersection_point(top_line, right_line)
    BL = _intersection_point(bottom_line, left_line)
    BR = _intersection_point(bottom_line, right_line)

    src_points = np.array([TL, TR, BL, BR], dtype=np.float32)
    return src_points, dict(
        top_line=top_line,
        bottom_line=bottom_line,
        left_line=left_line,
        right_line=right_line,
        anchors=dict(P_top=P_top, P_bottom=P_bottom, P_left=P_left, P_right=P_right),
    )

# sanity plot to verify EPM virtual corners:
def plot_epm_virtual_corners(arena_corners_raw, src_points, curr_filename_clean,
                            curr_results_path, spatial_unit="cm"):

    # pull points from dict
    A = arena_corners_raw["outer_A"]
    B = arena_corners_raw["outer_B"]
    C = arena_corners_raw["outer_C"]
    D = arena_corners_raw["outer_D"]
    E = arena_corners_raw["outer_E"]
    F = arena_corners_raw["outer_F"]
    G = arena_corners_raw["outer_G"]
    H = arena_corners_raw["outer_H"]

    plt.figure(figsize=(8, 6))

    # plot outer edges
    for (p, q, label) in [(A, B, "AB"), (C, D, "CD"), (E, F, "EF"), (G, H, "GH")]:
        plt.plot([p[0], q[0]], [p[1], q[1]], "-", label=label)

    # plot the construction helper lines (as in your comment)
    plt.plot([A[0], F[0]], [A[1], F[1]], ":", color="tab:blue", alpha=1.0, label="A-F")
    plt.plot([B[0], E[0]], [B[1], E[1]], ":", color="tab:orange", alpha=1.0, label="B-E")
    plt.plot([C[0], H[0]], [C[1], H[1]], ":", color="tab:green", alpha=1.0, label="C-H")
    plt.plot([D[0], G[0]], [D[1], G[1]], ":", color="tab:red", alpha=1.0, label="D-G")

    # plot quadrilateral through the virtual corners
    plt.plot([src_points[0][0], src_points[1][0]], [src_points[0][1], src_points[1][1]], "--", color="gray", alpha=0.5)
    plt.plot([src_points[2][0], src_points[3][0]], [src_points[2][1], src_points[3][1]], "--", color="gray", alpha=0.5)
    plt.plot([src_points[0][0], src_points[2][0]], [src_points[0][1], src_points[2][1]], "--", color="gray", alpha=0.5)
    plt.plot([src_points[1][0], src_points[3][0]], [src_points[1][1], src_points[3][1]], "--", color="gray", alpha=0.5)

    # label the virtual corners
    for (pt, label) in [(src_points[0], "TL"), (src_points[1], "TR"), (src_points[2], "BL"), (src_points[3], "BR")]:
        plt.plot(pt[0], pt[1], "o")
        plt.text(pt[0], pt[1], label)

    # plot all raw corner markers
    for corner_name, (corner_x, corner_y) in arena_corners_raw.items():
        plt.scatter(corner_x, corner_y, marker="x", s=100, label=corner_name)
        name_use = corner_name.split("_")[-1]
        plt.text(corner_x, corner_y, name_use)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(bbox_to_anchor=(1.15, 1), loc="upper left")
    plt.title(f"EPM virtual corners estimation\n{curr_filename_clean} (raw)")
    plt.xlabel(f"x ({spatial_unit})")
    plt.ylabel(f"y ({spatial_unit})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_EPM_virtual_corners.pdf"), dpi=300)
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_EPM_virtual_corners.png"), dpi=300, transparent=True)
    plt.close()

# build EPM valid mask function:
def _build_epm_valid_mask_from_objects(df_transformed: pd.DataFrame,
                                       bp_name: str,
                                       objects_geom: list[dict],
                                       likelihood_threshold: float) -> np.ndarray:
    """
    valid_mask(t) = (likelihood(bp)>thr) AND (point inside any EPM ROI polygon).
    For EPM we use the *inner* polygons of the objects as the physical arena union.
    """
    x = df_transformed[(bp_name, "x")].to_numpy(dtype=float)
    y = df_transformed[(bp_name, "y")].to_numpy(dtype=float)

    lik_ok = (df_transformed[(bp_name, "likelihood")].to_numpy(dtype=float) > float(likelihood_threshold))
    finite = np.isfinite(x) & np.isfinite(y)
    base_ok = lik_ok & finite

    inside_any = np.zeros_like(base_ok, dtype=bool)
    for obj in objects_geom:
        if obj.get("shape", "") != "polygon":
            continue
        poly = obj["inner"]["xy"]
        inside_any |= _polygon_contains_points(poly, x, y)

    return base_ok & inside_any

# put this in your FUNCTIONS section, after _polygon_contains_points() (because we reuse it)

def _epm_get_roi_polygons(objects_geom: list[dict]) -> dict[str, np.ndarray]:
    """
    Extract inner polygons for EPM ROIs from objects_geom.
    Assumes EPM ROIs are encoded as shape='polygon' with obj['inner']['xy'].
    Returns: roi_name -> (N,2) polygon array
    """
    rois = {}
    for obj in objects_geom:
        if obj.get("shape", "") != "polygon":
            continue
        name = obj.get("name", "")
        if not name:
            continue
        inner = obj.get("inner", {})
        xy = inner.get("xy", None)
        if xy is None:
            continue
        rois[name] = np.asarray(xy, dtype=float)
    return rois

def epm_compute_roi_masks_for_bps_OLD(
    df_points: pd.DataFrame,
    bp_list: list[str],
    objects_geom: list[dict],
    valid_mask: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a boolean DataFrame with columns (bp, roi) that indicate whether bp is inside roi at each frame.
    Frame index is df_points.index (must match any downstream cut index selection).
    valid_mask gates the result (False outside valid).
    """
    rois = _epm_get_roi_polygons(objects_geom)
    if len(rois) == 0:
        raise ValueError("EPM ROI polygons not found in objects_geom. Check EPM objects_DLC_bp_list construction.")

    idx = df_points.index
    out = {}

    if valid_mask is None:
        valid = np.ones(len(df_points), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)

    for bp in bp_list:
        if (bp, "x") not in df_points.columns or (bp, "y") not in df_points.columns:
            raise ValueError(f"EPM bp '{bp}' not found in df_points columns.")

        x = df_points[(bp, "x")].to_numpy(dtype=float)
        y = df_points[(bp, "y")].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)

        # base validity for this bp: finite coords AND global valid
        base = finite & valid

        for roi_name, poly_xy in rois.items():
            inside = _polygon_contains_points(poly_xy, x, y)
            inside = inside & base
            out[(bp, roi_name)] = inside

    mask_df = pd.DataFrame(out, index=idx)
    mask_df.columns = pd.MultiIndex.from_tuples(mask_df.columns, names=["bp", "roi"])
    return mask_df

def epm_compute_roi_masks_for_bps(
    df_points: pd.DataFrame,
    bp_list: list[str],
    objects_geom: list[dict],
    likelihood_threshold: float,
    valid_mask: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a boolean DataFrame with columns (bp, roi) that indicate whether bp is inside roi at each frame.
    A bp counts as "inside" only if:
      * x,y finite
      * likelihood(bp) > likelihood_threshold
      * (optional) valid_mask[t] is True
      * point inside roi polygon
    """
    rois = _epm_get_roi_polygons(objects_geom)
    if len(rois) == 0:
        raise ValueError("EPM ROI polygons not found in objects_geom. Check EPM objects_DLC_bp_list construction.")

    idx = df_points.index
    out = {}

    if valid_mask is None:
        valid_global = np.ones(len(df_points), dtype=bool)
    else:
        valid_global = np.asarray(valid_mask, dtype=bool)

    for bp in bp_list:
        # require the full DLC triplet to exist
        if (bp, "x") not in df_points.columns or (bp, "y") not in df_points.columns or (bp, "likelihood") not in df_points.columns:
            raise ValueError(f"EPM bp '{bp}' missing x/y/likelihood in df_points columns.")

        x = df_points[(bp, "x")].to_numpy(dtype=float)
        y = df_points[(bp, "y")].to_numpy(dtype=float)
        lik = df_points[(bp, "likelihood")].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        lik_ok = lik > float(likelihood_threshold)

        # base validity for this bp
        base = finite & lik_ok & valid_global

        for roi_name, poly_xy in rois.items():
            inside = _polygon_contains_points(poly_xy, x, y)
            out[(bp, roi_name)] = inside & base

    mask_df = pd.DataFrame(out, index=idx)
    mask_df.columns = pd.MultiIndex.from_tuples(mask_df.columns, names=["bp", "roi"])
    return mask_df

def epm_aggregate_roi_masks(
    mask_df: pd.DataFrame,
    mode: str = "kofn",
    k: int = 2) -> pd.DataFrame:
    """
    Aggregate (bp, roi) masks into per-roi masks.
    Returns DataFrame with columns = roi, index = frames.
    """
    if not isinstance(mask_df.columns, pd.MultiIndex):
        raise ValueError("mask_df must have MultiIndex columns (bp, roi).")

    mode = str(mode).lower().strip()

    # count how many bps are in each roi per frame
    #counts = mask_df.groupby(level="roi", axis=1).sum().astype(int)
    counts = mask_df.T.groupby(level="roi").sum().T

    if mode == "all":
        n_bp = mask_df.columns.get_level_values("bp").nunique()
        agg = (counts >= n_bp)
    elif mode == "kofn":
        agg = (counts >= int(k))
    else:
        raise ValueError(f"Unknown epm aggregation mode '{mode}'. Use 'all' or 'kofn'.")

    return agg

def epm_multi_bp_object_metrics_from_roi_masks(
    roi_bool_df: pd.DataFrame,
    frame_rate: float,
    prefix: str) -> dict:
    """
    Convert aggregated per-ROI boolean masks into the SAME key names that
    compute_object_zone_metrics() produces, so no new columns appear.

    Example keys:
      'center time_in_zone_s (cut)'
      'center time_on_object_s (cut)'
      'center zone_crossings (cut)'
      'center object_crossings (cut)'
    """
    out = {}
    for roi in roi_bool_df.columns:
        m = roi_bool_df[roi].to_numpy(dtype=bool)

        time_s = float(m.sum() / frame_rate)
        #crossings = int(np.sum(np.diff(m.astype(int)) != 0)) if m.size > 1 else 0
        crossings = _count_crossings(m)
        entries   = _count_entries(m)

        # IMPORTANT: use the same base key scheme as compute_object_zone_metrics()
        # For EPM, we treat "zone" and "object" identically because object_zone_margin=0
        # and ROIs are polygons.
        out[f"{roi} time_in_zone_s {prefix}".strip()] = time_s
        out[f"{roi} time_on_object_s {prefix}".strip()] = time_s
        out[f"{roi} zone_crossings {prefix}".strip()] = crossings
        out[f"{roi} object_crossings {prefix}".strip()] = crossings
        out[f"{roi} zone_entries {prefix}".strip()] = entries
        out[f"{roi} object_entries {prefix}".strip()] = entries

    return out

def epm_multi_bp_speed_series_OLD(
    df_all_transformed: pd.DataFrame,
    bp_list: list[str],
    frame_rate: float,
    time_step: float,
    likelihood_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a single multi-bp speed trace by averaging per-bp instantaneous speeds
    (nanmean across bps at each frame). Also returns a per-frame validity mask
    based on likelihood and finite coords.

    Returns
    -------
    speed_multi : np.ndarray
        shape (T,)
    valid_multi : np.ndarray (bool)
        shape (T,)
    """
    T = len(df_all_transformed)
    if T == 0:
        return np.array([]), np.array([], dtype=bool)

    per_bp_speeds = []
    per_bp_valids = []

    for bp in bp_list:
        if (bp, "x") not in df_all_transformed.columns:
            continue

        x = df_all_transformed[(bp, "x")].to_numpy(dtype=float)
        y = df_all_transformed[(bp, "y")].to_numpy(dtype=float)
        lik = df_all_transformed[(bp, "likelihood")].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        lik_ok = lik > float(likelihood_threshold)
        valid = finite & lik_ok

        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx * dx + dy * dy) / time_step

        speed[~valid] = np.nan

        per_bp_speeds.append(speed)
        per_bp_valids.append(valid)

    if len(per_bp_speeds) == 0:
        return np.full(T, np.nan), np.zeros(T, dtype=bool)

    S = np.vstack(per_bp_speeds)           # (n_bp, T)
    V = np.vstack(per_bp_valids)           # (n_bp, T)

    speed_multi = np.nanmean(S, axis=0)
    # validity: at least one bp valid at a frame
    valid_multi = np.any(V, axis=0) & np.isfinite(speed_multi)

    # if all bps nan at some frame, speed_multi is nan, valid_multi becomes False
    return speed_multi.astype(float), valid_multi.astype(bool)

def epm_multi_bp_speed_series(
    df_all_transformed: pd.DataFrame,
    bp_list: list[str],
    frame_rate: float,
    time_step: float,
    likelihood_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-bp speed trace with strict validity:
    valid_multi[t] == True only if ALL bps have finite coords AND likelihood > threshold at t.
    speed_multi[t] is the mean of per-bp speeds at t (only defined when valid_multi[t] True).
    """
    T = len(df_all_transformed)
    if T == 0:
        return np.array([]), np.array([], dtype=bool)

    per_bp_speeds = []
    per_bp_valids = []

    for bp in bp_list:
        # strict: require x,y,likelihood to exist
        if (bp, "x") not in df_all_transformed.columns or (bp, "y") not in df_all_transformed.columns or (bp, "likelihood") not in df_all_transformed.columns:
            raise ValueError(f"EPM bp '{bp}' missing x/y/likelihood in df_all_transformed.")

        x = df_all_transformed[(bp, "x")].to_numpy(dtype=float)
        y = df_all_transformed[(bp, "y")].to_numpy(dtype=float)
        lik = df_all_transformed[(bp, "likelihood")].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        lik_ok = lik > float(likelihood_threshold)
        valid = finite & lik_ok

        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx * dx + dy * dy) / time_step

        # invalid frames become NaN
        speed[~valid] = np.nan

        per_bp_speeds.append(speed)
        per_bp_valids.append(valid)

    # stack
    S = np.vstack(per_bp_speeds)  # (n_bp, T)
    V = np.vstack(per_bp_valids)  # (n_bp, T)

    """ # strict: ALL bps valid per frame
    valid_multi = np.all(V, axis=0)

    # speed defined only where all valid
    speed_multi = np.nanmean(S, axis=0)
    speed_multi[~valid_multi] = np.nan """
    
    # strict: ALL bps valid per frame
    valid_multi = np.all(V, axis=0)

    # speed defined only where all valid, avoid nanmean on all-NaN columns
    speed_multi = np.full(T, np.nan, dtype=float)
    if np.any(valid_multi):
        # On valid frames, there should be no NaNs in S[:, valid_multi] if your logic is consistent,
        # but using nanmean is still fine and robust.
        speed_multi[valid_multi] = np.nanmean(S[:, valid_multi], axis=0)

    return speed_multi.astype(float), valid_multi.astype(bool)

def _set_if_column_exists(row_dict: dict, col: str, value):
    """
    Setzt row_dict[col] = value nur dann, wenn die Spalte col in row_dict bereits existiert
    oder du explizit neue Spalten zulassen willst. Hier: wir wollen KEINE neuen Spalten.
    """
    if col in row_dict:
        row_dict[col] = value

# %% FETCHING DATA

# create the results folder if it does not exist:
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# scan DATA_PATH for all .csv files:
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
# exclude all files starting with '.' :
csv_files = [f for f in csv_files if not f.startswith('.')]
# check whether "_filtered" files are available; if so, just use those:
filtered_files = [f for f in csv_files if '_filtered' in f]
# check whether to filter for filtered data:
if use_filtered_data:
    # exclude all files from filtered_files, that do not contain "filtered" in their name:
    csv_files = [f for f in csv_files if 'filtered' in f]
else:
    # excluded all "filtered" files (if available):
    csv_files = [f for f in csv_files if 'filtered' not in f]
if filtered_files:
    csv_files = filtered_files
    print(f"Found filtered files; using those:")
else:
    print(f"No filtered files found; using all unfiltered instead:")
for file in csv_files:
    print(f"  {file}")

# load all DLC CSV files (cleaned & numeric) into a list "loaded_runs":
loaded_runs = []
for curr_filename in csv_files:
    # curr_filename = csv_files[0]
    curr_file = os.path.join(DATA_PATH, curr_filename)
    print(f"loading {curr_filename}...")

    """ 
    curr_file contains sometimes 2 and sometime 3 header rows, depending on
    whether the DLC project used social partner or not. Thus, we first need to detect,
    whether headers scorer, individuals, bodyparts (3 headers to skip) or only
    scorer, bodyparts (2 headers to skip) are present. 
    We do this by reading the first 3 lines of the file:
    """
    with open(curr_file, 'r') as f:
        first_line = f.readline().split(',')[0]
        second_line = f.readline().split(',')[0]
        third_line = f.readline().split(',')[0]
    first_3_columns = [first_line.strip().lower(), second_line.strip().lower(), third_line.strip().lower()]
    if first_3_columns == ['scorer', 'individuals', 'bodyparts']:
        header_rows = [1, 2, 3]  # 3 header rows to skip
    elif first_3_columns[:2] == ['scorer', 'bodyparts']:
        header_rows = [1, 2]     # 2 header rows to skip
    else:
        raise ValueError(f"Unexpected header format in file {curr_filename}.")

    # read CSV with multi-index columns
    df_in = pd.read_csv(curr_file, header=header_rows)
    #df_in = pd.read_csv(curr_file, header=[1, 2])

    # build clean name and results path:
    curr_filename_clean = curr_filename.split('DLC')[0]
    base_results_path   = os.path.join(RESULTS_PATH, curr_filename_clean)
    os.makedirs(base_results_path, exist_ok=True)

    # drop first or two (meta) column(s):
    df_cleaned = df_in.iloc[:, 1:].copy()
    #df_cleaned = df_in.iloc[:, len(header_rows)-1:].copy()

    # set proper multi-index on columns:
    df_cleaned.columns = pd.MultiIndex.from_tuples(df_cleaned.columns)

    # convert to numeric:
    #df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce')
    df_cleaned = df_cleaned.reset_index(drop=True)
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")
    
    """ 
    In case we had three header rows (scorer, individuals, bodyparts),
    and thus a multi-animal setup, we need to take a sub-dataframe of the
    Multiindex columns for the specific experimental mouse only (exp_mouse_key).
       """
    if len(header_rows) == 3:
        if exp_mouse_key is None:
            raise ValueError(f"File {curr_filename} contains multi-animal data, but no exp_mouse_key was provided.")
        available_keys = df_cleaned.columns.get_level_values(0).unique().tolist()
        if exp_mouse_key not in available_keys:
            raise ValueError(f"exp_mouse_key '{exp_mouse_key}' not found in file {curr_filename}. Available keys: {available_keys}")
        df_cleaned = df_cleaned.loc[:, pd.IndexSlice[[exp_mouse_key, "single"], :, :]].droplevel(0, axis=1)
        # convert multi-index dataframe into a single-index dataframe:
        df_cleaned.columns = pd.MultiIndex.from_tuples(df_cleaned.columns)
        

    loaded_runs.append({"filename": curr_filename,
                        "filename_clean": curr_filename_clean,
                        "base_results_path": base_results_path,
                        "df": df_cleaned})

print(f"loaded {len(loaded_runs)} file(s) for processing.")
# %% MAIN PROCESSING
# iterate over loaded datasets and run transforms, plots & metrics:
measurements_df = pd.DataFrame()

# einmalig, vor MAIN PROCESSING:
if objects_DLC_bp_list:
    object_bodyparts = _collect_object_bodyparts(objects_DLC_bp_list)
else:
    object_bodyparts = set()

for run in loaded_runs:
    ## %%
    # run = loaded_runs[0]
    curr_filename        = run["filename"]
    curr_filename_clean  = run["filename_clean"]
    base_results_path    = run["base_results_path"]
    df_cleaned_original  = run["df"].copy()

    epm_cache = None
    if arena_mode.lower().strip() == "epm":
        epm_cache = {
            "objects_geom": None,
            "df_all_transformed": None,
            "speed_df": None,
            "speed_cut_df": None,   # may remain None if cut_tracking False or cut fails
        }

    print(f"processing {curr_filename}...")

    body_parts = df_cleaned_original.columns.get_level_values(0).unique()
    
    # loop over all analysis bodyparts (user labels and DLC names)
    for bodypart_label, bp_name in mouse_center_point_DLC_bp.items():
        """ 
        bodypart_label = list(mouse_center_point_DLC_bp.keys())[0]
        bp_name = mouse_center_point_DLC_bp[bodypart_label] 
        """
    
        # create a per-bodypart results folder:
        bp_folder_name    = f"bodypart_{bodypart_label.replace(' ', '_')}"
        curr_results_path = os.path.join(base_results_path, bp_folder_name)
        os.makedirs(curr_results_path, exist_ok=True)

        # extract per-bodypart suffixes (useful for plot titles and optional global exports):
        title_suffix = f"{bodypart_label}"
        file_suffix  = f"_BP-{bodypart_label.replace(' ', '_')}"
        
        df_cleaned = df_cleaned_original.copy()
        
        if arena_mode.lower().strip() == "epm" and bodypart_label == "EPM_multi_bp":
            speed_df_run = epm_cache.get("speed_df", None) if epm_cache is not None else None
            if speed_df_run is None:
                print("EPM multi-bp: speed_df missing in cache. Skipping.")
                continue

            # optional, falls du im multi-bp Fall auch cut brauchst:
            speed_cut_df_run = epm_cache.get("speed_cut_df", None) if epm_cache is not None else None
            # speed_cut_df_run darf None sein, wenn cut_tracking False oder Cut fehlgeschlagen

            #continue

        # check whether expected body parts are present:
        if object_bodyparts is None:
            object_bodyparts = _collect_object_bodyparts(objects_DLC_bp_list)
        if use_LED_light:
            expected_body_parts = (set(mouse_center_point_DLC_bp.values())
                                | set(arena_corners_DLC_bp.values())
                                | set(LED_lights_DLC_bp.values())
                                | object_bodyparts)
        else:
            expected_body_parts = (set(mouse_center_point_DLC_bp.values())
                                | set(arena_corners_DLC_bp.values())
                                | object_bodyparts)
        detected_body_parts = set(body_parts)
        if not expected_body_parts.issubset(detected_body_parts):
            print(f"  warning: not all expected body parts were detected in {curr_filename}.")
            print(f"  expected: {expected_body_parts}")
            print(f"  detected: {detected_body_parts}")
            print(f"  missing body parts: {expected_body_parts - detected_body_parts}")
            print(f"  skipping {curr_filename}.")
            # continue    # enable if you want to skip files with missing parts
        else:
            print(f"  all expected body parts were detected in {curr_filename}.")

        # ---------- raw (before transform) ----------
        # arena_corners_raw = plot_arena_corners_and_mouse(
        #     df_cleaned, arena_corners_DLC_bp, mouse_center_point_DLC_bp, curr_filename_clean, curr_results_path)
        arena_corners_raw, objects_raw = plot_arena_corners_and_mouse(
                df_cleaned,
                arena_corners_DLC_bp,
                mouse_bp_name=bp_name,
                curr_filename_clean=curr_filename_clean,
                curr_results_path=curr_results_path,
                objects_DLC_bp_list=objects_DLC_bp_list,
                plot_object_shapes=True)

        # ---------- perspective transform into square arena ----------
        corner_coords = list(arena_corners_raw.values())
        corner_coords_arr = np.array(corner_coords, dtype=np.float32)

        dst_points = np.array([[0, 0], [arena_size, 0], [0, arena_size], [arena_size, arena_size]], dtype=np.float32)

        if arena_mode.lower().strip() == "epm":            
            A = arena_corners_raw["outer_A"]
            B = arena_corners_raw["outer_B"]
            C = arena_corners_raw["outer_C"]
            D = arena_corners_raw["outer_D"]
            E = arena_corners_raw["outer_E"]
            F = arena_corners_raw["outer_F"]
            G = arena_corners_raw["outer_G"]
            H = arena_corners_raw["outer_H"]
            """ 
            TL = _line_intersection(A, B, G, H)  # (AB) ∩ (GH)
            TR = _line_intersection(A, B, C, D)  # (AB) ∩ (CD)
            BL = _line_intersection(E, F, G, H)  # (EF) ∩ (GH)
            BR = _line_intersection(E, F, C, D)  # (EF) ∩ (CD)
            
            src_points = np.array([TL, TR, BL, BR], dtype=np.float32)"""
            
            src_points, dbg = epm_virtual_corners_shifted_long_lines(arena_corners_raw, y_down=True)
            
            # sanity plot to verify EPM virtual corners:
            plot_epm_virtual_corners(arena_corners_raw, src_points, curr_filename_clean, 
                                         curr_results_path, spatial_unit="cm")
        else:
            # Original heuristic for 4-corner rectangle arenas
            top_left_corner     = corner_coords_arr[np.argmin(corner_coords_arr[:, 0] + -1 * corner_coords_arr[:, 1])]
            top_right_corner    = corner_coords_arr[np.argmin(-1 * corner_coords_arr[:, 0] + -1 * corner_coords_arr[:, 1])]
            bottom_left_corner  = corner_coords_arr[np.argmin(corner_coords_arr[:, 0] +  corner_coords_arr[:, 1])]
            bottom_right_corner = corner_coords_arr[np.argmin(-1 * corner_coords_arr[:, 0] +  corner_coords_arr[:, 1])]
            src_points = np.array([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)

        for body_part in body_parts:
            x_raw = df_cleaned[(body_part, 'x')].copy()
            y_raw = df_cleaned[(body_part, 'y')].copy()
            likelihood = df_cleaned[(body_part, 'likelihood')].copy()

            pts = np.vstack((x_raw.values, y_raw.values)).T
            pts_tr = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M)
            df_cleaned[(body_part, 'x')] = pts_tr[:, 0, 0]
            df_cleaned[(body_part, 'y')] = pts_tr[:, 0, 1]

            mask = likelihood < likelihood_threshold
            df_cleaned.loc[mask, (body_part, 'x')] = np.nan
            df_cleaned.loc[mask, (body_part, 'y')] = np.nan

        arena_corners_transformed = {
            name: (
                cv2.perspectiveTransform(np.array([[list(coord)]], dtype=np.float32), M)[0][0][0],
                cv2.perspectiveTransform(np.array([[list(coord)]], dtype=np.float32), M)[0][0][1]
            )
            for name, coord in arena_corners_raw.items()
        }

        # keep a copy of all transformed frames (no center/LED filter) for "raw_all":
        df_all_transformed = df_cleaned.copy()

        # ---------- build object geometries ----------
        objects_raw_tr = transform_objects_raw(objects_raw, M)
        objects_geom = build_objects_geometry_from_raw(
            objects_raw_transformed=objects_raw_tr,
            object_zone_margin=object_zone_margin,
            arena_size=arena_size)
        
        # sanity plot to ensure object zones look correct:
        plot_objects_on_arena(objects_geom, arena_size, curr_filename_clean, curr_results_path,
                            object_zone_margin, spatial_unit=spatial_unit)


        # -------------------------------------------------------------------
        # EPM: compute multi bodypart metrics ONCE per run and reuse
        # -------------------------------------------------------------------
        if arena_mode.lower().strip() == "epm":
            if epm_cache["objects_geom"] is None:
                # cache the geometry and the full transformed timeline once
                epm_cache["objects_geom"] = objects_geom
                epm_cache["df_all_transformed"] = df_all_transformed

                # ROI masks per bp (with likelihood), then aggregate per ROI
                mask_df = epm_compute_roi_masks_for_bps(
                    df_points=df_all_transformed,
                    bp_list=epm_bp_list,
                    objects_geom=objects_geom,
                    likelihood_threshold=likelihood_threshold,
                    valid_mask=None
                )

                # aggregation: "all" means ALL bps in same ROI
                roi_agg_df = epm_aggregate_roi_masks(
                    mask_df,
                    mode=epm_aggregate_mode,
                    k=epm_k_of_n
                )

                # strict multi speed and strict validity: ALL bps valid
                speed_multi, valid_multi = epm_multi_bp_speed_series(
                    df_all_transformed=df_all_transformed,
                    bp_list=epm_bp_list,
                    frame_rate=frame_rate,
                    time_step=time_step,
                    likelihood_threshold=likelihood_threshold
                )
                
                # write EPM multi-bp ROI metrics with object-compatible key names
                epm_obj_metrics_uncut = epm_multi_bp_object_metrics_from_roi_masks(
                    roi_bool_df=roi_agg_df,
                    frame_rate=frame_rate,
                    prefix="(uncut)"
                )

                epm_cache["epm_multi_cache"] = dict(
                    roi_agg_df=roi_agg_df,
                    speed_multi=speed_multi,
                    valid_multi=valid_multi,
                    epm_obj_metrics_uncut=epm_obj_metrics_uncut
                )

            epm_multi_cache = epm_cache["epm_multi_cache"]
        else:
            epm_multi_cache = None

        # ---------- build "was mouse in arena" mask ----------
        # For rectangle arenas: valid = likelihood(bp)>thr (your previous behavior)
        # For EPM: valid = likelihood(bp)>thr AND inside union(center + arms)
        if arena_mode.lower().strip() == "epm":
            was_mouse_in_arena_all = _build_epm_valid_mask_from_objects(
                df_transformed=df_cleaned,   # df_cleaned is already transformed at this point
                bp_name=bp_name,
                objects_geom=objects_geom,
                likelihood_threshold=likelihood_threshold
            )
        else:
            was_mouse_in_arena_all = (df_cleaned[(bp_name, "likelihood")] > likelihood_threshold).to_numpy(dtype=bool)


        # ---------- filters (arena-valid & optional LED) ----------
        mouse_center_point_bp = bp_name
        df_raw = df_cleaned.copy()  # transformed, unfiltered timeline

        # df_raw_in_arena: frames that are "valid in arena" (used for raw speed plots)
        df_raw_in_arena = df_raw.loc[was_mouse_in_arena_all].copy()

        # analysis subset starts with arena-valid frames
        df_cleaned = df_raw_in_arena.copy()

        if use_LED_light:
            led_light_bp = LED_lights_DLC_bp['LED light']
            df_cleaned = df_cleaned[df_cleaned[(led_light_bp, 'likelihood')] > likelihood_threshold]
            print("  filtered data to only include frames where the LED light is on.")
        else:
            print("  using all data regardless of the LED light status.")

        # ---------- plot: tracks projective corrected ----------
        plt.figure(figsize=(12, 8))
        plt.scatter(df_cleaned[(bp_name, 'x')],
                    df_cleaned[(bp_name, 'y')],
                    s=10, label=bp_name + " points", alpha=0.5)
        for corner_name, (corner_x, corner_y) in arena_corners_transformed.items():
            plt.scatter(corner_x, corner_y, s=100, label=f'arena {corner_name}', edgecolor='black', alpha=0.5)
        # center-border boundary box:
        if arena_mode.lower().strip() != "epm":
            rect = plt.Rectangle(
                (border_margin, border_margin),
                arena_size - 2 * border_margin,
                arena_size - 2 * border_margin,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8,
                label=f'center-border boundary\n(border margin: {border_margin} cm)'
            )
            plt.gca().add_patch(rect)
        # arena boundary box:
        rect_arena = plt.Rectangle(
            (0, 0),
            arena_size,
            arena_size,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.8,
            label=f'arena boundary\n(size: {arena_size} cm)'
        )
        plt.gca().add_patch(rect_arena)
        # overlay objects (inner + zone), if present
        overlay_objects(plt.gca(), objects_geom, show_labels=True)
        plt.title(f"arena corners and mouse {bp_name} points in\n{curr_filename_clean}" + (" (LED light ON only)" if use_LED_light else ""))
        plt.xlabel(f"x ({spatial_unit})")
        plt.ylabel(f"y ({spatial_unit})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-arena_size * 0.05, arena_size * 1.05)
        plt.ylim(-arena_size * 0.05, arena_size * 1.05)
        plt.tight_layout()
        outname_tracks = f"{curr_filename_clean}_mouse_tracks_all_view_corrected" + ("_LED_filtered" if use_LED_light else "") + ".pdf"
        plt.savefig(os.path.join(curr_results_path, outname_tracks), dpi=300)
        plt.savefig(os.path.join(curr_results_path, outname_tracks.replace(".pdf", ".png")), dpi=300, transparent=True)
        plt.close()
        
        # ---------- plot: heatmap projective corrected ----------
        plt.figure(figsize=(12, 8))
        x_data = df_cleaned[(bp_name, 'x')].dropna()
        y_data = df_cleaned[(bp_name, 'y')].dropna()
        bins = 60
        hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=[[0, arena_size], [0, arena_size]])
        smoothed_hist = gaussian_filter(hist, sigma=2.5)
        plt.imshow(smoothed_hist.T, origin='lower', extent=[0, arena_size, 0, arena_size], cmap='viridis', aspect='auto')
        cbar = plt.colorbar(label=f'smoothed occupancy count\n(bin size = {arena_size/bins:.2f} {spatial_unit})', fraction=0.046, pad=0.005)
        cbar.ax.tick_params(labelsize=12)
        if heat_plot_scatter:
            plt.scatter(x_data, y_data, s=10, label=bp_name+" points", alpha=0.3, color='pink', lw=0)
        else: 
            # plot trajectory as a consecutive line:
            plt.plot(x_data, y_data, label=bp_name+" trajectory", color='pink', lw=1.0)
        # center-border boundary box:
        if arena_mode.lower().strip() != "epm":
            rect = plt.Rectangle(
                (border_margin, border_margin),
                arena_size - 2 * border_margin,
                arena_size - 2 * border_margin,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8,
                label=f'center-border boundary\n(border margin: {border_margin} cm)'
            )
            plt.gca().add_patch(rect)
        # arena boundary box:
        rect_arena = plt.Rectangle(
            (0, 0),
            arena_size,
            arena_size,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.8,
            label=f'arena boundary\n(size: {arena_size} cm)'
        )
        plt.gca().add_patch(rect_arena)
        # overlay objects (inner + zone), if present
        overlay_objects(plt.gca(), objects_geom, show_labels=True)
        plt.title(f"mouse heatmap in and {bp_name} points\n{curr_filename_clean}" + (" (LED light ON only)" if use_LED_light else ""))
        plt.xlabel(f"x ({spatial_unit})", fontsize=14)
        plt.ylabel(f"y ({spatial_unit})", fontsize=14)
        plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, arena_size)
        plt.ylim(0, arena_size)
        plt.tight_layout()
        plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_heatmap_smoothed.pdf"), dpi=300)
        plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_heatmap_smoothed.png"), dpi=300,
                    transparent=True)
        plt.close()
        

        # ---------- robust first detection: mouse in arena ----------
        max_false_frac    = 0.10
        max_false_streak  = 3
        win_frames        = max(1, int(round(mouse_first_track_delay * frame_rate))) if mouse_first_track_delay else 1

        #was_in = (df_raw[(bp_name, 'likelihood')] > likelihood_threshold).values.astype(bool)
        was_in = was_mouse_in_arena_all.astype(bool)
        # --- robust first detection using window_is_stable() ---
        idxs_true = np.flatnonzero(was_in)
        if idxs_true.size > 0 and win_frames > 0:
            first_true_idx = int(idxs_true[0])
            stable_idx = None
            last_start = len(was_in) - win_frames
            """ for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
                if window_is_stable(s):
                    stable_idx = s
                    break """
            for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
                if window_is_stable(was_in, s, win_frames, max_false_frac, max_false_streak):
                    stable_idx = s
                    break
            if stable_idx is None:  # Fallback: nimm erstes True
                stable_idx = first_true_idx

            first_detection_frame_idx = int(df_raw.index[stable_idx])
            first_detection_time = first_detection_frame_idx / frame_rate
        else:
            first_detection_frame_idx = None
            first_detection_time = None


        # ---------- speed & metrics ----------
        time_vec = df_cleaned.index / frame_rate
        speed_vec = np.sqrt(
            np.square(df_cleaned[(bp_name, 'x')].diff().fillna(0)) +
            np.square(df_cleaned[(bp_name, 'y')].diff().fillna(0))
        ) / time_step

        time_vec_raw = df_raw_in_arena.index / frame_rate
        speed_vec_raw = np.sqrt(
            np.square(df_raw_in_arena[(bp_name, 'x')].diff().fillna(0)) +
            np.square(df_raw_in_arena[(bp_name, 'y')].diff().fillna(0))
        ) / time_step

        # raw_all on transformed, unfiltered copy:
        time_vec_raw_all = df_all_transformed.index / frame_rate
        speed_vec_raw_all = np.sqrt(
            np.square(df_all_transformed[(bp_name, 'x')].diff().fillna(0)) +
            np.square(df_all_transformed[(bp_name, 'y')].diff().fillna(0))
        ) / time_step

        plt.figure(figsize=(12, 6))
        plt.plot(time_vec_raw, speed_vec_raw, label='speed (raw)', color='pink', lw=0.75)
        plt.plot(time_vec, speed_vec, label=('speed during LED' if use_LED_light else 'speed (analysis subset)'), color='black', lw=0.75)

        if first_detection_time is not None and len(time_vec) > 0:
            idx_plot = int(np.clip(np.searchsorted(time_vec, first_detection_time), 0, len(time_vec)-1))
            first_detection_speed = float(speed_vec.iloc[idx_plot])
            plt.annotate('mouse first\ndetection',
                        xy=(first_detection_time, first_detection_speed),
                        xytext=(first_detection_time + 0, first_detection_speed + 20),
                        arrowprops=dict(facecolor='darkslategrey', shrink=0.05, alpha=0.85, lw=0),
                        fontsize=12, color='black', ha='center')

        # indicate LED phase(s):
        if use_LED_light and (LED_lights_DLC_bp['LED light'], 'likelihood') in df_raw.columns:
            led_light_bp = LED_lights_DLC_bp['LED light']
            led_light_on = df_raw[(led_light_bp, 'likelihood')] > likelihood_threshold
            LED_time_vec = df_raw.index[led_light_on] / frame_rate
            if not LED_time_vec.empty:
                plt.fill_betweenx([0, ylim], LED_time_vec.min(), LED_time_vec.max(), 
                                color='yellow', alpha=0.2, label='LED light ON')

        moving = speed_vec >= movement_threshold
        moving_diff = np.diff(moving.astype(int))
        moving_start_indices = np.where(moving_diff == 1)[0] + 1
        moving_end_indices   = np.where(moving_diff == -1)[0] + 1
        if moving.iloc[0]:
            moving_start_indices = np.insert(moving_start_indices, 0, 0)
        if moving.iloc[-1]:
            moving_end_indices = np.append(moving_end_indices, len(moving))
        if len(moving_start_indices) > 0 and len(moving_end_indices) > 0:
            for i, (start_idx, end_idx) in enumerate(zip(moving_start_indices, moving_end_indices)):
                start_time = time_vec[start_idx]
                end_time   = time_vec[end_idx - 1] if end_idx < len(time_vec) else time_vec[-1]
                label = f'mouse moving\n(speed$\\geq${movement_threshold} {spatial_unit}/s)' if i == 0 else None
                plt.axhspan(ylim-21, ylim-18,
                            xmin=start_time/time_vec[-1], xmax=end_time/time_vec[-1],
                            color='darkturquoise', alpha=0.95, label=label, lw=0)
                plt.plot(time_vec[start_idx:end_idx], speed_vec.iloc[start_idx:end_idx], color='darkturquoise', lw=0.75)

        total_time_in_arena  = time_vec[-1] - time_vec[0]
        total_moving_time    = moving.sum()/frame_rate
        total_distance_moved = speed_vec[moving].sum() * time_step
        avg_speed_moving     = speed_vec[moving].mean()
        avg_speed_overall    = speed_vec.mean()
        max_speed            = speed_vec.max()
        total_movie_length   = len(df_raw) / frame_rate

        if use_LED_light:
            plt.annotate(
                f'total time in arena during LED: {total_time_in_arena:.2f} s (total recording time: {total_movie_length:.2f} s)\n'
                f'total moving time during LED: {total_moving_time:.2f} s (= {total_moving_time / total_time_in_arena * 100:.1f}%)\n'
                f'avg. speed during moving: {avg_speed_moving:.2f} {spatial_unit}/s ({avg_speed_moving/100* 3.6:.2f} km/h)\n'
                f'avg. speed overall: {avg_speed_overall:.2f} {spatial_unit}/s (= {avg_speed_overall/100* 3.6:.2f} km/h)\n'
                f'max. speed: {max_speed:.2f} {spatial_unit}/s (= {max_speed/100 * 3.6:.2f} km/h)\n'
                f'total distance moved: {total_distance_moved:.2f} {spatial_unit}',
                xy=(0.25, 0.98), xycoords='axes fraction', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0),
                verticalalignment='top', horizontalalignment='left')
        else:
            plt.annotate(
                f'total time in arena: {total_time_in_arena:.2f} s (total recording time: {total_movie_length:.2f} s)\n'
                f'total moving time: {total_moving_time:.2f} s (= {total_moving_time / total_time_in_arena * 100:.1f}%)\n'
                f'avg. speed during moving: {avg_speed_moving:.2f} {spatial_unit}/s ({avg_speed_moving/100* 3.6:.2f} km/h)\n'
                f'avg. speed overall: {avg_speed_overall:.2f} {spatial_unit}/s (= {avg_speed_overall/100* 3.6:.2f} km/h)\n'
                f'max. speed: {max_speed:.2f} {spatial_unit}/s (= {max_speed/100 * 3.6:.2f} km/h)\n'
                f'total distance moved: {total_distance_moved:.2f} {spatial_unit}',
                xy=(0.25, 0.98), xycoords='axes fraction', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0),
                verticalalignment='top', horizontalalignment='left')

        plt.title(f"mouse speed in {curr_filename_clean} ({title_suffix})")
        plt.xlabel('time (s)')
        plt.ylabel(f'speed ({spatial_unit}/s)')
        plt.grid()
        plt.xlim(0, time_vec[-1])
        plt.ylim(0, ylim)
        plt.legend(loc='upper left')
        plt.tight_layout()
        outname_speed = f"{curr_filename_clean}_mouse_speed" + ("_LED_filtered" if use_LED_light else "") + ".pdf"
        plt.savefig(os.path.join(curr_results_path, outname_speed), dpi=300)
        plt.savefig(os.path.join(curr_results_path, outname_speed.replace(".pdf", ".png")), dpi=300, transparent=True)
        plt.close()

        # ---------- time markers ----------
        if use_LED_light:
            led_light_bp = LED_lights_DLC_bp['LED light']
            first_LED_on_frame     = df_raw[(led_light_bp, 'likelihood')] > likelihood_threshold
            first_LED_on_time      = df_raw.index[first_LED_on_frame].min() / frame_rate
            first_LED_on_frame_idx = df_raw.index[first_LED_on_frame].min()
        else:
            first_LED_on_time      = None
            first_LED_on_frame_idx = None
        # first_Mouse_in_arena_frame     = df_raw[(bp_name, 'likelihood')] > likelihood_threshold
        # first_Mouse_in_arena_time      = df_raw.index[first_Mouse_in_arena_frame].min() / frame_rate
        # first_Mouse_in_arena_frame_idx = df_raw.index[first_Mouse_in_arena_frame].min()
        first_Mouse_in_arena_frame = pd.Series(was_mouse_in_arena_all, index=df_raw.index)
        if first_Mouse_in_arena_frame.any():
            first_Mouse_in_arena_frame_idx = int(first_Mouse_in_arena_frame[first_Mouse_in_arena_frame].index.min())
            first_Mouse_in_arena_time = first_Mouse_in_arena_frame_idx / frame_rate
        else:
            first_Mouse_in_arena_frame_idx = None
            first_Mouse_in_arena_time = None

        # ---------- prepare save measurements ----------
        measurements = {
            'filename': curr_filename_clean,
            'bodypart_label': bodypart_label,
            'bodypart_dlc_name': bp_name,
            'movie_frame_rate': frame_rate,
            'total_recording_time_in_s': total_movie_length,
            'total_time_in_arena_in_s': total_time_in_arena,
            'total_distance_moved_in_spatial_unit': total_distance_moved,
            'total_moving_time_in_s': total_moving_time,
            'avg_speed_moving': avg_speed_moving,
            'avg_speed_overall': avg_speed_overall,
            'max_speed': max_speed,
            'num_frames_with_mouse': len(df_cleaned),
            'num_frames_total': len(df_raw),
            'num_frames_moving': moving.sum(),
            'first_Mouse_in_arena_time_in_s': first_detection_time if first_detection_time is not None else np.nan,
            'first_Mouse_in_arena_frame_idx': first_detection_frame_idx if first_detection_frame_idx is not None else -1,
            'use_LED_light': use_LED_light,
            'spatial_unit': spatial_unit,
            'movement_threshold': movement_threshold,
            'arena_size': arena_size,
        }
        if border_margin is not None:
            measurements['border_margin'] = border_margin

        if use_LED_light:
            measurements.update({
                'num_frames_LED_on':  (df_raw[(LED_lights_DLC_bp['LED light'], 'likelihood')] > likelihood_threshold).sum(),
                'num_frames_LED_off': (df_raw[(LED_lights_DLC_bp['LED light'], 'likelihood')] <= likelihood_threshold).sum(),
                'first_LED_on_time_in_s': first_LED_on_time,
                'first_LED_on_frame_idx': first_LED_on_frame_idx,
            })


        # save time/speed series (all transformed frames) + flags:
        """ speed_df_dict = {
            'time': time_vec_raw_all,
            'speed': speed_vec_raw_all,
            'was_mouse_in_arena': df_raw[(bp_name, 'likelihood')] > likelihood_threshold,
            'was_mouse_moving':   speed_vec_raw_all >= movement_threshold} """
        speed_df_dict = {
            'time': time_vec_raw_all,
            'speed': speed_vec_raw_all,
            'was_mouse_in_arena': pd.Series(was_mouse_in_arena_all, index=df_all_transformed.index),
            'was_mouse_moving':   speed_vec_raw_all >= movement_threshold}
        if use_LED_light:
            speed_df_dict['was_LED_light_on'] = df_raw[(LED_lights_DLC_bp['LED light'], 'likelihood')] > likelihood_threshold
        speed_df = pd.DataFrame(speed_df_dict)
        
        if arena_mode.lower().strip() == "epm":
            epm_cache["speed_df"] = speed_df
            epm_cache["speed_cut_df"] = None  # may remain None if cut_tracking False or cut fails
        
        valid_uncut_obj = speed_df["was_mouse_in_arena"].astype(bool).to_numpy()
        if use_LED_light and "was_LED_light_on" in speed_df.columns:
            valid_uncut_obj &= speed_df["was_LED_light_on"].astype(bool).to_numpy()
        
        obj_metrics_uncut = compute_object_zone_metrics(
            df_points=df_all_transformed,
            bp_name=bp_name,
            frame_rate=frame_rate,
            objects_geom=objects_geom,
            valid_mask=valid_uncut_obj,
            prefix="(uncut)")
        measurements.update(obj_metrics_uncut)
            
        # ---------- freeze analysis (uncut) ----------
        speed_s = speed_df['speed'].rolling(freeze_smooth_win, center=True, min_periods=1).median()

        valid_uncut = speed_df['was_mouse_in_arena'].astype(bool)
        if use_LED_light and ('was_LED_light_on' in speed_df.columns):
            valid_uncut &= speed_df['was_LED_light_on'].astype(bool)

        is_freeze = valid_uncut & (speed_s < freeze_speed_threshold)

        # Bouts extrahieren (positionsbasiert)
        bouts_uncut, summary_uncut = extract_bouts(is_freeze.to_numpy(), frame_rate, freeze_min_duration_s, 
                                                freeze_gap_merge_max_s=freeze_gap_merge_max_s)

        # update measurements:
        measurements["freeze_speed_threshold"] = freeze_speed_threshold
        measurements["freeze_gap_merge_max_s"] = freeze_gap_merge_max_s
        measurements.update({f"{k} (uncut)": v for k, v in summary_uncut.items()})

        # Freeze-Flag
        speed_df['freezing_frame (uncut)'] = is_freeze.to_numpy()

        # Bout-Metadaten-Spalten
        speed_df['freezing_bout_id (uncut)']          = pd.Series(pd.NA, index=speed_df.index, dtype='Int64')
        speed_df['freezing_bout_start_s (uncut)']     = np.nan
        speed_df['freezing_bout_end_s (uncut)']       = np.nan
        speed_df['freezing_bout_duration_s (uncut)']  = np.nan

        col_id   = speed_df.columns.get_loc('freezing_bout_id (uncut)')
        col_s    = speed_df.columns.get_loc('freezing_bout_start_s (uncut)')
        col_e    = speed_df.columns.get_loc('freezing_bout_end_s (uncut)')
        col_dur  = speed_df.columns.get_loc('freezing_bout_duration_s (uncut)')

        for bid, b in enumerate(bouts_uncut, start=1):
            s_pos = int(b['start_frame'])
            e_pos = int(b['end_frame']) + 1

            start_global = float(speed_df.iloc[s_pos]['time'])
            end_global   = float(speed_df.iloc[e_pos-1]['time'])
            duration     = end_global - start_global

            speed_df.iloc[s_pos:e_pos, col_id]  = bid
            speed_df.iloc[s_pos:e_pos, col_s]   = start_global
            speed_df.iloc[s_pos:e_pos, col_e]   = end_global
            speed_df.iloc[s_pos:e_pos, col_dur] = duration

        # Safety: Frames ohne freezing bekommen keine Bout-Metadaten
        off = ~speed_df['freezing_frame (uncut)']
        """ speed_df.loc[off, ['freezing_bout_id (uncut)',
                        'freezing_bout_start_s (uncut)',
                        'freezing_bout_end_s (uncut)',
                        'freezing_bout_duration_s (uncut)']] = pd.NA """
        speed_df.loc[off, 'freezing_bout_id (uncut)'] = pd.NA
        speed_df.loc[off, ['freezing_bout_start_s (uncut)',
                        'freezing_bout_end_s (uncut)',
                        'freezing_bout_duration_s (uncut)']] = np.nan


        # save bout list:
        pd.DataFrame(bouts_uncut).to_csv(os.path.join(curr_results_path, 
                                                    f"{curr_filename_clean}_freeze_bouts_uncut.csv"), index=False)

        
        # ---------- save speed_df (uncut) ----------
        # save speed_df: 
        speed_df.to_csv(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_in_arena_speed.csv"), index=False)


        # ---------- center vs border metrics (uncut) ----------
        # use the full analysis window (df_cleaned) to compute uncut metrics:
        if arena_mode.lower().strip() != "epm":
            if not df_cleaned.empty:
                x_vals_full = df_cleaned[(bp_name, 'x')].values
                y_vals_full = df_cleaned[(bp_name, 'y')].values

                # inside-center boolean over the whole recording (same units as arena_size/border_margin):
                in_center_full = (
                    (x_vals_full >= border_margin) &
                    (x_vals_full <= arena_size - border_margin) &
                    (y_vals_full >= border_margin) &
                    (y_vals_full <= arena_size - border_margin)
                )
                in_border_full = ~in_center_full

                # times in seconds:
                time_in_center_full = in_center_full.sum() / frame_rate
                time_in_border_full = in_border_full.sum() / frame_rate

                # center-border crossings over the full recording:
                crossings_full = np.sum(np.diff(in_center_full.astype(int)) != 0)

                print(f"  uncut metrics: center={time_in_center_full:.2f}s, border={time_in_border_full:.2f}s, crossings={crossings_full}")

                # store alongside the other measurements:
                measurements.update({
                    'time_in_center_in_s (uncut)': float(time_in_center_full),
                    'time_in_border_in_s (uncut)': float(time_in_border_full),
                    'num_center_border_crossings (uncut)': int(crossings_full),
                })
            else:
                print("  uncut metrics: no frames after filtering; skipping.")
        else:
            print("  uncut metrics: EPM mode -> skipping center/border metrics.")

        # ---------- cut tracking (tolerant window search after first detection) ----------
        # requirements:
        # - speed_df must contain at least: 'time', 'speed', 'was_mouse_in_arena'
        # - parameters: cut_tracking (seconds), mouse_first_track_delay (seconds)
        # - tolerance parameters below can be tuned if needed
        if cut_tracking:
            speed_cut_df = speed_df.copy()

            # --- tolerance parameters (adjust as needed) ---
            max_false_frac   = 0.10     # max. Anteil an False im Fenster
            max_false_streak = 3        # max. Länge eines False-Blocks (in Frames)
            win_frames       = max(1, int(round(mouse_first_track_delay * frame_rate))) if mouse_first_track_delay else 0

            # Basis-Masken laden:
            if 'was_mouse_in_arena' not in speed_cut_df.columns or not speed_cut_df['was_mouse_in_arena'].notna().any():
                print("  cut_tracking: 'was_mouse_in_arena' column missing or empty; no cut applied.")
            else:
                in_arena = speed_cut_df['was_mouse_in_arena'].astype(bool).values
                if use_LED_light and 'was_LED_light_on' in speed_cut_df.columns:
                    led_on = speed_cut_df['was_LED_light_on'].astype(bool).values
                    was_valid_all = in_arena & led_on
                else:
                    if use_LED_light and 'was_LED_light_on' not in speed_cut_df.columns:
                        print("  cut_tracking: LED requested, but 'was_LED_light_on' missing -> falling back to in-arena only.")
                    was_valid_all = in_arena

                # Für window_is_stable() die globale Maske bereitstellen:
                was_in = was_valid_all  # <-- window_is_stable() liest diese Variable

                # Startfenster suchen: erster stabiler Abschnitt nach dem ersten True
                true_idx = np.flatnonzero(was_valid_all)
                if true_idx.size > 0 and win_frames > 0:
                    first_true_idx = int(true_idx[0])

                    start_idx_found = None
                    last_start = len(was_valid_all) - win_frames
                    """ for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
                        if window_is_stable(s):
                            start_idx_found = s
                            break """
                    for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
                        if window_is_stable(was_valid_all, s, win_frames, max_false_frac, max_false_streak):
                            start_idx_found = s
                            break

                    if start_idx_found is not None:
                        # Cut [t_start, t_start + cut_tracking]
                        t_start = float(speed_cut_df.iloc[start_idx_found]['time'])
                        t_end   = t_start + float(cut_tracking)

                        pre_len = len(speed_cut_df)
                        speed_cut_df = speed_cut_df[(speed_cut_df['time'] >= t_start) & (speed_cut_df['time'] <= t_end)].copy()
                        post_len = len(speed_cut_df)

                        # reset time
                        speed_cut_df['reset_time'] = speed_cut_df['time'] - t_start

                        # Valid-Frames im Cut (Arena ∧ evtl. LED)
                        if use_LED_light and 'was_LED_light_on' in speed_cut_df.columns:
                            valid = speed_cut_df['was_mouse_in_arena'].astype(bool) & speed_cut_df['was_LED_light_on'].astype(bool)
                        else:
                            valid = speed_cut_df['was_mouse_in_arena'].astype(bool)

                        # Moving/Nicht-Moving im Cut
                        speed_cut_df['was_mouse_moving'] = False
                        speed_cut_df.loc[valid, 'was_mouse_moving'] = (speed_cut_df.loc[valid, 'speed'] >= movement_threshold)

                        # Zeiten (s)
                        moving_time_cut     = speed_cut_df.loc[valid, 'was_mouse_moving'].sum() / frame_rate
                        nonmoving_time_cut  = (valid.sum() - speed_cut_df.loc[valid, 'was_mouse_moving'].sum()) / frame_rate
                        total_in_arena_cut  = valid.sum() / frame_rate

                        # further cut-speed metrics:
                        has_valid  = bool(valid.any())
                        has_moving = bool((valid & speed_cut_df['was_mouse_moving']).any()) if has_valid else False

                        avg_speed_moving_cut = float(
                            speed_cut_df.loc[valid & speed_cut_df['was_mouse_moving'], 'speed'].mean()
                        ) if has_moving else np.nan

                        avg_speed_overall_cut = float(
                            speed_cut_df.loc[valid, 'speed'].mean()
                        ) if has_valid else np.nan

                        max_speed_cut = float(
                            speed_cut_df.loc[valid, 'speed'].max()
                        ) if has_valid else np.nan

                        total_distance_moved_cut = float(
                            speed_cut_df.loc[valid & speed_cut_df['was_mouse_moving'], 'speed'].sum() * time_step
                        ) if has_valid else 0.0

                        # ins measurements-Dict schreiben
                        measurements.update({
                            'avg_speed_moving (cut)': avg_speed_moving_cut,
                            'avg_speed_overall (cut)': avg_speed_overall_cut,
                            'max_speed (cut)': max_speed_cut,
                            'total_distance_moved_in_spatial_unit (cut)': total_distance_moved_cut,
                            'num_frames_moving (cut)': int(speed_cut_df.loc[valid, 'was_mouse_moving'].sum()),
                            'num_frames_total (cut)': int(valid.sum()),
                        })
                        

                        print(
                            f"  cut_tracking: start at t={t_start:.3f}s (idx {start_idx_found}), "
                            f"window={mouse_first_track_delay:.2f}s, cut={cut_tracking:.2f}s, "
                            f"kept {post_len}/{pre_len} frames."
                        )

                        # in measurements notieren (für spätere Markierungen/Plots)
                        measurements.update({
                            'cut start in s': t_start,
                            'cut end in s': t_end,
                            'cut duration in s': cut_tracking,
                        })

                        # -------------------------------------------------------------------------
                        # PATCH/NEW: object zone metrics (cut)
                        # -------------------------------------------------------------------------
                        # Preconditions:
                        #   * objects_geom has been created earlier from df_all_transformed
                        #   * compute_object_zone_metrics(...) exists
                        #   * df_all_transformed is the fully transformed timeline (no likelihood filtering)
                        #
                        # Key idea:
                        #   speed_cut_df keeps the original frame index from speed_df, which matches
                        #   df_all_transformed.index. Therefore we can subset df_all_transformed by
                        #   speed_cut_df.index directly, without any time alignment.
                        #
                        if ('objects_geom' in locals()) and isinstance(objects_geom, list) and (len(objects_geom) > 0):
                            # frame-synchronous subset for the cut window
                            df_all_cut = df_all_transformed.loc[speed_cut_df.index].copy()

                            # validity aligned 1:1 to df_all_cut rows
                            valid_cut_obj = speed_cut_df['was_mouse_in_arena'].astype(bool).to_numpy()
                            if use_LED_light and ('was_LED_light_on' in speed_cut_df.columns):
                                valid_cut_obj &= speed_cut_df['was_LED_light_on'].astype(bool).to_numpy()

                            obj_metrics_cut = compute_object_zone_metrics(
                                df_points=df_all_cut,
                                bp_name=bp_name,
                                frame_rate=frame_rate,
                                objects_geom=objects_geom,
                                valid_mask=valid_cut_obj,
                                prefix="(cut)"
                            )
                            measurements.update(obj_metrics_cut)
                        else:
                            # If you want, you can keep this silent. I print once for clarity.
                            print("  object metrics (cut): objects_geom missing/empty -> skipping object-zone cut metrics.")
                        # -------------------------------------------------------------------------


                    else:
                        print("  cut_tracking: no stable segment (arena ± LED) found under current tolerances; no cut applied.")
                else:
                    print("  cut_tracking: no valid detection (arena ± LED) or window size=0; no cut applied.")

        
            if arena_mode.lower().strip() == "epm":
                epm_cache["speed_cut_df"] = speed_cut_df
            
            # ---------- freeze analysis (cut) ----------
            if cut_tracking and not speed_cut_df.empty:
                # 1) smoothed speed
                speed_s_cut = speed_cut_df['speed'].rolling(
                    freeze_smooth_win, center=True, min_periods=1
                ).median()

                # 2) gültige Frames (Arena & optional LED)
                valid_cut = speed_cut_df['was_mouse_in_arena'].astype(bool)
                if use_LED_light and ('was_LED_light_on' in speed_cut_df.columns):
                    valid_cut &= speed_cut_df['was_LED_light_on'].astype(bool)

                # 3) Freeze-Maske
                is_freeze_cut = valid_cut & (speed_s_cut < freeze_speed_threshold)

                # 4) Bouts aus der **positionellen** Maske extrahieren
                bouts_cut, summary_cut = extract_bouts(is_freeze_cut.to_numpy(), frame_rate, freeze_min_duration_s,
                                                    freeze_gap_merge_max_s=freeze_gap_merge_max_s)

                # 5) Measurements aktualisieren
                measurements.update({f"{k} (cut)": v for k, v in summary_cut.items()})

                # 6) Spalten (Frame-Flag + Bout-Metadaten)
                speed_cut_df['freezing_frame (cut)'] = is_freeze_cut.to_numpy()

                # NA-fähiger Integer für IDs
                speed_cut_df['freezing_bout_id (cut)']           = pd.Series(pd.NA, index=speed_cut_df.index, dtype='Int64')
                speed_cut_df['freezing_bout_start_s (cut)']      = np.nan
                speed_cut_df['freezing_bout_end_s (cut)']        = np.nan
                speed_cut_df['freezing_bout_duration_s (cut)']   = np.nan

                # Spaltenpositionen für iloc-Zuweisung
                col_id   = speed_cut_df.columns.get_loc('freezing_bout_id (cut)')
                col_s    = speed_cut_df.columns.get_loc('freezing_bout_start_s (cut)')
                col_e    = speed_cut_df.columns.get_loc('freezing_bout_end_s (cut)')
                col_dur  = speed_cut_df.columns.get_loc('freezing_bout_duration_s (cut)')

                # 7) Bout-Infos **positionsbasiert** eintragen
                for bid, b in enumerate(bouts_cut, start=1):
                    s_pos = int(b['start_frame'])
                    e_pos = int(b['end_frame']) + 1   # iloc: Ende exklusiv

                    # globale Zeiten aus der 'time'-Spalte holen
                    start_global = float(speed_cut_df.iloc[s_pos]['time'])
                    end_global   = float(speed_cut_df.iloc[e_pos-1]['time'])
                    duration     = end_global - start_global

                    speed_cut_df.iloc[s_pos:e_pos, col_id]  = bid
                    speed_cut_df.iloc[s_pos:e_pos, col_s]   = start_global
                    speed_cut_df.iloc[s_pos:e_pos, col_e]   = end_global
                    speed_cut_df.iloc[s_pos:e_pos, col_dur] = duration

                # 8) Safety: außerhalb von True-Frames keine Bout-Metadaten
                off = ~speed_cut_df['freezing_frame (cut)']
                speed_cut_df.loc[off, ['freezing_bout_id (cut)',
                                    'freezing_bout_start_s (cut)',
                                    'freezing_bout_end_s (cut)',
                                    'freezing_bout_duration_s (cut)']] = pd.NA

                # (optional) kleine Konsistenzprüfung für Debug-Ausgabe
                bad = speed_cut_df['freezing_bout_id (cut)'].notna() & (~speed_cut_df['freezing_frame (cut)'])
                if bad.any():
                    print(f"  WARNING: {bad.sum()} rows carry bout metadata while freezing_frame=False (indexing mismatch).")



            # ---------- save results (cut) ----------
            # save cut version:
            speed_cut_df.to_csv(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_in_arena_speed_cut.csv"), index=False)

        # ---------- replot tracks & heatmap for the cut interval ----------
        if cut_tracking and 'speed_cut_df' in locals() and isinstance(speed_cut_df, pd.DataFrame) and not speed_cut_df.empty:
            t_start_cut = float(speed_cut_df['time'].min())
            t_end_cut   = float(speed_cut_df['time'].max())

            # subset of the already filtered analysis DataFrame (df_cleaned) to the cut window
            mask_vis = ((df_cleaned.index / frame_rate >= t_start_cut) &
                        (df_cleaned.index / frame_rate <= t_end_cut))
            df_cut_vis = df_cleaned.loc[mask_vis].copy()

            if df_cut_vis.empty:
                print("  cut plots: no frames in cut window after filtering; skipping cut plots.")
            else:
                # ---------- calc. center vs border metrics ----------
                x_vals = df_cut_vis[(bp_name, 'x')].values
                y_vals = df_cut_vis[(bp_name, 'y')].values

                # boolean mask: inside center square
                in_center = (
                    (x_vals >= border_margin) &
                    (x_vals <= arena_size - border_margin) &
                    (y_vals >= border_margin) &
                    (y_vals <= arena_size - border_margin)
                )
                in_border = ~in_center

                # times: Anzahl der Frames * 1/framerate
                time_in_center = in_center.sum() / frame_rate
                time_in_border = in_border.sum() / frame_rate

                # crossings: Übergänge center <-> border zählen
                crossings = np.sum(np.diff(in_center.astype(int)) != 0)

                print(f"  cut metrics: center={time_in_center:.2f}s, border={time_in_border:.2f}s, crossings={crossings}")

                # update measurements list: 
                measurements.update({
                    'total_time_in_arena_s (cut)':     moving_time_cut + nonmoving_time_cut,
                    'total_moving_time_in_s (cut)':    moving_time_cut,
                    'total_nonmoving_time_in_s (cut)': nonmoving_time_cut,
                    'time_in_center_in_s (cut)': time_in_center,
                    'time_in_border_in_s (cut)': time_in_border,
                    'num_center_border_crossings (cut)': int(crossings),
                })
                
                
                
                # --- plot: tracks (cut) ---
                plt.figure(figsize=(12, 8))
                plt.scatter(df_cut_vis[(bp_name, 'x')],
                            df_cut_vis[(bp_name, 'y')],
                            s=10, label=bp_name+' points (cut)', alpha=0.5)
                for corner_name, (corner_x, corner_y) in arena_corners_transformed.items():
                    plt.scatter(corner_x, corner_y, s=100, label=f'arena {corner_name}', edgecolor='black', alpha=0.5)
                # center-border boundary box:
                if arena_mode.lower().strip() != "epm":
                    rect = plt.Rectangle(
                        (border_margin, border_margin),
                        arena_size - 2 * border_margin,
                        arena_size - 2 * border_margin,
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8,
                        label=f'center-border boundary\n(border margin: {border_margin} cm)'
                    )
                    plt.gca().add_patch(rect)
                # arena boundary box:
                rect_arena = plt.Rectangle(
                    (0, 0),
                    arena_size,
                    arena_size,
                    linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.8,
                    label=f'arena boundary\n(size: {arena_size} cm)'
                )
                plt.gca().add_patch(rect_arena)
                # overlay objects (inner + zone), if present
                overlay_objects(plt.gca(), objects_geom, show_labels=True)
                plt.title(f"arena corners and {bp_name} points (cut) in\n{curr_filename_clean} [cut {t_start_cut:.2f}-{t_end_cut:.2f}s]")
                plt.xlabel(f"x ({spatial_unit})")
                plt.ylabel(f"y ({spatial_unit})")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlim(-arena_size * 0.05, arena_size * 1.05)
                plt.ylim(-arena_size * 0.05, arena_size * 1.05)
                plt.tight_layout()
                outname_tracks_cut = f"{curr_filename_clean}_mouse_tracks_all_view_corrected_cut.pdf"
                plt.savefig(os.path.join(curr_results_path, outname_tracks_cut), dpi=300, transparent=True)
                plt.savefig(os.path.join(curr_results_path, outname_tracks_cut.replace(".pdf", ".png")), dpi=300, transparent=True)
                plt.close()

                # --- plot: heatmap (cut) ---
                plt.figure(figsize=(12, 8))
                x_data_cut = df_cut_vis[(bp_name, 'x')].dropna()
                y_data_cut = df_cut_vis[(bp_name, 'y')].dropna()
                if len(x_data_cut) == 0 or len(y_data_cut) == 0:
                    print("  cut heatmap: no valid x/y samples in cut window; skipping heatmap.")
                else:
                    bins = 60
                    hist_cut, xedges, yedges = np.histogram2d(
                        x_data_cut, y_data_cut, bins=bins, range=[[0, arena_size], [0, arena_size]])
                    smoothed_hist_cut = gaussian_filter(hist_cut, sigma=2.5)
                    plt.imshow(smoothed_hist_cut.T, origin='lower',
                            extent=[0, arena_size, 0, arena_size], cmap='viridis', aspect='auto')
                    cbar = plt.colorbar(label=f'smoothed occupancy count\n(bin size = {arena_size/bins:.2f} {spatial_unit})',
                                        fraction=0.046, pad=0.005)
                    cbar.ax.tick_params(labelsize=12)
                    if heat_plot_scatter:
                        plt.scatter(x_data_cut, y_data_cut, s=10, label=bp_name + ' points (cut)', alpha=0.5, color='pink')
                    else:
                        plt.plot(x_data_cut, y_data_cut, lw=0.5, alpha=0.7, label=bp_name + ' points (cut)', color='pink')
                    # center-border boundary box:
                    if arena_mode.lower().strip() != "epm":
                        rect = plt.Rectangle(
                            (border_margin, border_margin),
                            arena_size - 2 * border_margin,
                            arena_size - 2 * border_margin,
                            linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8,
                            label=f'center-border boundary\n(border margin: {border_margin} cm)')
                        plt.gca().add_patch(rect)
                    # arena boundary box:
                    rect_arena = plt.Rectangle(
                        (0, 0),
                        arena_size,
                        arena_size,
                        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.8,
                        label=f'arena boundary\n(size: {arena_size} cm)')
                    plt.gca().add_patch(rect_arena)
                    # overlay objects (inner + zone), if present
                    overlay_objects(plt.gca(), objects_geom, show_labels=True)
                    plt.title(f"mouse heatmap and {bp_name} points (cut) in\n{curr_filename_clean} [cut {t_start_cut:.2f}-{t_end_cut:.2f}s]")
                    plt.xlabel(f"x ({spatial_unit})", fontsize=14)
                    plt.ylabel(f"y ({spatial_unit})", fontsize=14)
                    plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
                    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim(0, arena_size)
                    plt.ylim(0, arena_size)
                    plt.tight_layout()
                    outname_heatmap_cut = f"{curr_filename_clean}_mouse_heatmap_smoothed_cut.pdf"
                    plt.savefig(os.path.join(curr_results_path, outname_heatmap_cut), dpi=300)
                    plt.savefig(os.path.join(curr_results_path, outname_heatmap_cut.replace(".pdf", ".png")), dpi=300, 
                                transparent=True)
                    plt.close()

        # ---------- cache EPM objects_geom and df_all_transformed ----------
        # EPM_multi_bp: overwrite measurements with multi-bp values using EXISTING keys only
        if arena_mode.lower().strip() == "epm" and bodypart_label == "EPM_multi_bp":
            if epm_multi_cache is None:
                raise RuntimeError("EPM_multi_bp requested but epm_multi_cache is None. Check placement of cache computation.")

            # overwrite uncut speed metrics
            speed_multi = epm_multi_cache["speed_multi"]
            valid_multi = epm_multi_cache["valid_multi"]

            # moving definition (same as elsewhere)
            moving_multi = (speed_multi >= movement_threshold) & valid_multi

            total_movie_length = len(df_all_transformed) / frame_rate
            total_time_in_arena = float(np.sum(valid_multi) / frame_rate)
            total_moving_time = float(np.sum(moving_multi) / frame_rate)

            total_distance_moved = float(np.nansum(speed_multi[moving_multi]) * time_step) if np.any(moving_multi) else 0.0
            avg_speed_moving = float(np.nanmean(speed_multi[moving_multi])) if np.any(moving_multi) else np.nan
            avg_speed_overall = float(np.nanmean(speed_multi[valid_multi])) if np.any(valid_multi) else np.nan
            max_speed = float(np.nanmax(speed_multi[valid_multi])) if np.any(valid_multi) else np.nan

            measurements.update({
                'total_recording_time_in_s': total_movie_length,
                'total_time_in_arena_in_s': total_time_in_arena,
                'total_distance_moved_in_spatial_unit': total_distance_moved,
                'total_moving_time_in_s': total_moving_time,
                'avg_speed_moving': avg_speed_moving,
                'avg_speed_overall': avg_speed_overall,
                'max_speed': max_speed,
                'num_frames_with_mouse': int(np.sum(valid_multi)),
                'num_frames_total': len(df_all_transformed),
                'num_frames_moving': int(np.sum(moving_multi)),
            })

            # overwrite uncut ROI metrics into the SAME column names as usual
            metrics_uncut = epm_multi_bp_object_metrics_from_roi_masks(
                roi_bool_df=roi_agg_df,
                frame_rate=frame_rate,
                prefix="(uncut)")
            measurements.update(metrics_uncut)

            # cut window metrics if cut exists
            if cut_tracking and ('cut start in s' in measurements) and ('cut end in s' in measurements):
                t_start = float(measurements['cut start in s'])
                t_end = float(measurements['cut end in s'])

                # build cut indices using time from df_all_transformed index
                t_vec = df_all_transformed.index.to_numpy(dtype=float) / frame_rate
                in_cut = (t_vec >= t_start) & (t_vec <= t_end)

                speed_multi_cut = speed_multi[in_cut]
                valid_multi_cut = valid_multi[in_cut]

                moving_multi_cut = (speed_multi_cut >= movement_threshold) & valid_multi_cut

                total_time_in_arena_cut = float(np.sum(valid_multi_cut) / frame_rate)
                total_moving_time_cut = float(np.sum(moving_multi_cut) / frame_rate)
                total_nonmoving_time_cut = float(total_time_in_arena_cut - total_moving_time_cut)

                total_distance_moved_cut = float(np.nansum(speed_multi_cut[moving_multi_cut]) * time_step) if np.any(moving_multi_cut) else 0.0
                avg_speed_moving_cut = float(np.nanmean(speed_multi_cut[moving_multi_cut])) if np.any(moving_multi_cut) else np.nan
                avg_speed_overall_cut = float(np.nanmean(speed_multi_cut[valid_multi_cut])) if np.any(valid_multi_cut) else np.nan
                max_speed_cut = float(np.nanmax(speed_multi_cut[valid_multi_cut])) if np.any(valid_multi_cut) else np.nan

                measurements.update({
                    'total_time_in_arena_s (cut)': total_time_in_arena_cut,
                    'total_moving_time_in_s (cut)': total_moving_time_cut,
                    'total_nonmoving_time_in_s (cut)': total_nonmoving_time_cut,
                    'avg_speed_moving (cut)': avg_speed_moving_cut,
                    'avg_speed_overall (cut)': avg_speed_overall_cut,
                    'max_speed (cut)': max_speed_cut,
                    'total_distance_moved_in_spatial_unit (cut)': total_distance_moved_cut,
                    'num_frames_moving (cut)': int(np.sum(moving_multi_cut)),
                    'num_frames_total (cut)': int(np.sum(valid_multi_cut)),
                })

                # cut ROI metrics
                roi_cut = roi_agg_df.loc[in_cut].copy()
                metrics_cut = epm_multi_bp_object_metrics_from_roi_masks(
                    roi_bool_df=roi_cut,
                    frame_rate=frame_rate,
                    prefix="(cut)"
                )
                measurements.update(metrics_cut)

        # ---------- finally save measurements ----------
        # update the all_mice measurements DataFrame:
        measurements_df = pd.concat([measurements_df, pd.DataFrame([measurements])], ignore_index=True)
        # save the current mouse's measurements:
        measurements_df_transposed = pd.DataFrame(measurements, index=[curr_filename_clean]).T
        measurements_df_transposed.to_csv(os.path.join(curr_results_path, f"{curr_filename_clean}_measurements.csv"), index=True)


        # ---------- plot: smoothed speed with movement segments & freezing bar (UNCUT only) ----------
        t_all = speed_df['time'].values
        speed_s_plot = speed_df['speed'].rolling(freeze_smooth_win, center=True, min_periods=1).median()

        # masks:
        in_arena_all = speed_df['was_mouse_in_arena'].astype(bool).values
        is_freeze_all = (in_arena_all) & (speed_s_plot.values < freeze_speed_threshold)
        is_moving_all = (in_arena_all) & ~is_freeze_all   # "moving" = NICHT freezing

        plt.figure(figsize=(12, 6))
        plt.plot(t_all, speed_s_plot, lw=0.8, alpha=0.7, label='smoothed speed', color='gray')
        
        # indicate LED phase(s):
        if use_LED_light and ('was_LED_light_on' in speed_df.columns):
            led_mask_all = speed_df['was_LED_light_on'].fillna(False).to_numpy()

            if led_mask_all.any():
                led_diff = np.diff(led_mask_all.astype(int))
                led_starts = np.where(led_diff == 1)[0] + 1
                led_ends   = np.where(led_diff == -1)[0] + 1
                if led_mask_all[0]:
                    led_starts = np.insert(led_starts, 0, 0)
                if led_mask_all[-1]:
                    led_ends = np.append(led_ends, len(led_mask_all))

                led_labeled = False
                for s, e in zip(led_starts, led_ends):
                    if e > s:
                        plt.fill_betweenx(
                            [0, ylim_smoothed],
                            t_all[s], t_all[e-1],
                            color='yellow', alpha=0.2,
                            label=('LED light ON' if not led_labeled else None),
                            zorder=0
                        )
                        led_labeled = True

        # moving segments (cyan) (only plot segments, not the whole line):
        moving_diff = np.diff(is_moving_all.astype(int))
        moving_starts = np.where(moving_diff == 1)[0] + 1
        moving_ends   = np.where(moving_diff == -1)[0] + 1
        if is_moving_all[0]:
            moving_starts = np.insert(moving_starts, 0, 0)
        if is_moving_all[-1]:
            moving_ends = np.append(moving_ends, len(is_moving_all))

        first_label_done = False
        for s, e in zip(moving_starts, moving_ends):
            if e > s:
                plt.plot(t_all[s:e], speed_s_plot.values[s:e],
                        lw=1.2,
                        color='cyan',
                        label='moving (≥ freeze thr.)' if not first_label_done else None)
                first_label_done = True

        # freezing-bar on top of the trace: 
        freeze_diff = np.diff(is_freeze_all.astype(int))
        freeze_starts = np.where(freeze_diff == 1)[0] + 1
        freeze_ends   = np.where(freeze_diff == -1)[0] + 1
        if is_freeze_all[0]:
            freeze_starts = np.insert(freeze_starts, 0, 0)
        if is_freeze_all[-1]:
            freeze_ends = np.append(freeze_ends, len(is_freeze_all))

        # indicate freezing bouts:
        bar_y0, bar_y1 = ylim_smoothed - 6, ylim_smoothed - 3
        bar_labeled = False
        if 'bouts_uncut' in locals() and len(bouts_uncut) > 0:
            for bout in bouts_uncut:
                # bouts_uncut-Elemente sind z.B. {'start_s': ..., 'end_s': ..., 'duration_s': ...}
                xs = float(bout['start_time_s'])
                xe = float(bout['end_time_s'])
                plt.axhspan(bar_y0, bar_y1,
                            xmin=xs / t_all[-1],
                            xmax=xe / t_all[-1],
                            alpha=0.9,
                            color='mediumpurple',
                            lw=0,
                            label='freezing bouts' if not bar_labeled else None)
                bar_labeled = True
        else:
            # Fallback (nur falls keine Bouts gebildet werden konnten):
            freeze_diff = np.diff(is_freeze_all.astype(int))
            freeze_starts = np.where(freeze_diff == 1)[0] + 1
            freeze_ends   = np.where(freeze_diff == -1)[0] + 1
            if is_freeze_all[0]:
                freeze_starts = np.insert(freeze_starts, 0, 0)
            if is_freeze_all[-1]:
                freeze_ends = np.append(freeze_ends, len(is_freeze_all))
            for s, e in zip(freeze_starts, freeze_ends):
                if e > s:
                    plt.axhspan(bar_y0, bar_y1,
                                xmin=t_all[s] / t_all[-1],
                                xmax=t_all[e-1] / t_all[-1],
                                alpha=0.9,
                                color='mediumpurple',
                                label='freezing bouts' if not bar_labeled else None)
                    bar_labeled = True
                
        # indicate freezings threshold:
        plt.axhline(freeze_speed_threshold, linestyle='--', lw=1.0, alpha=0.7, color='black',
                    label=f'freezing threshold = {freeze_speed_threshold:.2f} {spatial_unit}/s')

        # indicate first mouse in arena time point:
        if first_detection_time is not None and len(t_all) > 0:
            idx_plot = int(np.clip(np.searchsorted(t_all, first_detection_time), 0, len(t_all)-1))
            first_detection_speed = float(speed_s_plot.iloc[idx_plot])
            plt.annotate('mouse first\ndetection',
                        xy=(first_detection_time, first_detection_speed),
                        xytext=(first_detection_time + 0, first_detection_speed + ylim_smoothed/2),
                        arrowprops=dict(facecolor='darkslategrey', shrink=0.05, alpha=0.85, lw=0),
                        fontsize=12, color='black', ha='center', zorder=2)

        # indicate cut-start/end, if available:
        cut_start = measurements.get('cut start in s', None)
        cut_end   = measurements.get('cut end in s', None)
        if cut_tracking and (cut_start is not None) and (cut_end is not None):
            plt.axvline(cut_start, color='orange', lw=1.2, linestyle='--', label='cut start')
            plt.axvline(cut_end,   color='orange', lw=1.2, linestyle='-.',  label='cut end')

        plt.title(f"smoothed mouse speed in {curr_filename_clean} ({title_suffix})")
        plt.xlabel('time (s)')
        plt.ylabel(f'speed ({spatial_unit}/s)')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlim(0, t_all[-1] if len(t_all) else 0)
        plt.ylim(0, ylim_smoothed)
        plt.legend(loc='upper left')
        plt.tight_layout()
        outname_speed_smoothed = f"{curr_filename_clean}_mouse_speed_smoothed_freeze_uncut.pdf"
        plt.savefig(os.path.join(curr_results_path, outname_speed_smoothed), dpi=300)
        plt.savefig(os.path.join(curr_results_path, outname_speed_smoothed.replace(".pdf", ".png")), dpi=300, 
                    transparent=True)
        plt.close()


    # ---------- EPM multi-bodypart aggregation ----------
    if arena_mode.lower().strip() == "epm":
        if epm_cache is None or epm_cache["objects_geom"] is None:
            print("EPM multi-bp: missing cache (objects_geom). Skipping multi-bp EPM aggregation.")
        else:
            objects_geom_run = epm_cache["objects_geom"]
            df_all_tr_run = epm_cache["df_all_transformed"]
            speed_df_run = epm_cache["speed_df"]
            speed_cut_df_run = epm_cache["speed_cut_df"]

            # ensure requested bps exist
            missing_bps = [bp for bp in epm_bp_list if (bp, "x") not in df_all_tr_run.columns]
            if len(missing_bps) > 0:
                print(f"EPM multi-bp: missing bps {missing_bps} in data. Skipping.")
            else:
                # ------------------------
                # uncut: valid mask aligned to df_all_transformed
                # ------------------------
                valid_uncut = speed_df_run["was_mouse_in_arena"].astype(bool).to_numpy()
                if use_LED_light and ("was_LED_light_on" in speed_df_run.columns):
                    valid_uncut &= speed_df_run["was_LED_light_on"].astype(bool).to_numpy()

                mask_uncut_df = epm_compute_roi_masks_for_bps(
                    df_points=df_all_tr_run,
                    bp_list=epm_bp_list,
                    objects_geom=objects_geom_run,
                    valid_mask=valid_uncut,
                    likelihood_threshold=likelihood_threshold
                )

                agg_uncut_df = epm_aggregate_roi_masks(
                    mask_df=mask_uncut_df,
                    mode=epm_aggregate_mode,
                    k=epm_k_of_n,
                )

                if arena_mode.lower().strip() == "epm":
                    # uncut metrics kommen aus dem Cache
                    if epm_cache is not None and "epm_multi_cache" in epm_cache:
                        epm_uncut_metrics = epm_cache["epm_multi_cache"].get("epm_obj_metrics_uncut", {}) or {}

                    # cut metrics: nur wenn du sie berechnet hast
                    # (du hast aktuell noch KEINEN epm_obj_metrics_cut im Cache, dazu unten)
                    epm_cut_metrics = {}

                # ------------------------
                # cut: if available
                # ------------------------
                epm_cut_metrics = {}
                if cut_tracking and (speed_cut_df_run is not None) and (not speed_cut_df_run.empty):
                    df_all_cut = df_all_tr_run.loc[speed_cut_df_run.index].copy()

                    valid_cut = speed_cut_df_run["was_mouse_in_arena"].astype(bool).to_numpy()
                    if use_LED_light and ("was_LED_light_on" in speed_cut_df_run.columns):
                        valid_cut &= speed_cut_df_run["was_LED_light_on"].astype(bool).to_numpy()

                    mask_cut_df = epm_compute_roi_masks_for_bps(
                        df_points=df_all_cut,
                        bp_list=epm_bp_list,
                        objects_geom=objects_geom_run,
                        valid_mask=valid_cut,
                        likelihood_threshold=likelihood_threshold
                    )

                    agg_cut_df = epm_aggregate_roi_masks(
                        mask_df=mask_cut_df,
                        mode=epm_aggregate_mode,
                        k=epm_k_of_n,
                    )

                # ------------------------
                # write results into a run-level row
                # ------------------------
                """ epm_run_measurements = {
                    "filename": curr_filename_clean,
                    "bodypart_label": "EPM_multi_bp",
                    "bodypart_dlc_name": ",".join(epm_bp_list),
                    "epm_multi_bp_mode": epm_aggregate_mode,
                    "epm_multi_bp_k_of_n": int(epm_k_of_n) if epm_aggregate_mode == "kofn" else np.nan,
                    "use_LED_light": use_LED_light,
                    "cut_tracking": float(cut_tracking) if cut_tracking else 0.0,
                }
                epm_run_measurements.update(epm_uncut_metrics)
                epm_run_measurements.update(epm_cut_metrics) """
                
                epm_run_measurements = measurements.copy()

                # append as an additional row in the global measurements_df
                # measurements_df = pd.concat([measurements_df, pd.DataFrame([epm_run_measurements])], ignore_index=True)


                # save as a separate per-run CSV in base_results_path (run-level, not per-bodypart folder)
                epm_out_csv = os.path.join(base_results_path, f"{curr_filename_clean}_EPM_multi_bp_measurements.csv")
                pd.DataFrame([epm_run_measurements]).to_csv(epm_out_csv, index=False)

                # optional: save the aggregated boolean time series for debugging
                # (uncut always; cut only if present)
                dbg_out_uncut = os.path.join(base_results_path, f"{curr_filename_clean}_EPM_multi_bp_roi_mask_uncut.csv")
                agg_uncut_df.to_csv(dbg_out_uncut, index=True)
                if cut_tracking and (speed_cut_df_run is not None) and (not speed_cut_df_run.empty):
                    dbg_out_cut = os.path.join(base_results_path, f"{curr_filename_clean}_EPM_multi_bp_roi_mask_cut.csv")
                    agg_cut_df.to_csv(dbg_out_cut, index=True)

print("Processing complete for all files.")

# summary for all files:
out_all = os.path.join(RESULTS_PATH, "all_mice_OF_measurements_all_bodyparts.csv")
measurements_df.to_csv(out_all, index=False)
print(f"All measurements saved to {out_all}")
for bodypart_label in measurements_df["bodypart_label"].unique():
    sub = measurements_df[measurements_df["bodypart_label"] == bodypart_label].copy()
    out = os.path.join(RESULTS_PATH, f"all_mice_OF_measurements_BP-{bodypart_label.replace(' ', '_')}.csv")
    sub.to_csv(out, index=False)
print("Measurements per bodypart also saved.")
# %% END
print("All done.")