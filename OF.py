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
DATA_PATH = "/Users/husker/Workspace/Henrike DLC/OF 2025/"
RESULTS_PATH = "/Users/husker/Workspace/Henrike DLC/OF 2025/DLC_analysis/"
# DATA_PATH = "/Users/husker/Workspace/Denise DLC/cFC 2025/"
# RESULTS_PATH = "/Users/husker/Workspace/Denise DLC/cFC 2025/DLC_analysis/"

# define frame rate and time step:
frame_rate = 30  # fps
time_step = 1 / frame_rate

# arena size (assumed to be square):
arena_size = 49  # cm; adjust this to the size of your arena in cm
# arena_size = 25 # our cFC box

# define the size of a pixel (if available):
spatial_unit="cm"

# define whether to cut tracking data after mouse_first_track_delay seconds:
cut_tracking = 360 # False: no cut; otherwise, define number of seconds to define total tracking duration
                   # (rest will be cut)
mouse_first_track_delay = 2 # define the delay after which tracking starts (in seconds)
                            # i.e., after 'mouse_first_track_delay' seconds the mouse was first
                            # detected, we cut the tracking data to start from this point onward

# define likelihood threshold for valid points:
likelihood_threshold = 0.9 # this likelihood refers to the DLC assigned likelihood
                           # for each bodypart; it is a value between 0 and 1, where
                           # 1 means "very likely" and 0 means "not likely at all"
                           # "likely" roughly means "reliable"; by adjusting this 
                           # threshold, you can filter out low-confidence points.

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

# define bodypart-groups:
mouse_center_point_DLC_bp = {
    'center point': 'centerpoint'} # adjust 'center' to the name of the DLC body part that represents the center point of the mouse
""" mouse_center_point_DLC_bp = {
    'center point': 'center', # never change the KEY of this line, just its VALUE
    'tailbase': 'tailbase',   # all subsequent KEYS can be determined by yourself
    'ear_L': 'ear_L',
    'ear_R': 'ear_R',
    'headholder': 'headholder',
} """

arena_corners_DLC_bp = {
    'top left corner':      'A',
    'top right corner':     'B',
    'bottom left corner':   'C',
    'bottom right corner':  'D'} # adjust the names to the DLC body parts that represent the corners of the arena;

LED_lights_DLC_bp = {
    'LED light': 'LED_2P'} # adjust 'led' to the name of the DLC body part that represents the LED light;

# define the bodypart you'd like to analyze (must be a KEY in mouse_center_point_DLC_bp)
focus_bodypart = "center point"   # z.B. "center point", "tailbase", "ear_L", ...
# %% FUNCTIONS
def plot_arena_corners_and_mouse(df_cleaned, arena_corners_DLC_bp, mouse_center_point_DLC_bp, 
                                 curr_filename_clean, curr_results_path):
    """
    Calculate arena corners and plot mouse center points and arena corners.

    Parameters:
        df_cleaned (pd.DataFrame): Cleaned dataframe with DLC data.
        arena_corners_DLC_bp (dict): Dictionary mapping corner names to DLC body parts.
        mouse_center_point_DLC_bp (dict): Dictionary mapping mouse center point name to DLC body part.
        curr_filename_clean (str): Cleaned filename for saving plots.
        curr_results_path (str): Path to save the plots.

    Returns:
        dict: Dictionary with corner names as keys and (x, y) tuples as values.
    """
    arena_corners_raw = {}
    for corner_name, corner_bp in arena_corners_DLC_bp.items():
        corner_x = df_cleaned[(corner_bp, 'x')].mean()
        corner_y = df_cleaned[(corner_bp, 'y')].mean()
        arena_corners_raw[corner_name] = (corner_x, corner_y)

    plt.figure(figsize=(14, 8))
    plt.scatter(df_cleaned[(mouse_center_point_DLC_bp['center point'], 'x')],
                df_cleaned[(mouse_center_point_DLC_bp['center point'], 'y')],
                s=10, alpha=0.5)
    for corner_name, (corner_x, corner_y) in arena_corners_raw.items():
        plt.scatter(corner_x, corner_y, s=100, label=f'arena {corner_name}', edgecolor='black', alpha=0.5)
    plt.title(f'raw mouse center points (before transformation)\n{curr_filename_clean}')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_tracks_raw.pdf"), dpi=300)
    plt.close()

    return arena_corners_raw

# define a function for checking window stability:
def window_is_stable(start_i: int) -> bool:
    """Check whether the segment [start_i, start_i+win_frames) is stable enough."""
    seg = was_in[start_i : start_i + win_frames]
    if seg.size < win_frames:
        return False
    # condition 1: fraction of True must be high enough
    false_frac = 1.0 - (seg.sum() / float(win_frames))
    if false_frac > max_false_frac:
        return False
    # condition 2: longest consecutive False streak must be within limit
    longest_false = 0
    curr = 0
    for v in seg:
        if not v:
            curr += 1
            longest_false = max(longest_false, curr)
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
    curr_file = os.path.join(DATA_PATH, curr_filename)
    print(f"loading {curr_filename}...")

    # read CSV with multi-index columns
    df_in = pd.read_csv(curr_file, header=[1, 2])

    # build clean name and results path
    curr_filename_clean = curr_filename.split('DLC')[0]
    base_results_path   = os.path.join(RESULTS_PATH, curr_filename_clean)

    # Bodypart-Unterordner (sprechender Name aus dem Key)
    bp_folder_name      = f"bodypart_{focus_bodypart.replace(' ', '_')}"
    curr_results_path   = os.path.join(base_results_path, bp_folder_name)

    os.makedirs(curr_results_path, exist_ok=True)

    # drop first (meta) column
    df_cleaned = df_in.iloc[:, 1:].copy()

    # set proper multi-index on columns
    df_cleaned.columns = pd.MultiIndex.from_tuples(df_cleaned.columns)

    # convert to numeric
    df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce')

    loaded_runs.append({
        "filename": curr_filename,
        "filename_clean": curr_filename_clean,
        "results_path": curr_results_path,
        "df": df_cleaned
    })

print(f"loaded {len(loaded_runs)} file(s) for processing.")
# %% MAIN PROCESSING
# iterate over loaded datasets and run transforms, plots & metrics:
measurements_df = pd.DataFrame()

for run in loaded_runs:
    # %%
    # run = loaded_runs[3]
    curr_filename       = run["filename"]
    curr_filename_clean = run["filename_clean"]
    curr_results_path   = run["results_path"]
    df_cleaned          = run["df"].copy()

    print(f"processing {curr_filename}...")

    # determine available body parts:
    body_parts = df_cleaned.columns.get_level_values(0).unique()
    
    # get current focus bodypart:
    bp_name = mouse_center_point_DLC_bp[focus_bodypart]
    title_suffix = f"{focus_bodypart}"
    file_suffix  = f"_BP-{focus_bodypart.replace(' ', '_')}"

    # check whether expected body parts are present:
    if use_LED_light:
        expected_body_parts = set(mouse_center_point_DLC_bp.values()) | set(arena_corners_DLC_bp.values()) | set(LED_lights_DLC_bp.values())
    else:
        expected_body_parts = set(mouse_center_point_DLC_bp.values()) | set(arena_corners_DLC_bp.values())
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
    arena_corners_raw = plot_arena_corners_and_mouse(
        df_cleaned, arena_corners_DLC_bp, mouse_center_point_DLC_bp, curr_filename_clean, curr_results_path)

    # ---------- perspective transform into square arena ----------
    corner_coords      = list(arena_corners_raw.values())
    corner_coords_arr  = np.array(corner_coords)
    top_left_corner     = corner_coords_arr[np.argmin(corner_coords_arr[:, 0] + -1 * corner_coords_arr[:, 1])]
    top_right_corner    = corner_coords_arr[np.argmin(-1 * corner_coords_arr[:, 0] + -1 * corner_coords_arr[:, 1])]
    bottom_left_corner  = corner_coords_arr[np.argmin(corner_coords_arr[:, 0] +  corner_coords_arr[:, 1])]
    bottom_right_corner = corner_coords_arr[np.argmin(-1 * corner_coords_arr[:, 0] +  corner_coords_arr[:, 1])]

    src_points = np.array([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner], dtype=np.float32)
    dst_points = np.array([[0, 0], [arena_size, 0], [0, arena_size], [arena_size, arena_size]], dtype=np.float32)

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

    # ---------- filters (center & optional LED) ----------
    mouse_center_point_bp = bp_name # mouse_center_point_DLC_bp['center point']
    df_raw = df_cleaned.copy()  # transformed, unfiltered:
    df_cleaned = df_cleaned[df_cleaned[(mouse_center_point_bp, 'likelihood')] > likelihood_threshold]
    df_raw_in_arena = df_cleaned.copy()

    if use_LED_light:
        led_light_bp = LED_lights_DLC_bp['LED light']
        df_cleaned = df_cleaned[df_cleaned[(led_light_bp, 'likelihood')] > likelihood_threshold]
        print("  filtered data to only include frames where the LED light is on.")
    else:
        print("  using all data regardless of the LED light status.")

    # ---------- plot: tracks ----------
    plt.figure(figsize=(12, 8))
    plt.scatter(df_cleaned[(bp_name, 'x')],
                df_cleaned[(bp_name, 'y')],
                s=10, label=bp_name + " points", alpha=0.5)
    for corner_name, (corner_x, corner_y) in arena_corners_transformed.items():
        plt.scatter(corner_x, corner_y, s=100, label=f'arena {corner_name}', edgecolor='black', alpha=0.5)
    # center-border boundary box:
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
    plt.close()

    # ---------- plot: heatmap ----------
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
    plt.title(f"mouse heatmap in and {bp_name} points\n{curr_filename_clean}" + (" (LED light ON only)" if use_LED_light else ""))
    plt.xlabel(f"x ({spatial_unit})", fontsize=14)
    plt.ylabel(f"y ({spatial_unit})", fontsize=14)
    plt.legend(bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, arena_size)
    plt.ylim(0, arena_size)
    plt.tight_layout()
    plt.savefig(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_heatmap_smoothed.pdf"), dpi=300)
    plt.close()


    # ---------- robust first detection: mouse in arena ----------
    # Parameter wie im Cut-Modul verwenden:
    max_false_frac    = 0.10
    max_false_streak  = 3
    win_frames        = max(1, int(round(mouse_first_track_delay * frame_rate))) if mouse_first_track_delay else 1

    was_in = (df_raw[(bp_name, 'likelihood')] > likelihood_threshold).values.astype(bool)
    # --- robust first detection using window_is_stable() ---
    idxs_true = np.flatnonzero(was_in)
    if idxs_true.size > 0 and win_frames > 0:
        first_true_idx = int(idxs_true[0])
        stable_idx = None
        last_start = len(was_in) - win_frames
        for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
            if window_is_stable(s):
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
    first_Mouse_in_arena_frame     = df_raw[(bp_name, 'likelihood')] > likelihood_threshold
    first_Mouse_in_arena_time      = df_raw.index[first_Mouse_in_arena_frame].min() / frame_rate
    first_Mouse_in_arena_frame_idx = df_raw.index[first_Mouse_in_arena_frame].min()

    # ---------- prepare save measurements ----------
    measurements = {
        'filename': curr_filename_clean,
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
    speed_df_dict = {
        'time': time_vec_raw_all,
        'speed': speed_vec_raw_all,
        'was_mouse_in_arena': df_raw[(bp_name, 'likelihood')] > likelihood_threshold,
        'was_mouse_moving':   speed_vec_raw_all >= movement_threshold,
    }
    if use_LED_light:
        speed_df_dict['was_LED_light_on'] = df_raw[(LED_lights_DLC_bp['LED light'], 'likelihood')] > likelihood_threshold
    speed_df = pd.DataFrame(speed_df_dict)
        
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
    speed_df.loc[off, ['freezing_bout_id (uncut)',
                    'freezing_bout_start_s (uncut)',
                    'freezing_bout_end_s (uncut)',
                    'freezing_bout_duration_s (uncut)']] = pd.NA


    # save bout list:
    pd.DataFrame(bouts_uncut).to_csv(os.path.join(curr_results_path, 
                                                  f"{curr_filename_clean}_freeze_bouts_uncut.csv"), index=False)

    
    # ---------- save speed_df (uncut) ----------
    # save speed_df: 
    speed_df.to_csv(os.path.join(curr_results_path, f"{curr_filename_clean}_mouse_in_arena_speed.csv"), index=False)


    # ---------- center vs border metrics (uncut) ----------
    # use the full analysis window (df_cleaned) to compute uncut metrics:
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
                for s in range(first_true_idx, max(last_start, first_true_idx) + 1):
                    if window_is_stable(s):
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

                else:
                    print("  cut_tracking: no stable segment (arena ± LED) found under current tolerances; no cut applied.")
            else:
                print("  cut_tracking: no valid detection (arena ± LED) or window size=0; no cut applied.")

    
        
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
            plt.savefig(os.path.join(curr_results_path, outname_tracks_cut), dpi=300)
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
                plt.title(f"mouse heatmap and {bp_name} points (cut) in\n{curr_filename_clean} [cut {t_start_cut:.2f}-{t_end_cut:.2f}s]")
                plt.xlabel(f"x ({spatial_unit})", fontsize=14)
                plt.ylabel(f"y ({spatial_unit})", fontsize=14)
                plt.legend(bbox_to_anchor=(1.10, 1), loc='upper left')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlim(0, arena_size)
                plt.ylim(0, arena_size)
                plt.tight_layout()
                outname_heatmap_cut = f"{curr_filename_clean}_mouse_heatmap_smoothed_cut.pdf"
                plt.savefig(os.path.join(curr_results_path, outname_heatmap_cut), dpi=300)
                plt.close()

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
    plt.close()



# summary for all files:
measurements_df.to_csv(os.path.join(RESULTS_PATH, f"all_mice_OF_measurements{file_suffix}.csv"), index=False)
# %% END
print("All done.")