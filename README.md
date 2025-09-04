![GitHub Release](https://img.shields.io/github/v/release/FabrizioMusacchio/DeepLabCut_toolbox) [![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

# DeepLabCut Analysis Toolbox

This repository contains a collection of scripts designed to facilitate the analysis of DeepLabCut (DLC) results data. The primary focus is analyzing the DLC output tables, that contain the x and y coordinates of the tracked points, and analyzing them within the paradigm of their underlying behavioral task. 

The scripts are designed to be modular and can be easily adapted to suit your specific needs. If you have any suggestions or requests for new features, please feel free to open an issue or submit a pull request, or contact [me directly](mailto:fabrizio.musacchio@posteo.de).

New scripts will be added from time to time, and the repository will be updated with new features and improvements.

## ⬇️ Installation

```bash
conda create -n dlc_analysis -y python=3.12 mamba
conda activate dlc_analysis
mamba install -y ipykernel ipython numpy matplotlib pandas opencv scipy
```

## 📥 Square arena analysis (open field, contextual fear conditioning/freezing, w/ and w/o simultaneous imaging)
The script `OF.py` analyzes behavior in **square arenas** (e.g., open field (OF), contextual fear conditioning/freezing (cFC)). It supports experiments **with or without** simultaneous imaging/stimulation, detected via a tracked LED.

### Expected DLC input & naming conventions
We expect DLC CSVs with a **two-level header**: `(bodypart, attribute)` where `attribute ∈ {x, y, likelihood}`.

You should track at least:

**1) Arena corners (clockwise A → B → D → C):**
```
A----------------B
|                |
|      >o<      ,|
|       O      -o|
|       |       `|
|                |
|                |
C----------------D
```

* `A` = top-left, `B` = top-right, `D` = bottom-right, `C` = bottom-left.  
  Please stick to this A→B→D→C convention for reproducible rectification.


**2) Mouse bodyparts (at minimum a central, stable point):**
```
   >O<  ⟵ nose, ears, or head center
    O
   OOO  ⟵ center (AT LEAST)
    O    
    |   ⟵ tailbase
    L

```
- Typical names: `center`/`centerpoint` (required), plus optional `tailbase`, `ear_L`, `ear_R`, `nose`, etc.

**3) LED (optional):**
```
   ,
  -o  ⟵ led 1, led 2, ...
   `
```
- Track at least one LED landmark if you want to (a) visualize LED phases and/or (b) **restrict analysis** to frames where **LED is ON**.

> The pipeline uses `(x, y)` positions and the **DLC `likelihood`** to filter out low-confidence frames.

### Pre-processing: Rectification (scaling to a square arena)
**Goal:** map raw image coordinates onto a **metric, square arena** of known side length `arena_size` (e.g., 25 cm for cFC, 49 cm for OF).

1. **Corner aggregation**
   For each corner bodypart (A–D), we compute the mean `(x, y)` over time to reduce jitter.
2. **Homography**
   We compute a perspective transform from the four averaged corners to the target square. Every tracked bodypart (all frames) is transformed with the same matrix.
3. **Likelihood filtering**
   For each bodypart, frames with `likelihood < likelihood_threshold` get `NaN` for `(x, y)`.
4. **Units**
   After rectification, positions are in **centimeters** (or your chosen `spatial_unit`), so velocities are in `spatial_unit/s`.

> Ensure corner points are well distributed/non-collinear to avoid unstable transforms.

Corner order for homography is inferred geometrically from the four mean corner points; explicit A/B/C/D order is not required.

### Choosing the bodypart to analyze
By default, the analysis uses the **center point**. You can switch the focus via:

```python
# choose any KEY from mouse_center_point_DLC_bp:
focus_bodypart = "center point"   # e.g., "center point", "tailbase", "ear_L", ...
```

All plots, speeds, freezing calls, and per-mouse outputs are computed for **that focus bodypart**. Each mouse gets a **bodypart-specific subfolder** to keep results tidy:

```
<RESULTS_PATH>/<mouse_id>/bodypart_<focus_key_without_spaces>/
    <all figures & CSVs for this bodypart>
```

This makes it easy to re-run the pipeline with different focus bodyparts.

### Open field (OF) analysis
After rectification via homography, all bodypart coordinates lie in a square of side length `arena_size`. Units are centimeters if `arena_size` is specified in cm. All speeds are reported in cm/s. We then compute:

* **Tracks & heatmaps**
  Transformed XY traces and 2D occupancy maps (Gaussian-smoothed).
* **Center vs. border occupancy**
  With `border_margin`, we define a **central square** (`arena_size - 2*border_margin` per side). We report:
  * time in **center** (s)
  * time in **border** (s)
  * **center–border crossings**
* **Speed metrics** (uncut and/or LED-filtered)
  * total time analyzed
  * total **moving** time (speed ≥ movement threshold)
  * **total distance** moved
  * **average speed** (overall and during moving)
  * **max speed**
* **Freezing**
  Formal definition below, with smoothing and gap-tolerant bout extraction.



#### **Velocity of the focus bodypart**  
Let $(x_t, y_t)$ be rectified positions at frame $t$ and frame rate $f$ (Hz). With $\Delta t = 1/f$:

$$v_t = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\Delta t}$$

Frames where the DLC `likelihood` of the focus bodypart is below `likelihood_threshold` are set to `NaN` and excluded from velocity and distance calculations.

#### **Speed signals used for different calls**  
Movement labels use the **raw** frame-to-frame speed $v_t$ with a hard threshold `movement_threshold`.  

Freezing detection uses the **smoothed** speed $\tilde v_t$ (centered median over `freeze_smooth_win` frames) with threshold `freeze_speed_threshold` plus gap merging and minimum bout length.


#### **Movement vs. non-movement**  
A frame is labeled *moving* if

$$v_t \ge \theta_{\text{move}}$$

with $\theta_{\text{move}} = \texttt{movement\\_threshold}$ (e.g., 0.5 cm/s). The pipeline reports:

- total moving time = number of moving frames divided by $f$
- total distance moved = $\sum v_t \,\Delta t$ over moving frames
- average moving speed and overall average speed
- maximum speed

#### **Freezing (explained in more detail below)**  
We compute a smoothed speed $\tilde v_t$ using a centered median over `freeze_smooth_win` frames (default ≈ 0.25 s). A frame is *freezing* if $\tilde v_t < \theta_{\text{freeze}}$, with $\theta_{\text{freeze}} = \texttt{freeze\\_speed\\_threshold}$ (default: set this to  `movement_threshold`). Short False gaps up to `freeze_gap_merge_max_s` are merged, and bouts shorter than `freeze_min_duration_s` are discarded.


#### **Occupancy heatmap**  
We create a 2D histogram of $(x, y)$ in the rectified square with `bins = 60` and Gaussian smoothing (`sigma = 2.5`). The bin width is $ \text{arena\_size} / \text{bins} $. We overlay the center boundary defined by `border_margin`:

- center: $x,y \in [\text{border\\_margin}, \text{arena\\_size} - \text{border\\_margin}]$
- border: complement of the center

We report time in center, time in border, and center–border crossing count.


#### **Robust first detection (“$n$-second rule”) & optional cut**  
Many experiments analyze a fixed-length window **after** the animal is **reliably** detected. We implement a **robust first-detection**:

* Scan forward to find the first time the animal is continuously “present” for a **window** of length `mouse_first_track_delay`, with tolerance:
  allow up to `max_false_frac` fraction of missing detections and no **False** run longer than `max_false_streak` frames.
* Once found, define a **cut window** of length `cut_tracking` seconds starting at that time.
* All **cut** stats/plots use only frames inside that window (and still honor LED logic if enabled).

This avoids starting analysis on spurious single-frame detections.

#### **Missing data policy**  
Frames with `likelihood < likelihood_threshold` are set to `NaN`. We do **not** interpolate positions or speeds. All time, distance, movement, and freezing metrics are computed on the remaining valid frames only.



### Definition of freezing
In our analysis pipeline, **freezing** is defined as a behavioral state in which the selected bodypart remains below a velocity threshold for a minimum duration, after temporal smoothing and with tolerance for very short interruptions (gap merging). By default, the *center point* of the mouse is used as the reference bodypart, but this can be changed in the configuration.

Let $x(t), y(t)$ denote the position of the chosen bodypart at frame $t$, sampled at frame rate $f$ (frames per second).  

1. **Velocity**:  
   $$v(t) \;=\; \frac{\sqrt{(x(t) - x(t-1))^2 + (y(t) - y(t-1))^2}}{\Delta t}$$
   where $\Delta t = 1/f$ is the time step.
2. **Smoothing** (optional):  
   $$\tilde{v}(t) = \text{median\\_filter}(v(t), w)$$
   with a smoothing window $w$ (e.g. 250 ms).
3. **Thresholding**:  
   A frame is marked as *freezing* if
   $$\tilde{v}(t) < \theta$$
   where $\theta$ is the freezing speed threshold (typically equal to the movement threshold).
4. **Gap merging**:  
   Short interruptions shorter than $\tau_{\text{gap}}$ are merged into surrounding freezing periods.
5. **Bout definition**:  
   A continuous freezing segment is counted as a bout if its duration is at least
   $$\Delta t_{\min}$$
   (minimum bout duration).

#### Why we currently use a single bodypart
We deliberately implemented freezing detection based on a **single, central bodypart** (typically the center point or tailbase). The reasons are:

* **Simplicity and robustness**:  
  A single bodypart is easy to compute, stable, and widely used in the literature. It allows for straightforward thresholding and bout extraction, avoiding complex dependencies between landmarks.

* **Consistency with existing practice**:  
  Many published freezing pipelines define freezing via the movement of the body center. This facilitates comparability across studies.

* **Efficiency**:  
  Single-part freezing detection is fast and less sensitive to tracking noise from other landmarks (e.g. ears, nose, or tail tip).

**Why not multiple bodyparts (yet)?**  
In principle, freezing could be defined more strictly by requiring **multiple bodyparts** to remain immobile simultaneously (e.g. nose, tailbase, and ears). This has some advantages:

* Physiologically more realistic — the entire body is immobile, not just the center.  
* Robust to single landmark tracking errors.  
* Better captures subtle movements such as nose twitches or tail flicks.

However, it also introduces significant complexity:

* Requires a consensus rule (e.g. “at least N bodyparts must be still”).  
* Different bodyparts have different noise characteristics and typical amplitudes, so thresholds might need to be part-specific.  
* Risk of false negatives if one landmark is noisy while the rest indicate freezing.


For now, we use the **single-bodypart approach** as the default. This strikes a balance between robustness, interpretability, and comparability with existing studies. If needed, the pipeline can later be extended with a **multi-bodypart freezing logic** (e.g. majority voting or weighted consensus) to refine freezing detection further.

### LED on/off
The **optional** LED can encode **experimental phases**: e.g., miniscope on/off, optogenetic stimulation, task epochs, etc. If you track the LED as a DLC bodypart (e.g., `LED_2P`), you can:

* **Visualize** LED phases in speed/freezing plots (translucent yellow bands).
* **Restrict analysis** to frames where **both**:
  1. the animal’s **focus bodypart** is confidently detected (in-arena), and
  2. the **LED is on**.

**How the pipeline handles LED**  
* We build `was_LED_light_on` from the LED bodypart’s `likelihood > likelihood_threshold`.
* If `use_LED_light = True`, any analysis subset (tracks, heatmaps, speeds, freezing) uses **only** frames where
  `was_mouse_in_arena ∧ was_LED_light_on`.
* The **robust first-detection** and **cut** logic also honor LED if enabled (the stability window checks `in_arena ∧ LED_on`).

> The LED should be consistently visible to DLC during ON periods; otherwise, too many frames can be filtered out.


### Parameters (commonly tuned)

| Name                                            | Meaning                                         | Typical                         |
| ----------------------------------------------- | ----------------------------------------------- | ------------------------------- |
| `arena_size`                                    | Side length of square arena (units)             | 25 cm (cFC) or 49 cm (OF)       |
| `likelihood_threshold`                          | DLC confidence cutoff for valid points          | 0.9                             |
| `movement_threshold` / `freeze_speed_threshold` | Speed thresholds (same by default)              | e.g., 0.5 cm/s                  |
| `freeze_smooth_win`                             | Smoothing window for speed (frames)             | \~0.25 s @ 30 Hz → 8 frames     |
| `freeze_min_duration_s`                         | Minimum duration to count a freezing bout       | 0.5–2.0 s                       |
| `freeze_gap_merge_max_s`                        | Merge short False gaps inside freezing          | 0.25 s                          |
| `mouse_first_track_delay`                       | Stability window length for first detection     | 2 s                             |
| `cut_tracking`                                  | Duration of the cut window (seconds) or `False` | e.g., 306                       |
| `use_LED_light`                                 | Restrict analysis to LED ON ∧ in-arena          | `True/False`                    |
| `focus_bodypart`                                | Which bodypart to analyze                       | `center point` / `tailbase` / … |
| `border_margin`                                 | Half-thickness of border ring (units)           | e.g., 5 cm (cFC)                |

### Outputs

Per mouse (CSV stem = original DLC filename up to `DLC…`) inside
`<RESULTS_PATH>/<mouse_id>/bodypart_<focus_key>/`:

**Figures**

* `<mouse>_mouse_tracks_all_view_corrected[_LED_filtered].pdf`
* `<mouse>_mouse_heatmap_smoothed.pdf`
* `<mouse>_mouse_speed[_LED_filtered].pdf`
* `<mouse>_mouse_speed_smoothed_freeze_uncut.pdf`
* If `cut_tracking` is applied:

  * `<mouse>_mouse_tracks_all_view_corrected_cut.pdf`
  * `<mouse>_mouse_heatmap_smoothed_cut.pdf`

**Tables**

* `<mouse>_mouse_in_arena_speed.csv` (UNCUT):
  `time`, `speed`, `was_mouse_in_arena`, `was_mouse_moving`, optional `was_LED_light_on`, and **freezing annotations**:
  `freezing_frame (uncut)`, `freezing_bout_id (uncut)`, `freezing_bout_start_s (uncut)`, `freezing_bout_end_s (uncut)`, `freezing_bout_duration_s (uncut)`.
* If cut: `<mouse>_mouse_in_arena_speed_cut.csv` with analogous **(cut)** columns.
* `<mouse>_freeze_bouts_uncut.csv` and (if cut) `<mouse>_freeze_bouts_cut.csv`
  with per-bout `start_frame`, `end_frame`, `start_time_s`, `end_time_s`, `duration_s`.
* `<mouse>_measurements.csv` (key metrics for that mouse/bodypart).

**Group summary**

* `all_mice_OF_measurements.csv` at the root `RESULTS_PATH`.

### Quick start (cFC example)

```python
# parameters:
arena_size = 25
focus_bodypart = "center point"
use_LED_light = True
movement_threshold = 0.5         # cm/s
freeze_speed_threshold = movement_threshold
freeze_min_duration_s = 0.5
freeze_smooth_win = int(round(0.25 * frame_rate))
freeze_gap_merge_max_s = 0.25
mouse_first_track_delay = 2.0
cut_tracking = 306               # seconds (or False)
border_margin = 5.0

# Run the pipeline script on your DLC CSVs
# Results will be saved to:
# <RESULTS_PATH>/<mouse_id>/bodypart_center_point/
```


## 🚀 Velocity Analysis
The script `velocity_calculation.py` is designed to analyze the velocity of tracked points in a DLC output table. It calculates 

* the velocity of each point and 
* assesses based on an adjustable threshold whether the point is moving or not.

The script also generates a plot of the velocity over time, which can be useful for visualizing the movement patterns of the tracked points.

![img](/figures/velocity_calculation_example.png)
Example of the output plot generated by `velocity_calculation.py`

**Script parameters**:  
Before running `velocity_calculation.py`, define the following parameters at the beginning of the script to match your dataset and experimental setup:

| **Parameter** | **Value** | **Description** |
|------------------|-----------|-----------------------------|
| `DATA_PATH`           | `str`     | Path to the folder containing the DeepLabCut CSV output files. Should end with a slash `/`. <br> *Example:* `"./input_data/"` |
| `RESULTS_PATH`        | `str`     | Path to the folder where the results (plots and CSVs) will be saved. Should end with a slash `/`. <br> *Example:* `"./results/"` |
| `time_step`           | `float`   | Time between two frames, in seconds. Often equals the inverse of the video frame rate. <br> *Example:* `0.033` for 30 Hz |
| `pixel_size`          | `float`   | Physical size of one pixel, in mm. Use `1` if physical calibration is not required. <br> *Example:* `0.05` for 50 µm/pixel |
| `spatial_unit`        | `str`     | Unit for the spatial scale of velocity. Only used for labeling plots. <br> *Example:* `"mm"` |
| `likelihood_threshold`| `float`   | Minimum DeepLabCut likelihood score to consider a detection as valid. Values range from `0` to `1`. <br> *Example:* `0.9`                        |
| `movement_threshold`  | `float`   | Velocity threshold (in px/s or mm/s) to classify a body part as moving. <br> *Example:* `50` |
| `ylim`                  | `float` or `None`  | Sets the y-axis limit of the velocity plot. <br>`None` for automatic scaling; numeric (e.g., `1000`) for fixed scaling. |
| `bodypart_not_to_plot`  | `list[str]` or `None` | List of body parts to exclude from velocity plots. Set to `None` to include all. <br>Example: `['center', 'tail']` |
| `bodypart_groups`      | `dict` or `None` | Optional dictionary grouping body parts into named groups for group-wise velocity and movement analysis. <br> *Example:* `{'head': ['nose', 'neck'], 'rear': ['tail_base']}` |
| `time_intervals`       | `dict` or `None` | Optional dictionary defining named time intervals (in frames) for separate analysis over subsegments. <br> *Example:* `{'baseline': [0, 2499], 'stimulus': [2500, 4999]}` |


**Notes**:  
* Leave `bodypart_groups` and `time_intervals` as `None` if you do not want to use grouping or interval-based analysis.  When enabled, the script will compute separate statistics per group and/or per interval and save them in distinct output files.
* `bodypart_groups` is useful for analyzing the velocity of specific body parts, such as the head or tail, and can be used to compare their movement patterns.
* `time_intervals` is useful for analyzing the velocity of tracked points during specific time intervals, such as pre-stimulus and post-stimulus periods. The script will compute separate statistics for each interval and save them in distinct output files.

Make sure to adjust these values before running the analysis. Parameters such as `pixel_size` and `time_step` are essential for correct physical unit conversion.


**Output**:  
For each analyzed DeepLabCut CSV file, the script produces:

* A **plot** in PDF format showing:
  * The x- and y-positions of each body part
  * The velocity time series of each body part
  * Shaded regions indicating when body parts are moving
  * Annotations summarizing movement statistics
* A **CSV file** with the following columns:
  * Frame number and corresponding time
  * Velocity for each body part
  * Boolean mask whether a body part is moving (based on the velocity threshold)
  * Likelihood values and validity mask
  * An additional `any_bodypart_moving` column indicating whether any body part is moving in a given frame

If `bodypart_groups` are defined:
* Additional columns are added to the CSV with group-wise average velocities and movement masks

If `time_intervals` are defined:
* Separate CSV files and plots are generated **per interval**, with interval names appended to the filenames

Finally, an **aggregate summary CSV file** is generated, containing:
* Average number of moving frames per body part
* Fraction of total frames with movement
* If applicable, per group and per interval summaries


## ✍ How to cite
When using out toolbox, you can cite it using the following BibTeX entry:

```bibtex
@misc{dlc_analysis_toolbox,
  author = {Musacchio, Fabrizio},
  title = {DeepLabCut Analysis Toolbox},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FabrizioMusacchio/DeepLabCut_toolbox}},
  doi = {10.5281/zenodo.15245611},
}
```


The toolbox is archived on Zenodo, making it citable and persistent. You can always access the latest version via the concept DOI: [10.5281/zenodo.15245611](https://doi.org/10.5281/zenodo.15245611).

Zenodo also provides version-specific DOIs, allowing you to cite the exact release you used. Simply visit the [Zenodo record](https://doi.org/10.5281/zenodo.15245611) to find the DOI associated with the version of the toolbox corresponding to your project.

