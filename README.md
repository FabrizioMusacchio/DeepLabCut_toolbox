# DeepLabCut Analysis Toolbox

This repository contains a collection of scripts designed to facilitate the analysis of DeepLabCut (DLC) results data. The primary focus is analyzing the DLC output tables, that contain the x and y coordinates of the tracked points, and analyzing them within the paradigm of their underlying behavioral task. 

The scripts are designed to be modular and can be easily adapted to suit your specific needs. If you have any suggestions or requests for new features, please feel free to open an issue or submit a pull request, or contact [me directly](mailto:fabrizio.musacchio@posteo.de).

New scripts will be added from time to time, and the repository will be updated with new features and improvements.

## ⬇️ Installation

```bash
conda create -n dlc_analysis -y python=3.12 mamba
conda activate dlc_analysis
mamba install -y ipykernel ipython numpy matplotlib pandas
```

## 🚀 Velocity Analysis
The script `velocity_calculation.py` is designed to analyze the velocity of tracked points in a DLC output table. It calculates 

* the velocity of each point and 
* assesses based on an adjustable threshold whether the point is moving or not.

The script also generates a plot of the velocity over time, which can be useful for visualizing the movement patterns of the tracked points.

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
| `movement_threshold`  | `float`   | Velocity threshold (in px/s or mm/s) to classify a body part as moving. <br> *Example:* `50`                                                    |
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

