# Changes in DLC Toolbox scripts

## 🔧 Release v1.0.3 - Preparations for open-source-friendly restructuring
The release prepares the codebase for a more modular and open-source-friendly structure. 

Introduced a `publications` folder to house scripts specific to analyses performed for publications, allowing for better organization and separation from the main toolbox code.


--- 

## 🔧 Release v1.0.2 — Plot customization options added
This update introduces two new optional parameters to enhance the configurability and readability of the velocity plots:

* **ylim**: Allows the user to define a fixed y-axis limit for velocity plots (e.g. ylim = 1000) to improve comparability across animals or sessions. Can be set to None for automatic scaling.
* **bodypart_not_to_plot**: Enables selective exclusion of specific body parts (e.g. ['center', 'tail']) from the velocity plots for clarity or focus.

Both options are optional and can be adjusted at the top of the script. These enhancements improve the interpretability and visual comparability of the generated figures.

--- 

## 🔧 Release v1.0.1 — Added Zenodo DOI

* added the Zenodo DOI 

--- 

## 🐍 DeepLabCut Analysis Toolbox — Release v1.0.0

We are pleased to announce the initial public release of the DeepLabCut Analysis Toolbox (v1.0.0), a modular Python-based utility designed for post-processing of DeepLabCut (DLC) tracking data. This release establishes the toolbox as a standardized workflow for behavioral movement analysis in neuroscience research.

🧠 **Overview**

The toolbox provides:
* automated velocity computation from DLC-tracked points,
* detection of movement vs. non-movement periods using a customizable velocity threshold,
* grouping of body parts for selective or combined analysis (e.g., head-only tracking in freezing behavior),
* interval-based analysis over defined time windows (in frames), and
* publication-ready visualizations of movement traces and periods of activity.

📁 **Outputs**

For each analyzed DLC .csv file, the toolbox automatically generates:
* a .csv file with all computed velocities and movement flags,
* one .pdf plot visualizing position, velocity, and movement indicators over time,
* additional result files for each time interval if specified, and
* a global .csv containing the average velocities across all DLC files in the dataset.

🛠️ **User-defined Parameters**

Users can configure the following key parameters:
* pixel_size, time_step, movement_threshold (unit conversions and thresholding),
* bodypart_groups (for grouped analyses),
* time_intervals (for interval-based sub-analyses),
* and path definitions for data input/output.

A detailed table of parameters and their descriptions is provided in the updated [README](https://github.com/FabrizioMusacchio/DeepLabCut_toolbox/blob/main/README.md).

📜 **Citation**

If you use this toolbox in your work, please cite it as:

Musacchio, Fabrizio. DeepLabCut Analysis Toolbox. Version 1.0.0. 2025. https://github.com/FabrizioMusacchio/MotilA

A CITATION.cff file is included for automated citation parsing. You may also use the [Zenodo record](https://doi.org/...) once archived.
