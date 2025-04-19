# DeepLabCut Analysis Toolbox

This repository contains a collection of scripts designed to facilitate the analysis of DeepLabCut (DLC) results data. The primary focus is analyzing the DLC output tables, that contain the x and y coordinates of the tracked points, and analyzing them within the paradigm of their underlying behavioral task. 

The scripts are designed to be modular and can be easily adapted to suit your specific needs. If you have any suggestions or requests for new features, please feel free to open an issue or submit a pull request, or contact [me directly](mailto:fabrizio.musacchio@posteo.de).

New scripts will be added from time to time, and the repository will be updated with new features and improvements.

## ⬇️ Installation

```bash
conda create -n dlc_analysis -y python=3.11 mamba
conda activate dlc_analysis
mamba install -y ipykernel ipython numpy matplotlib pandas
```

## 🚀 Velocity Analysis
The script `velocity_calculation.py` is designed to analyze the velocity of tracked points in a DLC output table. It calculates 

* the velocity of each point and 
* assesses based on an adjustable threshold whether the point is moving or not.

The script also generates a plot of the velocity over time, which can be useful for visualizing the movement patterns of the tracked points.


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
}
```
