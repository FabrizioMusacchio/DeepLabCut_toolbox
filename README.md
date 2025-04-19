# Deeplabcut 

## Installation on Apple Silicon:
For a manual installation on Apple Silicon, you can use the following commands:

```bash
conda create -n dlc_m1 -y python=3.11 mamba
conda activate dlc_m1
conda install -y mamba
mamba install -y ipykernel ipython jupyter notebook nb_conda ffmpeg pytables==3.8.0
pip install deeplabcut[gui,tf]
```

You can also use the provided environment file:

```bash
conda env create -f dlc_m1.yaml
conda activate dlc
```


How to remove the environment:

```bash
conda env remove -n dlc_m1
```

list all installed packages:

```bash
conda list
```

clear pip-cache:

```bash
python3.11 -mpip cache purge
```