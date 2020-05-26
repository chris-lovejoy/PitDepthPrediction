# Sizing pits in heat exchanger tubes with eddy current measurements


## The project
Heat exchangers, which are used in a wide range of industries, have an estimated global market of $16 billion. They operate under stress and are therefore prone to developing faults. Failures account for an estimated 0.25% loss of the entire GDP of industrialised nations.

Existing methods are able to detect the presence of flaws but aren't able to reliably predict their depth. This makes decisions around replacing the heat exchanger tubes challenging; unnecessary taking tubes out of action reduces efficiency while not removing a faulty tube may lead to expensive damages.

We set out to develop machine learning models that would reliably predict depth of flaws, to enable better decision-making regarding flaw depth prediciton.

This project was designed by, sponsored by, and undertaken in collaboration with, the Electric Power Research Institute (EPRI).

Team: Chris Lovejoy, Andrea Grafmueller, Chirag Garg, Theodore Hermann


## Table of contents
* [General info](#general-info)
* [File Structure](#file-structure)
* [Setup](#setup)
* [Usage](#usage)
* [References](#references)

## General info
### Problem statement

Heat exchanger tubes are commonly inspected with eddy current (EC) techniques. While detection of defects is not a challenge, characterization (that is, determining the depth of wall loss) is done based on a single parameter (amplitude or phase); this mono-parametric approach lacks accuracy and leads to tubes being unnecessarily removed from service. The main defect of interest here is microbiologically-influenced corrosion (MIC). Since this is not reproducible in manufactured tubes, we will start with inner diameter (ID) pits.


### Objectives

Primary goal is to develop machine learning models to estimate the depth of defects based on the EC waveforms that perform better than current industry practice. Estimates using current methods (non-ML) perform with a RMSE (root-mean-squared-error) of ~15% of the thickness.

Secondary goals pertain to other interesting characterizations (such as shape classification, length and width characterization, etc) but to be only pursued if:
1) They support the primary goal; or
2) Primary goal has already been satisfactorily achieved


### Outcome

We tested our models on a hold-out test set, and the top performing models achieved an RMSE of approximately 7%.


## File-Structure
```bash
│   .gitignore
│   README.md
│
├───data
│   │
│   ├─── complete_raw     
│   │        "holds dataframes of the complete original dataset and different subsets"
│   │       full_tube_and_channel_data.csv
│   │       filtered_data.csv
│   │       raw_data.csv
│   │       filtered_cg_data.csv
│   │
│   ├───filtered
│   │       "original filtered data from EPRI"
│   │       [TubeID_SN_FlawID_Angle_filt.csv] x multiple  
│   │       
│   ├───filtered_CG
│   │       "processed data subtracting background differently and fitting peaks"
│   │
│   ├───interim
│   │       "dataframes that hold a collection of the colected features
│   │       with one row per original measurement"
│   │
│   └───raw
│           "original raw data from EPRI"
│           [TubeID_SN_FlawID_Angle_raw.csv] x multiple
│
├───demos
│       "notebooks demonstrating how different functions can be used"
│
├───models
│
├───notebooks
│
└───src
    │
    ├───data
    ├───features
    ├───models
    └───visualisation
```

## Setup

The code was developed using an anaconda virtual environment. To create the environment, please install Anaconda (anaconda.org) and in a command line interface run:

```sh
conda env create --file environment-py37.yml
```
If successful, the command returns instructions on how to activate the environment, for example: `conda activate conda-env3_7`.

The input data are separate .csv files, each corresponding to one defect, in folder `data/raw`. From these the code will generate the following interim data:
1. A single .csv file holding a compilation of all defects of interest created with the CompleteData() class. e.g. filtered_data.csv, stored in `complete_raw` folder.
2. A processed .csv file in the `interim` folder, summarizing the features of each defect measurement in one line, produced with the FeatureData() class. The file contains the following columns: `Tube_Alias 	 Flaw_ID 	Angle 	Feature_1 .... Feature_N  Flaw_Depth 	Pct_Depth 	Flaw_Volume 	Flaw_Area`

*Note: Notebook 1 contains the code required to do this*


## Usage
The codebase is organized into custom-build functions located in the `src` directory and Jupyter notebooks in the `notebooks` directory. The notebooks show the results and rely on the functions from the `src` directory.


Note: the data used to train our model is not available, and remains the intellectual property of the Electric Power Research Institute (EPRI).


## References
[[1](https://www.nde-ed.org/EducationResources/CommunityCollege/EddyCurrents/cc_ec_index.htm)] Background info on eddy current measurements.
