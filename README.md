This repository contains the code and data for the manuscript "A Bayesian observer model reveals a prior for natural daylights in hue perception".

## File structure
```
experiment directory 
│
|
└───config
|   |   par_a                 # directory for parameter files shared by subjects
|   |   par_s                 # directory for subjects' parameter files
│   |       └───s00           # directory for s00's parameter files
│   |   resources             # directory for monitor settings and calibration files
│   |   subjects              # directory for subjects' colorspace files
|   |   exp_config.yaml       # experiment config YAML
│   └───stim_set_demo.yaml    # documentation of paratmeter sets 
|
└───data
│   |   noise_data            # directory for data in pre-tests (noise level determination)
│   |   unique_hue            # directory for supplementray data in unique hue identification tests
|   └───all_sel_data.csv      # a csv pooling all subjects' data for analysis
|
└───exp                       # directory for scripts for experiments
|
|
└───data analysis             # directory for data analysis scripts and results
|   |   model_estimates_v2    # directory for modeling estimates
|   └───pf_estimates          # directory for psychometric function estimates

