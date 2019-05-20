# DemPref

This is the accompanying code for the paper "Learning Reward Functions by Integrating Human Demonstrations and Preferences", which will appear at RSS 2019.

## Environment

To play around with our code, it will be easiest to use a Conda environment on an OSX system. Simply install Anaconda and then run the following commands (in the given order) from the `DemPrefCode` directory to setup and activate the environment.

~~~~
conda create --name dempref --file requirements.txt
source activate dempref
~~~~

## Experiments

To re-run any simulation experiment, run the following two commands from the `DemPrefCode` directory, in the given order:

~~~~
cd experiments/{experiment_name}_experiment
python {experiment_name}_experiment.py {domain_name}
~~~~

where `experiment_name` is one of `main`, `update_func`, `iterated_corr` and `domain_name` is one of `driver`, `lander`, `fetch_reach`.

## Raw Data and Processing Files

For each of the simulation experiments, the raw data and Jupyter notebooks used to plot the figures seen in this paper have been provided. They can be found in the relevant directory for each experiments. (Enter the `experiments` directory and you will see directories for the three simulation experiments.)