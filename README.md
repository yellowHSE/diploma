# Robustness of unsupervised methods for surface-anomaly detection

PyTorch implementation of "Robustness of unsupervised methods for surface-anomaly detection" (Under review)

## Environment preparation

    conda env create -f requirements.yml
    conda activate unsupervised_robustness

## Dataset preparation

The scripts inside the ```splits``` directory will prepare the datasets for furhter use. The splits rely on the paths set in the ```path_constants.py``` file, however only the output path has to be set for each user. After the path is set, run the following commands to prepare a local copy of the datasets (will take some time):

    cd splits
    python join_data.py
    python prepare_perlin_source_images.py
    python make_splits.py

## Experiment running

To generate a batch script that will run the experiment use the following python script: ```generate_experiments.py```. The details on how to set the parameters for the experiments are set in the file. The file generates a ```experiment.sh```. In order to run it, the following commands must be made

    python generate_experiments.py
    chmod +x experiment.sh
    ./experiment.sh

This will run a lot of experiments at once. To cancel them use ```htop``` and delete them manually. As the experiments also take a lot of time the usage of ```tmux``` is advised.

## Result reading and plot generation

After the experiments finish the output images are generated with the following command (change the parameters for which experiments and setups inside the ```read_results.py``` file).

    python read_results.py

the images are save in the ```plots``` directory.

## Adding new methods and datasets

New datasets and methods can be added in the  ```methods``` and ```datasets_python``` directories. Check other files for the template.

