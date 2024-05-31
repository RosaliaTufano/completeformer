# Completeformer
> This repository contains the code for training and evaluating the model in our paper.


T5 Checkpoints: [![DOI T5 Checkpoints](https://zenodo.org/badge/DOI/10.5281/zenodo.7105131.svg)](https://doi.org/10.5281/zenodo.7105131)

ALiBi Checkpoints: [![DOI ALiBi Checkpoints](https://zenodo.org/badge/DOI/10.5281/zenodo.7089528.svg)](https://doi.org/10.5281/zenodo.7089528)

Datasets and Sinusoidal and xPOS Checkpoints: [![DOI Datasets and Sinusoidal and xPOS Checkpoints](https://zenodo.org/badge/DOI/10.5281/zenodo.7833996.svg)](https://doi.org/10.5281/zenodo.7833996)

## Usage

### Installing the library:

```bash
git clone <repository_url>
cd completeformer
pip install -e .
```

### Data Preparation

We have provided scripts to download our model checkpoints. Specifically, run the following commands to download the models:

```bash
completeformer-download --path path/to/where/the/data/will/be/downloaded
```

<!-- data in Zenodo. Please download the data from [here](https://zenodo.org/record/4550000#.YQZ8Y2hKiUk) and store it in a `data` folder outside the repository (this is to work with docker in the next step). -->

### Docker

Depending on your usecase, you can either startup our same development environment or produce our results using docker containers.

#### Development

We've included a `start.sh` script that you can run to start the docker container. Specifically, you can run:
```
bash start.sh /path/to/where/you/want/to/store/the/data <PORT_NUMBER>
```

The port number will be used to access the Jupyter notebook environment that this repository was built using. Once the container is running, you should be able to go to http://localhost:<PORT_NUMBER> to access the notebook server. Importantly, there are two folders `work` and `data` in the container. The `work` folder is where the repository's code is located, and the `data` folder is where the data will be stored.

<!-- Open the notebook located in `work/nbs/06_experiments.paper` and run its code if you want to reproduce the results in the paper. -->

#### Results

To reproduce the results in the paper, you can run the following command for measuring a complexity trained model:
```
python scripts/test_checkpoint.py --checkpoint_path path/to/checkpoint --output_dir path/to/output/dir
```

Or for measuring a length trained model:
```
python scripts/test_length_checkpoint.py --checkpoint_path path/to/checkpoint --output_dir path/to/output/dir
```
