This repository contains the source code of the paper entitled:

> **A PAC-Bayes Analysis of Adversarial Robustness**<br/>
> Paul Viallard, Guillaume Vidot, Amaury Habrard, Emilie Morvant<br/>
> NeurIPS 2021, 2021

### Running the experiments 

The datasets can be downloaded and preprocessed by running the following command in your bash shell.
```bash
./generate_data
```
Then, to run the experiments, you have to run the following command in your bash shell.
(The outputs of the experiment are in data/merge)
```bash
./run_all
```

To generate the plots of the paper, you have to run the following command in your bash shell.
```bash
cd plot/ 
python generate_bar.py
python generate_plot.py
cd ../
```

### Dependencies

The code was tested on GNU Bash 4.4.20 and Python 3.6.10 with the packages
* h5py (2.10.0)
* matplotlib (3.3.4)
* numpy (1.18.1)
* pandas (1.1.5)
* scikit_learn (0.23.1)
* seaborn (0.11.1)
* torch (1.6.0)
* torchvision (0.7.0)

These dependencies can be installed (using pip) with the following command.
> pip install h5py==2.10.0 matplotlib==3.3.4 numpy==1.18.1 pandas==1.1.5 scikit_learn==0.23.1 seaborn==0.11.1 torch==1.6.0 torchvision==0.7.0
