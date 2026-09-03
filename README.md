# Machine Learning and Control Theory for Accurate and Reliable Temperature Estimation in Power Transformers

## Table of Contents

1. [Project Overview](#project-overview)
2. [License Notice](#license-notice)
3. [Thesis Abstract](#thesis-abstract)
4. [Files in This Repository](#files-in-this-repository)
5. [How to Install the Project](#how-to-install-the-project)
6. [How to Run the Simulation Process](#how-to-run-the-simulation-process)
7. [How to Run the Training Process](#how-to-run-the-training-process)

## Project Overview

This repository contains my Master Thesis' Project, conducted at the **Royal Institute of Technology of Stockholm**
(Kungliga Tekniska Högskolan), in collaboration with **Hitachi Energy Ltd.** and with the supervision of the
**University of Trento** (Università degli Studi di Trento). The title of the thesis is:

> *Machine Learning and Control Theory for Accurate and Reliable Temperature Estimation in Power Transformers*

The goal of the project is to analyze the "IEC thermal model", traditionally employed in the simulation of Power Transformers,
and test the efficacy of possible improvements selected from both *Control Theory* (Luemberger Observer, Kalman Filter) and
*Machine Learning* (Recurrent Neural Networks).

## License Notice

This project is licensed under the *Apache-2.0 License*, which covers all the files in this repository **except** the ones
contained in the `dataset/` folder. The reason is that this folder contains extracts from the
[CIGRE dataset](https://gitlab.com/cigrea2.60/cigre-dttm-benchmarking-platform), which is licensed under
*GNU General Public License Version 3*. Hence the `dataset/` folder, and only that folder, is licensed under the
*GNU General Public License Version 3*.

## Thesis Abstract

Power Transformers are critical elements of the electrical network; the elevate
temperatures they reach while processing current can provoke an excessive
degradation of their most delicate components, resulting in premature system
aging. Hence, it is of crucial importance to properly model the evolution of the
*hotspot temperature* reached by the machine, in order to estimate the associated
*loss of life* and optimize the maintenance schedule accordingly. The project
begins with the implementation of the "IEC Model", a popular mathematical
representation of Power Transformer's thermal behavior, and tests the efficacy
of possible improvement techniques that do not compromise the simplicity of
the model.

From the field of *Control Theory* the Luenberger Observer and Kalman
Filter are tested. They have an extremely similar effect on the system, and
cause an error reduction of -65% over the Mean Absolute Error (MAE) and
of -53% over the Root Mean Square Error (RMSE). On the other hand,
the Recurrent Neural Network is selected to represent the field of *Machine Learning*.
Six architecture are tested, and the most effective one gains
a reduction of -48% over the MAE and -45% over the RMSE. Finally,
the two domains are fused into an hybrid approach which obtains the best
performance, -75% over MAE and -66% over RMSE, demonstrating that
different techniques can be stacked without one shadowing the effects of the
other.

The research leaves space for many types of continuations, including
the employment of simulation algorithms guaranteeing better numerical
precision, the implementation on physical Power Transformer machines, and
the experimentation of additional Recurrent Neural Network Architectures.

**Keywords**: Power transformers, State estimation, Machine learning, Recurrent neural
networks, Control theory, State observers, Kalman filters

## Files in This Repository

In this project, you will find the following files and folders:

- `dataset/`: folder containing the dataset used in the experiments.
  - `CIGRE/`: subfolder reporting the extracts from [CIGRE dataset](https://gitlab.com/cigrea2.60/cigre-dttm-benchmarking-platform) used in this project.
  - `parsed/`: subfolder where the preprocessed dataset are stored.
  - `dataset.py`: python script to extract the time series from the files in `CIGRE/` and store them in `parsed/` as `CSV` format.
  - `dataset_params.py`: python script to extract the parameters from the files in `CIGRE/` and store them in `parsed/` as `CSV` format.
  - `LICENSE`: GNU General Public License file, which covers **exclusively** the `dataset/` folder.

- `docs/`: output folder for the Doxygen documentation.

- `info/`: folder containing additional information about the project.
  - `Alex Pegoraro - Thesis Presentation.pdf`: the slide presentation of the project.
  - `Alex Pegoraro - KTH Thesis.pdf`: the thesis file (with KTH frontispiece), which act also as a project report.
  - `Alex Pegoraro - UniTN Thesis.pdf`: the thesis file (with UniTN frontispiece), which act also as a project report.

- `modules/`: folder containing all the python modules employed for training and simulation.
  - `control.py`: python module defining the functions regarding the deterministic control theory aspects.
  - `iec.py`: python module defining the functions to deal with the IEC model.
  - `inout.py`: python module defining the functions to manage input and output.
  - `plots.py`: python module defining the functions to produce and manage the plots.
  - `simulation.py`: python module defining the functions to simulate a dynamical system.
  - `train_main.py`: python module defining a single function: the training loop algorithm.
  - `train_rnn.py`: python module defining all the RNN classes used in the project.
  - `train_utils.py`: python module defining the functions used in the training process.

- `out/`: folder reporting the outputs of some relevant simulations.
  - `simulation-full/`: simulation run on the full dataset.
  - `simulation-iec-kalman/`: simulation on the test dataset with a Kalman Filter.
  - `simulation-iec-observer/`: simulation on the test dataset with a Luemberger Observer.
  - `simulation-iec-simple/`: simulation on the test dataset with no improvement.
  - `simulation-rnn-a-simple/`: simulation on the test dataset with the improvements obtained from `RNN_a`.
  - `simulation-rnn-b-simple/`: simulation on the test dataset with the improvements obtained from `RNN_b`.
  - `simulation-rnn-L1-simple/`: simulation on the test dataset with the improvements obtained from `RNN_tot_L1`.
  - `simulation-rnn-L2-simple/`: simulation on the test dataset with the improvements obtained from `RNN_tot_L2`.
  - `simulation-rnn-p5-simple/`: simulation on the test dataset with the improvements obtained from `RNN_p5`.
  - `simulation-rnn-p6-kalman/`: simulation with the improvements obtained from `RNN_p6` and a Kalman Filter.
  - `simulation-rnn-p6-observer/`: simulation with the improvements obtained from `RNN_p6` and a Luemberger Observer.
  - `simulation-rnn-p6-simple/`: simulation on the test dataset with the improvements obtained from `RNN_p6`.
  - `train-rnn-a/`: training with the `RNN_a` network.
  - `train-rnn-b/`: training with the `RNN_b` network.
  - `train-rnn-L1/`: training with the `RNN_tot_L1` network.
  - `train-rnn-L2/`: training with the `RNN_tot_L2` network.
  - `train-rnn-p5/`: training with the `RNN_p5` network.
  - `train-rnn-p6/`: training with the `RNN_p6` network.

- `.gitignore`: to ignore temporary folders in version controlling.
- `Doxyfile`: configuration file for the Doxygen documentation.
- `LICENSE`: Apache-2.0 license file, which cover all the repository **except** the `dataset/` folder.
- `main.py`: the main file for the simulation process.
- `main_params.py`: parameter file imported by `main.py`.
- `NOTICE`: license notice file, it clarifies the range of action of the two `LICENSE` files.
- `README.md`: entry point of the project and home page of the Doxygen documentation.
- `requirements.txt`: plain text reporting the python requirements to run this project.
- `train.py`: the main file for the training process.
- `train_params.py`: parameter file imported by `train.py`.

## How to Install the Project

To clone this repository on your local machine, you can use the traditional Git cloning:

```bash
git clone git@github.com:AlphaNightLight/power-transformers.git
```

The project have been designed to run on Python's *virtual environments*, you can create one on the terminal with the
following command:

```bash
# Create a virtual environment
python -m venv myvenv
```

Before doing any operation with the virtual environment, you first need to activate it; the method depends on your terminal:

```
# Activate a virtual environment
source myvenv/bin/activate # Linux and MacOS
# venv\Scripts\Activate.ps1 # Windows PowerShell
# venv\Scripts\activate.bat # Windows CMD
```

The next step is to install the requirements, which are stored in the `requirements.txt` file in the root of the repository:

```
# Install the requirements
pip install -r requirements.txt
```

Now, you are ready to launch all the simulation and trainings you want. When you're done, remember to exit from the virtual
environment with the command `deactivate`.

**NOTE**: the simulations do not work directly on the [CIGRE dataset](https://gitlab.com/cigrea2.60/cigre-dttm-benchmarking-platform),
but on the preprocessed files stored
as `CSV` in `dataset/parsed/`. If for any reason those files gets corrupted, you can regenerate them from the raw dataset running
the two appropriate scripts:

```bash
# Parse the CIGRE dataset
python3 dataset.py
python3 dataset_params.py
```

## How to Run the Simulation Process

To run a simulation you first need to activate your virtual environment. Then, edit the file `main_params.py` to set all the
parameters of your simulation. Of particular importance are the matrices `A_prime` and `B_prime`, which contains the improvements
to sum to the IEC model; you usually obtain them as the output of the Recurrent Neural Network.

**NOTE**: If you wish to run a simulation without improvement matrices do NOT remove them from the parameter file, as the main
script will load them anyway, but simply set them to zero.

When you are satisfied with your setup, run the simulation is as easy as invoking the `main.py` script without any argument:

```bash
# Run a simulation
python3 main.py
```

All the plots, metrics and also the `CSV` of the output will be stored in the output location you indicated in the parameter file.

## How to Run the Training Process

Again, the precondition to be able to run the training process is to activate the virtual environment first. After this, you can
modify the parameter file for training, which is `train_params.py`, and launch the training with the `train.py` script:

```bash
# Run the training
python3 train.py
```

In the output location you specify, you will see the following subfolders:

- `after/`: A simulation of the system after the training, i.e. with the final values of the improvement matrices
- `before/`: A simulation of the system before the training, i.e. with the improvement matrices equal to zero
- `train-pics/`: Reports the plots of the simulations executed for periodic tests
- `train-state/`: Reports the value of the improvements and the loss measured for the periodic tests

The model output is represented by the `primes.txt` file containing the improvement matrices. You will also see a `logs.txt`
file reporting the training parameters for reproducibility.

**New RNN**: If you want to create your own Recurrent Neural Network architecture to be used in this project, you
will need to act on the `modules/train_rnn.py` file. This module defines the `RNN_base` class and contains the instructions
to subclass it to your own architecture. Once you have implemented it, you will be able to seamlessly include it in
the `train_params.py` file just like the other subclasses that are already implemented.

## Maintainer

Alex Pegoraro
