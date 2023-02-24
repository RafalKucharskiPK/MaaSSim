# MaaSSimGym

MaaSSimGym is a GymAPI-compatible environment used to train Reinforcement Learning models to be used in the MaaSSim simulator.


### Usage

In `demo.ipynb` you can find how to use the environment to train a model from Stable Baselines library as MaaSSim driver decision agent along with its evaluation and visualisation.


### Installation

#### Prerequisites

- `Python 3.8`

#### Requirements

To use MaaSSimGym you first need to install requirements listed in the parent directory and then the requirements listed in this subpackage.

`pip install -r ../requirements.txt requirements.txt`


### Implementation details

MaaSSimGym relies on multithreaded execution of the original MaaSSim simulation and the environment wrapper at the same time.
Communication between them is conducted through `threading.Event` instances. Environment thread is the master thread, but if there is any exception in only one of the threads, the execution overall would not halt and should be terminated manually.

