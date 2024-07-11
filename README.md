# Partial Tracing geometrically using Convolution

Code for the algorithm to calculate the partial trace of a multi-qubit/qudit ($D$-level) state geometrically using convolution. You can find the paper [here]().

## Installation Instructions

* Download the repository.
* Create a conda environment with the necessary dependencies using the `environment.yaml` file provided in the repository using the command:

```bash
conda env create -f environment.yaml
```

## Usage

* Activate the conda environment using the command:

```bash
conda activate cpt
```

* The `main.py` file contains the code to calculate the partial trace of a multi-qubit/qudit state geometrically using convolution. Sample code snippets are provided in the docstrings.

## Module List

* `main.py`: Contains the code to calculate the partial trace of a multi-qubit/qudit state geometrically using convolution.
* `utils.py`: Contains utility functions to calculate either the computation time for partially tracing systems or to obtain the reduced density matrix after partial tracing.
* `InitialState.py`: Contains the code to generate the initial state of the multi-qubit/qudit system.
* `QuditPartialTrace.py`: Contains the code to calculate the partial trace of a multi-qubit/qudit state using convolution and block-splitting method.
* `Plotter.py`: Contains the code to plot the time results of the partial trace calculation.

***

**Note**: Further details about the modules can be found in the docstrings or comments in the respective files.