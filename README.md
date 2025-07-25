# Convolutional and computer vision methods for accelerating partial tracing operation in quantum mechanics for general qudit systems

## - Aaditya Rudra $^{1}$ and M. S. Ramkarthik $^{2*}$

1 - Department of Electrical Engineering, Visvesvaraya National Institute of Technology, South Ambazari Road, Nagpur, 440010, Maharashtra, India. ([adityarudra02@gmail.com](mailto:adityarudra02@gmail.com))

2* - Department of Physics, Visvesvaraya National Institute of Technology, South Ambazari Road, Nagpur, 440010, Maharashtra, India. ([msramkarthik@phy.vnit.ac.in](mailto:msramkarthik@phy.vnit.ac.in))

---

## Abstract: 

Partial trace is a mathematical operation used extensively in quantum mechanics to study the subsystems of a composite quantum system and in several other applications such as calculation of entanglement measures. Calculating partial trace proves to be a computational challenge with an increase in the number of qubits as the Hilbert space dimension scales up exponentially and more so as we go from two-level systems (qubits) to $D$-level systems. In this paper, we present a novel approach to the partial trace operation that provides a geometrical insight into the structures and features of the partial trace operation. We utilise these facts to propose a new method to calculate partial trace using signal processing concepts, namely convolution, filters and multigrids. Our proposed method of partial tracing significantly reduces the computational complexity by directly selecting the features of the reduced subsystem rather than eliminating the traced-out subsystems. We give a detailed description of our method and provide some explicit examples of the computation. Our method can be generalized further to a general $D$-level system, and a similar reduction in computation time is obtained. We also observe various geometrical patterns and self-forming fractal structures, which we discuss here. We give numerical evidence to all the claims.


## About this Repository:

Code for the algorithm to calculate the partial trace of a multi-qubit/qudit ($D$-level) state geometrically using convolution. 

## Installation Instructions:

* Download the repository.
* Create a conda environment with the necessary dependencies using the `environment.yaml` file provided in the repository using the command:

```bash
conda env create -f environment.yaml
```

## Usage:

* Activate the conda environment using the command:

```bash
conda activate cpt
```

* The `main.py` file contains the code to calculate the partial trace of a multi-qubit/qudit state geometrically using convolution. Sample code snippets are provided in the docstrings.

## Module List:

* `main.py`: Contains the code to calculate the partial trace of a multi-qubit/qudit state geometrically using convolution.
* `utils.py`: Contains utility functions to calculate either the computation time for partially tracing systems or to obtain the reduced density matrix after partial tracing.
* `InitialState.py`: Contains the code to generate the initial state of the multi-qubit/qudit system.
* `QuditPartialTrace.py`: Contains the code to calculate the partial trace of a multi-qubit/qudit state using convolution and block-splitting method.
* `Plotter.py`: Contains the code to plot the time results of the partial trace calculation.

***

**Note**: Further details about the modules can be found in the docstrings or comments in the respective files.