
# OTTO-benchmark

This is a fork of [OTTO](https://github.com/C0PEP0D/otto) used for benchmarking
solvers on the olfactory search POMDP in the paper "Deep reinforcement learning for the olfactory search POMDP: a quantitative benchmark", by Aurore Loisy and Robin A. Heinonen (submitted).


Refer to the original repository for tutorials and extensive documentation.

New features:
- a new "windy" setup has been added to the original "isotropic" one;
- alpha-vector policies obtained from solvers using point-based value iteration (PBVI), namely Sarsop and Perseus, can be loaded;
- CNN and reward shaping have been implemented in the "windy" setup (not used in the benchmark).

## Installation

### Requirements

OTTO-benchmark requires Python 3.8 or greater.
Dependencies are listed in [requirements.txt](https://github.com/C0PEP0D/otto/blob/main/requirements.txt),
missing dependencies will be installed automatically.

### Conda users

If you use conda to manage your Python environments, you can install OTTO-benchmark in a dedicated environment `otto-benchmark`

``` bash
conda create --name otto-benchmark python=3.8
conda activate otto-benchmark
```

### Installing

``` bash
python3 setup.py install
```

## Downloading policies

The policies computed using DRL, Sarsop and Perseus can be downloaded [here](ADDURL)  TODO!!!

Decompress the file and place the `zoo` folder at the root of OTTO-benchmark (at the same level as `isotropic`, `windy`, 
and `converter-pbvi-to-otto` folders).



## Usage

The software contains 2 main directories, "isotropic" and "windy", containing the 2 variants of the POMDP.
They organized in the exact same way. They contain three subdirectories:

- `evaluate`: for **evaluating the performance** of a policy
- `learn`: for **learning a DRL policy** for the task
- `visualize`: for **visualizing a search** episode

The code organization and usage is self-explanatory. It is explained through examples below.

The four test cases used in the benchmark are:
- isotropic-19x19
- isotropic-53x53
- windy-medium-detections
- windy-low-detections

### Visualize

To visualize a search for the "windy-medium-detections" case, go to `windy/visualize` and run

```bash
python3 visualize.py -i windy-medium-detections
```

The policy is selected by modifying the variable `POLICY` in the file 
`windy/visualize/parameters/windy-medium-detections.py` 
and, if needed, setting the path to the desired policy.

### Evaluate

To evaluate a policy for the "isotropic-19x19", go to `isotropic/evaluate` and run

```bash
python3 evaluate.py -i isotropic-19x19
```

To change the policy, edit the `isotropic/evaluate/parameters/isotropic-19x19.py` file as done above for the visualization.

### Learn

To learn a DRL policy for "isotropic-53x53" case, go to `isotropic/learn` and run
```bash
python3 learn.py -i isotropic-53x53
```

Change hyperparameters by editing the `isotropic/learn/parameters/isotropic-53x53.py` file.


## Using new PBVI policies

PVBI policies in the benchmark have been computed using [Sarsop](https://github.com/rheinonen/sarsop/) and a custom implementation of [Perseus](https://github.com/rheinonen/PerseusPOMDP/). 

The policies obtained from these solvers can be used by `OTTO-benchmark` after conversion. For that, go to the folder `converter-pbvi-to-otto`, edit the file `convert_perseus_file.py` or `convert_sarsop_file.py` to set the correct paths at the beginning of the file, and run it.

There is no need to convert the policies that are [downloadable](https://github.com/auroreloisy/otto-benchmark#downloading-policies).
