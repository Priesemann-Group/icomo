# Installation

## Stable release

To install icomo, run this command in your
terminal:

```bash
$ pip install icomo
```

To run all the examples, you might also want to install the optional dependencies:

```bash
$ pip install icomo[opt]
```

You might want to create a virtual environment first, e.g. using
[mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html):

```bash
$ mamba create -n icomo_env icomo
$ mamba activate icomo_env
```

or directly using venv:

```bash
$ python -m venv /path/to/new/icomo_env
$ source /path/to/new/icomo_env/bin/activate
$ pip install icomo
```

You might also want to install [jupyterlab](https://jupyter.org) to run the examples:

```bash
$ pip install jupyterlab
```

## From sources

The sources for icomo can be downloaded from
the [Github repo](https://github.com/Priesemann-Group/icomo).

You can clone the public repository:

```bash
$ git clone https://github.com/Priesemann-Group/icomo
```

Enter the directory and install with pip:

```bash
$ cd icomo
$ pip install -e .
```
This enables you to edit the code and have the changes directly available in your python
environment.
To include the optional and development dependencies, use:

```bash
$ cd icomo
$ pip install -e .[dev]
```
