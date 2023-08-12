# Installation

## Stable release

To install comodi, run this command in your
terminal:

```bash
$ pip install comodi
```

You might want to create a virtual environment first, e.g. using
[mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html):

```bash
$ mamba create -n comodi_env comodi
$ mamba activate comodi_env
```

or directly using venv:

```bash
$ python -m venv /path/to/new/comodi_env
$ source /path/to/new/comodi_env/bin/activate
$ pip install comodi
```

This is the preferred method to install comodi,
as it will always install the most recent stable release.

## From sources

The sources for comodi can be downloaded from
the [Github repo](https://github.com/Priesemann-Group/comodi).

You can either clone the public repository:

```bash
$ git clone https://github.com/Priesemann-Group/comodi
```

Or download the
[tarball](https://github.com/Priesemann-Group/comodi/tarball/main):

```bash
$ curl -OJL https://github.com/Priesemann-Group/comodi/tarball/main
```

Enter the directory and install with pip:

```bash
$ cd comodi
$ pip install -e .
```
This enables you to edit the code and have the changes directly available in your python
environment.
