
```{image} https://raw.githubusercontent.com/Priesemann-Group/icomo/main/docs/images/icomo_logo250px.png
:alt: My Logo
:class: logo, mainlogo, dark-light p-2
:align: center
:height: 160px
```

# Introduction of ICoMo

This toolbox aims to simplify the construction of compartmental models and the inference of their parameters.

The aim isn't to provide a complete package that will build models from A to Z, but rather
provide different helper functions examples and guidelines to help leverage modern python
packages like [JAX](https://jax.readthedocs.io/en/latest/),
[Diffrax](https://docs.kidger.site/diffrax/) and
[PyMC](https://www.pymc.io/welcome.html) to build, automatically differentiate and fit
compartmental models.


A central part of the toolbox is the possibility to wrap JAX functions to be
used in PyMC models (see [here](api/jax2pytensor)), which
is used tro wrap the Diffrax ODE solvers, but might be also useful for other projects.

## Features

* Facilitate the construction of compartmental models by only defining flow between compartments, and
  automatically generating the corresponding ODEs.
* Plot the graph of the compartmental model to verify the correctness of the model.
* Integrate the ODEs using diffrax, automatically generating the Jacobian of the parameters of the ODE
* Fit the parameters using minimization algorithms or build a Bayesian model using PyMC.

## Content

```{toctree}
:maxdepth: 3
:caption: Getting started

installation
Getting started <example>
Delaying flows <erlang_kernel>
```

```{toctree}
:maxdepth: 3
:caption: API

api/comp_model
api/jax2pytensor
api/diffrax_wrapper
api/tree_tools
api/experimental
```

```{toctree}
:maxdepth: 3
:caption: Contributing

contributing
```


### Indices and tables

* [Index](genindex)
* [Search](search)

## Credits

Logo by [Fabian Mikulasch](https://scholar.google.com/citations?user=ZWWBIoUAAAAJ&hl=en)
