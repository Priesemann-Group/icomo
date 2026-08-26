<div align="center">
<img src="https://raw.githubusercontent.com/Priesemann-Group/icomo/main/docs/images/icomo_logo250px.png" width="550" alt="logo"></img>
</div>


# Inference of Compartmental Models toolbox

## Leverage the power of JAX libraries for PyMC models

This toolbox aims to simplify the construction of compartmental models and the inference of their parameters.

The aim isn't to provide a complete package that will build models from A to Z, but rather
provide different helper functions examples and guidelines to help leverage modern python
packages like [JAX](https://jax.readthedocs.io/en/latest/),
[Diffrax](https://docs.kidger.site/diffrax/) and
[PyMC](https://www.pymc.io/welcome.html) to build, automatically differentiate and fit
compartmental models.

A central part of the toolbox is the possibility to wrap JAX functions to be
used in PyMC models (see [here](https://icomo.readthedocs.
io/en/stable/api/jax2pytensor.html)), which
is used tro wrap the [Diffrax ODE solvers](https://docs.kidger.site/diffrax/api/diffeqsolve/), but might be also useful for other
projects.

* Documentation: https://icomo.readthedocs.io.

### Features

* Facilitate the construction of compartmental models by only defining flow between compartments, and
  automatically generating the corresponding ODEs.
* Plot the graph of the compartmental model to verify the correctness of the model.
* Integrate the ODEs using diffrax, automatically generating the Jacobian of the parameters of the ODE
* Fit the parameters using minimization algorithms or build a Bayesian model using PyMC.

### Citation

If you use this toolbox in your research, please find the citation information on
the right sidebar.

### Credits

Logo by [Fabian Mikulasch](https://scholar.google.com/citations?user=ZWWBIoUAAAAJ&hl=en)




