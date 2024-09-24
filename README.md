# Inference of Compartmental Models (ICoMo) Toolbox

This toolbox aims to simplify the construction of compartmental models and the inference of their parameters.

The aim isn't to provide a complete package that will build models from A to Z, but rather
provide different helper functions examples and guidelines to help leverage modern python
packages like [JAX](https://jax.readthedocs.io/en/latest/),
[Diffrax](https://docs.kidger.site/diffrax/) and
[PyMC](https://www.pymc.io/welcome.html) to build, automatically differentiate and fit
compartmental models.

* Documentation: https://icomo.readthedocs.io.

### Features

* Facilitate the construction of compartmental models by only defining flow between compartments, and
  automatically generating the corresponding ODEs.
* Plot the graph of the compartmental model to verify the correctness of the model.
* Integrate the ODEs using diffrax, automatically generating the Jacobian of the parameters of the ODE
* Fit the parameters using minimization algorithms or build a Bayesian model using PyMC.




