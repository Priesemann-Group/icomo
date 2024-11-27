Transform Jax to Pytensor
=========================

This function wraps a `JAX <https://jax.readthedocs.io/>`_ function, such that it can
be used in a `PyTensor <https://pytensor.readthedocs.io/>`_ graph.
This
allows to build more complex models with PyMC, for instance including the solution
of differential equations, or the solution of a non-linear system of equations, as
the many JAX libraries can be used in the model.

API
---


.. autofunction:: icomo.jax2pytensor

