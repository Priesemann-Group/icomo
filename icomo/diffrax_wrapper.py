"""Wrapper of diffrax functions for convenience."""

import inspect
from collections.abc import Callable
from typing import Optional

import diffrax
import jax
from jaxtyping import Array, ArrayLike, PyTree


def diffeqsolve(
    *args,
    ts_out: Optional[ArrayLike] = None,
    ODE: Callable[[ArrayLike, PyTree, PyTree], PyTree] | None = None,
    fixed_step_size: bool = False,
    **kwargs,
) -> diffrax.Solution:
    """Solves a system of differential equations.

    Wrapper of :meth:`diffrax.diffeqsolve`.
    It accepts the same parameters as :meth:`diffrax.diffeqsolve`.
    Additionally, for convenience, it allows the specification of the output
    timesteps ``ts_out`` and
    the system of differential equations ``ODE`` as keyword arguments. If not specified,
    it uses the :class:`diffrax.Tsit5` solver. For a simple ODE, it is enough to
    specify the following keyword arguments:

    Parameters
    ----------
    ts_out :
        The timesteps at which the output is returned. Equivalent to
        ``diffrax.diffeqsolve(..., saveat=diffrax.SaveAt(ts=ts_out))``. It sets
        additionally the initial time ``t0``, the final time ``t1`` and the time step
        ``dt0`` of the solver if not specified separately.
    ODE :
        The function `f(t, y, args)` that returns the derivatives of the variables of
        the system of differential equations. Equivalent to
        ``diffrax.diffeqsolve(..., terms=diffrax.ODETerm(ODE))``.
    y0 :
        The initial values of the variables of the system of differential equations.
    args : PyTree of ArrayLike or Callable
        The arguments of the system of differential equations. Passed as the third
        argument to the ODE function.

    Other Parameters
    ----------------
    fixed_step_size :
        If True, the solver uses a fixed step size of ``dt0``, i.e it uses
        ``stepsize_controller=diffrax.ConstantStepSize()``.

    Returns
    -------
    sol :
        The solution of the system of differential equations. sol.ys contains the
        variables of the system of differential equations at the output timesteps.

    """
    signature = inspect.signature(diffrax.diffeqsolve)
    signature_bound = signature.bind_partial(*args, **kwargs)

    if fixed_step_size:
        if "stepsize_controller" in signature_bound.arguments:
            raise TypeError(
                "Don't simultaneously specify `icomo.diffeqsolve(..., "
                "fixed_step_size=True, "
                "stepsize_controller=...)`"
            )
        kwargs["stepsize_controller"] = diffrax.ConstantStepSize()

    if ts_out is not None:
        if "saveat" in signature_bound.arguments:
            raise TypeError(
                "Don't simultaneously specify `icomo.diffeqsolve(..., "
                "ts_out=..., saveat=...)`"
            )
        kwargs["saveat"] = diffrax.SaveAt(ts=ts_out)
        if "t0" not in signature_bound.arguments:
            kwargs["t0"] = ts_out[0]
        if "t1" not in signature_bound.arguments:
            kwargs["t1"] = ts_out[-1]
        if "dt0" not in signature_bound.arguments:
            kwargs["dt0"] = ts_out[1] - ts_out[0]
    else:
        if "t0" not in signature_bound.arguments:
            raise TypeError(
                "Specify `icomo.diffeqsolve(..., t0=...)` and/or "
                "`icomo.diffeqsolve(..., ts_out=...)`"
            )
        if "t1" not in signature_bound.arguments:
            raise TypeError(
                "Specify `icomo.diffeqsolve(..., t1=...)` and/or "
                "`icomo.diffeqsolve(..., ts_out=...)`"
            )
        if "dt0" not in signature_bound.arguments:
            raise TypeError(
                "Specify `icomo.diffeqsolve(..., dt0=...)` and/or "
                "`icomo.diffeqsolve(..., ts_out=...)`, or "
                "`icomo.diffeqsolve(..., ts_solver=...)`"
            )

    if ODE is not None:
        if "terms" in signature_bound.arguments:
            raise TypeError(
                "Don't simultaneously specify `icomo.diffeqsolve(..., "
                "ODE=..., terms=...)`"
            )
        kwargs["terms"] = diffrax.ODETerm(ODE)
    else:
        if "terms" not in signature_bound.arguments:
            raise TypeError(
                "Specify either `icomo.diffeqsolve(..., ODE=...)` or "
                "`icomo.diffeqsolve(..., terms=...)`"
            )

    if "solver" not in signature_bound.arguments:
        kwargs["solver"] = diffrax.Tsit5()

    return diffrax.diffeqsolve(*args, **kwargs)


def interpolate_func(
    ts_in: ArrayLike,
    values: ArrayLike,
    method: str = "cubic",
    ret_gradients: bool = False,
) -> Callable[[ArrayLike], Array]:
    """
    Return a diffrax-interpolation function that can be used to interpolate pytensors.

    Wrapper of :class:`diffrax.CubicInterpolation` and
    :class:`diffrax.LinearInterpolation`.

    Parameters
    ----------
    ts_in :
        The timesteps at which the time-dependent variable is given.
    vales :
        The time-dependent variable.
    method :
        The interpolation method used. Can be "cubic" or "linear".
    ret_gradients :
        If True, the function returns the gradient of the interpolation function.

    Returns
    -------
    interp :
        The interpolation function. Call ``interp(t)`` to evaluate the
        interpolated variable at time `t`. `t` can be a float or an ArrayLike.

    """
    ts_in = jax.numpy.array(ts_in)
    if method == "cubic":
        coeffs = diffrax.backward_hermite_coefficients(ts_in, values)
        interp = diffrax.CubicInterpolation(ts_in, coeffs)
    elif method == "linear":
        interp = diffrax.LinearInterpolation(ts_in, values)
    else:
        raise RuntimeError(
            f'Interpolation method {method} not known, possibilities are "cubic" or '
            f'"linear"'
        )
    if ret_gradients:
        # return jax.vmap(interp.derivative, 0, 0)
        return interp.derivative
    else:
        # return jax.vmap(interp.evaluate, 0, 0)
        return interp.evaluate
