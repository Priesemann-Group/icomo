"""Tools to create compartmental models and ODE systems to use with pymc."""

import inspect
import logging
from collections.abc import Callable, Sequence
from types import EllipsisType

import diffrax
import graphviz
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, PyTree

# import jax.numpy as jnp
from icomo.jax2pytensor import jax2pytensor

logger = logging.getLogger(__name__)


def erlang_kernel(
    comp: ArrayLike, rate: ArrayLike, inflow: ArrayLike = 0
) -> tuple[Array, ArrayLike]:
    r"""Model the compartments delayed by an Erlang kernel.

    Utility function to model an Erlang kernel for a compartmental model. The shape is
    determined by the length of the last dimension of comp. For example, if comp C
    is an array of shape (...,3), the function implements the following system of ODEs:

    .. math::
       \begin{align*}
       \mathrm{rate\_indiv} &= 3 \cdot \mathrm{rate},&\\
        \frac{\mathrm dC^{(1)}(t)}{\mathrm dt} &= \mathrm{inflow} &-\mathrm{
        rate\_indiv}
        \cdot
            C^{(1)},\\
        \frac{\mathrm dC^{(2)}(t)}{\mathrm dt} &= \mathrm{rate\_indiv}  \cdot
            C^{(1)} &-\mathrm{rate\_indiv}  \cdot C^{(2)},\\
        \frac{\mathrm dC^{(3)}(t)}{\mathrm dt} &= \mathrm{rate\_indiv}  \cdot
            C^{(2)} &-\mathrm{rate\_indiv}  \cdot C^{(3)},\\
        \mathrm{outflow} &= \mathrm{rate\_indiv}  \cdot C^{(3)}
       \end{align*}
    ..

    Parameters
    ----------
    comp: jnp.ndarray of shape (..., n)
        The compartment on which the Erlang kernel is applied. The last dimension n is
        the length of the kernel.
    rate: float or ndarray
        The rate of the kernel, 1/rate is the mean time spent in total in all
        compartments

    Returns
    -------
    d_comp: jnp.ndarray of shape (..., n)
        The derivatives erlangerlangof the compartments
    outflow: float or ndarray
        The outflow from the last compartment
    """
    if not isinstance(comp, ArrayLike):
        raise AttributeError(
            f"comp has to be an array-like object, not of type {type(comp)}"
        )
    length = jnp.shape(comp)[-1]
    d_comp = jnp.zeros_like(comp)
    rate_indiv = rate * length
    d_comp = d_comp.at[..., 0].add(inflow - rate_indiv * comp[..., 0])
    d_comp = d_comp.at[..., 1:].add(
        -rate_indiv * comp[..., 1:] + rate_indiv * comp[..., :-1]
    )

    outflow = rate_indiv * comp[..., -1]

    return d_comp, outflow


class CompModel:
    """Class to help building a compartmental model.

    The model is built by adding flows between compartments. The model is then compiled
    into a function that can be used in an ODE.
    """

    def __init__(self, y_dict: PyTree[ArrayLike] = None):
        """Initialize the CompModel class.

        Parameters
        ----------
        y_dict: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        if y_dict is None:
            y_dict = {}
        self.y = y_dict
        self._init_graph()

    def _init_graph(self):
        self.graph = graphviz.Digraph("CompModel")
        for key in self.y.keys():
            self.graph.node(key)
        self.graph.attr(rankdir="LR")

    @property
    def y(self) -> PyTree[Array]:
        """Returns the compartments of the model.

        Returns
        -------
        y: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        return self._y

    @y.setter
    def y(self, y_dict: PyTree[ArrayLike]) -> None:
        """Set the compartments of the model.

        Also resets the derivatives of the compartments to zero.

        Parameters
        ----------
        y_dict: dict
            Dictionary of compartments. Keys are the names of the compartments and
            values are floats or ndarrays that represent their value.
        """
        self._y = jax.tree_util.tree_map(lambda x: jnp.array(x), y_dict)
        self._dy = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.y)
        self._init_graph()

    @property
    def dy(self):
        """Returns the derivative of the compartments.

        This should be returned by the function that defines the system of ODEs.

        Returns
        -------
        dComp: dict
        """
        return self._dy

    def flow(
        self,
        start_comp: str | int | Sequence[str | int],
        end_comp: str | int | Sequence[str | int],
        rate: ArrayLike,
        label: str | None = None,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """
        Add a flow from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            Key of the start compartment.
        end_comp: str
            Key of the end compartment
        rate: float or ndarray
            rate of the flow to add between compartments, is multiplied by start_comp,
            so it should be broadcastable whith it.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_erlang: bool, default: False
            If True, end_comp points to a compartment with an Erlang distributed
            dwelling time, i.e., as the last dimension of the compartment is used
            for the Erlang distribution modeling. The flow is then only added to the
            first element of the last dimension, i.e. to self.y[end_comp][...,0].

        Returns
        -------
        None
        """
        self.add_deriv(start_comp, -rate * nested_indexing(self.y, start_comp))
        self.add_deriv(
            end_comp, rate * nested_indexing(self.y, start_comp), end_comp_is_erlang
        )

        self.graph.edge(start_comp, end_comp, label=label)

    def erlang_flow(
        self,
        start_comp: str | int | Sequence[str | int],
        end_comp: str | int | Sequence[str | int],
        rate: ArrayLike,
        label: str | None = None,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """Add a flow with erlang kernel from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            start compartments of the flow. The length of last dimension the list is
            the shape of the
            Erlang kernel.
        end_comp: str or list
            end compartment of the flow
        rate: float or ndarray
            rate of the flow, equal to the inverse of the mean time spent in the Erlang
            kernel.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_erlang: bool, default: False
            If True, end_comp points to a compartment with an Erlang distributed
            dwelling time, i.e., as the last dimension of the compartment is used
            for the Erlang distribution modeling. The flow is then only added to the
            first element of the last dimension, i.e. to self.y[end_comp][...,0].
        """
        d_comp, outflow = erlang_kernel(
            comp=nested_indexing(self.y, start_comp), rate=rate
        )
        self.add_deriv(start_comp, d_comp)
        self.add_deriv(end_comp, outflow, end_comp_is_erlang)

        self.graph.edge(start_comp, end_comp, label=label)

    def delayed_copy(
        self,
        comp_to_copy: str | int | Sequence[str | int],
        delayed_comp: str | int | Sequence[str | int],
        tau_delay: ArrayLike,
    ) -> None:
        """Add a delayed copy of a compartment."""
        d_delayed_comp = delayed_copy_kernel(
            initial_comp=nested_indexing(self.y, comp_to_copy),
            delayed_comp=nested_indexing(self.y, delayed_comp),
            tau_delay=tau_delay,
        )
        self.add_deriv(delayed_comp, d_delayed_comp)

    def add_deriv(
        self,
        y_key: str | int | Sequence[str | int],
        additive_dy: ArrayLike,
        end_comp_is_erlang: bool = False,
    ) -> None:
        """Add a derivative to a compartment.

        Add a derivative to a compartment. This is useful if the derivative is not
        directly modelled by a flow between compartments.

        Parameters
        ----------
        y_key: str
            Key of the compartment
        additive_dy: float or ndarray
            Derivative to add to the compartment
        comp_is_list: bool, default: False

        """
        nested_indexing(
            tree=self.dy,
            indices=y_key,
            add=additive_dy,
            at=(Ellipsis, 0) if end_comp_is_erlang else None,
        )

    def view_graph(self, on_display: bool = True) -> None:
        """Display a graph of the compartmental model.

        Requires Graphviz (a non-python software) to be installed
        (https://www.graphviz.org/). It is also available in the conda-forge channel.
        See https://github.com/xflr6/graphviz?tab=readme-ov-file#installation.

        Parameters
        ----------
        on_display : bool
            If True, the graph is displayed in the notebook, otherwise it is saved as a
            pdf in the current folder and opened with the default pdf viewer.
        """
        if on_display:
            try:
                from IPython.display import display

                display(self.graph)
            except Exception:
                self.graph.view()
        else:
            self.graph.view()


def nested_indexing(
    tree: PyTree,
    indices: str | int | Sequence[str | int],
    add: ArrayLike | None = None,
    at: int | EllipsisType | slice | None | Sequence[EllipsisType | slice | int] = None,
):
    """Return the element of a nested structure of lists or tuples.

    Parameters
    ----------
    tree :
        The nested structure of lists or tuples.
    indices :
        The indices of the element to return.
    add : optional
        The element to add to the nested structure of lists or tuples.
    at : optional
        Specifies the position where to add the element.

    Returns
    -------
    element : object
        The element of the nested structure of lists or tuples.

    """
    element = tree
    if not isinstance(indices, tuple | list):
        indices = [indices]
    for depth, index in enumerate(indices):
        if not depth == len(indices) - 1:
            element = element[index]
        else:
            if add is None:
                return element[index]
            else:
                if at is not None:
                    element[index] = element[index].at[at].add(add)
                else:
                    element[index] = element[index] + add


def delayed_copy_kernel(
    initial_comp: ArrayLike, delayed_comp: ArrayLike, tau_delay: ArrayLike
) -> Array:
    """Return the derivative to model the delayed copy of a compartment.

    The delay has the form of an Erlang kernel with shape parameter
    delayed_comp.shape[-1].

    Parameters
    ----------
    initial_comp : jnp.ndarray, shape: (...)
        The compartment that is copied
    delayed_comp : jnp.ndarray, shape (..., n)
        Compartment that is a delayed copies of initial_var, the last element of the
        last dimension is the compartment which has the same total content as
        initial_comp over time, but is delayed by tau_delay.
    tau_delay : float or ndarray
        The mean delay of the copy.

    Returns
    -------
    d_delayed_vars : list of floats or list of ndarrays
        The derivatives of the delayed compartments

    """
    length = jnp.shape(delayed_comp)[-1]
    inflow = initial_comp / tau_delay * length
    d_delayed_vars, outflow = erlang_kernel(inflow, delayed_comp[:], 1 / tau_delay)
    return d_delayed_vars


def interpolate_func(
    ts_in: ArrayLike,
    values: ArrayLike,
    method: str = "cubic",
    ret_gradients: bool = False,
) -> Callable[[ArrayLike], Array]:
    """
    Return a diffrax-interpolation function that can be used to interpolate pytensors.

    Parameters
    ----------
    ts_in : array-like
        The timesteps at which the time-dependent variable is given.
    vales : array-like
        The time-dependent variable.
    method : str
        The interpolation method used. Can be "cubic" or "linear".
    ret_gradients : bool
        If True, the function returns the gradient of the interpolation function.

    Returns
    -------
    interp : Callable
        The interpolation function. Call `interp(t)` to evaluate the
        interpolated variable at time `t`. t can be a float or an array-like.

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


def diffeqsolve(
    *args,
    ts_out: ArrayLike | None = None,
    ODE: Callable[[ArrayLike, PyTree, PyTree], PyTree] | None = None,
    fixed_step_size: bool = False,
    **kwargs,
) -> diffrax.Solution:
    """Solves a system of differential equations.

    Wrapper of `diffrax.diffeqsolve`.
    It accepts the same parameters as `diffrax.diffeqsolve`. Additionally,
    for convenience, it allows the specification of the output timesteps `ts_out` and
    the system of differential equations `ODE` as keyword arguments. For a simple ODE,
    use the following keyword arguments:

    Parameters
    ----------
    ts_out : array-like
        The timesteps at which the output is returned. Same as
        icomo.diffeqsolve(..., saveat=diffrax.SaveAt(ts=ts_out)). It sets additionally
        the initial time `t0`, the final time `t1` and the time step `dt0` of the solver
        if not specified separately.
    ODE : function(t, y, args)
        The function that returns the derivatives of the variables of the system of
        differential equations. Same as
        icomo.diffeqsolve(..., terms=diffrax.ODETerm(ODE)).
    y0 : PyTree of array-likes
        The initial values of the variables of the system of differential equations.
    args : PyTree of array-likes or Callable
        The arguments of the system of differential equations. Passed as the third
        argument to the ODE function.

    Other Parameters
    ----------------
    fixed_step_size : bool, default is False
        If True, the solver uses a fixed step size of `dt0`, i.e it uses
        stepsize_controller=diffrax.ConstantStepSize().

    Returns
    -------
    sol : diffrax.Solution
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


class ODEIntegrator:
    """Creates an integrator for compartmental models.

    The integrator is a function that
    takes as input a function that returns the derivatives of the variables of the
    system of differential equations, and returns a function that solves the system of
    differential equations. For the initilization of the integrator object, the
    timesteps of the solver and the output have to be specified.

    """

    def __init__(
        self,
        ts_out,
        ts_solver=None,
        ts_arg=None,
        interp="cubic",
        solver=None,
        t_0=None,
        t_1=None,
        **kwargs,
    ):
        """Initialize the ODEIntegrator class.

        Parameters
        ----------
        ts_out : array-like
            The timesteps at which the output is returned.
        ts_solver : array-like or None
            The timesteps at which the solver will be called. If None, it is set to
            ts_out.
        ts_arg : array-like or None
            The timesteps at which the time-dependent argument of the system of
            differential equations are given. If None, it is set to ts_solver.
        interp : str
            The interpolation method used to interpolate_pytensor the time-dependent
            argument of the system of differential equations.
            Can be "cubic" or "linear".
        solver : :class:`diffrax.AbstractStepSizeController`
            The solver used to integrate the system of differential equations.
            Default is diffrax.Tsit5(), a 5th order Runge-Kutta method.
        t_0 : float or None
            The initial time of the integration. If None, it is set to ts_solve[0].
        t_1 : float or None
            The final time of the integration. If None, it is set to ts_solve[-1].
        **kwargs
            Arguments passed to the solver, see :func:`diffrax.diffeqsolve` for more
            details.
        """
        self.ts_out = ts_out
        if ts_solver is None:
            self.ts_solver = self.ts_out
        else:
            self.ts_solver = ts_solver
        if solver is None:
            solver = diffrax.Tsit5()
        if t_0 is None:
            self.t_0 = float(self.ts_solver[0])
        else:
            self.t_0 = t_0
        if self.t_0 > self.ts_out[0]:
            raise ValueError("t_0 should be smaller than the first element of ts_out")
        if t_1 is None:
            self.t_1 = float(self.ts_solver[-1])
        else:
            self.t_1 = t_1
        if self.t_1 < self.ts_out[-1]:
            raise ValueError("t_1 should be larger than the last element of ts_out")
        self.ts_arg = ts_arg
        self.interp = interp
        self.solver = solver
        self.kwargs_solver = kwargs

    def get_func(self, ODE, list_keys_to_return=None):
        """
        Return a function that solves the system of differential equations.

        Parameters
        ----------
        ODE : function(t, y, args)
            A function that returns the derivatives of the variables of the system of
            differential equations. The function has to take as input the time `t`, the
            variables `y` and the arguments `args=(arg_t, constant_args)` of the system
            of differential equations.
            `t` is a float, `y` is a list or dict of floats or ndarrays, or in general,
            a pytree, see :mod:`jax.tree_util` for more details. The return value of the
            function has to be a pytree/list/dict with the same structure as `y`.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that are
            returned by the integrator. If set, the integrator returns a list of the
            variables of the system of differential equations in the order of the keys.
            If `None`, the output is returned as is.

        Returns
        -------
        integrator : function(y0, arg_t=None, constant_args=None)
            A function that solves the system of differential equations and returns the
            output at the specified timesteps. The function takes as input `y0` the
            initial values of the variables of the system of differential equations, the
            time-dependent argument of the system of differential equations `arg_t`, and
            the constant arguments `constant_args` of the system of differential
            equations. `t`, `y0` and `(arg_t, constant_args)` are passed to the ODE
            function as its three arguments. If `arg_t` is `None`, only `constant_args`
            are passed to the ODE function and vice versa, without being in a tuple.

        """

        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                if not callable(arg_t):
                    if self.ts_arg is None:
                        raise RuntimeError("Specify ts_arg to use a non-callable arg_t")
                    arg_t_func = interpolation_func(
                        ts=self.ts_arg, x=arg_t, method=self.interp
                    ).evaluate
                else:
                    logger.warning(
                        "arg_t is callable, but ts_arg is not None. ts_arg"
                        " won't be used."
                    )
                    arg_t_func = arg_t

            if arg_t is None and self.ts_arg is not None:
                logger.warning(
                    "You did specify ts_arg, but arg_t is None. "
                    "Did you mean to do this?"
                )
            term = diffrax.ODETerm(ODE)

            if arg_t is None:
                args = constant_args
            elif constant_args is None:
                args = arg_t_func
            else:
                args = (
                    arg_t_func,
                    constant_args,
                )
            saveat = diffrax.SaveAt(ts=self.ts_out)  # jnp.array?

            stepsize_controller = (
                diffrax.StepTo(ts=self.ts_solver)  # jnp.array?
                if "stepsize_controller" not in self.kwargs_solver
                else self.kwargs_solver["stepsize_controller"]
            )

            dt0 = (
                None
                if isinstance(stepsize_controller, diffrax.StepTo)
                else self.ts_solver[1] - self.ts_solver[0]
            )

            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                self.t_0,
                self.t_1,
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=y0,
                args=args,
                saveat=saveat,
                **self.kwargs_solver,
                # adjoint=diffrax.BacksolveAdjoint(),
            )
            if list_keys_to_return is None:
                return sol.ys
            else:
                return tuple([sol.ys[key] for key in list_keys_to_return])

        return integrator

    def get_op(
        self,
        ODE,
        return_shapes=((),),
        list_keys_to_return=None,
        name=None,
    ):
        """Return a pytensor operator that solves the system of differential equations.

        Same as get_func, but returns a pytensor operator that can be used in a pymc
        model. Beware that for this operator the output of the integration of the ODE
        can only be a single or a list variables. If the output is a dict,
        set list_keys_to_return to specify the keys of the variables that are
        returned by the integrator. These return values aren't allowed to be further
        nested.

        Parameters
        ----------
        ODE : function(t, y, args)
            A function that returns the derivatives of the variables of the system of
            differential equations. The function has to take as input the time `t`, the
            variables `y` and the arguments `args=(arg_t, constant_args)` of the system
            of differential equations. `t` is a float, `y` is a list or dict of
            floats or ndarrays, or in general, a pytree, see :mod:`jax.tree_util` for
            more details. The return value of the function has to be a
            pytree/list/dict with the same structure as `y`.
        return_shapes : tuple of tuples, default is ((),)
            Depreceated, the return shape had to be specified before, now it is inferred
            automatically. This argument isn't used anymore.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that will
            be chosen to be returned by the integrator.
            If `None`, the output is returned as is.
        name :
            The name under which the operator is registered in pymc.

        Returns
        -------
        pytensor_op : :class:`pytensor.graph.op.Op`
            A :mod:`pytensor` operator that can be used in a :class:`pymc.Model`.

        """
        integrator = self.get_func(ODE, list_keys_to_return=list_keys_to_return)

        if list_keys_to_return is None:
            output_shape_def = lambda y0, **kwargs: tree_map(
                lambda shape: (len(self.ts_out),) + shape, y0
            )
        else:
            output_shape_def = lambda y0, **kwargs: tuple(
                [(len(self.ts_out),) + y0[key] for key in list_keys_to_return]
            )

        pytensor_op = jax2pytensor(
            integrator, output_shape_def=output_shape_def, name=name
        )

        return pytensor_op


def interpolation_func(ts, x, method="cubic"):
    """
    Return a diffrax-interpolation function that can be used to interpolate pytensors.

    Parameters
    ----------
    ts : array-like
        The timesteps at which the time-dependent variable is given.
    x : array-like
        The time-dependent variable.
    method
        The interpolation method used. Can be "cubic" or "linear".

    Returns
    -------
    interp : :class:`diffrax.CubicInterpolation` or :class:`diffrax.LinearInterpolation`
        The interpolation function. Call `interp.evaluate(t)` to evaluate the
        interpolated variable at time `t`. t can be a float or an array-like.

    """
    # ts = jnp.array(ts)
    if method == "cubic":
        coeffs = diffrax.backward_hermite_coefficients(ts, x)
        interp = diffrax.CubicInterpolation(ts, coeffs)
    elif method == "linear":
        interp = diffrax.LinearInterpolation(ts, x)
    else:
        raise RuntimeError(
            f'Interpoletion method {method} not known, possibilities are "cubic" or '
            f'"linear"'
        )
    return interp


def interpolate_pytensor(
    ts_in, ts_out, y, method="cubic", ret_gradients=False, name=None
):
    """
    Interpolate the time-dependent variable `y` at the timesteps `ts_out`.

    Parameters
    ----------
    ts_in : array-like
        The timesteps at which the time-dependent variable is given.
    ts_out : array-like
        The timesteps at which the time-dependent variable should be interpolated.
    y : array-like
        The time-dependent variable.
    method : str
        The interpolation method used. Can be "cubic" or "linear".
    ret_gradients : bool
        If True, the gradients of the interpolated variable are returned. Default is
        False.

    Returns
    -------
    y_interp : array-like
        The interpolated variable at the timesteps `ts_out`.

    """

    def interpolator(ts_out, y, ts_in=ts_in):
        interp = interpolation_func(ts_in, y, method)
        if ret_gradients:
            return jax.vmap(interp.derivative, 0, 0)(ts_out)
        else:
            return jax.vmap(interp.evaluate, 0, 0)(ts_out)

    interpolator_op = jax2pytensor(
        interpolator,
        output_shape_def=lambda y, **kwargs: (len(ts_out),) + y[1:],
        name=name,
    )

    return interpolator_op(ts_out, y)
