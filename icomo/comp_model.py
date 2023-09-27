"""Tools to create compartmental models and compile them into functions that can be used."""

import logging

import diffrax
import graphviz
from pytensor.tensor.type import TensorType

from icomo.pytensor_op import create_and_register_jax

logger = logging.getLogger(__name__)


def SIR(t, y, args):
    """
    Differential equations of an SIR model.

    Parameters
    ----------
    t: float
        time variable
    y: dict
        dictionary of compartments. Has to include keys "S", "I", "R". The value of "S"
        is the number of susceptible individuals, "I" is the number of infected
        individuals and "R" is the number of recovered individuals.
    args: tuple
        tuple of arguments. The first argument is the function β(t) that describes the
        infection rate. The second argument is a dictionary of constant arguments. It
        has to include the key "gamma" which is the recovery rate and the key "N" which
        is the total population size.

    Returns
    -------
    dy: dict
        dictionary of derivatives with keys "S", "I", "R".

    """
    β, const_arg = args
    γ = const_arg["gamma"]
    N = const_arg["N"]
    dS = -β(t) * y["I"] * y["S"] / N
    dI = β(t) * y["I"] * y["S"] / N - γ * y["I"]
    dR = γ * y["I"]
    dy = {"S": dS, "I": dI, "R": dR}
    return dy


def Erlang_SEIRS(t, y, args):
    """Return the derivatives of the Erlang SEIRS model.

    This function defines the differential equations of an SEIRS model with Erlang
    distributed latent, infectious and recovering (recovered-to-susceptible) periods.

    Parameters
    ----------
    t: float
        time variable
    y: dict
        dictionary of compartments. Has to include keys "S", "Es", "Is", "Rs". The value
        of "Es", "Is", "Rs" have to be a list of length `n` where `n` is the number of
        compartments/shape of the Erlang distribution/kernel.
    args: tuple: (callable, dict)
        Callable is a function with argument t that returns the value of the transmission
        rate beta(t)
        dict is the constant arguments of the model. Has to include keys "N", "rate_latent",
        "rate_infectious" and "rate_recovery".
    Returns
    -------
    dComp: dict
        Same as y, but with the derivatives of the compartments.
    """
    beta, const_arg = args
    N = const_arg["N"]
    dComp = {}

    I = sum(y["Is"])
    dComp["S"] = -beta(t) * I * y["S"] / N

    # Latent period
    dEs, outflow = erlang_kernel(
        inflow=beta(t) * I * y["S"] / N,
        Vars=y["Es"],
        rate=const_arg["rate_latent"],
    )
    dComp["Es"] = dEs

    # Infectious period
    dIs, outflow = erlang_kernel(
        inflow=outflow,
        Vars=y["Is"],
        rate=const_arg["rate_infectious"],
    )
    dComp["Is"] = dIs

    # Recovered/non-susceptible period
    dRs, outflow = erlang_kernel(
        inflow=outflow,
        Vars=y["Rs"],
        rate=const_arg["rate_recovery"],
    )
    dComp["Rs"] = dRs

    dComp["S"] = dComp["S"] + outflow

    return dComp


def erlang_kernel(inflow, Vars, rate):
    """Model the compartments delayed by an Erlang kernel.

    Utility function to model an Erlang kernel for a compartmental model. The shape is
    determined by the length of Vars.

    Parameters
    ----------
    inflow: float or ndarray
        The inflow into the first compartment of the kernel
    Vars: list of floats or list of ndarrays
        The compartments of the kernel
    rate: float or ndarray
        The rate of the kernel, 1/rate is the mean time spent in total in all compartments
    Returns
    -------
    dVars: list of floats or list of ndarrays
        The derivatives of the compartments
    out_flow: float or ndarray
        The outflow from the last compartment
    """
    dVars = []
    m = len(Vars)
    for i, Var_i in enumerate(Vars):
        dVars.append(-m * rate * Var_i)
        if i == 0:
            dVars[0] = dVars[0] + inflow
        else:
            dVars[i] = dVars[i] + m * rate * Vars[i - 1]
    out_flow = m * rate * Vars[-1]
    return dVars, out_flow


class CompModel:
    """Class to help building a compartmental model.

    The model is built by adding flows between compartments. The model is then compiled
    into a function that can be used in an ODE.
    """

    def __init__(self, Comp_dict):
        """Initialize the CompModel class.

        Parameters
        ----------
        Comp_dict: dict
            Dictionary of compartments. Keys are the names of the compartments and values
            are floats or ndarrays that represent their value.
        """
        self.Comp = Comp_dict
        self.dComp = {}
        self.graph = graphviz.Digraph("comp_model")
        for key in self.Comp.keys():
            self.graph.node(key)
        self.graph.attr(rankdir="LR")

    def flow(self, start_comp, end_comp, rate, label=None, end_comp_is_list=False):
        """
        Add a flow from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            Key of the start compartment. Can also be a list of keys in which case an
            identical flow is added from each compartment of the list to end_com
        end_comp: str
            Key of the end compartment
        rate: float or ndarray
            rate of the flow to add between compartments, is multiplied by start_comp, so it should
            be broadcastable whith it.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_list: bool, default: False
            If True, end_comp points to a list of compartments, and the outflow is added to
            the first of them. This is typically the case when the end compartment is
            used with a Erlang kernel.

        Returns
        -------
        None
        """
        if isinstance(start_comp, str):
            start_comp_list = [start_comp]
        else:
            start_comp_list = start_comp
        for start_comp in start_comp_list:
            self._add_dy_to_comp(start_comp, -rate * self.Comp[start_comp])
            self._add_dy_to_comp(
                end_comp, rate * self.Comp[start_comp], end_comp_is_list
            )

            self.graph.edge(start_comp, end_comp, label=label)

    def erlang_flow(
        self, start_comp, end_comp, rate, label=None, end_comp_is_list=False
    ):
        """
        Add a flow with erlang kernel from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str
            start compartments of the flow. The length the list is the shape of the
            Erlang kernel.
        end_comp: str
            end compartment of the flow
        rate: float or ndarray
            rate of the flow, equal to the inverse of the mean time spent in the Erlang
            kernel.
        label: str, optional
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.
        end_comp_is_list: bool, default: False
            If True, end_comp points to a list of compartments, and the outflow is added to
            the first of them. This is typically the case when the end compartment is
            used with a Erlang kernel.
        """
        if not (
            isinstance(self.Comp[start_comp], list)
            or isinstance(self.Comp[start_comp], tuple)
        ):
            raise RuntimeError(
                "start_comp should refer to a list of compartments to allow"
                "the modelling of an erlang kernel across those compartments"
            )

        dy, outflow = erlang_kernel(inflow=0, Vars=self.Comp[start_comp], rate=rate)
        self._add_dy_to_comp(start_comp, dy, comp_is_list=True)
        self._add_dy_to_comp(end_comp, outflow, comp_is_list=end_comp_is_list)

        self.graph.edge(start_comp, end_comp, label=label)

    def _add_dy_to_comp(self, comp_key, dy, comp_is_list=False):
        if not comp_is_list:
            if not comp_key in self.dComp.keys():
                self.dComp[comp_key] = 0
            self.dComp[comp_key] = self.dComp[comp_key] + dy
        else:
            if not comp_key in self.dComp.keys():
                self.dComp[comp_key] = [0 for _ in self.Comp[comp_key]]
            if isinstance(dy, list):
                if not len(dy) == len(self.Comp[comp_key]):
                    raise RuntimeError(
                        "dy should have the same length as the compartment"
                    )
                for i, _ in enumerate(self.Comp[comp_key]):
                    self.dComp[comp_key][i] = self.dComp[comp_key][i] + dy[i]
            else:
                # Add only to first comp
                self.dComp[comp_key][0] = self.dComp[comp_key][0] + dy

    @property
    def dy(self):
        """Returns the derivative of the compartments.

        This should be returned by the function that defines the system of ODEs.

        Returns
        -------
        dComp: dict
        """
        return self.dComp

    def view_graph(self, on_display=True):
        """Display a graph of the compartmental model.

        Parameters
        ----------
        on_display : bool
            If True, the graph is displayed in the notebook, otherwise it is saved as a
            pdf in the current folder and opened with the default pdf viewer.

        Returns
        -------
        None

        """
        if on_display:
            try:
                from IPython.display import display

                display(self.graph)
            except:
                self.graph.view()
        else:
            self.graph.view()


def delayed_copy(initial_var, delayed_vars, tau_delay):
    """Return the derivative to model the delayed copy of a compartment.

    The delay has the form of an Erlang kernel with shape parameter len(delayed_vars).

    Parameters
    ----------
    initial_var : float or ndarray
        The compartment that is copied
    delayed_vars : list of floats or list of ndarrays
        List of compartments that are delayed copies of initial_var, the last element
        is the compartment which has the same total content as initial_var over time, but
        is delayed by tau_delay.
    tau_delay : float or ndarray
        The mean delay of the copy.

    Returns
    -------
    d_delayed_vars : list of floats or list of ndarrays
        The derivatives of the delayed compartments

    """
    shape = len(delayed_vars)
    inflow = initial_var / tau_delay * shape
    if shape == 1:
        d_delayed_vars = []
        d_delayed_vars_last = inflow
    elif shape > 1:
        d_delayed_vars, outflow = erlang_kernel(
            inflow, delayed_vars[:-1], 1 / tau_delay * shape
        )
        d_delayed_vars_last = outflow
    return d_delayed_vars + [d_delayed_vars_last]


class ODEIntegrator:
    """Creates an integrator for compartmental models.

    The integrator is a function that
    takes as input a function that returns the derivatives of the variables of the system
    of differential equations, and returns a function that solves the system of
    differential equations. For the initilization of the integrator object, the timesteps
    of the solver and the output have to be specified.

    """

    def __init__(
        self,
        ts_out,
        ts_solver=None,
        ts_arg=None,
        interp="cubic",
        solver=diffrax.Tsit5(),
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
            The timesteps at which the solver will be called. If None, it is set to ts_out.
        ts_arg : array-like or None
            The timesteps at which the time-dependent argument of the system of differential
            equations are given. If None, it is set to ts_solver.
        interp : str
            The interpolation method used to interpolate_pytensor the time-dependent argument of the
            system of differential equations. Can be "cubic" or "linear".
        solver : :class:`diffrax.AbstractStepSizeController`
            The solver used to integrate the system of differential equations. Default is
            diffrax.Tsit5(), a 5th order Runge-Kutta method.
        t_0 : float or None
            The initial time of the integration. If None, it is set to ts_solve[0].
        t_1 : float or None
            The final time of the integration. If None, it is set to ts_solve[-1].
        **kwargs
            Arguments passed to the solver, see :func:`diffrax.diffeqsolve` for more details.
        """
        self.ts_out = ts_out
        if ts_solver is None:
            self.ts_solver = self.ts_out
        else:
            self.ts_solver = ts_solver
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
            `t` is a float, `y` is a list or dict of floats or ndarrays, or in general, a
            pytree, see :mod:`jax.tree_util` for more details. The return value of the function
            has to be a pytree/list/dict with the same structure as `y`.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that are
            returned by the integrator. If set, the integrator returns a list of the
            variables of the system of differential equations in the order of the keys.
            If `None`, the output is returned as is.

        Returns
        -------
        integrator : function(y0, arg_t=None, constant_args=None)
            A function that solves the system of differential equations and returns the
            output at the specified timesteps. The function takes as input `y0` the initial
            values of the variables of the system of differential equations, the
            time-dependent argument of the system of differential equations `arg_t`, and
            the constant arguments `constant_args` of the system of differential equations. `t`, `y0` and
            `(arg_t, constant_args)` are passed to the ODE function as its three arguments.
            If `arg_t` is `None`, only `constant_args` are passed to the ODE function and
            vice versa, without being in a tuple.

        """
        def arg_t_extractor(arg_t, ts_arg):
            if arg_t is None:
                raise RuntimeError("Specifying arg_t of None is not supported within tuple of arg_t")
            if not callable(arg_t):
                if self.ts_arg is None:
                    raise RuntimeError("Specify ts_arg to use a non-callable arg_t")
                arg_t_func = interpolation_func(
                    ts=ts_arg, x=arg_t, method=self.interp
                ).evaluate
                return arg_t_func
            logger.warning(
                "arg_t is callable, but ts_arg is not None. ts_arg"
                " won't be used."
            )
            return arg_t
        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                if isinstance(arg_t, tuple) and isinstance(self.ts_arg, tuple):
                    if len(arg_t) != len(self.ts_arg):
                        raise RuntimeError("Mismatch of elements in `arg_t` and `ts_arg`")
                    arg_t_funcs = tuple(map(arg_t_extractor, arg_t, self.ts_arg))
                elif isinstance(arg_t, tuple):
                    arg_t_funcs = tuple(map(lambda f : arg_t_extractor(f, self.ts_arg), arg_t))
                else:
                    arg_t_funcs = arg_t_extractor(arg_t, self.ts_arg)

            if arg_t is None and self.ts_arg is not None:
                logger.warning(
                    "You did specify ts_arg, but arg_t is None. Did you mean to do this?"
                )
            term = diffrax.ODETerm(ODE)

            if arg_t is None:
                args = constant_args
            elif constant_args is None:
                args = arg_t_funcs
            else:
                args = (
                    arg_t_funcs,
                    constant_args,
                )
            saveat = diffrax.SaveAt(ts=self.ts_out)

            stepsize_controller = (
                diffrax.StepTo(ts=self.ts_solver)
                if not "stepsize_controller" in self.kwargs_solver
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

        Same as get_func, but returns a pytensor operator that can be used in a pymc model.
        Beware that for this operator the output of the integration of the ODE can only
        be a single or a list variables. If the output is a dict, set list_keys_to_return
        to specify the keys of the variables that are returned by the integrator. These
        return values aren't allowed to be further nested.

        Parameters
        ----------
        ODE : function(t, y, args)
            A function that returns the derivatives of the variables of the system of
            differential equations. The function has to take as input the time `t`, the
            variables `y` and the arguments `args=(arg_t, constant_args)` of the system
            of differential equations.
            `t` is a float, `y` is a list or dict of floats or ndarrays, or in general, a
            pytree, see :mod:`jax.tree_util` for more details. The return value of the function
            has to be a pytree/list/dict with the same structure as `y`.
        return_shapes : tuple of tuples, default is ((),)
            The shapes (except the time dimension) of the variables of the system of
            differential equations that are returned by the integrator. If
            `list_keys_to_return` is `None`, the shapes have to be given in the same order
            as the variables are returned by the integrator. If `list_keys_to_return` is
            not `None`, the shapes have to be given in the same order as the keys in
            `list_keys_to_return`. The default `((),)` means a single variable with only a
            time dimension is returned.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that will
            be chosen to be returned by the integrator. Necessary if the ODE returns a
            `dict`, as :mod:`pytensor` only accepts single outputs or a list of outputs.
            If `None`, the output is returned as is.
        name :
            The name under which the operator is registered in pymc.

        Returns
        -------
        pytensor_op : :class:`pytensor.graph.op.Op`
            A :mod:`pytensor` operator that can be used in a :class:`pymc.Model`.

        """
        integrator = self.get_func(ODE, list_keys_to_return=list_keys_to_return)

        pytensor_op = create_and_register_jax(
            integrator,
            output_types=[
                TensorType(
                    dtype="float64", shape=tuple([len(self.ts_out)] + list(shape))
                )
                for shape in return_shapes
            ],
            name=name,
        )
        return pytensor_op


def interpolation_func(ts, x, method="cubic"):
    """
    Return a diffrax-interpolation function that can be used to interpolate_pytensor the time-dependent variable.

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
        The interpolation function. Call `interp.evaluate(t)` to evaluate the interpolated
        variable at time `t`. t can be a float or an array-like.

    """
    if method == "cubic":
        coeffs = diffrax.backward_hermite_coefficients(ts, x)
        interp = diffrax.CubicInterpolation(ts, coeffs)
    elif method == "linear":
        interp = diffrax.LinearInterpolation(ts, x)
    else:
        raise RuntimeError(
            f'Interpoletion method {method} not known, possibilities are "cubic" or "linear"'
        )
    return interp


def interpolate_pytensor(ts_in, ts_out, y, method="cubic", ret_gradients=False):
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

    def interpolator(ts_out, y):
        interp = interpolation_func(ts_in, y, method)
        if ret_gradients:
            return interp.derivative(ts_out)
        else:
            return interp.evaluate(ts_out)

    interpolator_op = create_and_register_jax(
        interpolator,
        output_types=[
            TensorType(dtype="float64", shape=(len(ts_out),)),
        ],
    )

    return interpolator_op(ts_out, y)
