"""Convert a jax function to a pytensor compatible function."""

import functools as ft
import logging
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytensor.compile.builders
import pytensor.tensor as pt
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

log = logging.getLogger(__name__)

_filter_ptvars = lambda x: isinstance(x, pt.Variable)


def jax2pytensor(jaxfunc, name=None):
    """Return a Pytensor from a JAX jittable function.

    This decorator transforms any JAX jittable function into a function that accepts
    and returns `pytensor.Variables`. The jax jittable function can accept any
    nested python structure (pytrees) as input, and return any nested Python structure.

    It requires to define the output types of the returned values as pytensor types. A
    unique name should also be passed in case the name of the jaxfunc is identical to
    some other node. The design of this function is based on
    https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jaxfunc : jax jittable function
        function for which the node is created, can return multiple tensors as a tuple.
        It is required that all return values are able to transformed to
        pytensor.Variable.
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Only used internally in the pytensor graph.

    Returns
    -------
        A function which can be used in a pymc.Model as function, is differentiable
        and the resulting model can be compiled either with the default C backend, or
        the JAX backend.

    Examples
    --------
    A simple example on how the :func:`jax.numpy.sum` function is used in a PyMC model.

    >>> import jax.numpy as jnp                                # doctest: +ELLIPSIS
    >>> import pymc as pm                                      # doctest: +ELLIPSIS
    >>> import icomo                                           # doctest: +ELLIPSIS
    >>> import numpyro                                         # doctest: +ELLIPSIS
    >>> numpyro.set_host_device_count(4)
    >>> with pm.Model() as model:
    ...     x = pm.Normal("input", mu=1, size=3)
    ...     sum_pt = icomo.jax2pytensor(jnp.sum)
    ...     sum_x = sum_pt(x)
    ...     obs = pm.Normal("obs", sum_x, observed=3)
    >>> trace = pm.sample(model=model, nuts_sampler="numpyro") # doctest: +ELLIPSIS
    >>> print(np.round(np.mean(trace.posterior["input"].to_numpy()),1))
    1.0

    Or a more complex example. One might want to build a model which includes the
    solution of non-linear equation. For instance the intersection of `log(x)` and
    `1/x`:

    >>> import optimistix
    >>> with pm.Model() as model:
    ...     arg1 = pm.HalfNormal("arg1")
    ...     arg1 = pm.math.clip(arg1,0.1,10)
    ...     f1 = lambda x, arg1: jnp.log(x*arg1)
    ...     f2 = lambda x: 1/x
    ...     @icomo.jax2pytensor
    ...     def find_intersection(funcs, arg1):
    ...         f1, f2 = funcs["f1"], funcs["f2"]
    ...         loss = lambda x, _: (f1(x, arg1) - f2(x))**2
    ...               # Loss is squared to make it more convex
    ...         res = optimistix.minimise(fn = loss,
    ...                                   solver=optimistix.BFGS(rtol=1e-8, atol=1e-8),
    ...                                   y0=3.)
    ...         return res
    ...
    ...     intersection_res = find_intersection({"f1": f1, "f2": f2}, arg1)
    ...     intersection = pm.Deterministic("inters", intersection_res.value)
    ...     obs = pm.Normal("obs", mu=intersection, sigma=0.1, observed=3)
    >>> trace = pm.sample(model = model, nuts_sampler="numpyro") # doctest: +ELLIPSIS
    >>> print(f"Inters. = {np.round(np.mean(trace.posterior['inters'].to_numpy()),1)}")
    Inters. = 3.0
    >>> print(f"Std. int. = {np.round(np.std(trace.posterior['inters'].to_numpy()),1)}")
    Std. int. = 0.1

    Jax2pytensor also wraps returned functions from transformed functions, such that
    they can be used either in another transformed function, or evaluated directly to
    obtain the pytensor result:

    >>> with pm.Model() as model:
    ...     arg1 = pm.HalfNormal("arg1")
    ...     arg1 = pm.math.clip(arg1,0.1,10)
    ...     @icomo.jax2pytensor
    ...     def f1_creator(arg1):
    ...         def f_log(x):
    ...             return jnp.log(x*arg1)
    ...         return f_log
    ...     f1 = f1_creator(arg1)
    ...
    ...     f2 = lambda x: 1/x
    ...
    ...     @icomo.jax2pytensor
    ...     def find_intersection(f1, f2):
    ...         loss = lambda x, _: (f1(x) - f2(x))**2
    ...         res = optimistix.minimise(fn = loss,
    ...                                   solver=optimistix.BFGS(rtol=1e-8, atol=1e-8),
    ...                                   y0=3.)
    ...         return res
    ...
    ...     # Pass wrapped f2 function to find_intersection, it creates a pytensor op
    ...     # that also includes arg1 as its input.
    ...     intersection_res = find_intersection(f1, f2)
    ...     intersection = pm.Deterministic("inters", intersection_res.value)
    ...
    ...     # Or evaluate f2 directly
    ...     log_var = pm.Deterministic("log_var", f2(intersection))
    ...     obs = pm.Normal("obs", mu=intersection, sigma=0.1, observed=3)
    >>> trace = pm.sample(model = model, nuts_sampler="numpyro")   # doctest: +ELLIPSIS
    >>> print(f"Inters. = {np.round(np.mean(trace.posterior['inters'].to_numpy()),1)}")
    Inters. = 3.0
    >>> print(f"Std. int. = {np.round(np.std(trace.posterior['inters'].to_numpy()),1)}")
    Std. int. = 0.1
    >>> print(f"log_var = {np.round(np.mean(trace.posterior['log_var'].to_numpy()),1)}")
    log_var = 0.3
    >>> # It also works with the default sampler backend
    >>> trc2 = pm.sample(model = model, cores=1, progressbar=False) # doctest: +ELLIPSIS
    >>> print(f"log_var = {np.round(np.mean(trc2.posterior['log_var'].to_numpy()),1)}")
    log_var = 0.3

    This feature of wrapping returned functions hasn't been tested extensively,
    so please report any issues you might encounter.

    Notes
    -----
    The function is based on a blog post by Ricardo Vieira and Adrian Seyboldt,
    available at
    `pymc-labls.io <https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick
    -examples/>`__.
    To accept functions and non pytensor variables as input, the function make use
    of :func:`equinox.partition` and :func:`equinox.combine` to split and combine the
    variables. Shapes are inferred using
    :func:`pytensor.compile.builders.infer_shape` and :func:`jax.eval_shape`.
    """

    def func(*args, **kwargs):
        """Return a pytensor from a jax jittable function."""
        ### Split variables: in the ones that will be transformed to JAX inputs,
        ### pytensor.Variables; _WrappedFunc, that are functions that have been returned
        ### from a transformed function; and the rest, static variables that are not
        ### transformed.

        pt_vars, static_vars_tmp = eqx.partition(
            (args, kwargs), _filter_ptvars, is_leaf=callable
        )
        # is_leaf=callable is used, as libraries like diffrax or equinox might return
        # functions that are still seen as a nested pytree structure. We consider them
        # as wrappable functions, that will be wrapped with _WrappedFunc.

        func_vars, static_vars = eqx.partition(
            static_vars_tmp, lambda x: isinstance(x, _WrappedFunc), is_leaf=callable
        )
        vars_from_func = tree_map(lambda x: x.get_vars(), func_vars)
        pt_vars = dict(vars=pt_vars, vars_from_func=vars_from_func)
        """
        def func_unwrapped(vars_all, static_vars):
            vars, vars_from_func = vars_all["vars"], vars_all["vars_from_func"]
            func_vars_evaled = tree_map(
                lambda x, y: x.get_func_with_vars(y), func_vars, vars_from_func
            )
            args, kwargs = eqx.combine(vars, static_vars, func_vars_evaled)
            return self.jaxfunc(*args, **kwargs)
        """

        pt_vars_flat, vars_treedef = tree_flatten(pt_vars)
        pt_vars_types_flat = [var.type for var in pt_vars_flat]
        shapes_vars_flat = pytensor.compile.builders.infer_shape(pt_vars_flat, (), ())
        shapes_vars = tree_unflatten(vars_treedef, shapes_vars_flat)

        dummy_inputs_jax = jax.tree_util.tree_map(
            lambda var, shape: jnp.empty(
                [int(dim.eval()) for dim in shape], dtype=var.type.dtype
            ),
            pt_vars,
            shapes_vars,
        )

        # Combine the static variables with the inputs, and split them again in the
        # output. Static variables don't take part in the graph, or might be a
        # a function that is returned.
        jaxfunc_partitioned, static_out_dic = _partition_jaxfunc(
            jaxfunc, static_vars, func_vars
        )

        func_flattened = _flatten_func(jaxfunc_partitioned, vars_treedef)

        jaxtypes_outvars = jax.eval_shape(
            ft.partial(jaxfunc_partitioned, vars=dummy_inputs_jax),
        )

        jaxtypes_outvars_flat, outvars_treedef = tree_flatten(jaxtypes_outvars)

        pttypes_outvars = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in jaxtypes_outvars_flat
        ]

        ### Call the function that accepts flat inputs, which in turn calls the one that
        ### combines the inputs and static variables.
        jitted_sol_op_jax = jax.jit(func_flattened)
        len_gz = len(pttypes_outvars)

        vjp_sol_op_jax = _get_vjp_sol_op_jax(func_flattened, len_gz)
        jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

        if name is None:
            curr_name = jaxfunc.__name__
        else:
            curr_name = name

        # Get classes that creates a Pytensor Op out of our function that accept
        # flattened inputs. They are created each time, to set a custom name for the
        # class.
        SolOp, VJPSolOp = _return_pytensor_ops_classes(curr_name)

        local_op = SolOp(
            vars_treedef,
            outvars_treedef,
            input_types=pt_vars_types_flat,
            output_types=pttypes_outvars,
            jitted_sol_op_jax=jitted_sol_op_jax,
            jitted_vjp_sol_op_jax=jitted_vjp_sol_op_jax,
        )

        @jax_funcify.register(SolOp)
        def sol_op_jax_funcify(op, **kwargs):
            return local_op.perform_jax

        @jax_funcify.register(VJPSolOp)
        def vjp_sol_op_jax_funcify(op, **kwargs):
            return local_op.vjp_sol_op.perform_jax

        ### Evaluate the Pytensor Op and return unflattened results
        output_flat = local_op(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        outvars = tree_unflatten(outvars_treedef, output_flat)

        static_outfuncs, static_outvars = eqx.partition(
            static_out_dic["out"], callable, is_leaf=callable
        )

        static_outfuncs_flat, treedef_outfuncs = jax.tree_util.tree_flatten(
            static_outfuncs, is_leaf=callable
        )
        for i_func, _ in enumerate(static_outfuncs_flat):
            static_outfuncs_flat[i_func] = _WrappedFunc(
                jaxfunc, i_func, *args, **kwargs
            )

        static_outfuncs = jax.tree_util.tree_unflatten(
            treedef_outfuncs, static_outfuncs_flat
        )
        static_vars = eqx.combine(static_outfuncs, static_outvars, is_leaf=callable)

        output = eqx.combine(outvars, static_vars, is_leaf=callable)

        return output

    return func


class _WrappedFunc:
    def __init__(self, exterior_func, i_func, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.i_func = i_func
        vars, static_vars = eqx.partition(
            (self.args, self.kwargs), _filter_ptvars, is_leaf=callable
        )
        self.vars = vars
        self.static_vars = static_vars
        self.exterior_func = exterior_func

    def __call__(self, *args, **kwargs):
        # If called, assume that args and kwargs are pytensors, so return the result
        # as pytensors.
        def f(func, *args, **kwargs):
            res = func(*args, **kwargs)
            return res

        return jax2pytensor(f)(self, *args, **kwargs)

    def get_vars(self):
        return self.vars

    def get_func_with_vars(self, vars):
        # Use other variables than the saved ones, to generate the function. This
        # is used to transform vars externally from pytensor to JAX, and use the
        # then create the function which is returned.

        args, kwargs = eqx.combine(vars, self.static_vars, is_leaf=callable)
        output = self.exterior_func(*args, **kwargs)
        outfuncs, _ = eqx.partition(output, callable, is_leaf=callable)
        outfuncs_flat, _ = jax.tree_util.tree_flatten(outfuncs, is_leaf=callable)
        interior_func = outfuncs_flat[self.i_func]
        return interior_func


def _get_vjp_sol_op_jax(jaxfunc, len_gz):
    def vjp_sol_op_jax(args):
        y0 = args[:-len_gz]
        gz = args[-len_gz:]
        if len(gz) == 1:
            gz = gz[0]
        func = lambda *inputs: jaxfunc(inputs)
        primals, vjp_fn = jax.vjp(func, *y0)
        gz = tree_map(
            lambda g, primal: jnp.broadcast_to(g, jnp.shape(primal)),
            gz,
            primals,
        )
        if len(y0) == 1:
            return vjp_fn(gz)[0]
        else:
            return tuple(vjp_fn(gz))

    return vjp_sol_op_jax


def _partition_jaxfunc(jaxfunc, static_vars, func_vars):
    """Partition the jax function into static and non-static variables.

    Returns a function that accepts only non-static variables and returns the non-static
    variables. The returned static variables are stored in a dictionary and returned,
    to allow the referencing after creating the function

    Additionally wrapped functions saved in func_vars are regenerated with
    vars["vars_from_func"] as input, to allow the transformation of the variables.
    """
    static_out_dic = {"out": None}

    def jaxfunc_partitioned(vars):
        vars, vars_from_func = vars["vars"], vars["vars_from_func"]
        func_vars_evaled = tree_map(
            lambda x, y: x.get_func_with_vars(y), func_vars, vars_from_func
        )
        args, kwargs = eqx.combine(
            vars, static_vars, func_vars_evaled, is_leaf=callable
        )

        out = jaxfunc(*args, **kwargs)
        outvars, static_out = eqx.partition(out, eqx.is_array, is_leaf=callable)
        static_out_dic["out"] = static_out
        return outvars

    return jaxfunc_partitioned, static_out_dic


### Construct the function that accepts flat inputs and returns flat outputs.
def _flatten_func(jaxfunc, vars_treedef):
    def func_flattened(vars_flat):
        vars = tree_unflatten(vars_treedef, vars_flat)
        outvars = jaxfunc(vars)
        outvars_flat, _ = tree_flatten(outvars)
        return _normalize_flat_output(outvars_flat)

    return func_flattened


def _normalize_flat_output(output):
    if len(output) > 1:
        return tuple(
            output
        )  # Transform to tuple because jax makes a difference between
        # tuple and list and not pytensor
    else:
        return output[0]


def _return_pytensor_ops_classes(name):
    class SolOp(Op):
        def __init__(
            self,
            input_treedef,
            output_treeedef,
            input_types,
            output_types,
            jitted_sol_op_jax,
            jitted_vjp_sol_op_jax,
        ):
            self.vjp_sol_op = None
            self.input_treedef = input_treedef
            self.output_treedef = output_treeedef
            self.input_types = input_types
            self.output_types = output_types
            self.jitted_sol_op_jax = jitted_sol_op_jax
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax

        def make_node(self, *inputs):
            self.num_inputs = len(inputs)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
            self.num_outputs = len(outputs)

            self.vjp_sol_op = VJPSolOp(
                self.input_treedef,
                self.input_types,
                self.jitted_vjp_sol_op_jax,
            )

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            results = self.jitted_sol_op_jax(inputs)
            if self.num_outputs > 1:
                for i in range(self.num_outputs):
                    outputs[i][0] = np.array(results[i], self.output_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.output_types[0].dtype)

        def perform_jax(self, *inputs):
            results = self.jitted_sol_op_jax(inputs)
            return results

        def grad(self, inputs, output_gradients):
            # If a output is not used, it is disconnected and doesn't have a gradient.
            # Set gradient here to zero for those outputs.
            for i in range(self.num_outputs):
                if isinstance(output_gradients[i].type, DisconnectedType):
                    if None not in self.output_types[i].shape:
                        output_gradients[i] = pt.zeros(
                            self.output_types[i].shape, self.output_types[i].dtype
                        )
                    else:
                        output_gradients[i] = pt.zeros((), self.output_types[i].dtype)
            result = self.vjp_sol_op(inputs, output_gradients)

            if self.num_inputs > 1:
                return result
            else:
                return (result,)  # Pytensor requires a tuple here

    # vector-jacobian product Op
    class VJPSolOp(Op):
        def __init__(
            self,
            input_treedef,
            input_types,
            jitted_vjp_sol_op_jax,
        ):
            self.input_treedef = input_treedef
            self.input_types = input_types
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax

        def make_node(self, y0, gz):
            y0 = [
                pt.as_tensor_variable(
                    _y,
                ).astype(self.input_types[i].dtype)
                for i, _y in enumerate(y0)
            ]
            gz_not_disconntected = [
                pt.as_tensor_variable(_gz)
                for _gz in gz
                if not isinstance(_gz.type, DisconnectedType)
            ]
            outputs = [in_type() for in_type in self.input_types]
            self.num_outputs = len(outputs)
            return Apply(self, y0 + gz_not_disconntected, outputs)

        def perform(self, node, inputs, outputs):
            results = self.jitted_vjp_sol_op_jax(tuple(inputs))
            if len(self.input_types) > 1:
                for i, result in enumerate(results):
                    outputs[i][0] = np.array(result, self.input_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.input_types[0].dtype)

        def perform_jax(self, *inputs):
            results = self.jitted_vjp_sol_op_jax(tuple(inputs))
            if self.num_outputs == 1:
                if isinstance(results, Sequence):
                    return results[0]
                else:
                    return results
            else:
                return tuple(results)

    SolOp.__name__ = name
    SolOp.__qualname__ = ".".join(SolOp.__qualname__.split(".")[:-1] + [name])

    VJPSolOp.__name__ = "VJP_" + name
    VJPSolOp.__qualname__ = ".".join(
        VJPSolOp.__qualname__.split(".")[:-1] + ["VJP_" + name]
    )

    return SolOp, VJPSolOp
