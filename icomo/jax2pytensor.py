"""Convert a jax function to a pytensor compatible function."""

import functools as ft
import inspect
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


class _WrappedFunc:
    def __init__(self, func, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        vars, static_vars = eqx.partition((self.args, self.kwargs), _filter_ptvars)
        self.vars = vars
        self.static_vars = static_vars
        self.func = func

    def __call__(self, *args, **kwargs):
        func_internal = self.func(*self.args, **self.kwargs)
        return func_internal(*args, **kwargs)

    def get_vars(self):
        return self.vars

    def set_vars(self, vars):
        self.vars = vars
        self.args, self.kwargs = eqx.combine(self.vars, self.static_vars)

    def get_func_with_vars(self, vars):
        args, kwargs = eqx.combine(vars, self.static_vars)
        return self.func(*args, **kwargs)


def jax2pyfunc(func_gen):
    """Return a pytensor function from a jax jittable function."""

    @ft.wraps(func_gen)
    def func_gen_wrapper(*args, **kwargs):
        return _WrappedFunc(func_gen, *args, **kwargs)

    return func_gen_wrapper


class _Jax2Pytensor:
    def __init__(
        self,
        jaxfunc,
        name=None,
    ):
        self.jaxfunc = jaxfunc
        self.name = name

    def __call__(self, *args, **kwargs):
        """Return a pytensor from a jax jittable function."""
        ### Construct the function to return that is compatible with pytensor but has
        ### the same signature as the jax function.
        pt_vars, static_vars_tmp = eqx.partition((args, kwargs), _filter_ptvars)
        func_vars, static_vars = eqx.partition(
            static_vars_tmp, lambda x: isinstance(x, _WrappedFunc)
        )
        vars_from_func = tree_map(lambda x: x.get_vars(), func_vars)
        pt_vars = dict(vars=pt_vars, vars_from_func=vars_from_func)

        def func(vars_all, static_vars):
            vars, vars_from_func = vars_all["vars"], vars_all["vars_from_func"]
            func_vars_evaled = tree_map(
                lambda x, y: x.get_func_with_vars(y), func_vars, vars_from_func
            )
            args, kwargs = eqx.combine(vars, static_vars, func_vars_evaled)
            return self.jaxfunc(*args, **kwargs)

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

        # jaxtypes_out = eqx.filter_eval_shape(
        #    jaxfunc, *dummy_inputs_jax[0], **dummy_inputs_jax[1]
        # )

        static_outvars = None

        def _jaxfunc_partitioned(vars, static_vars):
            # args, kwargs = eqx.combine(vars, static_vars)
            out = func(vars, static_vars)
            nonlocal static_outvars
            outvars, static_outvars = eqx.partition(
                out,
                eqx.is_array,  # lambda x: isinstance(x, jax.Array)
            )
            # print("static outvars: {}", static_outvars)
            # jax.debug.print("outvars: {}", outvars)
            return outvars

        def _func_flattened(vars_flat, vars_treedef, static_vars):
            # jax.debug.print("vars flat: {}", vars_flat)
            vars = tree_unflatten(vars_treedef, vars_flat)
            # jax.debug.print("vars: {}", vars)
            outvars = _jaxfunc_partitioned(vars, static_vars)
            outvars_flat, _ = tree_flatten(outvars)
            # jax.debug.print("outvars flat: {}", outvars_flat)
            return _normalize_flat_output(outvars_flat)

        _func_flattened_partial = ft.partial(
            _func_flattened,
            vars_treedef=vars_treedef,
            static_vars=static_vars,
        )

        jaxtypes_outvars = jax.eval_shape(
            ft.partial(
                _jaxfunc_partitioned, vars=dummy_inputs_jax, static_vars=static_vars
            ),
        )

        """
        # Evaluate shape, could have also used jax.eval_shape or
        # equinox.filter_eval_shape, for future...
        _, jaxtypes_outvars = jax.make_jaxpr(
            ft.partial(_jaxfunc_partitioned, static_vars=static_vars),
            return_shape=True,
        )(dummy_inputs_jax)
        """

        # jaxtypes_outvars_flat = (
        #    (jaxtypes_outvars_flat,)
        #    if isinstance(jaxtypes_outvars_flat, jax.ShapeDtypeStruct)
        #    else jaxtypes_outvars_flat
        # )

        jaxtypes_outvars_flat, outvars_treedef = tree_flatten(jaxtypes_outvars)

        # print("jaxtypes_outvars: ", jaxtypes_outvars)

        pttypes_outvars = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in jaxtypes_outvars_flat
        ]
        # print("pttypes_outvars: ", pttypes_outvars)

        ### Create the Pytensor Op, the normal one and the vector-jacobian-
        ### product (vjp)
        # flat_func = _flattened_input_func(
        #     jaxfunc, input_treedef, inputnames_list, other_args_dic
        # )

        jitted_sol_op_jax = jax.jit(
            ft.partial(
                _func_flattened,
                vars_treedef=vars_treedef,
                static_vars=static_vars,
            )
        )
        len_gz = len(pttypes_outvars)

        def vjp_sol_op_jax(args):
            y0 = args[:-len_gz]
            gz = args[-len_gz:]
            if len(gz) == 1:
                gz = gz[0]
            primals, vjp_fn = jax.vjp(local_op.perform_jax, *y0)
            gz = tree_map(
                lambda g, primal: jnp.broadcast_to(g, jnp.shape(primal)),
                gz,
                primals,
            )
            if len(y0) == 1:
                return vjp_fn(gz)[0]
            else:
                return tuple(vjp_fn(gz))

        jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

        if self.name is None:
            self.name = self.jaxfunc.__name__

        # Get classes that creates a Pytensor Op out of our function that accept
        # flattened inputs. They are created each time, to set a custom name for the
        # class.
        SolOp, VJPSolOp = _return_pytensor_ops_classes(self.name)

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
        # print("pt_vars_flat: ", pt_vars_flat)
        output_flat = local_op(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        outvars = tree_unflatten(outvars_treedef, output_flat)
        output = eqx.combine(outvars, static_outvars)
        # output = outvars
        # print("output: ", output)
        return output


def _flattened_input_func(
    func, input_treedef, inputnames_list, other_args_dic, flatten_output=True
):
    """Build a function that accepts flat inputs and returns flat outputs.

    Returns a function that accepts flattened inputs, and optionnially returns flattened
    outputs. The function is used to create a Pytensor Op, as Pytensor requires a
    flat inputs and outputs.

    Parameters
    ----------
    func : function
        function to be converted that accepts non-flat inputs
    input_treedef : treedef
        treedef of the inputs, as returned by jax.tree_util.tree_flatten
    inputnames_list : list of str
        parameter names of the inputs
    other_args_dic : dict
        dictionary of other arguments that are not inputs for the graph. Aren't affected
        by the flattening.
    flatten_output : bool
        whether the output should be flattened or not. Default is True.

    Returns
    -------
    function
        function that accepts flat inputs and optionally returns flat outputs.
    """

    def new_func(inputs_list_flat):
        inputs_for_graph = tree_unflatten(input_treedef, inputs_list_flat)
        inputs_for_graph = tree_map(lambda x: jnp.array(x), inputs_for_graph)
        inputs_for_graph_dic = {
            arg: val
            for arg, val in zip(inputnames_list, inputs_for_graph, strict=False)
        }
        results = func(**inputs_for_graph_dic, **other_args_dic)
        if not flatten_output:
            return results
        else:
            results, output_treedef_local = tree_flatten(results)
            return _normalize_flat_output(results)

    return new_func


def _normalize_flat_output(output):
    if len(output) > 1:
        return tuple(
            output
        )  # Transform to tuple because jax makes a difference between
        # tuple and list and not pytensor
    else:
        return output[0]


def _split_arguments(func_signature, args, kwargs, args_for_graph):
    """Split the arguments into inputs for the graph and other arguments.

    Parameters
    ----------
    func_signature : inspect.Signature
        signature of the function
    args : tuple
        arguments of the function as collected by *args
    kwargs : dict
        keyword arguments of the function as collected by **kwargs
    args_for_graph : list of str or "all"
        If "all", all arguments are used for the graph. Otherwise specify a list of
        argument names to use for the graph.

    Returns
    -------
    inputs_for_graph_list : list
        list of inputs for the graph as a list of non-flat pytensors trees.
    inputnames_for_graph_list : list
        list of the argument names of the inputs for the graph, same length as
        inputs_for_graph_list.
    other_args_dic : dict
        dictionary of arguments not used for the graph.
    """
    for key in kwargs.keys():
        if key not in func_signature.parameters and key in args_for_graph:
            raise RuntimeError(
                f"Keyword argument <{key}> not found in function signature. "
                f"**kwargs are not supported in the definition of the function,"
                f"because the order is not guaranteed."
            )
    arguments_bound = func_signature.bind(*args, **kwargs)
    arguments_bound.apply_defaults()

    # Check whether there exist an used **kwargs in the function signature
    for arg_name in arguments_bound.signature.parameters:
        if (
            arguments_bound.signature.parameters[arg_name]
            == inspect._ParameterKind.VAR_KEYWORD
        ):
            var_keyword = arg_name
    else:
        var_keyword = None
    arg_names = [
        key for key in arguments_bound.arguments.keys() if not key == var_keyword
    ]
    arg_names_from_kwargs = [key for key in arguments_bound.kwargs.keys()]

    if args_for_graph == "all":
        args_for_graph_from_args = arg_names
        args_for_graph_from_kwargs = arg_names_from_kwargs
    else:
        args_for_graph_from_args = []
        args_for_graph_from_kwargs = []
        for arg in args_for_graph:
            if arg in arg_names:
                args_for_graph_from_args.append(arg)
            elif arg in arg_names_from_kwargs:
                args_for_graph_from_kwargs.append(arg)
            else:
                raise ValueError(f"Argument {arg} not found in the function signature.")

    inputs_for_graph_list = [
        arguments_bound.arguments[arg] for arg in args_for_graph_from_args
    ]
    inputs_for_graph_list += [
        arguments_bound.kwargs[arg] for arg in args_for_graph_from_kwargs
    ]
    inputnames_for_graph_list = args_for_graph_from_args + args_for_graph_from_kwargs

    other_args_dic = {
        arg: arguments_bound.arguments[arg]
        for arg in arg_names
        if arg not in inputnames_for_graph_list
    }
    other_args_dic.update(
        **{
            arg: arguments_bound.kwargs[arg]
            for arg in arg_names_from_kwargs
            if arg not in inputnames_for_graph_list
        }
    )

    return (
        inputs_for_graph_list,
        inputnames_for_graph_list,
        other_args_dic,
    )


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
            # print("make_node inputs: ", inputs)

            self.num_inputs = len(inputs)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
            self.num_outputs = len(outputs)

            self.vjp_sol_op = VJPSolOp(
                self.input_treedef,
                self.input_types,
                self.jitted_vjp_sol_op_jax,
            )
            # print("make_node outputs: ", outputs)

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            # print("perform inputs: ", inputs)
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


def jax2pytensor(*args, **kwargs):
    """Return a pytensor from a jax jittable function.

    It requires to define the output types of the returned values as pytensor types. A
    unique name should also be passed in case the name of the jaxfunc is identical to
    some other node. The design of this function is based on
    https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jaxfunc : jax jittable function
        function for which the node is created, can return multiple tensors as a tuple.
        It is required that all return values are able to transformed to
        pytensor.TensorVariable.
    output_shape_def : function
        Function that returns the shape of the output. If None, the shape is expected to
        be the same as the shape of the args_for_graph arguments. If not None, the
        function should return a tuple of shapes, it will receive as input the shapes of
        the args_for_graph as tuples. Shapes are defined as tuples of integers or None.
    args_for_graph : list of str or "all"
        If "all", all arguments except arguments passed via **kwargs are used for the
        graph. Otherwise specify a list of argument names to use for the graph.
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Only used internally in the pytensor graph.

    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """
    return _Jax2Pytensor(*args, **kwargs)
