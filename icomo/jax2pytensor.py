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
import pytensor.scalar as ps
import pytensor.tensor as pt
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

log = logging.getLogger(__name__)

_filter = lambda x: isinstance(x, pt.Variable)


class _Shape(tuple):
    pass


def _jax2pytensor(
    jaxfunc,
    output_shape_def=None,
    args_for_graph="all",
    name=None,
    static_argnames=(),
    input_dtype=None,
):
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
    input_dtype: pytensor type
        dtype of the inputs, if None, it is inferred from the input types, by
        upcasting all inputs. If output_shape_def is not None, the input_dtype also
        defines the outputs dtype.

    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """

    ### Construct the function to return that is compatible with pytensor but has the
    ### same signature as the jax function.
    def new_func(*args, **kwargs):
        vars, static_vars = eqx.partition((args, kwargs), _filter)

        nonlocal input_dtype
        if input_dtype is None:
            # infer dtype by finding the upcast of all input dtypes. As inputs might be
            # not numpy or pytensors, we temporarily convert them to pytensor first
            # to obtain the dtype. We transform all inputs to the same dtype, to
            # avoid issues if a variable with integer type is given as input, which
            # leads to error when differentiating the function.
            inputs_list_tmp = tree_map(lambda x: pt.as_tensor_variable(x), vars)
            inputs_flat_tmp, _ = tree_flatten(inputs_list_tmp)
            input_types_flat_tmp = [inp.type for inp in inputs_flat_tmp]
            input_dtype = ps.upcast(
                *[inp_type.dtype for inp_type in input_types_flat_tmp]
            )
            del inputs_list_tmp, inputs_flat_tmp, input_types_flat_tmp

        # Convert our inputs to symbolic variables
        pt_vars = tree_map(lambda x: pt.as_tensor_variable(x, dtype=input_dtype), vars)
        # inputs_flat, input_treedef = tree_flatten(inputs_list)
        # input_types_flat = [inp.type for inp in inputs_flat]

        ### Infer output shape and type from jax function, it works by passing
        ### pt.TensorTyp variables, as jax only needs type and shape information.

        # Convert static_argnames to static_argnums because make_jaxpr requires it.
        # static_argnums = tuple(
        #    i
        #    for i, (k, param) in enumerate(func_signature.parameters.items())
        #    if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        #    and k in static_argnames
        # )

        # shapes = pytensor.compile.builders.infer_shape(inputs_flat, (), ())
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
            args, kwargs = eqx.combine(vars, static_vars)
            out = jaxfunc(*args, **kwargs)
            nonlocal static_outvars
            outvars, static_outvars = eqx.partition(
                out,
                eqx.is_array,  # lambda x: isinstance(x, jax.Array)
            )
            # jax.debug.print("static outvars: {}", static_outvars)
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
            ft.partial(_jaxfunc_partitioned, static_vars=static_vars), dummy_inputs_jax
        )

        # jaxtypes_outvars_flat = (
        #    (jaxtypes_outvars_flat,)
        #    if isinstance(jaxtypes_outvars_flat, jax.ShapeDtypeStruct)
        #    else jaxtypes_outvars_flat
        # )

        jaxtypes_outvars_flat, outvars_treedef = tree_flatten(jaxtypes_outvars)

        print("jaxtypes_outvars: ", jaxtypes_outvars)

        pttypes_outvars = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in jaxtypes_outvars_flat
        ]
        print("pttypes_outvars: ", pttypes_outvars)

        ### Create the Pytensor Op, the normal one and the vector-jacobian product (vjp)
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
                lambda g, primal: jnp.broadcast_to(g, jnp.shape(primal)), gz, primals
            )
            if len(y0) == 1:
                return vjp_fn(gz)[0]
            else:
                return tuple(vjp_fn(gz))

        jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

        nonlocal name
        if name is None:
            name = jaxfunc.__name__

        # Get classes that creates a Pytensor Op out of our function that accept
        # flattened inputs. They are created each time, to set a custom name for the
        # class.
        SolOp, VJPSolOp = _return_pytensor_ops_classes(name)

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
        print("pt_vars_flat: ", pt_vars_flat)
        output_flat = local_op(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        outvars = tree_unflatten(outvars_treedef, output_flat)
        output = eqx.combine(outvars, static_outvars)
        # output = outvars
        print("output: ", output)
        return output

    return new_func


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
            arg: val for arg, val in zip(inputnames_list, inputs_for_graph)
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


def _get_output_shape_from_user(
    output_shape_def, input_treedef, inputnames_list, input_types_flat
):
    """Get the output shapes from the user defined output_shape_def function.

    Returns the output shapes and treedef from the user defined output_shape_def
    function.

    Parameters
    ----------
    output_shape_def : function
        function that returns the shape of the output.
    input_treedef : treedef
        treedef of the inputs, as returned by jax.tree_util.tree_flatten
    inputnames_list : list of str
        parameter names of the inputs
    input_types_flat : list of types
        types of the inputs

    Returns
    -------
    output_shapes_flat : list of tuples
        list of output shapes

    """
    # Convert shape tuples to dummy shape objects, such that a tree_map on
    # such an object doesn't interpret the tuple as part of the tree
    input_shapes_flat = [_Shape(t.shape) for t in input_types_flat]

    input_shapes_list = tree_unflatten(input_treedef, input_shapes_flat)
    input_types_dic = {
        arg: shapes for arg, shapes in zip(inputnames_list, input_shapes_list)
    }
    output_shape = output_shape_def(**input_types_dic)

    # For flattening the output shapes, we need to redefine what is a leaf, so
    # that the shape tuples don't get also flattened.
    is_leaf = lambda x: isinstance(x, Sequence) and (
        len(x) == 0 or x[0] is None or isinstance(x[0], int)
    )
    output_shapes_flat, output_treedef = tree_flatten(output_shape, is_leaf=is_leaf)

    if len(output_shapes_flat) == 0 or not isinstance(output_shapes_flat[0], Sequence):
        output_shapes_flat = (output_shapes_flat,)

    return output_shapes_flat, output_treedef


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
            print("make_node inputs: ", inputs)

            self.num_inputs = len(inputs)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
            self.num_outputs = len(outputs)

            self.vjp_sol_op = VJPSolOp(
                self.input_treedef,
                self.input_types,
                self.jitted_vjp_sol_op_jax,
            )
            print("make_node outputs: ", outputs)

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            print("perform inputs: ", inputs)
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


class jax2pytensor:
    """Return a pytensor from a jax jittable function."""

    def __init__(self, jaxfunc, *args, **kwargs):
        self.jaxfunc = jaxfunc
        self.args = args
        self.kwargs = kwargs
        self.pytensor_func = _jax2pytensor(jaxfunc, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Return a pytensor from a jax jittable function."""
        return self.pytensor_func(*args, **kwargs)
