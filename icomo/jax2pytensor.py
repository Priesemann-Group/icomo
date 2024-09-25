"""Convert a jax function to a pytensor compatible function."""

import inspect
import logging
from collections.abc import Sequence

import jax

# import jax.tree
import jax.numpy as jnp
import numpy as np
import pytensor.scalar as ps
import pytensor.tensor as pt
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

log = logging.getLogger(__name__)


class _Shape(tuple):
    pass


def jax2pytensor(
    jaxfunc,
    output_shape_def=None,
    args_for_graph="all",
    name=None,
    static_argnames=(),
    dtype=None,
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
    dtype: pytensor type
        dtype of the in- and outputs, if None, it is inferred from the input types, by
        upcasting all inputs.

    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """

    ### Construct the function to return that is compatible with pytensor but has the
    ### same signature as the jax function.
    def new_func(*args, **kwargs):
        func_signature = inspect.signature(jaxfunc)

        (
            inputs_list,
            inputnames_list,
            other_args_dic,
        ) = _split_arguments(func_signature, args, kwargs, args_for_graph)

        nonlocal dtype
        if dtype is None:
            # infer dtype by finding the upcast of all input dtypes. As inputs might be
            # not numpy or pytensors, we temporarily convert them to pytensor first
            # to obtain the dtype. We transform all inputs to the same dtype, to
            # avoid issues if a variable with integer type is given as input, which
            # leads to error when differentiating the function.
            inputs_list_tmp = tree_map(lambda x: pt.as_tensor_variable(x), inputs_list)
            inputs_flat_tmp, _ = tree_flatten(inputs_list_tmp)
            input_types_flat_tmp = [inp.type for inp in inputs_flat_tmp]
            dtype = ps.upcast(*[inp_type.dtype for inp_type in input_types_flat_tmp])
            del inputs_list_tmp, inputs_flat_tmp, input_types_flat_tmp

        # Convert our inputs to symbolic variables
        inputs_list = tree_map(
            lambda x: pt.as_tensor_variable(x, dtype=dtype), inputs_list
        )
        inputs_flat, input_treedef = tree_flatten(inputs_list)
        input_types_flat = [inp.type for inp in inputs_flat]

        ### Create internal function that accepts flattened inputs to use for pytensor.
        def conv_input_to_jax(func, flatten_output=True):
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

                    if len(results) > 1:
                        return tuple(
                            results
                        )  # Transform to tuple because jax makes a difference between
                        # tuple and list and not pytensor
                    else:
                        return results[0]

            return new_func

        jitted_sol_op_jax = jax.jit(
            conv_input_to_jax(jaxfunc),
            static_argnames=static_argnames,
        )

        # Convert static_argnames to static_argnums because make_jaxpr requires it.
        static_argnums = tuple(
            i
            for i, (k, param) in enumerate(func_signature.parameters.items())
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and k in static_argnames
        )

        for input_var, inputname in zip(inputs_list, inputnames_list):
            for i, input_elem in enumerate(tree_leaves(input_var)):
                type = input_elem.type
                if None in type.shape:
                    if output_shape_def is None:
                        raise RuntimeError(
                            f"A dimension of input {inputname}, element {i} is "
                            f"undefined: {type.shape} You need to provide the "
                            f"output_shape_def function to define the shape of the "
                            f"output, or set the shape of the tensor to a integer with "
                            f"input_var.type.shape = (10,) for example."
                        )

        if output_shape_def is not None:
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
            output_shapes_flat, output_treedef = tree_flatten(
                output_shape, is_leaf=is_leaf
            )

            if len(output_shapes_flat) == 0 or not isinstance(
                output_shapes_flat[0], Sequence
            ):
                output_shapes_flat = (output_shapes_flat,)

            output_types = [
                pt.type.TensorType(dtype=dtype, shape=shape)
                for shape in output_shapes_flat
            ]

        else:
            ### Infer output shape and type from jax function, it works by passing
            ### pt.TensorTyp variables, as jax only needs type and shape information.
            _, output_shape_jax = jax.make_jaxpr(
                conv_input_to_jax(jaxfunc, flatten_output=False),
                static_argnums=static_argnums,
                return_shape=True,
            )(tree_map(lambda x: x.type, inputs_flat))

            out_shape_jax_flat, output_treedef = tree_flatten(output_shape_jax)

            output_types = [
                pt.TensorType(dtype=var.dtype, shape=var.shape)
                for var in out_shape_jax_flat
            ]

        ### Create the Pytensor Op, the normal one and the vector-jacobian product (vjp)
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

        SolOp, VJPSolOp = _return_pytensor_ops(name)

        local_op = SolOp(
            input_treedef,
            output_treedef,
            input_arg_names=inputnames_list,
            input_types=input_types_flat,
            output_types=output_types,
            jitted_sol_op_jax=jitted_sol_op_jax,
            jitted_vjp_sol_op_jax=jitted_vjp_sol_op_jax,
            other_args=other_args_dic,
        )

        @jax_funcify.register(SolOp)
        def sol_op_jax_funcify(op, **kwargs):
            return local_op.perform_jax

        @jax_funcify.register(VJPSolOp)
        def vjp_sol_op_jax_funcify(op, **kwargs):
            return local_op.vjp_sol_op.perform_jax

        ### Evaluate the Pytensor Op and return unflattened results
        output_flat = local_op(*inputs_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        output = tree_unflatten(output_treedef, output_flat)
        len_gz = len(output_types)

        return output

    return new_func


def _split_arguments(func_signature, args, kwargs, args_for_graph):
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


def _return_pytensor_ops(name):
    class SolOp(Op):
        def __init__(
            self,
            input_treedef,
            output_treeedef,
            input_arg_names,
            input_types,
            output_types,
            jitted_sol_op_jax,
            jitted_vjp_sol_op_jax,
            other_args,
        ):
            self.vjp_sol_op = None
            self.input_treedef = input_treedef
            self.output_treedef = output_treeedef
            self.input_arg_names = input_arg_names
            self.input_types = input_types
            self.output_types = output_types
            self.jitted_sol_op_jax = jitted_sol_op_jax
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax
            self.other_args = other_args

        def make_node(self, *inputs):
            self.num_inputs = len(inputs)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
            self.num_outputs = len(outputs)

            self.vjp_sol_op = VJPSolOp(
                self.input_treedef,
                self.input_types,
                self.jitted_vjp_sol_op_jax,
                self.other_args,
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
            # raise NotImplementedError()
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
            self, input_treedef, input_types, jitted_vjp_sol_op_jax, other_args
        ):
            self.input_treedef = input_treedef
            self.input_types = input_types
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax
            self.other_args = other_args

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
            # self.num_gz_not_disconnected = len(gz_not_disconntected)

            outputs = [in_type() for in_type in self.input_types]
            self.num_outputs = len(outputs)
            return Apply(self, y0 + gz_not_disconntected, outputs)

        def perform(self, node, inputs, outputs):
            # inputs = tree_unflatten(self.full_input_treedef_def, inputs)
            # y0 = inputs[:-self.num_gz]
            # gz = inputs[-self.num_gz:]
            results = self.jitted_vjp_sol_op_jax(tuple(inputs))
            if len(self.input_types) > 1:
                for i, result in enumerate(results):
                    outputs[i][0] = np.array(result, self.input_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.input_types[0].dtype)

        def perform_jax(self, *inputs):
            # inputs = tree_unflatten(self.full_input_treedef_def, inputs)

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
