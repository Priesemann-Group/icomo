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
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Only used internally in the pytensor graph.

    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """

    def func(*args, **kwargs):
        """Return a pytensor from a jax jittable function."""
        ### Construct the function to return that is compatible with pytensor but has
        ### the same signature as the jax function.
        pt_vars, static_vars = eqx.partition((args, kwargs), _filter_ptvars)

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

        ### Combine the static variables with the inputs, and split them again in the
        ### output. Static variables don't take part in the graph, or might be a
        ### a function that is returned.
        static_outvars = None

        def _jaxfunc_partitioned(vars, static_vars):
            args, kwargs = eqx.combine(vars, static_vars)

            out = jaxfunc(*args, **kwargs)
            nonlocal static_outvars
            outvars, static_outvars = eqx.partition(
                out,
                eqx.is_array,  # lambda x: isinstance(x, jax.Array)
            )
            # print("static outvars: {}", static_outvars)
            # jax.debug.print("outvars: {}", outvars)
            return outvars

        ### Construct the function that accepts flat inputs and returns flat outputs.
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

        jaxtypes_outvars_flat, outvars_treedef = tree_flatten(jaxtypes_outvars)

        pttypes_outvars = [
            pt.TensorType(dtype=var.dtype, shape=var.shape)
            for var in jaxtypes_outvars_flat
        ]

        ### Call the function that accepts flat inputs, which in turn calls the one that
        ### combines the inputs and static variables.
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
        # print("pt_vars_flat: ", pt_vars_flat)
        output_flat = local_op(*pt_vars_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        outvars = tree_unflatten(outvars_treedef, output_flat)
        output = eqx.combine(outvars, static_outvars)

        return output

    return func


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
