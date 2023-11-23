"""Convert a jax function to a pytensor Op."""

import inspect

import jax
import numpy as np
import pytensor.tensor as pt
from jax.tree_util import tree_flatten, tree_unflatten
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from functools import partial


def create_and_register_jax(
    jax_func,
    output_types=(),
    input_dtype="float64",
    name=None,
    static_argnums=(),
):
    """Return a pytensor from a jax jittable function.

    It requires to define the output types of the returned values as pytensor types. A
    unique name should also be passed in case the name of the jax_func is identical to
    some other node. The design of this function is based on
    https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jax_func : jax jittable function
        function for which the node is created, can return multiple tensors as a tuple.
    output_types : list of pt.TensorType
        The shape of the TensorType has to be defined.
    input_dtype : str
        inputs are converted to this dtype
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Should be unique so that jax_juncify won't ovewrite another when registering it
    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """
    jitted_sol_op_jax = partial(jax.jit, static_argnums=static_argnums)(jax_func)
    len_gz = len(output_types)

    def vjp_sol_op_jax(*args):
        y0 = args[:-len_gz]
        gz = args[-len_gz:]
        if len(gz) == 1:
            gz = gz[0]
        _, vjp_fn = jax.vjp(sol_op.perform_jax, *y0)
        if len(y0) == 1:
            return vjp_fn(gz)[0]
        else:
            return vjp_fn(gz)

    jitted_vjp_sol_op_jax = partial(jax.jit, static_argnums=static_argnums)(
        vjp_sol_op_jax
    )

    class SolOp(Op):
        def __init__(self):
            self.vjp_sol_op = None

        def make_node(self, *inputs, **kwinputs):
            # Convert keyword inputs to positional arguments
            func_signature = inspect.signature(jax_func)
            for key in kwinputs.keys():
                if not key in func_signature.parameters:
                    raise RuntimeError(
                        f"Keyword argument <{key}> not found in function signature. "
                        f"**kwargs are not supported in the definition of the function"
                    )
            arguments = func_signature.bind(*inputs, **kwinputs)
            arguments.apply_defaults()
            all_inputs = arguments.args

            # Convert our inputs to symbolic variables
            all_inputs_flat, self.input_tree = tree_flatten(all_inputs)
            all_inputs_flat = [
                pt.as_tensor_variable(inp).astype(input_dtype)
                for inp in all_inputs_flat
            ]
            self.num_inputs = len(all_inputs_flat)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in output_types]

            self.vjp_sol_op = VJPSolOp(self.input_tree)

            return Apply(self, all_inputs_flat, outputs)

        def perform(self, node, inputs, outputs):
            # This function is called by the C backend, thus the numpy conversion.
            inputs = tree_unflatten(self.input_tree, inputs)
            results = jitted_sol_op_jax(*inputs)
            if len(output_types) > 1:
                for i, _ in enumerate(output_types):
                    outputs[i][0] = np.array(results[i], output_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, output_types[0].dtype)

        def perform_jax(self, *inputs):
            inputs = tree_unflatten(self.input_tree, inputs)
            results = jitted_sol_op_jax(*inputs)
            if len(output_types) > 1:
                return tuple(
                    results
                )  # Transform to tuple because jax makes a difference between tuple and list
            else:
                return results

        def grad(self, inputs, output_gradients):
            # If a output is not used, it is disconnected and doesn't have a gradient.
            # Set gradient here to zero for those outputs.
            for i, otype in enumerate(output_types):
                if isinstance(output_gradients[i].type, DisconnectedType):
                    output_gradients[i] = pt.zeros(otype.shape, otype.dtype)

            inputs_unflat = tree_unflatten(self.input_tree, inputs)
            result = self.vjp_sol_op(inputs_unflat, output_gradients)

            if self.num_inputs > 1:
                return result
            else:
                return (result,)  # Pytensor requires a tuple here

    # vector-jacobian product Op
    class VJPSolOp(Op):
        def __init__(self, input_tree_def):
            self.input_tree_def = input_tree_def

        def make_node(self, y0, gz):
            y0_flat, self.input_tree_def = tree_flatten(y0)

            inputs = [
                pt.as_tensor_variable(
                    _y,
                ).astype(input_dtype)
                for _y in y0_flat
            ] + [
                pt.as_tensor_variable(
                    _gz,
                )
                for _gz in gz
            ]

            # inputs_unflat = tuple(list(y0) + list(gz))
            # _, self.full_input_tree_def = tree_flatten(inputs_unflat)

            outputs = [input.type() for input in y0_flat]
            self.num_outputs = len(outputs)
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            # inputs = tree_unflatten(self.full_input_tree_def, inputs)
            results = jitted_vjp_sol_op_jax(*inputs)
            results, _ = tree_flatten(results)
            if self.num_outputs > 1:
                for i, result in enumerate(results):
                    outputs[i][0] = np.array(result, input_dtype)
            else:
                outputs[0][0] = np.array(results, input_dtype)

        def perform_jax(self, *inputs):
            # inputs = tree_unflatten(self.full_input_tree_def, inputs)
            results = jitted_vjp_sol_op_jax(*inputs)
            results, _ = tree_flatten(results)
            if self.num_outputs == 1:
                results = results[0]
            return results

    if name is None:
        name = jax_func.__name__
    SolOp.__name__ = name
    SolOp.__qualname__ = ".".join(SolOp.__qualname__.split(".")[:-1] + [name])

    VJPSolOp.__name__ = "VJP_" + name
    VJPSolOp.__qualname__ = ".".join(
        VJPSolOp.__qualname__.split(".")[:-1] + ["VJP_" + name]
    )

    sol_op = SolOp()

    @jax_funcify.register(SolOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op.perform_jax

    @jax_funcify.register(VJPSolOp)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return sol_op.vjp_sol_op.perform_jax

    return sol_op
