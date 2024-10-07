"""
Tests for jax2pytensor package.
"""

import jax
import jax.numpy as jnp
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from icomo import jax2pytensor


@pytest.fixture
def test_models():
    # 2 parameters input, tuple output
    with pm.Model() as model1:
        x, y = pm.Normal("input", size=2)

        @jax2pytensor
        def f(x, y):
            return jax.nn.sigmoid(x + y), y * 2

        out, _ = f(x, y)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input, single output
    with pm.Model() as model2:
        x, y = pm.Normal("input", size=2)

        def f2(x, y):
            return jax.nn.sigmoid(x + y)

        f2_op = jax2pytensor(f2)
        out = f2_op(x, y)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input, list output
    with pm.Model() as model3:
        x, y = pm.Normal("input", size=2)

        def f(x, y):
            return [jax.nn.sigmoid(x + y), y * 2]

        f_op = jax2pytensor(f)
        out, _ = f_op(x, y)
        pm.Normal("obs", out, observed=3)

    # single 1d input, tuple output
    with pm.Model() as model4:
        x, y = pm.Normal("input", size=2)

        def f4(x):
            return jax.nn.sigmoid(x), x * 2

        f4_op = jax2pytensor(f4)
        out, _ = f4_op(x)
        pm.Normal("obs", out, observed=3)

    # single 0d input, tuple output
    with pm.Model() as model5:
        x = pm.Normal("input", size=())

        def f5(x):
            return jax.nn.sigmoid(x), x

        f5_op = jax2pytensor(f5)
        out, _ = f5_op(x)
        pm.Normal("obs", out, observed=3)

    # single input, list output
    with pm.Model() as model6:
        x, y = pm.Normal("input", size=2)

        def f(x):
            return [jax.nn.sigmoid(x), 2 * x]

        f_op = jax2pytensor(f)
        out, _ = f_op(x)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input with pytree, tuple output
    with pm.Model() as model7:
        x, y = pm.Normal("input", size=2)
        y_tmp = {"y": y, "y2": [y**2]}

        def f(x, y):
            return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

        f_op = jax2pytensor(f)
        out, _ = f_op(x, y_tmp)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input with pytree, pytree output
    with pm.Model() as model8:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=(1,))
        y_tmp = {"a": y, "b": [y**2]}

        def f(x, y):
            jax.debug.print("x: {}", x)
            jax.debug.print("y: {}", y)
            return x, jax.tree_util.tree_map(
                lambda x: jnp.exp(x), y
            )  # {"a": y["y4"], "b": y["y3"]}  # , jax.tree_map(jnp.exp, y)

        f_op = jax2pytensor(f)
        out_x, out_y = f_op(x, y_tmp)
        # for_model = out_y["y3"]
        pm.Normal("obs", out_x, observed=(3, 2, 3))
        # pm.Normal("obs2", out_y["y3"], observed=(3,))

    # 2 parameters input with pytree, pytree output and non-graph argument
    with pm.Model() as model9:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=1)
        y_tmp = {"a": y, "b": [y**2]}

        def f(x, y, non_model_arg):
            print(non_model_arg)
            return x, jax.tree_util.tree_map(jax.nn.sigmoid, y)

        f_op = jax2pytensor(
            f,
            args_for_graph=["x", "y"],
        )
        out_x, out_y = f_op(x, y_tmp, "Hello World!")

        pm.Normal("obs", out_y["b"][0], observed=(3,))

    # Use "None" in shape specification and have a non-used output of higher rank
    with pm.Model() as model10:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=3)

        def f(x, y):
            return x[:, None] @ y[None], x

        f_op = jax2pytensor(f, args_for_graph=["x", "y"])
        out_x, out_y = f_op(x, y)

        pm.Normal("obs", out_y[0], observed=(3,))

    with pm.Model() as model11:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=3)

        # Now x has an unknown shape
        x, _ = pytensor.map(pt.exp, [x])
        x.type.shape = (3,)

        def f11(x, y):
            return x[:, None] @ y[None]

        f_op = jax2pytensor(f11)
        out_x = f_op(x, y)
        # out_x = x[:, None] @ y[None]

        pm.Normal("obs", out_x[0, 1], observed=2)

    # Test broadcasting with unknown shape
    with pm.Model() as model12:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=3)

        shape_before = x.type.shape

        # Now x has an unknown shape
        x = pt.cumsum(x)

        x.type.shape = shape_before

        def f12(x, y):
            return x * jnp.ones(3)

        f_op = jax2pytensor(f12)
        out_x = f_op(x, y)
        # out_x = x[:, None] @ y[None]

        pm.Normal("obs", out_x[0], observed=2)

    with pm.Model() as model13:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=3)

        shape_before = x.type.shape

        # Now x has an unknown shape
        x = pt.cumsum(x)

        def f12(x, y):
            return x * jnp.ones(3)

        f_op = jax2pytensor(f12, output_shape_def=lambda **_: (3,))
        out_x = f_op(x, y)
        # out_x = x[:, None] @ y[None]

        pm.Normal("obs", out_x[0], observed=2)

    return (
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        model7,
        model8,
        model9,
        model10,
        model11,
        model12,
        model13,
    )


def test_jax_compilation(test_models):
    for i, model in enumerate(test_models):
        print(f"Test model {i + 1}")

        ip = model.initial_point()
        # Setting the mode to fast_compile shouldn't make a difference in the test
        # coverage
        logp_fn = model.compile_fn(model.logp(sum=False), mode="FAST_COMPILE")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="FAST_COMPILE")
        dlogp_fn(ip)

        ip = model.initial_point()
        logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
        dlogp_fn(ip)


def test_jax_without_jit(test_models):
    with jax.disable_jit():
        for i, model in enumerate(test_models):
            print(f"Test model {i + 1}")

            ip = model.initial_point()
            logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
            logp_fn(ip)
            dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
            dlogp_fn(ip)
