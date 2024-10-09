import diffrax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

from icomo import jax2pytensor


@pytest.fixture
def ode_models():
    with pm.Model() as model1:
        prey_0, predator_0 = pm.Normal("y0", size=2)
        args = pm.Normal("args", size=4)

        def vector_field(t, y, args):
            prey, predator = y
            α, β, γ, δ = args
            d_prey = α * prey - β * prey * predator
            d_predator = -γ * predator + δ * prey * predator
            d_y = d_prey, d_predator
            return d_y

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        t0 = 0
        t1 = 140
        dt0 = 0.1
        y0 = (prey_0, predator_0)
        # args = (0.1, 0.02, 0.4, 0.02)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))

        diffeqsolve = jax2pytensor(diffrax.diffeqsolve)

        sol = diffeqsolve(term, solver, t0, t1, dt0, y0=y0, args=args, saveat=saveat)
        pm.Normal("obs", sol.ys, observed=3 * np.ones((2, 100)))

    with pm.Model() as model2:
        y0 = pm.Normal("y0", size=2)
        args = pm.Normal("args", size=4)

        def vector_field(t, y, args):
            prey, predator = y
            α, β, γ, δ = args
            d_prey = α * prey - β * prey * predator
            d_predator = -γ * predator + δ * prey * predator
            d_y = jnp.array([d_prey, d_predator])
            return d_y

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        t0 = 0
        t1 = 140
        dt0 = 0.1
        # args = (0.1, 0.02, 0.4, 0.02)
        # saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
        diffeqsolve = jax2pytensor(diffrax.diffeqsolve)

        sol = diffeqsolve(term, solver, t0, t1, dt0, y0=y0, args=args, saveat=saveat)
        pm.Normal("obs", sol.ys, observed=np.array([[3, 3]]))

    return (model1, model2)


def test_jax_compilation(ode_models):
    for i, model in enumerate(ode_models):
        print(f"ODE model {i + 1}")

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
