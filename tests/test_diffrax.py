import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest

import icomo
from icomo import interpolate_func, jax2pytensor


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
        t1 = 14
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
        t1 = 14
        dt0 = 0.1
        # args = (0.1, 0.02, 0.4, 0.02)
        # saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
        diffeqsolve = jax2pytensor(diffrax.diffeqsolve)

        sol = diffeqsolve(term, solver, t0, t1, dt0, y0=y0, args=args, saveat=saveat)
        pm.Normal("obs", sol.ys, observed=np.array([[3, 3]]))

    def Erlang_SEIR_v2(t, y, args):
        beta_t_func = args["beta_t_func"]

        comp_model = icomo.CompModel(y)

        comp_model.flow(
            start_comp="S",
            end_comp="Es",
            rate=y["I"] / args["N"] * beta_t_func(t),
            label="beta(t) * I/N",  # label of the graph edge
            end_comp_is_erlang=True,
        )  # One has to specify that "Es" refers to
        # a list of compartments
        comp_model.erlang_flow(
            "Es", "I", args["rate_latent"], label="rate_latent (erlang)"
        )
        comp_model.flow("I", "R", args["rate_infectious"], label="rate_infectious")
        return comp_model.dy

    with pm.Model() as model3:
        len_sim = 30
        num_points = len_sim // 10

        t_out = np.linspace(0, len_sim, num_points)

        N = 1e5
        R0 = 1.5
        duration_latent = 3
        duration_infectious = 7
        beta0 = R0 / duration_infectious

        beta_t = pm.Normal("beta_t", mu=beta0, sigma=0.1, shape=len(t_out))
        beta_t_func = icomo.jax2pytensor(interpolate_func)(ts_in=t_out, values=beta_t)
        args = {
            "beta_t_func": beta_t_func,
            "N": N,
            "rate_latent": 1 / duration_latent,
            "rate_infectious": 1 / duration_infectious,
        }

        y0 = {
            "Es": np.array([100, 100, 100]),
            "I": pm.Normal("I0", mu=10, sigma=1),
            "R": 0,
        }
        # y0["S"] = N - jax.tree_util.tree_reduce(lambda x, y: x + y, y0)
        y0["S"] = N - y0["R"] - np.sum(y0["Es"])

        solution = icomo.jax2pytensor(icomo.diffeqsolve)(
            diffrax.ODETerm(Erlang_SEIR_v2),
            saveat=diffrax.SaveAt(ts=t_out),
            t0=min(t_out),
            t1=max(t_out),
            dt0=np.diff(t_out)[0],
            y0=y0,
            args=args,
            solver=diffrax.Tsit5(),
        )
        pm.Normal("obs", solution.ys["I"], observed=3 * np.ones(num_points))
        beta_t_interp = icomo.jax2pytensor(lambda f, x: f(x))(beta_t_func, t_out)
        pm.Deterministic("beta_t_interp", beta_t_interp)

    return (
        model1,
        model2,
        model3,
    )


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


def test_jax_without_jit(ode_models):
    with jax.disable_jit():
        for i, model in enumerate(ode_models):
            print(f"Test model {i + 1}")

            ip = model.initial_point()
            logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
            logp_fn(ip)
            dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
            dlogp_fn(ip)
