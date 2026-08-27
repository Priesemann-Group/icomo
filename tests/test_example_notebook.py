"""Executable, reduced-data copy of the documentation example.

This intentionally keeps the numerical workflow close to docs/example.ipynb
whilst omitting its plotting and explanatory cells. The pytest test re-executes
this file so that JAX sees the XLA configuration before import.
"""

import os
import subprocess
import sys
from pathlib import Path

# The second quarter of the 2022 England case data used in the notebook. This
# retains real, contiguous observations while reducing the ODE horizon by about
# a factor of four.
DATA = (
    54791,
    47462,
    37592,
    42179,
    53727,
    49101,
    44750,
    38709,
    32988,
    27885,
    30952,
    37371,
    33771,
    31685,
    26638,
    21615,
    19924,
    20144,
    25223,
    25772,
    21457,
    18464,
    15703,
    13210,
    14715,
    17108,
    14560,
    13055,
    11608,
    10012,
    8354,
    9034,
    11907,
    13318,
    11672,
    10221,
    8954,
    7517,
    9062,
    10702,
    9748,
    8725,
    7642,
    6942,
    6120,
    7973,
    9417,
    8286,
    7638,
    6820,
    5821,
    5019,
    5997,
    7313,
    6679,
    6293,
    5753,
    5409,
    4988,
    5680,
    7115,
    7093,
    7011,
    6209,
    6635,
    7492,
    9815,
    11747,
    11411,
    11784,
    11319,
    10784,
    10077,
    12693,
    15272,
    15141,
    14837,
    14225,
    13919,
    13623,
    16997,
    19942,
    20896,
    20873,
    19590,
    18349,
    16800,
    20778,
    25104,
    25041,
    24601,
)


def erlang_seir(t, y, args):
    """Define the Erlang-SEIR dynamics used throughout the notebook."""
    beta_t_func = args["beta_t_func"]
    infection_rate = beta_t_func(t) * y["I"] * y["S"] / args["N"]
    d_es, outflow = icomo.erlang_kernel(
        inflow=infection_rate,
        comp=y["Es"],
        rate=args["rate_latent"],
    )
    return {
        "S": -infection_rate,
        "Es": d_es,
        "I": outflow - args["rate_infectious"] * y["I"],
        "R": args["rate_infectious"] * y["I"],
    }


comp_model = diffrax = icomo = jax = jnp = jaxopt = np = optax = pm = pt = None


def erlang_seir_comp_model(t, y, args):
    """Define the same dynamics with icomo.CompModel."""
    beta_t_func = args["beta_t_func"]
    comp_model.y = y
    comp_model.flow(
        start_comp="S",
        end_comp="Es",
        rate=y["I"] / args["N"] * beta_t_func(t),
        label="beta(t) * I/N",
        end_comp_is_erlang=True,
    )
    comp_model.erlang_flow(
        "Es",
        "I",
        args["rate_latent"],
        label="rate_latent (erlang)",
    )
    comp_model.flow(
        "I",
        "R",
        args["rate_infectious"],
        label="rate_infectious",
    )
    return comp_model.dy


def all_finite(tree):
    """Return whether every array leaf in a PyTree is finite."""
    return all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(tree)
    )


def run_example():
    """Run the numerical parts of the example and assert their key outcomes."""
    global comp_model, diffrax, icomo, jax, jnp
    global jaxopt, np, optax, pm, pt

    os.environ["XLA_FLAGS"] = (
        "--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=65536"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import diffrax
    import jax
    import jax.numpy as jnp
    import jaxopt
    import numpy as np
    import optax
    import pymc as pm
    import pytensor.tensor as pt

    import icomo

    comp_model = icomo.CompModel()
    data = np.asarray(DATA)
    len_sim = len(data)
    t_out = np.linspace(0, len_sim, len_sim)
    t_beta = np.linspace(0, len_sim, len_sim // 14)

    population = 1e5
    reproduction_number = 1.5
    duration_latent = 3
    duration_infectious = 4
    beta0 = reproduction_number / duration_infectious

    args = {
        "N": population,
        "rate_latent": 1 / duration_latent,
        "rate_infectious": 1 / duration_infectious,
    }
    y0 = {
        "Es": np.array([100, 100, 100]),
        "I": 300,
        "R": 0,
    }
    y0["S"] = population - y0["R"] - y0["I"] - np.sum(y0["Es"])
    args["beta_t_func"] = icomo.interpolate_func(
        ts_in=t_beta,
        values=beta0 * np.ones(len(t_beta)),
    )

    solution = icomo.diffeqsolve(
        ts_out=t_out,
        y0=y0,
        args=args,
        ODE=erlang_seir,
    )
    comp_model_solution = icomo.diffeqsolve(
        ts_out=t_out,
        y0=y0,
        args=args,
        ODE=erlang_seir_comp_model,
    )
    assert solution.ys["I"].shape == (len_sim,)
    np.testing.assert_allclose(
        solution.ys["I"],
        comp_model_solution.ys["I"],
        rtol=1e-7,
        atol=1e-7,
    )

    population_england = 50e6

    def simulation(args_optimization):
        beta_t = args_optimization["beta_t"]
        infected0 = args_optimization["I0"] / 2
        exposed0 = args_optimization["I0"] / 6 * jnp.ones(3)
        simulation_args = {
            "N": population_england,
            "rate_latent": 1 / duration_latent,
            "rate_infectious": 1 / duration_infectious,
            "beta_t_func": icomo.interpolate_func(ts_in=t_beta, values=beta_t),
        }
        simulation_y0 = {
            "Es": exposed0,
            "I": infected0,
            "R": 0,
        }
        simulation_y0["S"] = (
            population_england
            - jnp.sum(simulation_y0["Es"])
            - simulation_y0["I"]
            - simulation_y0["R"]
        )

        output = icomo.diffeqsolve(
            ts_out=t_out,
            y0=simulation_y0,
            args=simulation_args,
            ODE=erlang_seir_comp_model,
            adjoint=diffrax.DirectAdjoint(),
        ).ys
        output["beta_t_interpolated"] = simulation_args["beta_t_func"](t_out)
        return output

    @jax.jit
    def loss(args_optimization):
        new_infected = -jnp.diff(simulation(args_optimization)["S"])
        return jnp.mean((new_infected - data[1:]) ** 2 / (new_infected + 1))

    init_params = {
        "beta_t": beta0 * np.ones_like(t_beta),
        "I0": np.array(float(data[0] * duration_infectious)),
    }
    value_and_grad_loss = jax.jit(jax.value_and_grad(loss))
    value_and_grad_loss(init_params)
    solver = jaxopt.ScipyMinimize(
        fun=value_and_grad_loss,
        value_and_grad=True,
        method="L-BFGS-B",
        jit=False,
    )
    result = solver.run(init_params)
    scipy_loss = float(np.asarray(result.state.fun_val))
    assert bool(np.asarray(result.state.success)), result.state
    assert np.isfinite(scipy_loss)
    assert scipy_loss < 1_500, scipy_loss

    schedule = optax.exponential_decay(
        init_value=5e-2,
        transition_steps=1000,
        decay_rate=1 / 2,
        transition_begin=50,
        staircase=False,
        end_value=None,
    )
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(init_params)
    params_adam = init_params
    adam_loss = np.inf
    for _ in range(2000):
        adam_loss, grads = value_and_grad_loss(params_adam)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_adam = optax.apply_updates(params_adam, updates)
    adam_loss = float(np.asarray(adam_loss))
    assert np.isfinite(adam_loss)
    assert adam_loss < 1_500, adam_loss

    t_out_bayes = np.arange(len_sim)
    t_beta_bayes = np.linspace(
        t_out_bayes[0],
        t_out_bayes[-1],
        len(t_out_bayes) // 14,
    )
    with pm.Model(coords={"time": t_out_bayes, "t_beta": t_beta_bayes}) as model:
        duration_latent_var = pm.LogNormal(
            "duration_latent",
            mu=np.log(duration_latent),
            sigma=0.1,
        )
        duration_infectious_var = pm.LogNormal(
            "duration_infectious",
            mu=np.log(duration_infectious),
            sigma=0.3,
        )
        r0_var = pm.LogNormal("R0", np.log(1), 1)
        beta0_var = r0_var / duration_infectious_var
        beta_t_var = beta0_var * pt.exp(
            pt.cumsum(
                icomo.experimental.hierarchical_priors(
                    "beta_t_log_diff",
                    dims=("t_beta",),
                )
            )
        )
        model_args = {
            "N": population_england,
            "rate_latent": 1 / duration_latent_var,
            "rate_infectious": 1 / duration_infectious_var,
        }
        infections0_var = pm.LogNormal(
            "infections_0",
            mu=np.log(data[0] * duration_infectious),
            sigma=2,
        )
        model_y0 = {
            "Es": infections0_var / 3 * np.ones(3),
            "I": infections0_var / 2,
            "R": 0,
        }
        model_y0["S"] = (
            population_england - pt.sum(model_y0["Es"]) - model_y0["I"] - model_y0["R"]
        )
        beta_t_func = icomo.jax2pytensor(icomo.interpolate_func)(
            ts_in=t_beta_bayes,
            values=beta_t_var,
        )
        model_args["beta_t_func"] = beta_t_func

        # Deliberately use Diffrax's default RecursiveCheckpointAdjoint here. It
        # works with vectorized NumPyro chains, unlike DirectAdjoint under pmap.
        output = icomo.jax2pytensor(icomo.diffeqsolve)(
            ts_out=t_out_bayes,
            y0=model_y0,
            args=model_args,
            ODE=erlang_seir,
        ).ys
        pm.Deterministic("I", output["I"])
        new_cases = -pt.diff(output["S"])
        pm.Deterministic("new_cases", new_cases)
        sigma_error = pm.HalfCauchy("sigma_error", beta=1)
        pm.StudentT(
            "cases_observed",
            nu=4,
            mu=new_cases,
            sigma=sigma_error * pt.sqrt(new_cases + 1),
            observed=data[1:],
        )
        beta_t_interp = beta_t_func(t_out_bayes)
        pm.Deterministic("beta_t_interp", beta_t_interp)

    initial_point = model.initial_point()
    logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
    dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
    assert all_finite(logp_fn(initial_point))
    assert all_finite(dlogp_fn(initial_point))

    trace = pm.sample(
        model=model,
        tune=2,
        draws=2,
        chains=2,
        nuts_sampler="numpyro",
        nuts={"chain_method": "vectorized"},
        target_accept=0.6,
        progressbar=False,
        compute_convergence_checks=False,
        random_seed=123,
    )
    assert trace.posterior.sizes["chain"] == 2
    assert trace.posterior.sizes["draw"] == 2


def test_example_notebook():
    """Run the copied example in a clean process with its JAX configuration."""
    test_file = Path(__file__)

    # XLA_FLAGS is read when JAX initializes. Other test modules may import JAX
    # during pytest collection, so setting it in this process would be too late.
    # Re-executing this file guarantees that run_example configures XLA first.
    subprocess.run(
        [sys.executable, str(test_file)],
        check=True,
        cwd=test_file.parents[1],
        timeout=1_800,
    )


if __name__ == "__main__":
    run_example()
