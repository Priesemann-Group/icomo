import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm

import icomo


def delayed_copy_ode(t, y, args):
    f_input_grad = args["f_input_grad"]
    dy = {}
    dy["start_comp"] = f_input_grad(t)
    dy["erlang_comp"] = icomo.delayed_copy_kernel(
        initial_comp=y["start_comp"],
        delayed_comp=y["erlang_comp"],
        tau_delay=1 / args["rate_latent"],
    )

    return dy


def delayed_copy_manual_ode(t, y, args):
    dy = {}
    dy["start_comp"] = args["f_input_grad"](t)
    # rate = args["rate_latent"]*len(y["erlang_comp"])
    dy["erlang_comp"], outflow = icomo.erlang_kernel(
        inflow=y["start_comp"], comp=y["erlang_comp"], rate=args["rate_latent"]
    )
    # dy["end_comp"] = outflow - rate*y["end_comp"]
    return dy


def erlang_flow_ode(t, y, args):
    comp_model = icomo.CompModel(y)
    f_input_grad = args["f_input_grad"]
    comp_model.add_deriv("start_comp", f_input_grad(t), end_comp_is_erlang=True)
    comp_model.erlang_flow(
        start_comp="start_comp", end_comp="end_comp", rate=args["rate_latent"]
    )
    return comp_model.dy


def test_erlang_stuff():
    std = 0.01
    pos = 0
    f_input = (
        lambda t: 1
        / np.sqrt(2 * np.pi * std**2)
        * jnp.exp(-((t - pos) ** 2) / (2 * std**2))
    )
    f_input_grad = jax.grad(f_input)
    args = {"f_input_grad": f_input_grad, "rate_latent": 1 / 3}
    t_out = np.linspace(-1, 10, 1000)
    k_erlang = 3
    for ode_name in ["delayed_copy", "erlang_flow", "delayed_copy_manual"]:
        if ode_name == "delayed_copy":
            ode = delayed_copy_ode
            y0 = {"start_comp": 0, "erlang_comp": jnp.array([0] * k_erlang)}
            solution_delayed_copy = icomo.diffeqsolve(
                ts_out=t_out, y0=y0, args=args, ODE=ode
            )
        elif ode_name == "erlang_flow":
            ode = erlang_flow_ode
            y0 = {"start_comp": jnp.array([0] * k_erlang), "end_comp": 0}
            solution_erlang_flow = icomo.diffeqsolve(
                ts_out=t_out, y0=y0, args=args, ODE=ode
            )
        elif ode_name == "delayed_copy_manual":
            ode = delayed_copy_manual_ode
            y0 = {"start_comp": 0, "erlang_comp": jnp.array([0] * k_erlang)}
            solution_delayed_copy_manual = icomo.diffeqsolve(
                ts_out=t_out, y0=y0, args=args, ODE=ode
            )

    assert jnp.allclose(
        solution_delayed_copy.ys["erlang_comp"][:, -1],
        solution_erlang_flow.ys["end_comp"],
    )
    assert jnp.allclose(
        solution_delayed_copy_manual.ys["erlang_comp"][:, -1]
        * k_erlang
        * args["rate_latent"],
        solution_erlang_flow.ys["end_comp"],
    )
    gamma_dist = lambda t: np.exp(
        pm.logp(
            pm.Gamma.dist(alpha=k_erlang, beta=args["rate_latent"] * k_erlang), value=t
        ).eval()
    )
    assert jnp.allclose(
        solution_erlang_flow.ys["end_comp"],
        gamma_dist(t_out),
        atol=1e-4,
    )
