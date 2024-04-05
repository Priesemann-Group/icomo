"""
Tests for icomo package.
"""
import pytest

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import pymc as pm
import pytensor.tensor as pt

import icomo


def Erlang_SEIR(t, y, args):
    beta_t, const_arg = args

    N = const_arg["N"]
    dy = {}
    dy["S"] = -beta_t(t) * y["I"] * y["S"] / N

    dEs, outflow = icomo.erlang_kernel(
        inflow=beta_t(t) * y["I"] * y["S"] / N,
        Vars=y["Es"],
        rate=const_arg["rate_latent"],
    )
    dy["Es"] = dEs

    dy["I"] = outflow - const_arg["rate_infectious"] * y["I"]

    dy["R"] = const_arg["rate_infectious"] * y["I"]

    return dy


@pytest.fixture
def initialization_integrator_simple():
    len_sim = 365
    num_points = len_sim

    t_out = np.linspace(0, len_sim, num_points)
    t_solve_ODE = np.linspace(0, len_sim, num_points // 2)
    t_beta = np.linspace(0, len_sim, num_points // 14)

    N = 1e5
    R0 = 1.5
    duration_latent = 3
    duration_infectious = 7
    beta0 = R0 / duration_infectious

    arg_t = beta0 * np.ones(len(t_beta))
    const_args = {
        "N": N,
        "rate_latent": 1 / duration_latent,
        "rate_infectious": 1 / duration_infectious,
    }

    y0 = {
        "Es": [100, 100, 100],
        "I": 300,
        "R": 0,
    }
    y0["S"] = N - jax.tree_util.tree_reduce(lambda x, y: x + y, y0)
    y0["S"] = N - y0["R"] - np.sum(y0["Es"])

    integrator_object = icomo.ODEIntegrator(
        ts_out=t_out,
        t_0=min(t_solve_ODE),
        ts_solver=t_solve_ODE,
        ts_arg=t_beta,
    )
    return integrator_object, y0, arg_t, const_args, t_beta, t_out


def test_integration(initialization_integrator_simple):
    (
        integrator_object,
        y0,
        arg_t,
        const_args,
        t_beta,
        t_out,
    ) = initialization_integrator_simple

    SEIR_integrator = integrator_object.get_func(Erlang_SEIR)

    output = SEIR_integrator(y0=y0, arg_t=arg_t, constant_args=const_args)
    assert np.isclose(output["I"][-1], 1.05142425e-01)


def Erlang_SEIR_v2(t, y, args):
    beta_t, const_arg = args

    comp_model = icomo.CompModel(y)

    comp_model.flow(
        start_comp="S",
        end_comp="Es",
        rate=y["I"] / const_arg["N"] * beta_t(t),
        label="beta(t) * I/N",  # label of the graph edge
        end_comp_is_list=True,
    )  # One has to specify that "Es" refers to
    # a list of compartments
    comp_model.erlang_flow(
        "Es", "I", const_arg["rate_latent"], label="rate_latent (erlang)"
    )
    comp_model.flow("I", "R", const_arg["rate_infectious"], label="rate_infectious")
    comp_model.view_graph()
    return comp_model.dy


def test_alternative_builder(initialization_integrator_simple):
    (
        integrator_object,
        y0,
        arg_t,
        const_args,
        t_beta,
        t_out,
    ) = initialization_integrator_simple

    SEIR_integrator_v2 = integrator_object.get_func(Erlang_SEIR_v2)
    output2 = SEIR_integrator_v2(y0=y0, arg_t=arg_t, constant_args=const_args)
    assert np.isclose(output2["I"][-1], 1.05142425e-01)


# fmt: off
data = [236576, 188567, 118275, 181068, 229901, 275647, 222174, 172980, 120287,
        93615, 95790, 132959, 115245, 103219, 96412, 86325, 77950, 98097, 131514,
        118857, 111760, 100324, 91730, 81733, 102365, 127364, 113201, 108668, 96200,
        84026, 73512, 87702, 105051, 93949, 88418, 76644, 63264, 52305, 60552,
        76483, 69661, 64339, 52293, 44559, 37529, 41755, 54745, 52502, 51572, 44716,
        33241, 34647, 37720, 45131, 39188, 36476, 30631, 27837, 25089, 31360, 44649,
        44795, 46179, 43761, 41408, 38759, 49026, 68520, 70111, 73531, 70510, 67030,
        62078, 75738, 99832, 93708, 91752, 82853, 75285, 67417, 81054, 109286,
        99095, 94185, 82905, 72430, 61456, 68336, 92857, 80126, 73643, 54791, 47462,
        37592, 42179, 53727, 49101, 44750, 38709, 32988, 27885, 30952, 37371, 33771,
        31685, 26638, 21615, 19924, 20144, 25223, 25772, 21457, 18464, 15703, 13210,
        14715, 17108, 14560, 13055, 11608, 10012, 8354, 9034, 11907, 13318, 11672,
        10221, 8954, 7517, 9062, 10702, 9748, 8725, 7642, 6942, 6120, 7973, 9417,
        8286, 7638, 6820, 5821, 5019, 5997, 7313, 6679, 6293, 5753, 5409, 4988,
        5680, 7115, 7093, 7011, 6209, 6635, 7492, 9815, 11747, 11411, 11784, 11319,
        10784, 10077, 12693, 15272, 15141, 14837, 14225, 13919, 13623, 16997, 19942,
        20896, 20873, 19590, 18349, 16800, 20778, 25104, 25041, 24601, 24240, 23650,
        21961, 27780, 33704, 31415, 29698, 26454, 24611, 21246, 25096, 29826, 26946,
        23856, 21100, 18359, 15017, 17236, 18718, 16670, 16169, 14158, 12378, 9934,
        11275, 13299, 11725, 10544, 9329, 8508, 7325, 8426, 9835, 9170, 8353, 7296,
        6548, 5420, 6547, 8068, 7309, 6748, 6193, 5339, 4495, 5363, 6424, 5559,
        5032, 4564, 4012, 3502, 4085, 5224, 4778, 4554, 3796, 3451, 3112, 3439,
        4662, 5226, 4615, 4350, 3808, 3417, 4288, 5125, 4578, 4235, 3747, 3553,
        3268, 4347, 5307, 5147, 4973, 4634, 4134, 3521, 4339, 6166, 7828, 7553,
        6815, 6485, 5719, 7240, 8617, 8251, 8414, 7895, 7615, 7065, 9204, 11692,
        10965, 10231, 9389, 8950, 7053, 8558, 10497, 9461, 8768, 8134, 7970, 6442,
        7780, 10002, 8233, 7598, 6576, 6297, 4968, 5523, 6807, 5665, 5449, 4855,
        4401, 3583, 4271, 5130, 4549, 4014, 3678, 3107, 2665, 3267, 4321, 3829,
        3576, 3205, 2893, 2433, 2912, 3923, 3534, 3260, 3037, 2773, 2430, 2890,
        3937, 3805, 3616, 3329, 2902, 2551, 3448, 4639, 4386, 3980, 3505, 3596,
        2748, 3786, 5325, 5241, 5195, 4537, 4216, 3570, 4631, 6986, 7111, 6533,
        6094, 6084, 4743, 5772, 9073, 8631, 7802, 6877, 5741, 4132, 3694, 4950,
        6605, 8419, 7718]
# fmt: on
data = np.array(data)
N_England = 50e6


def test_fitting(initialization_integrator_simple):
    (
        integrator_object,
        y0,
        arg_t,
        const_args,
        t_beta,
        t_out,
    ) = initialization_integrator_simple

    SEIR_integrator = integrator_object.get_func(Erlang_SEIR)

    def simulation(args_optimization):
        beta_t = args_optimization["beta_t"]

        I0 = args_optimization["I0"] / 2
        Es_0 = [args_optimization["I0"] / 6 for _ in range(3)]

        const_args["N"] = N_England

        y0 = {
            "Es": Es_0,
            "I": I0,
            "R": 0,
        }
        y0["S"] = N_England - jax.tree_util.tree_reduce(lambda x, y: x + y, y0)

        output = SEIR_integrator(y0=y0, arg_t=beta_t, constant_args=const_args)

        beta_t_interpolated = icomo.interpolation_func(
            t_beta, beta_t, "cubic"
        ).evaluate(t_out)
        output["beta_t_interpolated"] = beta_t_interpolated

        return output

    @jax.jit
    def loss(args_optimization):
        output = simulation(args_optimization)
        new_infected = -jnp.diff(output["S"])

        loss = jnp.mean((new_infected - data[1:]) ** 2 / (new_infected + 1))

        return loss

    init_params = {
        "beta_t": arg_t,
        "I0": np.array(float(data[0] * 7)),
    }

    value_and_grad_loss = jax.jit(jax.value_and_grad(loss))
    value_and_grad_loss(init_params)

    solver = jaxopt.ScipyMinimize(
        fun=value_and_grad_loss, value_and_grad=True, method="L-BFGS-B", jit=False
    )

    res = solver.run(init_params)

    assert np.isclose(res.params["beta_t"][-2], 0.24915876, atol=1e-5)
    assert np.isclose(res.params["beta_t"][1], 0.10921172, atol=1e-5)


def test_bayes():
    t_out_bayes = np.arange(100)
    data_bayes = data[t_out_bayes]
    t_solve_ODE_bayes = np.linspace(
        t_out_bayes[0], t_out_bayes[-1], len(t_out_bayes) // 2
    )
    t_beta_bayes = np.linspace(t_out_bayes[0], t_out_bayes[-1], len(t_out_bayes) // 14)

    integrator_object_bayes = icomo.ODEIntegrator(
        ts_out=t_out_bayes,
        ts_solver=t_solve_ODE_bayes,
        ts_arg=t_beta_bayes,
    )

    duration_latent = 3
    duration_infectious = 7

    with pm.Model(coords={"time": t_out_bayes, "t_beta": t_beta_bayes}) as model:
        # We also allow the other rates of the compartments to vary
        duration_latent_var = pm.LogNormal(
            "duration_latent", mu=np.log(duration_latent), sigma=0.1
        )
        duration_infectious_var = pm.LogNormal(
            "duration_infectious", mu=np.log(duration_infectious), sigma=0.3
        )

        R0 = pm.LogNormal("R0", np.log(1), 1)
        beta_0_var = 1 * R0 / duration_infectious_var
        beta_t_var = beta_0_var * pt.exp(
            pt.cumsum(icomo.hierarchical_priors("beta_t_log_diff", dims=("t_beta",)))
        )

        const_args_var = {
            "N": N_England,
            "rate_latent": 1 / duration_latent_var,
            "rate_infectious": 1 / duration_infectious_var,
        }
        infections_0_var = pm.LogNormal(
            "infections_0", mu=np.log(data_bayes[0] * duration_infectious), sigma=2
        )

        y0_var = {
            "Es": [infections_0_var / 6 for _ in range(3)],
            "I": infections_0_var / 2,
            "R": 0,
        }
        y0_var["S"] = N_England - jax.tree_util.tree_reduce(lambda x, y: x + y, y0_var)

        SEIR_integrator_op = integrator_object_bayes.get_op(
            Erlang_SEIR,
            return_shapes=[() for _ in range(2)],
            list_keys_to_return=["S", "I"],
        )

        S, I = SEIR_integrator_op(
            y0=y0_var, arg_t=beta_t_var, constant_args=const_args_var
        )

        pm.Deterministic("I", I)
        new_cases = -pt.diff(S)
        pm.Deterministic("new_cases", new_cases)

        sigma_error = pm.HalfCauchy("sigma_error", beta=1)
        pm.StudentT(
            "cases_observed",
            nu=4,
            mu=new_cases,
            sigma=sigma_error * pt.sqrt(new_cases + 1),
            observed=data_bayes[1:],
        )

        # Like before,w we also want to save the interpolated beta_t
        beta_t_interp = icomo.interpolate_pytensor(
            t_beta_bayes, t_out_bayes, beta_t_var
        )
        pm.Deterministic("beta_t_interp", beta_t_interp)

    # Test only successful compilation

    ip = model.initial_point()
    # Setting the mode to fast_compile shouldn't make a difference in the test coverage
    logp_fn = model.compile_fn(model.logp(sum=False), mode="FAST_COMPILE")
    logp_fn(ip)
    dlogp_fn = model.compile_fn(model.dlogp(), mode="FAST_COMPILE")
    dlogp_fn(ip)

    ip = model.initial_point()
    logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
    logp_fn(ip)
    dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
    dlogp_fn(ip)

    with jax.disable_jit():
        ip = model.initial_point()
        logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
        dlogp_fn(ip)
