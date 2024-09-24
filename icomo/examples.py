"""Example of the inference of compartmental models."""

import diffrax
import numpy as np
import pymc as pm
import pymc.sampling.jax
import pytensor.tensor as pt

import icomo


def model_cases_seropositivity(
    N,
    cases_data,
    t_cases_data,
    seropos_data,
    t_seropos_data,
    sim_model=False,
    fact_subs=4,
    num_cps_reporting=3,
    num_cps_R=8,
):
    """Return example model for cases and seropositivity data."""
    end_sim = max(t_cases_data)

    t_solve_ODE = np.arange(-20, end_sim, fact_subs)

    coords = {
        "cp_reporting_id": np.arange(num_cps_reporting),
        "cp_R_id": np.arange(num_cps_R),
        "t_solve_ODE": t_solve_ODE,
        "t_seropos_data": t_seropos_data,
        "t_cases_data": t_cases_data,
    }

    with pm.Model(coords=coords) as model:
        R0 = pm.LogNormal("R0", np.log(1), 1) if not sim_model else 1.2
        inv_gamma = (
            pm.Gamma("inv_gamma", alpha=140, beta=1 / 0.1)
            if not sim_model
            else pt.as_tensor_variable(14)
        )
        gamma = pm.Deterministic("gamma", 1 / inv_gamma)

        eta_base = pm.Normal("eta_base", 0, 1) if not sim_model else 0.3
        t_pos_rep, Delta_rhos_rep, transients_rep = icomo.priors_for_cps(
            cp_dim="cp_reporting_id",
            time_dim="t_cases_data",
            name_positions="t_pos_rep",
            name_magnitudes="Delta_rhos",
            name_durations="transients_rep",
            beta_magnitude=1,
            sigma_magnitude_fix=0.1 if sim_model else None,
        )

        eta_report = pt.sigmoid(
            eta_base
            + icomo.slow_modulation.sigmoidal_changepoints(
                ts_out=t_cases_data,
                positions_cp=t_pos_rep,
                magnitudes_cp=Delta_rhos_rep,
                durations_cp=transients_rep,
            )
        )
        pm.Deterministic("eta_report", eta_report, dims=("t_cases_data",))

        t_pos_R, Delta_rhos_R, transients_R = icomo.priors_for_cps(
            cp_dim="cp_R_id",
            time_dim="t_solve_ODE",
            name_positions="t_pos_R",
            name_magnitudes="Delta_rhos_R",
            name_durations="transients_R",
            beta_magnitude=1,
            sigma_magnitude_fix=0.1 if sim_model else None,
        )

        reproduction_scale_t = pt.exp(
            icomo.slow_modulation.sigmoidal_changepoints(
                ts_out=t_solve_ODE,
                positions_cp=t_pos_R,
                magnitudes_cp=Delta_rhos_R,
                durations_cp=transients_R,
            )
        )
        beta_t = R0 * gamma * reproduction_scale_t

        I_0_raw = pm.LogNormal("I_0_raw", np.log(100), 2) if not sim_model else 100
        I_0 = pm.Deterministic("I_0", I_0_raw / eta_report[0])
        # I_0 = I_0_raw
        S_0 = N - I_0
        R_0 = 0

        pm.Deterministic("beta_t", beta_t, dims=("t_solve_ODE",))

        def SIR(t, y, args):
            S, I, R = y
            β, (γ, N) = args
            dS = -β(t) * I * S / N
            dI = β(t) * I * S / N - γ * I
            dR = γ * I
            return dS, dI, dR

        integrator = icomo.ODEIntegrator(
            ts_out=t_solve_ODE,
            t_0=min(t_solve_ODE),
            ts_solver=t_solve_ODE,
            ts_arg=t_solve_ODE,
            interp="cubic",
            solver=diffrax.Bosh3(),  # a 3rd order method
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=len(t_solve_ODE)),
        )
        SIR_integrator = integrator.get_op(
            SIR, return_shapes=[() for _ in range(3)], name="SIR"
        )

        S, I, R = SIR_integrator(
            y0=(S_0, I_0, R_0), arg_t=beta_t, constant_args=(gamma, N)
        )

        pm.Deterministic("S", S, dims=("t_solve_ODE",))
        pm.Deterministic("I", I, dims=("t_solve_ODE",))
        pm.Deterministic("R", R, dims=("t_solve_ODE",))

        new_positive = icomo.interpolate_pytensor(
            ts_in=t_solve_ODE[:-1] + 0.5 * np.diff(t_solve_ODE),
            ts_out=t_cases_data,
            y=-pt.diff(S) / np.diff(t_solve_ODE),
            ret_gradients=False,
            method="cubic",
        )

        pm.Deterministic("new_positive", new_positive)
        new_reported = new_positive * eta_report
        pm.Deterministic("new_reported", new_reported)

        error_rep = pm.HalfCauchy("error_report", beta=1)
        pm.Normal(
            "new_reported_data",
            new_reported,
            pt.sqrt(new_reported + 1e-5) * error_rep,
            observed=cases_data if not sim_model else None,
        )

        sero_at_data = icomo.interpolate_pytensor(
            ts_in=t_solve_ODE,
            ts_out=t_seropos_data,
            y=R,
            ret_gradients=False,
            method="cubic",
        )

        error_sero = pm.HalfNormal("error_sero", sigma=0.01)
        pm.Normal(
            "sero_data",
            sero_at_data / N,
            error_sero,
            observed=seropos_data if not sim_model else None,
        )

        sero_at_cases = icomo.interpolate_pytensor(
            ts_in=t_solve_ODE,
            ts_out=t_cases_data,
            y=R,
            ret_gradients=False,
            method="cubic",
        )

        pm.Deterministic("Sero_t", sero_at_cases / N)

    return model
