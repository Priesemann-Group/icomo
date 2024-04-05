"""Helper functions to model a slow modulation of a time series."""

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.sort import ArgSortOp

from .tools import hierarchical_priors


def sigmoidal_changepoints(
    ts_out, positions_cp, magnitudes_cp, durations_cp, reorder_cps=False
):
    r"""Modulation of a time series by sigmoidal changepoints.

    The changepoints are defined by their position, magnitude and duration. The resulting equation is:

    .. math::
     f(t) = \sum_{i=1}^{\mathrm{num_cps}} \frac{\mathrm{magnitudes\_cp}[i]}{1 + exp(-4 \cdot \mathrm{slope}[i] \cdot (t - \mathrm{positions_cp}[i]))}
    ..

    where slope[i] = magnitudes_cp[i] / durations_cp[i].

    Parameters
    ----------
    t_out: 1d-array
        timepoints where modulation is evaluated, shape: `(time, )`
    t_cp: nd-array
        timepoints of the changepoints, shape: `(num_cps, further dims...)`
    magnitudes: nd-array
        magnitude of the changepoints, shape: `(num_cps, further dims...)`
    durations: nd-array
        magnitude of the changepoints, shape: `(num_cps, further dims...)`
    reorder_cps: bool, default=False
        reorder changepoints such that their timepoints are linearly increasing
    Returns
    -------
    modulation_t: (n+1)d-array
        shape: `(time, further dims...)`

    """
    if reorder_cps:
        order = pt.argsort(positions_cp, axis=0)
        positions_cp = positions_cp[order, ...]
        magnitudes_cp = magnitudes_cp[order, ...]
        durations_cp = durations_cp[order, ...]

    # add necessary empty dimensions to time axis
    ts_out = np.expand_dims(ts_out, axis=tuple(range(1, max(1, positions_cp.ndim) + 1)))
    slope_cp = 1 / durations_cp
    modulation_t = (
        pt.sigmoid((ts_out - positions_cp) * slope_cp * 4) * magnitudes_cp
    )  # 4*slope_cp because the derivative of the sigmoid at zero is 1/4, we want to set it to slope_cp

    modulation_t = pt.sum(modulation_t, axis=1)

    return modulation_t


def priors_for_cps(
    cp_dim,
    time_dim,
    name_positions,
    name_magnitudes,
    name_durations,
    beta_magnitude=1,
    sigma_magnitude_fix=None,
    dist_magnitudes=pm.Normal,
    absolute_magnitude_parametrization=False,
    empirical_bayes_hyper_sigma=False,
    centered_parametrization=False,
    model=None,
):
    """Create priors for changepoints.

    Their positions are uniformly distributed between
    the first and last timepoint. The magnitudes are sampled from a hierarchical prior
    with a beta distribution. The durations are sampled from a normal distribution with
    mean equal to the mean distance between changepoints and standard deviation equal to
    the standard deviation of the distance between changepoints.

    Parameters
    ----------
    cp_dim : str
        Dimension of the :class:`pymc.Model` for the changepoints. Define it by passing
        `coords={cp_dim: np.arange(num_cps)}` to :class:`pymc.Model` at creation. The length of this
        dimension determines the number of changepoints.
    time_dim : str
        Dimension of the :class:`pymc.Model` for the time.
    name_positions : str
        Name under which the positions of the changepoints are stored in :class:`pymc.Model`
    name_magnitudes : str
        Name under which the magnitudes of the changepoints are stored in :class:`pymc.Model`
    name_durations : str
        Name under which the durations of the changepoints are stored in :class:`pymc.Model`
    beta_magnitude : float, default=1
        Beta parameter of the hierarchical prior for the magnitudes
    sigma_magnitude_fix : float, default=None
        If not `None`, the standard deviation from which the magnitudes are sampled is fixed
    dist_magnitudes : :class:`pymc.Distribution`, default=pm.Normal
        Distribution from which the magnitudes are sampled. Can for example be
        functools.partial(pm.StudentT, nu=4) to sample from a StudentT distribution for
        a more robust model.
    absolute_magnitude_parametrization : bool, default=False
        Whether to use an parametrization that is absolute or relative to previous
        values for the magnitudes.
    empirical_bayes_hyper_sigma : bool, default=False
        Whether to set the standard deviation of the hierarchical magnitudes to the maximum
        likelihood estimate instead of sampling. Corresponds to an empirical Bayes approach.
    centered_parametrization : bool, default=False
        Whether to use a centered parametrization for the hierarchical priors of the magnitudes.
    model : :class:`pymc.Model`, default=None
        pm.Model in which the priors are created. If None, the pm.Model is taken from the
        the context.

    Returns
    -------
    positions, magnitudes, durations : :class:`pytensor.Variable`
        Variables of dim `cp_dim` that define the positions, magnitudes and durations of the changepoints

    """
    model = pm.modelcontext(model)

    time_arr = model.coords[time_dim]
    num_cps = len(model.coords[cp_dim])
    interval_cps = (max(time_arr) - min(time_arr)) / (num_cps + 1)

    ### Positions
    std_Delta_pos = interval_cps / 3
    Delta_pos = pm.Normal(f"Delta_{name_positions}", 0, std_Delta_pos, dims=(cp_dim,))
    positions = Delta_pos + np.arange(1, num_cps + 1) * interval_cps
    pm.Deterministic(f"{name_positions}", positions)

    ### Magnitudes:
    magnitudes_tmp = hierarchical_priors(
        name_magnitudes,
        dims=(cp_dim,),
        beta=beta_magnitude,
        fix_hyper_sigma=sigma_magnitude_fix,
        dist_values=dist_magnitudes,
        centered_parametrization=centered_parametrization,
        empirical_sigma=empirical_bayes_hyper_sigma,
    )
    if not absolute_magnitude_parametrization:
        magnitudes = magnitudes_tmp
    else:
        magnitudes = pt.diff(pt.concatenate([[0.0], magnitudes_tmp]))

    ### Durations:
    mean_duration_len = interval_cps / 3
    std_duration_len = interval_cps / 6
    softplus_scaling = interval_cps / 12
    durations_raw = pm.Normal(
        f"{name_durations}_raw", mean_duration_len, std_duration_len, dims=(cp_dim,)
    )
    durations = pm.Deterministic(
        f"{name_durations}",
        pt.softplus(durations_raw / softplus_scaling) * softplus_scaling,
    )

    order = pt.argsort(positions, axis=0)
    positions = positions[order, ...]
    magnitudes = magnitudes[order, ...]
    durations = durations[order, ...]

    return positions, magnitudes, durations


@jax_funcify.register(ArgSortOp)
def _jax_funcify_Argsort(op, node, **kwargs):
    def argsort(x, axis=None):
        return jnp.argsort(x, axis=axis)

    return argsort
