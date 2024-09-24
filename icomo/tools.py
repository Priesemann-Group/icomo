"""Auxiliary functions."""

import pymc as pm
import pytensor.tensor as pt


def hierarchical_priors(
    name,
    dims,
    beta=1,
    fix_hyper_sigma=None,
    dist_values=pm.Normal,
    centered_parametrization=False,
    empirical_sigma=False,
):
    """Create hierarchical priors for a variable.

    Create an n-dimensional variable with hierarchical prior with name `name` and
    dimensions `dims`. The prior is a normal distribution with standard deviation
    `sigma` which is sample from a half-cauchy distribution with beta parameter `beta`.

    Parameters
    ----------
    name : str
        Name under which the variable is stored in pm.Model
    dims : tuple of str
        Dimensions over which the variable is defined. Define a dimension by passing
        coords={dim_name: np.arange(size)} to pm.Model at creation.
    beta : float, default=1
        Beta parameter of the half-cauchy distribution
    fix_hyper_sigma : float, default=None
        If not None, the standard deviation from which the variable is sampled is fixed
    dist_values : :class:`pymc.Distribution`, default=pm.Normal
        Distribution from which the values are sampled. Can for example be
        functools.partial(pm.StudentT, nu=4) to sample from a StudentT distribution for
        a more robust model.
    centered_parametrization : bool, default=False
        Whether to use a centered or non-centered parametrization for the hierarchical
        prior
    empirical_sigma : bool, default=False
        Whether to set the standard deviation of the normal distribution to the standard
        deviation of the sampled value. Corresponds to an empirical Bayes approach.

    Returns
    -------
    values : pm.Variable
        pm.Variable for the variable with name `name`

    """
    if empirical_sigma:
        if not dist_values == pm.Normal:
            raise RuntimeError("Empirical sigma only works with normal distribution")
    if not empirical_sigma:
        sigma = (
            pm.HalfCauchy(f"sigma_{name}", beta=beta)
            if fix_hyper_sigma is None
            else fix_hyper_sigma
        )
    if not centered_parametrization:
        values = (dist_values(f"{name}_raw", mu=0, sigma=1, dims=dims)) * sigma
        values = pm.Deterministic(f"{name}", values, dims=dims)

    else:
        values = dist_values(f"{name}", mu=0, sigma=sigma, dims=dims)
    if empirical_sigma:
        values = pm.Flat(f"{name}", dims=dims)
        sigma = pt.std(values)
        logp_reg = pm.logp(dist_values.dist(mu=values, sigma=sigma), 0)
        pm.Potential(f"{name}_regularization", logp_reg, dims=dims)

    return values
