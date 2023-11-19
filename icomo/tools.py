"""Auxiliary functions."""

import pymc as pm


def hierarchical_priors(
    name, dims, beta=1, fix_hyper_sigma=None, dist_values=pm.Normal
):
    """Create hierarchical priors for a variable.

    Create an n-dimensional variable with hierarchical prior with name `name` and
    dimensions `dims`. The prior is a normal distribution with standard deviation `sigma`
    which is sample from a half-cauchy distribution with beta parameter `beta`.

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

    Returns
    -------
    values : pm.Variable
        pm.Variable for the variable with name `name`

    """
    sigma = (
        pm.HalfCauchy(f"sigma_{name}", beta=beta)
        if fix_hyper_sigma is None
        else fix_hyper_sigma
    )
    values = (dist_values(f"{name}_raw", mu=0, sigma=1, dims=dims)) * sigma
    values = pm.Deterministic(f"{name}", values, dims=dims)
    return values
