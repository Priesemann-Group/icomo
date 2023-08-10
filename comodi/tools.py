import pymc as pm


def hierarchical_priors(name, dims, beta=1, fix_hyper_sigma=None):
    """Create hierarchical priors for a variable.

    Create an n-dimensional variable with hierarchical prior with name `name` and
    dimensions `dims`. The prior is a normal distribution with standard deviation `sigma`
    which is sample from a half-cauchy distribution with beta parameter `beta`.

    Parameters
    ----------
    name : str
        name under which the variable is stored in pm.Model
    dims : tuple of str
        dimensions over which the variable is defined. Define a dimension by passing
        coords={dim_name: np.arange(size)} to pm.Model at creation.
    beta : float, default=1
        beta parameter of the half-cauchy distribution
    fix_hyper_sigma : float, default=None
        if not None, the standard deviation from which the variable is sampled is fixed

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
    values = (pm.Normal(f"{name}_raw", 0, 1, dims=dims)) * sigma
    values = pm.Deterministic(f"{name}", values, dims=dims)
    return values
