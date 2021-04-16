import numpy as np

import pymc3 as pm


def build_logistic_model(df, null_model=True, priors={
                                'Intercept': {'mu': 0, 'sigma': 5},
                                'Format Slope': {'mu': 0, 'sigma': 5},
                                'Order Slope': {'mu': 0, 'sigma': 5},
                                'Interaction Slope': {'mu': 0, 'sigma': 5}
                                }
                         ):
    """Ad-Hoc function for building the computational graph for a
    logistic model in PyMC3 evaluating the impact of the binary factors
    FnoA Order and Format and their interaction on number of normatively
    correct responses. More precisely the model is a binomial regression where
    given the number of successes and total trials we try to estimate the
    parameter p using a logitic model.

        Args:
            df (DataFrame): pandas DataFrame containing the data to analyze.
            null_model (bool): boolean specifying if the generated model is
                intercept only or will include the predictors.
            priors (dict): a dictionary specifying the parameters of the
                gaussian priors for the model. By default they are set to be
                    only weakly informative. Keys are components of the model
                    while values are dictionaries reporting the mu and sigma
                    values for the gaussians.

        Returns:
            log_reg (PyMC3 model): a PyMC3 model containing the computational
                graph for a logistic regression.
    """
    with pm.Model() as log_reg:

        form = pm.Data(
            name='Format',
            value=np.array(
                [1 if i == 'Disease' else 0 for i in df['format'].values]
            )
        )
        order = pm.Data(
            name='Order',
            value=np.array(
                [1 if i == 'FnoA Second' else 0 for i
                    in df['fnoa_order'].values]
            )
        )

        n = pm.Data(
            name='N',
            value=df['total_norm_corr'].values
        )
        observed = pm.Data(
            name='Observed',
            value=df['n_norm_corr'].values
        )

        intercept = pm.Normal(
            mu=priors['Intercept']['mu'],
            sd=priors['Intercept']['sigma'],
            name='Intercept'
        )

        if not null_model:
            slope_form = pm.Normal(
                mu=priors['Format Slope']['mu'],
                sd=priors['Format Slope']['sigma'],
                name='Format Slope'
            )
            slope_order = pm.Normal(
                mu=priors['Order Slope']['mu'],
                sd=priors['Order Slope']['sigma'],
                name='Order Slope'
            )
            slope_interaction = pm.Normal(
                mu=priors['Interaction Slope']['mu'],
                sd=priors['Interaction Slope']['sigma'],
                name='Interaction Slope'
            )

            theta = intercept + slope_form*form + slope_order*order + \
                slope_interaction*form*order
        else:
            theta = intercept

        p = pm.math.sigmoid(
            theta
        )

        outcome = pm.Binomial(
            name='Outcome',
            n=n,
            p=p,
            observed=observed
        )

    return log_reg
