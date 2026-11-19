Linear models
=============

.. autoclass:: sem2.model.models.linear.SEM2Linear
   :members: coef_matrix, mean_vector, cov_matrix
   :exclude-members: __init__

   .. extra line prevents docs of __init__ of SEM2 displayed right after docs of SEM2Linear

Sampling and graph inspection follow :class:`~sem2.model.models.base.SEM2`.

Correlated Gaussian exogenous variables
---------------------------------------

.. code-block:: python

   from sem2.model.models.linear_gaussian import SEM2LinearGaussian

   model = SEM2LinearGaussian(
       "EX ~ Normal(0,1); X <- 0 + 1*EX; Y <- 0 + X + 1*EY; EX <-> EY"
   )
   covariance = model.cov_matrix(["X", "Y"])

``EX`` has a fixed marginal law. ``EY`` has mean zero and a learned standard
deviation initialized at one. The edge permits their covariance to be learned;
its initial value is zero. Joint draws are reused throughout each observation.
An intervention replaces the named equation without conditioning the remaining
exogenous variables. Use unit noise loadings when learning marginal variances
to avoid redundant scales.

``noise_covariance()`` returns the covariance in ``noise_names`` order.
With a symmetric, zero-diagonal matrix ``W`` of free edge weights, the
correlation matrix is ``I + W / (1 - lambda_min(W))``. Absent edges
remain exactly zero. This parameterization covers the positive-definite
interior; singular covariances are limiting cases. Gaussian likelihood
requires a nonsingular observed covariance.

.. autoclass:: sem2.model.models.linear_gaussian.SEM2LinearGaussian
   :members: noise_covariance
