# Advanced likelihood distributions

These distribution objects are for custom likelihoods and posthoc estimators.
They are not needed for the standard workflow examples.

## Distribution objects

::: bami.inference.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - Bernoulli
        - Binomial
        - Categorical
        - Multinomial
        - Normal
        - Gaussian
        - LogNormal
        - Poisson
        - VonMises
        - IID
        - Joint
        - Mixture

## Examples

```python
import numpy as np

from bami.inference.distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    Gaussian,
    IID,
    Joint,
    LogNormal,
    Mixture,
    Multinomial,
    Normal,
    Poisson,
    VonMises,
)


candidate_p = np.array([0.3, 0.7])
candidate_mu = np.array([0.0, 1.0])
candidate_sigma = np.array([1.0, 1.0])

bernoulli = Bernoulli()
binomial = Binomial()
categorical = Categorical()
multinomial = Multinomial()
normal = Normal()
gaussian = Gaussian()
lognormal = LogNormal()
poisson = Poisson()
von_mises = VonMises()

bernoulli_logp = bernoulli.log_prob(1, p=candidate_p)
binomial_logp = binomial.log_prob(k=7, n=10, p=candidate_p)
categorical_logp = categorical.log_prob(1, p=np.array([[0.3, 0.7], [0.6, 0.4]]))
multinomial_logp = multinomial.log_prob([4, 6], p=np.array([[0.4, 0.6], [0.7, 0.3]]))
normal_logp = normal.log_prob(0.2, mu=candidate_mu, sigma=candidate_sigma)
gaussian_logp = gaussian.log_prob(0.2, mu=candidate_mu, sigma=candidate_sigma)
lognormal_logp = lognormal.log_prob(1.5, mu=candidate_mu, sigma=candidate_sigma)
poisson_logp = poisson.log_prob(3, rate=np.array([2.5, 3.5]))
von_mises_logp = von_mises.log_prob(0.1, mu=candidate_mu, kappa=np.array([1.0, 2.0]))

joint = Joint(rt=normal, correct=bernoulli)
iid = IID(normal)
mixture = Mixture(weights=[0.6, 0.4], components=[normal, Gaussian()])

joint_logp = joint.log_prob(
    {"rt": 0.5, "correct": 1},
    rt={"mu": candidate_mu, "sigma": candidate_sigma},
    correct={"p": candidate_p},
)
iid_logp = iid.log_prob(np.array([0.0, 0.5]), mu=candidate_mu[:, None], sigma=1.0)
mixture_logp = mixture.log_prob(0.2, mu=candidate_mu, sigma=candidate_sigma)
```
