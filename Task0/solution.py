import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    normal_prior = 0.35
    laplace_prior = 0.25
    student_prior = 0.4

    normal_likelihood = 0
    laplace_likelihood = 0
    student_likelihood = 0

    for point in x:
        normal_prob = norm.pdf(point, 0, np.sqrt(2))
        laplace_prob = laplace.pdf(point, 0, 1)
        student_prob = t.pdf(point, 4)

        normal_likelihood = normal_likelihood + np.log(normal_prob)
        laplace_likelihood = laplace_likelihood + np.log(laplace_prob)
        student_likelihood = student_likelihood + np.log(student_prob)

    evidence = logsumexp(
        [
            np.log(normal_prior) + normal_likelihood,
            np.log(laplace_prior) + laplace_likelihood,
            np.log(student_prior) + student_likelihood
        ]
    )

    nominators = np.array([normal_likelihood + np.log(normal_prior),
                           laplace_likelihood + np.log(laplace_prior),
                           student_likelihood + np.log(student_prior)
                           ])
    log_p = nominators - evidence

    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 50 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
