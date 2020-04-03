"""
This (currently) a toy script which computes likelihood of a patient x given
a (sub-)set of N similar patients defined by M-dimensional feature vectors.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals
from scipy.stats import multivariate_normal

def main(cfg):
    N,M = cfg["N"], cfg["M"]

    # Get means
    mu = np.random.rand(M,)
    if M <= 5:
        print("Means:")
        print(mu, "\n")

    # Get covariance matrix
    A = np.random.rand(M,M)
    cov = A * np.transpose(A) + np.eye(M)
    if M <= 5:
        print("Covariance Matrix:")
        print(cov)

    D = multivariate_normal(mu, cov)
    data = D.rvs(N)

    # Get sample distribution
    mu_hat = np.mean(data, axis=0)
    cov_hat = np.cov( np.transpose(data) )
    D_hat = multivariate_normal(mu_hat, cov_hat)

    print("RMSE, mean: %0.3f" % np.sqrt( np.linalg.norm(mu - mu_hat) ))
    print("RMSE, cov eigenvalues: %0.3f" % (np.sqrt( np.linalg.norm( \
            eigvals(cov) - eigvals(cov_hat) ) ) ))

    x = np.random.rand(M,)
    if M <= 5:
        print("x: ", x)
    print("|| x - mu_hat ||: %0.3f" % np.linalg.norm(x- mu_hat))

    print("Likelihood of mu_hat: %0.3f" % D_hat.pdf(mu_hat))
    print("Likelihood of x: %0.3f" % D_hat.pdf(x))

    X = np.random.rand(50, M)
    dists = [np.linalg.norm(x - mu_hat) for x in X]
    probs = [D_hat.pdf(x) for x in X]
    plt.scatter(dists, probs)
    plt.title("P(mu_hat) = %0.3f" % D_hat.pdf(mu_hat))
    plt.xlabel("|| x - mu_hat ||")
    plt.ylabel("Probability density")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=10)
    parser.add_argument("-M", type=int, default=4)
    cfg = vars( parser.parse_args() )
    main(cfg)

