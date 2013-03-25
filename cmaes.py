from __future__ import division
import numpy as np
# import pprint as pp
# import ipdb


def cmaes(strfitnessfct, N):
    # input parameters
    xmean = np.random.rand(N)
    sigma = 0.5
    stopfitness = 1e-30
    stopeval = 1e4 * N ** 2

    # strategy parameter setting: selection
    lamda = int(4 + 3 * np.log(N))
    mu = lamda / 2
    weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, int(mu) + 1)])
    mu = int(mu)
    weights = weights / np.sum(weights)
    mueff = 1 / np.sum(weights ** 2)

    # strategy parameter setting: adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros((1, N))
    ps = np.zeros((1, N))
    B = np.eye(N)
    D = np.eye(N)
    C = np.identity(N)
    eigenval = 0
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    # generation loop
    counteval = 0
    arx = np.empty([N, lamda])
    arz = np.empty([N, lamda])
    arfitness = np.empty(lamda)
    while counteval < stopeval:
        for k in range(lamda):
            arz[:, k] = np.random.randn(N)
            # arx[:, k] = np.random.multivariate_normal(xmean, sigma ** 2 * C)
            arx[:, k] = xmean + sigma * (np.dot(np.dot(B, D), arz[:, k]))
            arfitness[k] = strfitnessfct(arx[:, k])
            counteval += 1
        # sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(arfitness)
        arfitness = arfitness[arindex]
        # xold = xmean
        xmean = np.dot(arx[:, arindex[:mu]], weights)
        zmean = np.dot(arz[:, arindex[:mu]], weights)

        ps = np.dot((1 - cs), ps) + np.dot((np.sqrt(cs * (2 - cs) * mueff)), np.dot(B, zmean))
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lamda)) / chiN < 1.4 + 2 / (N + 1)
        pc = np.dot((1 - cc), pc) + np.dot(np.dot(hsig, np.sqrt(cc * (2 - cc) * mueff)), np.dot(np.dot(B, D), zmean))
        # adapt covariance matrix C
        C = np.dot((1 - c1 - cmu), C) \
            + np.dot(c1, ((pc * pc.T)
            + np.dot((1 - hsig) * cc * (2 - cc), C))) \
            + np.dot(cmu,
                     np.dot(np.dot(np.dot(np.dot(B, D), arz[:, arindex[:mu]]),
                            np.diag(weights)), (np.dot(np.dot(B, D), arz[:, arindex[:mu]])).T))

        # adapt step size sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # diagonalization
        if counteval - eigenval > lamda / (c1 + cmu) / N / 10:
            eigenval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            #ipdb.set_trace()
            D, B = np.linalg.eig(C)
            D = np.diag(np.sqrt(D))
            # invsqrtC = np.dot(np.dot(B, np.diag(D ** -1)), B.T)

        if arfitness[0] <= stopfitness:
            break

    return arx[:, arindex[0]], counteval


def f2(x):
    """
    Rosenbrock banana function
    minimum: (1, 1)
    f2((1,1)) == 0.0
    """
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == '__main__':
    res, evals = cmaes(f2, 2)
    print "result:", res
    print "evals:", evals
    print "function value:", f2(res)
