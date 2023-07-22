import numpy as np


def compute_mip_batch(atoms, query, sigma, delta, batch_size=16):
    """
    does same thing as previous, but instead of doing multiplication between single element of A and x,
    it sequentially slices 'batch_size' elements from left to right, and performs inner product to
    pull an arm.
    """
    dim = len(query)
    n_atoms = len(atoms)
    solution_mask = np.ones(n_atoms, dtype="bool")
    mu = np.zeros(n_atoms)
    solutions = np.nonzero(solution_mask)[0]
    d_used = 0
    n_samples = 0  # instrumentation

    while len(solutions) > 1 and d_used < dim - batch_size:
        tmp = (
            atoms[solution_mask, d_used : d_used + batch_size]
            @ query[d_used : d_used + batch_size]
        )
        mu[np.ix_(solutions)] = (d_used * mu[np.ix_(solutions)] + tmp) / (
            d_used + batch_size
        )
        n_samples += len(solutions) * batch_size

        if d_used > 0:
            C = sigma * np.sqrt(
                2 * np.log(4 * n_atoms * d_used**2 / delta) / (d_used + 1)
            )
            max_mu = np.max(mu[solution_mask])
            solution_mask = solution_mask & (mu + C >= max_mu - C)
            solutions = np.nonzero(solution_mask)[0]

        d_used += batch_size

    if len(solutions) > 1:
        best_val, best_ind = np.NINF, -1
        for i in solutions:
            if mu[i] > best_val:
                best_val = mu[i]
                best_ind = i
    else:
        best_ind = solutions[0]

    return best_ind, n_samples


def compute_mip(atoms, query, sigma, delta):
    """

    :param atoms: 'Parameter' of softmax function (n_classes * n_features matrix) (denoted as A below)
    :param query: 'datapoint' of softmax function (n_features dimensional vector) (denoted as x below)
    :param sigma: upper bound of standard deviation for all element sampling?
    :param delta: error probability in PAC estimation
    :return: best index(explained below), number of samples used

    takes parameter for softmax function and the datapoint to calculate the softmax on,
    and performs best-arm identification to find the index with maximum softmax score.

    In the best-arm identification problem, each arm is index,
    and each arm pull, for each arm i, is calculating A_ij * x_j and integrate it to the estimation of M_i,
    where M = Ax, and j is a sample from uniform distribution ranging from 1 to n_features.


    """

    dim = len(query) #n_features
    n = len(atoms) #n_atoms(n_classes?)
    S_solution = set(range(n))
    J = np.random.permutation(dim)
    mu = np.zeros(n)
    d_used = 0
    n_iter = 0
    for j in J:
        if len(S_solution) == 1:
            break

        for i in S_solution:
            mu[i] = (d_used * mu[i] + query[j] * atoms[i][j]) / (d_used + 1)

        n_iter += len(S_solution)

        if d_used > 0:
            C = sigma * np.sqrt(2 * np.log(4 * n * d_used**2 / delta) / (d_used + 1))
            max_mu = np.max(mu)
            S_solution = [i for i in S_solution if mu[i] + C >= max_mu - C]

        d_used += 1

    if len(S_solution) > 1:  #Possibly a placeholder for exact computation? (Probably not?)
        best_val, best_ind = np.NINF, -1
        for i in S_solution:
            if mu[i] > best_val:
                best_val = mu[i]
                best_ind = i
    else:
        best_ind = S_solution.pop()

    return best_ind, n_iter

def estimate_softmax_normalization(atoms, query, beta, epsilon, delta, sigma):
    n = atoms.shape[0]
    d = query.shape[0]

    T0 = 48 * beta**2 * sigma**2 * np.log(6 * n / delta)

    mu_hat = (
        atoms[:, :T0]
        @ query[:T0]
    )
    C = np.sqrt(1 / 24 * beta**2) #equivalent with line 3 of adaApprox alogrithm in proposal

    mu_hat_exp = np.exp(mu_hat - C)
    alpha = mu_hat_exp / np.sum(mu_hat_exp)

    T = (
        34 * beta**2 * sigma**2 * np.log(6 * n / delta) * n
        + 8 * sigma**2 * np.log(6 * n / delta) * beta**2 * n / epsilon
        + 16 * beta**2 * sigma**2 * np.log(12 / delta) / epsilon**2
    )

    n_samples = np.minimum(alpha * T, d)

    mu_hat_refined = np.zeros(n)

    for i in n:
        mu_hat_refined += atoms[i, :n_samples[i]] @ query[:n_samples[i]] #TODO: what if n_samples[i] == d?

    return np.sum(np.exp(beta * mu_hat_refined))

def ada_softmax(A, x, beta, epsilon, delta):
    #TODO: how to figure out sigma?
    #sigma = compute_sigma()
    sigma = 100

    S_hat = estimate_softmax_normalization(A, x, beta, epsilon / 2, delta / 3)

    best_index_hat, _ = compute_mip(A, x, sigma, delta / 3)

    n_arm_pull = 8 * sigma**2 * beta**2 * np.log(6 / delta) / epsilon**2

    mu_best_hat = A[:n_arm_pull] @ x[:n_arm_pull] #TODO: is there a case of overflow?

    y_best_hat = np.exp(-1 * beta * mu_best_hat)

    return best_index_hat, y_best_hat / S_hat
