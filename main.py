import numpy as np
import matplotlib.pyplot as plt
from bandit_lin_softmax import compute_mip, compute_mip_batch
import time


# TODO: move this to config file
def generate_data(n_atoms, n_features):
    query = np.random.normal(0.0, 1.0, n_features)
    atoms = np.zeros((n_atoms, n_features))
    r1, r2 = np.random.choice(n_atoms, 2, replace=False)
    # offset fixed atoms by some amount, to simulate environment where
    # bandit can quickly identify best arm
    atoms[r1][query > 0] = 1.0
    atoms[r1][query < 0] = -1.0
    atoms[r2][query > 0] = 0.5
    atoms[r2][query < 0] = -0.5
    atoms += np.random.normal(0.0, 1.0, (n_atoms, n_features))
    return query, atoms


# TODO: move this to unit test
def validate_mip():
    query, atoms = generate_data(20, 1000)
    best_ind = np.argmax(atoms @ query)
    print('best_ind = ', best_ind)

    # TODO: how do we estimate $sigma$ in general
    best_ind_bandit, n_samples = compute_mip(atoms, query, sigma=1, delta=0.05)
    print('best_ind (bandit) = ', best_ind_bandit)
    print('n_iter (bandit) = ', n_samples)

    best_ind_bandit_batch, n_samples_batch = compute_mip_batch(atoms, query, sigma=1, delta=0.05)
    print('best_ind (bandit_batch) = ', best_ind_bandit)
    print('n_iter (bandit_batch) = ', n_samples_batch)


def run_experiment(n_atoms, n_features, sigma, n_trials=1):
    query, atoms = generate_data(n_atoms, n_features)
    start_time = time.time()
    _ = np.argmax(atoms @ query)
    naive_time = time.time() - start_time

    print("naive_time = ", naive_time)

    # TODO: how do we estimate $sigma$ in general
    start_time = time.time()
    _, n_samples = compute_mip(atoms, query, sigma, delta=0.05)
    bandit_time = time.time() - start_time

    print("bandit_time = ", bandit_time)

    start_time = time.time()
    _, n_samples_batch = compute_mip_batch(atoms, query, sigma, delta=0.05)
    bandit_batch_time = time.time() - start_time

    print("bandit_batch_time = ", bandit_batch_time)

    return naive_time, bandit_time, n_samples, bandit_batch_time, n_samples_batch


if __name__ == "__main__":
    np.random.seed(42)

    N_ATOMS = 1000
    N_FEATURES_EXP = np.array([5000, 10000, 20000, 40000, 80000])
    (
        bandit_n_samples,
        bandit_times,
        bandit_batch_n_samples,
        bandit_batch_times,
        naive_times,
    ) = ([], [], [], [], [])
    for n_features in N_FEATURES_EXP:
        (
            naive_time,
            bandit_time,
            bandit_ns,
            bandit_batch_time,
            bandit_batch_ns,
        ) = run_experiment(N_ATOMS, n_features, sigma=1.0)
        bandit_n_samples.append(bandit_ns)
        bandit_times.append(bandit_time)
        bandit_batch_n_samples.append(bandit_batch_ns)
        bandit_batch_times.append(bandit_batch_time)
        naive_times.append(naive_time)

    plt.plot(
        N_FEATURES_EXP, np.array(bandit_n_samples), marker="o", label="samples (bandit)"
    )
    plt.plot(
        N_FEATURES_EXP,
        np.array(bandit_batch_n_samples),
        marker="o",
        label="samples (bandit-batch)",
    )
    plt.plot(
        N_FEATURES_EXP, N_ATOMS * N_FEATURES_EXP, marker="s", label="samples (total)"
    )
    plt.legend()
    plt.savefig("iter_vs_dim_(mips).png")
    plt.clf()

    plt.plot(N_FEATURES_EXP, np.array(naive_times), marker="o", label="time (naive)")
    plt.plot(N_FEATURES_EXP, np.array(bandit_times), marker="o", label="time (bandit)")
    plt.plot(
        N_FEATURES_EXP,
        np.array(bandit_batch_times),
        marker="o",
        label="time (bandit-batch)",
    )
    plt.legend()
    plt.savefig("iter_vs_time.png")
    plt.clf()
