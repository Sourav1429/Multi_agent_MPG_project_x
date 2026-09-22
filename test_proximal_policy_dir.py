import numpy as np

from multi_agent import project_x_tabular


def evaluate_deterministic(problem, pi):
    np.random.seed(12345)
    return problem.evaluate_policy(pi)


def objective_for_policy(problem, pi, pi_old):
    Phi, Jc = evaluate_deterministic(problem, pi)

    lam = problem.lam
    b = problem.b
    mu = problem.mu

    t = min(
        Phi / lam,
        b - Jc
    )

    proximal_penalty = (
        1.0 / (2.0 * mu)
    ) * np.sum((pi - pi_old) ** 2)

    objective = t - proximal_penalty

    return objective, t, Phi, Jc


class Args:
    env_nm = "pollution"
    T = 20
    M = 2
    horizon = 10
    gamma = 0.9
    mu = 0.8
    lamdba = 20
    eta = 0.1
    b = 12

    init_pi = np.zeros((M, 2, 2))

    for i in range(M):
        for s in range(2):
            init_pi[i, s, 0] = 0.7
            init_pi[i, s, 1] = 0.3


args = Args()
problem = project_x_tabular(args)

pi_old = np.array(args.init_pi, dtype=float)

print("\n========== BASE POLICY ==========")

base_obj, base_t, base_phi, base_jc = \
    objective_for_policy(problem, pi_old, pi_old)

print("Policy:")
print(pi_old)
print(f"Phi       = {base_phi:.10f}")
print(f"Jc        = {base_jc:.10f}")
print(f"t         = {base_t:.10f}")
print(f"Objective = {base_obj:.10f}")


print("\n========== MANUAL PERTURBATIONS ==========")

epsilons = [
    -0.15,
    -0.10,
    -0.05,
    -0.02,
     0.00,
     0.02,
     0.05,
     0.10,
     0.15,
]

for eps in epsilons:

    pi_test = pi_old.copy()

    # Shift probability between action 0 and action 1
    pi_test[:, :, 0] = 0.7 - eps
    pi_test[:, :, 1] = 0.3 + eps

    obj, t, phi, jc = \
        objective_for_policy(problem, pi_test, pi_old)

    print(
        f"\neps = {eps:+.3f}"
    )
    print(
        f"policy = [{0.7-eps:.3f}, {0.3+eps:.3f}]"
    )
    print(
        f"Phi = {phi:.10f}, "
        f"Jc = {jc:.10f}, "
        f"t = {t:.10f}, "
        f"objective = {obj:.10f}, "
        f"delta = {obj-base_obj:+.10f}"
    )