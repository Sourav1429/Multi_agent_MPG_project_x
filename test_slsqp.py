import numpy as np
from scipy.optimize import minimize

from multi_agent import project_x_tabular

def evaluate_deterministic(problem, pi):
    np.random.seed(12345)
    return problem.evaluate_policy(pi)
def test_slsqp(problem):
    """
    Diagnose ONE SLSQP proximal subproblem without running
    the full outer algorithm.
    """

    pi = np.array(problem.args.init_pi, dtype=float)

    mu = problem.mu
    lam = problem.lam
    b = problem.b
    S = problem.S
    A = problem.A
    M = problem.num_players

    n_policy = M * S * A

    # ---------------------------------------------------------
    # 1. Evaluate current policy
    # ---------------------------------------------------------
    Phi_current, Jc_current = problem.evaluate_policy(pi)

    t0 = min(
        Phi_current / lam,
        b - Jc_current
    )

    print("\n========== INITIAL POINT ==========")
    print("pi =")
    print(pi)
    print(f"Phi = {Phi_current:.10f}")
    print(f"Jc  = {Jc_current:.10f}")
    print(f"lambda = {lam}")
    print(f"b = {b}")
    print(f"mu = {mu}")
    print(f"t0 = {t0:.10f}")

    print("\nInitial constraints:")
    print(
        f"potential constraint = "
        f"{Phi_current / lam - t0:.10f}"
    )
    print(
        f"cost constraint = "
        f"{b - Jc_current - t0:.10f}"
    )

    # ---------------------------------------------------------
    # 2. Unpack vector
    # ---------------------------------------------------------
    def unpack(x):
        pi_prime = x[:-1].reshape(M, S, A)
        t = x[-1]
        return pi_prime, t

    # ---------------------------------------------------------
    # 3. Objective
    # ---------------------------------------------------------
    def objective(x):
        pi_prime, t = unpack(x)

        proximal_penalty = (
            1.0 / (2.0 * mu)
        ) * np.sum((pi_prime - pi) ** 2)

        return -(t - proximal_penalty)

    # ---------------------------------------------------------
    # 4. Potential constraint
    # ---------------------------------------------------------
    def potential_constraint(x):
        pi_prime, t = unpack(x)

        Phi_prime, _ = evaluate_deterministic(problem, pi_prime)

        value = Phi_prime / lam - t

        return value

    # ---------------------------------------------------------
    # 5. Cost constraint
    # ---------------------------------------------------------
    def cost_constraint(x):
        pi_prime, t = unpack(x)

        _, Jc_prime = evaluate_deterministic(problem, pi_prime)

        value = b - Jc_prime - t

        return value

    # ---------------------------------------------------------
    # 6. Probability simplex constraint
    # ---------------------------------------------------------
    def probability_constraints(x):
        pi_prime, _ = unpack(x)

        return (
            np.sum(pi_prime, axis=2) - 1.0
        ).ravel()

    # ---------------------------------------------------------
    # 7. Initial vector
    # ---------------------------------------------------------
    x0 = np.concatenate(
        [
            pi.flatten(),
            [t0]
        ]
    )

    # ---------------------------------------------------------
    # 8. Check x0 manually
    # ---------------------------------------------------------
    print("\n========== CHECK x0 ==========")

    print("Objective:", objective(x0))

    print(
        "Potential constraint:",
        potential_constraint(x0)
    )

    print(
        "Cost constraint:",
        cost_constraint(x0)
    )

    print(
        "Probability max error:",
        np.max(np.abs(probability_constraints(x0)))
    )

    # ---------------------------------------------------------
    # 9. SLSQP
    # ---------------------------------------------------------
    constraints = [
        {
            "type": "eq",
            "fun": probability_constraints
        },
        {
            "type": "ineq",
            "fun": potential_constraint
        },
        {
            "type": "ineq",
            "fun": cost_constraint
        }
    ]

    bounds = (
        [(0.0, 1.0)] * n_policy
        + [(None, None)]
    )

    print("\n========== RUNNING SLSQP ==========")

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 100,
            "ftol": 1e-8,
            "disp": True
        }
    )

    # ---------------------------------------------------------
    # 10. Diagnose returned solution
    # ---------------------------------------------------------
    pi_hat, t_hat = unpack(result.x)

    Phi_hat, Jc_hat = evaluate_deterministic(problem, pi_hat)

    pot_res = Phi_hat / lam - t_hat
    cost_res = b - Jc_hat - t_hat

    prob_res = np.max(
        np.abs(
            np.sum(pi_hat, axis=2) - 1.0
        )
    )

    print("\n========== FINAL DIAGNOSIS ==========")

    print(f"success = {result.success}")
    print(f"message = {result.message}")
    print(f"iterations = {result.nit}")
    print(f"function evaluations = {result.nfev}")

    print(f"\nt_hat = {t_hat:.10f}")
    print(f"Phi_hat = {Phi_hat:.10f}")
    print(f"Jc_hat = {Jc_hat:.10f}")

    print("\nConstraint residuals:")
    print(f"potential = {pot_res:.10f}")
    print(f"cost      = {cost_res:.10f}")
    print(f"prob max  = {prob_res:.10e}")

    print("\nPolicy:")
    print(pi_hat)

    return result


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Use the SAME arguments/configuration as your experiment
    # ---------------------------------------------------------

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

    test_slsqp(problem)