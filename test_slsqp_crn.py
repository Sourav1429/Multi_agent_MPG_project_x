import numpy as np
from scipy.optimize import minimize

from multi_agent import project_x_tabular


# ============================================================
# 1. Fixed common random numbers
# ============================================================

def generate_common_random_numbers(problem, seed=12345):
    """
    Generate all random numbers ONCE.

    These random numbers are reused for every policy evaluation
    during the SLSQP solve.
    """

    rng = np.random.default_rng(seed)

    S = problem.S
    N = problem.num_samples
    H = problem.horizon
    M = problem.num_players

    # One uniform random number for every:
    # starting state, trajectory, time step, player
    U = rng.random((S, N, H, M))

    return U


# ============================================================
# 2. Deterministic CRN policy evaluation
# ============================================================

def evaluate_policy_crn(problem, pi, U):

    pi = np.asarray(pi, dtype=float).copy()

    # Same numerical protection as your original evaluator
    pi = np.clip(pi, 0.0, 1.0)

    row_sums = np.sum(pi, axis=2, keepdims=True)

    pi /= np.maximum(row_sums, 1e-15)

    potential_values = {}
    cost_values = {}

    S = problem.S
    N = problem.num_samples
    H = problem.horizon
    M = problem.num_players
    A = problem.A
    gamma = problem.gamma

    for start_state in range(S):

        potential_total = 0.0
        cost_total = 0.0

        for n in range(N):

            current_state = problem.env.reset(state=start_state)

            episode_potential = 0.0
            episode_cost = 0.0

            for t in range(H):

                joint_action = []

                for i in range(M):

                    # IMPORTANT:
                    # Use the SAME random number U for this
                    # state/sample/time/player every evaluation.
                    u = U[start_state, n, t, i]

                    probabilities = pi[i, current_state]

                    cumulative = np.cumsum(probabilities)

                    action = np.searchsorted(
                        cumulative,
                        u,
                        side="right"
                    )

                    # Numerical safety
                    action = min(action, A - 1)

                    joint_action.append(action)

                (
                    new_state,
                    rewards,
                    cost,
                    potential,
                    done
                ) = problem.env.step(joint_action)

                episode_potential += (
                    gamma ** t
                ) * potential

                episode_cost += (
                    gamma ** t
                ) * cost

                current_state = new_state

                if done:
                    break

            potential_total += episode_potential
            cost_total += episode_cost

        potential_values[start_state] = (
            potential_total / N
        )

        cost_values[start_state] = (
            cost_total / N
        )

    Phi = np.mean(
        list(potential_values.values())
    )

    Jc = np.mean(
        list(cost_values.values())
    )

    return Phi, Jc


# ============================================================
# 3. SLSQP diagnostic
# ============================================================

def test_slsqp_crn(problem):

    pi = np.array(
        problem.args.init_pi,
        dtype=float
    )

    mu = problem.mu
    lam = problem.lam
    b = problem.b

    S = problem.S
    A = problem.A
    M = problem.num_players

    n_policy = M * S * A

    # --------------------------------------------------------
    # Generate random numbers ONCE
    # --------------------------------------------------------

    U = generate_common_random_numbers(
        problem,
        seed=12345
    )

    # --------------------------------------------------------
    # Current policy
    # --------------------------------------------------------

    Phi_current, Jc_current = evaluate_policy_crn(
        problem,
        pi,
        U
    )

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

    # --------------------------------------------------------
    # Unpack
    # --------------------------------------------------------

    def unpack(x):

        pi_prime = x[:-1].reshape(
            M,
            S,
            A
        )

        t = x[-1]

        return pi_prime, t

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(x):

        pi_prime, t = unpack(x)

        proximal_penalty = (
            1.0 / (2.0 * mu)
        ) * np.sum(
            (pi_prime - pi) ** 2
        )

        return -(
            t - proximal_penalty
        )

    # --------------------------------------------------------
    # Evaluate Phi and Jc TOGETHER
    #
    # This is important.
    # Both constraints use exactly the same evaluation.
    # --------------------------------------------------------

    evaluation_cache = {
        "x": None,
        "Phi": None,
        "Jc": None
    }

    def evaluate_x(x):

        pi_prime, _ = unpack(x)

        # Cache based on the policy vector.
        # This avoids evaluating Phi/Jc twice for the
        # same SLSQP point.

        policy_flat = x[:-1]

        if (
            evaluation_cache["x"] is not None
            and np.array_equal(
                evaluation_cache["x"],
                policy_flat
            )
        ):
            return (
                evaluation_cache["Phi"],
                evaluation_cache["Jc"]
            )

        Phi_prime, Jc_prime = evaluate_policy_crn(
            problem,
            pi_prime,
            U
        )

        evaluation_cache["x"] = policy_flat.copy()
        evaluation_cache["Phi"] = Phi_prime
        evaluation_cache["Jc"] = Jc_prime

        return Phi_prime, Jc_prime

    # --------------------------------------------------------
    # Potential constraint
    # --------------------------------------------------------

    def potential_constraint(x):

        _, t = unpack(x)

        Phi_prime, _ = evaluate_x(x)

        return (
            Phi_prime / lam
        ) - t

    # --------------------------------------------------------
    # Cost constraint
    # --------------------------------------------------------

    def cost_constraint(x):

        _, t = unpack(x)

        _, Jc_prime = evaluate_x(x)

        return (
            b - Jc_prime
        ) - t

    # --------------------------------------------------------
    # Probability constraints
    # --------------------------------------------------------

    def probability_constraints(x):

        pi_prime, _ = unpack(x)

        return (
            np.sum(
                pi_prime,
                axis=2
            ) - 1.0
        ).ravel()

    # --------------------------------------------------------
    # Initial vector
    # --------------------------------------------------------

    x0 = np.concatenate(
        [
            pi.flatten(),
            [t0]
        ]
    )

    # --------------------------------------------------------
    # Check x0
    # --------------------------------------------------------

    print("\n========== CHECK x0 ==========")

    print(
        "Objective:",
        objective(x0)
    )

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
        np.max(
            np.abs(
                probability_constraints(x0)
            )
        )
    )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    bounds = (
        [(0.0, 1.0)] * n_policy
        +
        [(None, None)]
    )

    # --------------------------------------------------------
    # SLSQP
    # --------------------------------------------------------

    print(
        "\n========== RUNNING SLSQP WITH CRN =========="
    )

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

    # --------------------------------------------------------
    # Final solution
    # --------------------------------------------------------

    pi_hat, t_hat = unpack(result.x)

    Phi_hat, Jc_hat = evaluate_policy_crn(
        problem,
        pi_hat,
        U
    )

    pot_res = (
        Phi_hat / lam
    ) - t_hat

    cost_res = (
        b - Jc_hat
    ) - t_hat

    prob_res = np.max(
        np.abs(
            np.sum(
                pi_hat,
                axis=2
            ) - 1.0
        )
    )

    # --------------------------------------------------------
    # Print diagnosis
    # --------------------------------------------------------

    print(
        "\n========== FINAL CRN DIAGNOSIS =========="
    )

    print(
        f"success = {result.success}"
    )

    print(
        f"message = {result.message}"
    )

    print(
        f"iterations = {result.nit}"
    )

    print(
        f"function evaluations = {result.nfev}"
    )

    print(
        f"\nt_hat = {t_hat:.10f}"
    )

    print(
        f"Phi_hat = {Phi_hat:.10f}"
    )

    print(
        f"Jc_hat = {Jc_hat:.10f}"
    )

    print("\nConstraint residuals:")

    print(
        f"potential = {pot_res:.10f}"
    )

    print(
        f"cost      = {cost_res:.10f}"
    )

    print(
        f"prob max  = {prob_res:.10e}"
    )

    print("\nPolicy:")

    print(pi_hat)

    return result


# ============================================================
# 4. Same configuration as your experiment
# ============================================================

if __name__ == "__main__":

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

        init_pi = np.zeros(
            (M, 2, 2)
        )

        for i in range(M):

            for s in range(2):

                init_pi[i, s, 0] = 0.7
                init_pi[i, s, 1] = 0.3

    args = Args()

    problem = project_x_tabular(args)

    test_slsqp_crn(problem)