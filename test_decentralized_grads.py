"""
test_decentralized_grads_env.py

Standalone gradient test for the PollutionTaxEnv.

IMPORTANT:
    This script uses the SAME PollutionTaxEnv from envs.py.
    It does NOT recreate the environment dynamics.

It tests whether the Monte-Carlo likelihood-ratio gradient used
by the decentralized implementation agrees with finite differences
of the actual environment objective.

Run:
    python test_decentralized_grads_env.py
"""

import numpy as np

from envs import (
    PollutionTaxEnv
)


# ============================================================
# CONFIGURATION
# ============================================================

M = 2
N_STATES = 2
N_ACTIONS = 2

HORIZON = 10
GAMMA = 1.0

N_SAMPLES = 10000
SEED = 12345


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================

def make_env():
    """
    Create the exact PollutionTaxEnv from envs.py.
    """

    return PollutionTaxEnv(
        num_players=M,
        num_states=N_STATES,
        num_actions=N_ACTIONS,
    )


# ============================================================
# POLICY ACTION SAMPLING
# ============================================================

def sample_joint_action(Pi, state, rng):
    """
    Sample one action for every agent from the decentralized
    joint policy.

    Pi shape:
        (M, N_STATES, N_ACTIONS)
    """

    actions = []

    for i in range(M):

        probs = np.asarray(
            Pi[i, state],
            dtype=float,
        )

        action = rng.choice(
            N_ACTIONS,
            p=probs,
        )

        actions.append(int(action))

    return tuple(actions)


# ============================================================
# EXACT EVALUATION USING THE ACTUAL ENVIRONMENT
# ============================================================

def exact_evaluation(Pi, horizon=HORIZON, gamma=GAMMA):
    """
    Evaluate the policy exactly by enumerating all possible
    joint actions.

    IMPORTANT:
        Reward, cost, and transition are obtained from the
        actual envs.py implementation.

    No reward/cost/transition function is recreated here.

    Because PollutionTaxEnv is deterministic given state/action,
    we can propagate the state distribution exactly.
    """

    env = make_env()

    state_prob = np.zeros(N_STATES, dtype=float)

    # PollutionTaxEnv.reset() with no state starts FREE.
    initial_state = env.reset()

    state_prob[initial_state] = 1.0

    total_rewards = np.zeros(M, dtype=float)
    total_cost = 0.0

    # All possible joint actions.
    joint_actions = []

    for a0 in range(N_ACTIONS):
        for a1 in range(N_ACTIONS):
            joint_actions.append((a0, a1))

    for t in range(horizon):

        next_state_prob = np.zeros(
            N_STATES,
            dtype=float,
        )

        discount = gamma ** t

        for state in range(N_STATES):

            probability_state = state_prob[state]

            if probability_state == 0:
                continue

            for joint_action in joint_actions:

                # Probability of this joint action under
                # independent policies.
                probability_action = 1.0

                for i in range(M):

                    probability_action *= (
                        Pi[i, state, joint_action[i]]
                    )

                if probability_action == 0:
                    continue

                probability = (
                    probability_state
                    * probability_action
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Use the REAL environment.
                # ------------------------------------------------

                env.reset(state)

                next_state, rewards, cost, potential, done = (
                    env.step(joint_action)
                )

                rewards = np.asarray(
                    rewards,
                    dtype=float,
                )

                # Expected discounted reward.
                total_rewards += (
                    probability
                    * discount
                    * rewards
                )

                # Expected discounted cost.
                total_cost += (
                    probability
                    * discount
                    * float(cost)
                )

                # State transition.
                next_state_prob[next_state] += probability

        state_prob = next_state_prob

    return total_rewards, total_cost


# ============================================================
# MONTE-CARLO REINFORCE GRADIENT
# ============================================================

def reinforce_gradient(
    Pi,
    horizon=HORIZON,
    gamma=GAMMA,
    n_samples=N_SAMPLES,
    seed=SEED,
):
    """
    Monte-Carlo likelihood-ratio gradient.

    Uses:

        grad log pi(a|s) = 1 / pi(a|s)

    for the selected action coordinate.

    The environment is the actual PollutionTaxEnv from envs.py.
    """

    rng = np.random.default_rng(seed)

    reward_gradient = np.zeros_like(
        Pi,
        dtype=float,
    )

    cost_gradient = np.zeros_like(
        Pi,
        dtype=float,
    )

    reward_estimate = np.zeros(
        M,
        dtype=float,
    )

    cost_estimate = 0.0

    for sample in range(n_samples):

        env = make_env()

        state = env.reset()

        trajectory = []

        # --------------------------------------------------------
        # Generate trajectory
        # --------------------------------------------------------

        for t in range(horizon):

            actions = []

            log_prob_data = []

            for i in range(M):

                p_vector = np.asarray(
                    Pi[i, state],
                    dtype=float,
                )

                action = rng.choice(
                    N_ACTIONS,
                    p=p_vector,
                )

                actions.append(int(action))

                # Probability of the ACTION THAT WAS ACTUALLY
                # SAMPLED.
                p_action = float(
                    p_vector[action]
                )

                log_prob_data.append(
                    (
                        i,
                        state,
                        action,
                        p_action,
                    )
                )

            joint_action = tuple(actions)

            # ----------------------------------------------------
            # Actual environment
            # ----------------------------------------------------

            next_state, rewards, cost, potential, done = (
                env.step(joint_action)
            )

            rewards = np.asarray(
                rewards,
                dtype=float,
            )

            trajectory.append(
                {
                    "state": state,
                    "actions": joint_action,
                    "rewards": rewards,
                    "cost": float(cost),
                    "log_prob_data": log_prob_data,
                }
            )

            state = next_state

        # --------------------------------------------------------
        # Compute returns from every timestep.
        # --------------------------------------------------------

        reward_returns_from_t = np.zeros(
            (horizon, M),
            dtype=float,
        )

        cost_returns_from_t = np.zeros(
            horizon,
            dtype=float,
        )

        for t in range(horizon):

            for k in range(t, horizon):

                discount = gamma ** (k - t)

                reward_returns_from_t[t] += (
                    discount
                    * trajectory[k]["rewards"]
                )

                cost_returns_from_t[t] += (
                    discount
                    * trajectory[k]["cost"]
                )

        # --------------------------------------------------------
        # Gradient accumulation
        # --------------------------------------------------------

        for t in range(horizon):

            entry = trajectory[t]

            state = entry["state"]
            log_prob_data = entry["log_prob_data"]

            outer_discount = gamma ** t

            for (
                i,
                s,
                action,
                p_action,
            ) in log_prob_data:

                score = 1.0 / max(
                    p_action,
                    1e-12,
                )

                # ----------------------------------------------
                # Reward gradient
                # ----------------------------------------------

                reward_gradient[
                    i,
                    s,
                    action
                ] += (
                    outer_discount
                    * score
                    * reward_returns_from_t[t, i]
                )

                # ----------------------------------------------
                # Cost gradient
                # ----------------------------------------------

                cost_gradient[
                    i,
                    s,
                    action
                ] += (
                    outer_discount
                    * score
                    * cost_returns_from_t[t]
                )

        # --------------------------------------------------------
        # Objective estimates
        # --------------------------------------------------------

        trajectory_reward = np.zeros(
            M,
            dtype=float,
        )

        trajectory_cost = 0.0

        for t in range(horizon):

            discount = gamma ** t

            trajectory_reward += (
                discount
                * trajectory[t]["rewards"]
            )

            trajectory_cost += (
                discount
                * trajectory[t]["cost"]
            )

        reward_estimate += trajectory_reward
        cost_estimate += trajectory_cost

    # ------------------------------------------------------------
    # Average
    # ------------------------------------------------------------

    reward_gradient /= n_samples
    cost_gradient /= n_samples

    reward_estimate /= n_samples
    cost_estimate /= n_samples

    return (
        reward_gradient,
        cost_gradient,
        reward_estimate,
        cost_estimate,
    )


# ============================================================
# FINITE DIFFERENCE
# ============================================================

def finite_difference_directional_derivative(
    Pi,
    agent,
    state,
    direction,
    epsilon,
    objective,
):
    """
    Compute the directional derivative using:

        [J(Pi + eps*d) - J(Pi - eps*d)] / (2 eps)

    The direction is tangent to the probability simplex.

    For two actions:

        direction = [1, -1]

    means:

        increase P(action 0)
        decrease P(action 1)
    """

    Pi_plus = Pi.copy()
    Pi_minus = Pi.copy()

    Pi_plus[
        agent,
        state
    ] += epsilon * direction

    Pi_minus[
        agent,
        state
    ] -= epsilon * direction

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    if np.any(Pi_plus < 0):
        raise ValueError(
            "Pi_plus contains negative probabilities."
        )

    if np.any(Pi_minus < 0):
        raise ValueError(
            "Pi_minus contains negative probabilities."
        )

    if not np.isclose(
        Pi_plus[agent, state].sum(),
        1.0,
    ):
        raise ValueError(
            "Pi_plus does not lie on simplex."
        )

    if not np.isclose(
        Pi_minus[agent, state].sum(),
        1.0,
    ):
        raise ValueError(
            "Pi_minus does not lie on simplex."
        )

    # ------------------------------------------------------------
    # Exact evaluation using envs.py
    # ------------------------------------------------------------

    rewards_plus, cost_plus = exact_evaluation(
        Pi_plus
    )

    rewards_minus, cost_minus = exact_evaluation(
        Pi_minus
    )

    if objective == "reward":

        return (
            rewards_plus[agent]
            - rewards_minus[agent]
        ) / (2.0 * epsilon)

    elif objective == "cost":

        return (
            cost_plus
            - cost_minus
        ) / (2.0 * epsilon)

    else:
        raise ValueError(
            "objective must be 'reward' or 'cost'."
        )


# ============================================================
# GRADIENT CHECK
# ============================================================

def check_gradient(
    Pi,
    agent,
    state,
    epsilon,
    reward_gradient,
    cost_gradient,
):
    """
    Compare the finite-difference directional derivative
    with the Monte-Carlo gradient directional derivative.
    """

    # Valid tangent direction for a 2-action simplex.
    direction = np.array(
        [1.0, -1.0],
        dtype=float,
    )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    fd_reward = finite_difference_directional_derivative(
        Pi=Pi,
        agent=agent,
        state=state,
        direction=direction,
        epsilon=epsilon,
        objective="reward",
    )

    mc_reward = np.dot(
        reward_gradient[agent, state],
        direction,
    )

    reward_abs_error = abs(
        fd_reward - mc_reward
    )

    reward_relative_error = (
        reward_abs_error
        / max(
            1.0,
            abs(fd_reward),
            abs(mc_reward),
        )
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    fd_cost = finite_difference_directional_derivative(
        Pi=Pi,
        agent=agent,
        state=state,
        direction=direction,
        epsilon=epsilon,
        objective="cost",
    )

    mc_cost = np.dot(
        cost_gradient[agent, state],
        direction,
    )

    cost_abs_error = abs(
        fd_cost - mc_cost
    )

    cost_relative_error = (
        cost_abs_error
        / max(
            1.0,
            abs(fd_cost),
            abs(mc_cost),
        )
    )

    return {
        "fd_reward": fd_reward,
        "mc_reward": mc_reward,
        "reward_abs_error": reward_abs_error,
        "reward_relative_error": reward_relative_error,
        "fd_cost": fd_cost,
        "mc_cost": mc_cost,
        "cost_abs_error": cost_abs_error,
        "cost_relative_error": cost_relative_error,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    np.set_printoptions(
        precision=8,
        suppress=True,
    )

    # --------------------------------------------------------
    # Interior policy
    # --------------------------------------------------------

    Pi = np.zeros(
    (M, N_STATES, N_ACTIONS),
    dtype=float,
    )

    Pi[:, :, 0] = 1.0

    print("=" * 70)
    print("POLLUTION GAME — GRADIENT CHECK")
    print("USING THE ACTUAL envs.py")
    print("=" * 70)

    print("\nPolicy:")
    print(Pi)

    print("\nPolicy row sums:")
    print(Pi.sum(axis=2))

    # --------------------------------------------------------
    # Verify environment directly
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DIRECT ENVIRONMENT CHECK")
    print("-" * 70)

    env = make_env()

    test_actions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    for state in [0, 1]:

        print(f"\nState = {state}")

        for action in test_actions:

            env.reset(state)

            next_state, rewards, cost, potential, done = (
                env.step(action)
            )

            print(
                f"  action={action}"
                f"  reward={rewards}"
                f"  cost={cost}"
                f"  next_state={next_state}"
            )

    # --------------------------------------------------------
    # Exact objective
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EXACT OBJECTIVE")
    print("-" * 70)

    exact_rewards, exact_cost = exact_evaluation(
        Pi
    )

    print(
        "Expected discounted rewards:",
        exact_rewards,
    )

    print(
        "Expected discounted cost:",
        exact_cost,
    )

    # --------------------------------------------------------
    # Monte-Carlo gradient
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MONTE-CARLO GRADIENT")
    print("-" * 70)

    print(
        f"Number of trajectories: {N_SAMPLES}"
    )

    (
        reward_gradient,
        cost_gradient,
        mc_rewards,
        mc_cost,
    ) = reinforce_gradient(
        Pi=Pi,
        horizon=HORIZON,
        gamma=GAMMA,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    print("\nMonte-Carlo reward objective:")
    print(mc_rewards)

    print(
        "\nMonte-Carlo cost objective:",
        mc_cost,
    )

    print("\nMonte-Carlo reward gradient:")
    print(reward_gradient)

    print("\nMonte-Carlo cost gradient:")
    print(cost_gradient)

    # --------------------------------------------------------
    # Finite difference tests
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 70)

    epsilons = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
    ]

    for agent in range(M):

        for state in range(N_STATES):

            print("\n" + "-" * 70)

            print(
                f"Agent {agent}, State {state}"
            )

            print(
                "Policy:",
                Pi[agent, state],
            )

            print(
                "Direction: [ +1, -1 ]"
            )

            print("-" * 70)

            print(
                f"{'epsilon':>12}"
                f"{'FD reward':>18}"
                f"{'MC reward':>18}"
                f"{'FD cost':>18}"
                f"{'MC cost':>18}"
            )

            for epsilon in epsilons:

                result = check_gradient(
                    Pi=Pi,
                    agent=agent,
                    state=state,
                    epsilon=epsilon,
                    reward_gradient=reward_gradient,
                    cost_gradient=cost_gradient,
                )

                print(
                    f"{epsilon:12.1e}"
                    f"{result['fd_reward']:18.8f}"
                    f"{result['mc_reward']:18.8f}"
                    f"{result['fd_cost']:18.8f}"
                    f"{result['mc_cost']:18.8f}"
                )

    # --------------------------------------------------------
    # Detailed check
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DETAILED CHECK")
    print("=" * 70)

    epsilon = 1e-4

    for agent in range(M):

        for state in range(N_STATES):

            result = check_gradient(
                Pi=Pi,
                agent=agent,
                state=state,
                epsilon=epsilon,
                reward_gradient=reward_gradient,
                cost_gradient=cost_gradient,
            )

            print(
                f"\nAgent {agent}, State {state}"
            )

            print("\nReward:")

            print(
                "  finite difference =",
                result["fd_reward"],
            )

            print(
                "  MC gradient       =",
                result["mc_reward"],
            )

            print(
                "  absolute error    =",
                result["reward_abs_error"],
            )

            print(
                "  relative error    =",
                result["reward_relative_error"],
            )

            print("\nCost:")

            print(
                "  finite difference =",
                result["fd_cost"],
            )

            print(
                "  MC gradient       =",
                result["mc_cost"],
            )

            print(
                "  absolute error    =",
                result["cost_abs_error"],
            )

            print(
                "  relative error    =",
                result["cost_relative_error"],
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()