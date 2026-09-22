import numpy as np
from scipy.optimize import minimize

from multi_agent import project_x_tabular


# ============================================================
# Configuration
# ============================================================

class Args:
    env_nm = "pollution"

    T = 20
    horizon = 10

    gamma = 0.9
    mu = 0.8
    lamdba = 20

    eta = 0.1
    M = 2
    b = 12


args = Args()
problem = project_x_tabular(args)


# ============================================================
# Exact finite-horizon evaluator
# ============================================================

def evaluate_exact(pi):

    M = problem.num_players
    S = problem.S
    A = problem.A
    gamma = problem.gamma
    H = problem.horizon

    P = np.zeros((S, S))
    r_phi = np.zeros(S)
    r_cost = np.zeros(S)

    import itertools

    joint_actions = list(
        itertools.product(range(A), repeat=M)
    )

    for s in range(S):

        for joint_action in joint_actions:

            # Probability of joint action
            prob = 1.0

            for i in range(M):
                prob *= pi[i, s, joint_action[i]]

            if prob == 0:
                continue

            problem.env.reset(state=s)

            next_state, rewards, cost, potential, done = \
                problem.env.step(list(joint_action))

            r_phi[s] += prob * potential
            r_cost[s] += prob * cost

            P[s, next_state] += prob

    # Finite horizon
    V_phi = r_phi.copy()
    V_cost = r_cost.copy()

    for _ in range(1, H):
        V_phi = r_phi + gamma * P @ V_phi
        V_cost = r_cost + gamma * P @ V_cost

    return np.mean(V_phi), np.mean(V_cost)


# ============================================================
# Initial policy
# ============================================================

pi0 = np.array([
    [[0.7, 0.3],
     [0.7, 0.3]],

    [[0.7, 0.3],
     [0.7, 0.3]]
], dtype=float)


Phi0, Jc0 = evaluate_exact(pi0)

t0 = min(
    Phi0 / args.lamdba,
    args.b - Jc0
)

print("\n========== EXACT INITIAL POINT ==========")

print("pi =")
print(pi0)

print(f"Phi = {Phi0:.10f}")
print(f"Jc  = {Jc0:.10f}")
print(f"t0  = {t0:.10f}")


# ============================================================
# Reduced parameterization
#
# x contains:
#
# p[0,0], p[0,1],
# p[1,0], p[1,1],
# t
#
# pi[i,s,0] = p[i,s]
# pi[i,s,1] = 1-p[i,s]
# ============================================================

def unpack(x):

    p = x[:-1].reshape(M, S)
    t = x[-1]

    pi = np.zeros((M, S, 2))

    pi[:, :, 0] = p
    pi[:, :, 1] = 1.0 - p

    return pi, t


M = problem.num_players
S = problem.S


# ============================================================
# Objective
# ============================================================

def objective(x):

    pi, t = unpack(x)

    proximal_penalty = (
        1.0 / (2.0 * args.mu)
    ) * np.sum((pi - pi0) ** 2)

    return -(t - proximal_penalty)


# ============================================================
# Potential constraint
#
# Phi(pi)/lambda - t >= 0
# ============================================================

def potential_constraint(x):

    pi, t = unpack(x)

    Phi, _ = evaluate_exact(pi)

    return Phi / args.lamdba - t


# ============================================================
# Cost constraint
#
# b - Jc(pi) - t >= 0
# ============================================================

def cost_constraint(x):

    pi, t = unpack(x)

    _, Jc = evaluate_exact(pi)

    return args.b - Jc - t


# ============================================================
# Initial x
# ============================================================

p0 = pi0[:, :, 0]

x0 = np.concatenate([
    p0.flatten(),
    [t0]
])


# ============================================================
# Bounds
#
# probabilities in [0,1]
# t is free
# ============================================================

bounds = (
    [(0.0, 1.0)] * (M * S)
    + [(None, None)]
)


# ============================================================
# Check initial point
# ============================================================

print("\n========== CHECK x0 ==========")

print(f"Objective = {objective(x0):.12f}")
print(
    f"Potential constraint = "
    f"{potential_constraint(x0):.12f}"
)
print(
    f"Cost constraint = "
    f"{cost_constraint(x0):.12f}"
)


# ============================================================
# Run SLSQP
# ============================================================

print("\n========== RUNNING EXACT SLSQP ==========")

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=[
        {
            "type": "ineq",
            "fun": potential_constraint
        },
        {
            "type": "ineq",
            "fun": cost_constraint
        }
    ],
    options={
        "maxiter": 1000,
        "ftol": 1e-10,
        "disp": True
    }
)


# ============================================================
# Final result
# ============================================================

pi_hat, t_hat = unpack(result.x)

Phi_hat, Jc_hat = evaluate_exact(pi_hat)

print("\n========== RESULT ==========")

print("success     =", result.success)
print("message     =", result.message)
print("iterations  =", result.nit)
print("evaluations =", result.nfev)

print(f"\nt_hat       = {t_hat:.12f}")
print(f"Phi_hat     = {Phi_hat:.12f}")
print(f"Jc_hat      = {Jc_hat:.12f}")

print(
    f"objective   = "
    f"{objective(result.x):.12f}"
)

print("\nConstraint residuals:")

print(
    f"potential   = "
    f"{potential_constraint(result.x):.12e}"
)

print(
    f"cost        = "
    f"{cost_constraint(result.x):.12e}"
)

print("\nOriginal policy:")
print(pi0)

print("\nProximal policy:")
print(pi_hat)

print("\nPolicy change:")
print(pi_hat - pi0)