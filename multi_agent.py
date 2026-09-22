import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from itertools import product,combinations
from envs import (
    DemandResponseMarketEnv,
    PollutionTaxEnv,
    multi_gridW,
    congestion_game,
    POLLUTION_FREE,
    POLLUTED,
    CLEAN,
    PROFIT_PER_ITEM,
    TAX_BY_NUM_PLAYERS,
)
from scipy.optimize import minimize
import pickle

class Player:
    def __init__(self, env):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.policy = (np.ones((self.num_states, self.num_actions) )/ self.num_actions)
        if isinstance(env, PollutionTaxEnv):
            self.policy[:, 0] = 0.7
            self.policy[:, 1] = 0.3
        elif isinstance(env, DemandResponseMarketEnv):
            self.policy[:, 0] = 1.0
            self.policy[:, 1:] = 0.0
    def get_action(self, state):
        return np.random.choice(self.num_actions,p=self.policy[state])
    def get_policy(self):
        return self.policy

class project_x_tabular:
    def __init__(self, args):
        self.args = args
        self.T = args.T
        self.horizon = args.horizon
        self.gamma = args.gamma
        self.mu = args.mu
        self.lam = args.lamdba
        self.eta = args.eta
        self.num_players = args.M
        self.b = args.b
        self.S,self.A,self.num_joint_actions = None,None,None
        if args.env_nm == "pollution":
            self.env = PollutionTaxEnv(num_players=self.num_players)
            self.stepsize = {2: 0.005,4: 0.002,8: 0.0007}[self.num_players]
            self.num_samples = {2: 1000, 4: 1000, 8: 2500}[self.num_players]
            self.max_iters = 20
            self.S = 2
            self.A = 2
            self.num_joint_actions = self.A**self.num_players
        elif args.env_nm == "energy":
            self.env = DemandResponseMarketEnv(num_players=self.num_players)
            self.stepsize = {2: 0.002,4: 0.001,8: 0.0003}[self.num_players]
            self.num_samples = {2: 100,4: 150,8: 100}[self.num_players]
            self.max_iters = 30
            self.S = 5
            self.A = 5
            self.num_joint_actions = self.A**self.num_players
        elif args.env_nm == "grid":
            self.env = multi_gridW()
            self.stepsize = 0.002
            self.num_samples = 10
            self.max_iters = 60
            self.gs = len(self.env.grid_objs[0].grid)
            self.S = self.gs*self.gs
            self.A = 4
            self.num_joint_actions = self.A**2 # 2 agents
        elif args.env_nm == "congestion":
            self.env = congestion_game()
            self.stepsize = 0.001
            self.num_samples = 1000
            self.max_iters = 10
            self.S = 2
            self.A = len(self.env.actions)
        else:
            raise ValueError(f"Unknown environment: {args.env_nm}")
        self.players = [Player(self.env) for _ in range(self.num_players)]
        self.pi = np.array([player.get_policy() for player in self.players])
        self.store_values = {'potential': [],'Cost': []}
    def play_episode(self):
        pot_expected = 0.0
        cost_expected = 0.0
        for _ in range(self.num_samples):
            pot_exp = []
            cost_exp = []
            state = self.env.reset()
            for t in range(self.T):
                joint_action = [player.get_action(state) for player in self.players]
                new_state, rewards, cost, potential, done = self.env.step(joint_action)
                pot_exp.append(potential)
                cost_exp.append(cost)
                state = new_state
                if done:
                    break
            pot_exp = np.asarray(pot_exp)
            cost_exp = np.asarray(cost_exp)
            discount = self.gamma ** np.arange(len(pot_exp))
            pot_expected += np.dot(pot_exp,discount)
            cost_expected += np.dot(cost_exp,discount)
        pot_expected /= self.num_samples
        cost_expected /= self.num_samples
        return pot_expected, cost_expected
    def evaluate_policy_exact(self, pi):
        """
        Exact finite-horizon policy evaluation for the Energy environment.

        The Energy environment has stochastic transitions through
            w ~ Uniform{0, ..., S-1}
        and a 0.9 / 0.1 transition mixture.

        We explicitly enumerate w, so there is no Monte-Carlo noise.
        """

        M = self.num_players
        S = self.S
        A = self.A
        gamma = self.gamma
        H = self.horizon

        pi = np.asarray(pi, dtype=float)

        # ---------------------------------------------------------
        # Sanity check
        # ---------------------------------------------------------
        if isinstance(self.env, DemandResponseMarketEnv):
            if S != 5 or A != 5:
                raise ValueError(
                    f"Energy environment requires S=5, A=5, "
                    f"but got S={S}, A={A}"
                )

        # Policy-induced transition matrix
        P_pi = np.zeros((S, S))

        # Expected immediate potential and cost
        r_phi = np.zeros(S)
        r_cost = np.zeros(S)

        joint_actions = list(product(range(A), repeat=M))

        # ---------------------------------------------------------
        # Construct P_pi, r_phi, r_cost
        # ---------------------------------------------------------
        for s in range(S):

            for joint_action in joint_actions:

                # Probability of joint action under independent policies
                prob_action = 1.0

                for i in range(M):
                    prob_action *= pi[i, s, joint_action[i]]

                if prob_action <= 0.0:
                    continue

                # =================================================
                # ENERGY ENVIRONMENT
                # =================================================
                if isinstance(self.env, DemandResponseMarketEnv):

                    action_sum = sum(joint_action)

                    # ---------------------------------------------
                    # Enumerate all possible w values.
                    #
                    # w is uniform over {0,...,S-1}.
                    # ---------------------------------------------
                    for w in range(S):

                        prob_w = 1.0 / S

                        # -----------------------------------------
                        # Branch 1: probability 0.9
                        # -----------------------------------------
                        next_state_1 = round(
                            2 * action_sum / M + w
                        )

                        next_state_1 = max(
                            min(next_state_1, S - 1),
                            0
                        )

                        prob_transition_1 = (
                            prob_action * prob_w * 0.9
                        )

                        # Potential uses the NEXT state
                        potential_1 = (
                            2 * action_sum
                            - 0.25 * sum(a**2 for a in joint_action)
                            - 0.25 * sum(
                                a1 * a2
                                for a1, a2 in combinations(joint_action, 2)
                            )
                            - M * (1.25 ** next_state_1)
                        )

                        cost = action_sum

                        P_pi[s, next_state_1] += prob_transition_1
                        r_phi[s] += prob_transition_1 * potential_1
                        r_cost[s] += prob_transition_1 * cost

                        # -----------------------------------------
                        # Branch 2: probability 0.1
                        # -----------------------------------------
                        next_state_2 = w

                        prob_transition_2 = (
                            prob_action * prob_w * 0.1
                        )

                        potential_2 = (
                            2 * action_sum
                            - 0.25 * sum(a**2 for a in joint_action)
                            - 0.25 * sum(
                                a1 * a2
                                for a1, a2 in combinations(joint_action, 2)
                            )
                            - M * (1.25 ** next_state_2)
                        )

                        P_pi[s, next_state_2] += prob_transition_2
                        r_phi[s] += prob_transition_2 * potential_2
                        r_cost[s] += prob_transition_2 * cost

                # =================================================
                # POLLUTION ENVIRONMENT
                # =================================================
                elif isinstance(self.env, PollutionTaxEnv):

                    self.env.reset(state=s)

                    (
                        next_state,
                        rewards,
                        cost,
                        potential,
                        done
                    ) = self.env.step(list(joint_action))

                    P_pi[s, next_state] += prob_action
                    r_phi[s] += prob_action * potential
                    r_cost[s] += prob_action * cost

                else:
                    raise NotImplementedError(
                        "Exact evaluation is not implemented for "
                        f"{type(self.env).__name__}"
                    )

        # ---------------------------------------------------------
        # Check that transition matrix is stochastic
        # ---------------------------------------------------------
        row_sums = np.sum(P_pi, axis=1)

        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise RuntimeError(
                f"Invalid transition matrix. Row sums = {row_sums}"
            )

        # ---------------------------------------------------------
        # Finite-horizon Bellman recursion
        #
        # V_1(s) = r(s)
        #
        # V_h(s) = r(s) + gamma P V_{h-1}(s)
        # ---------------------------------------------------------
        V_phi = r_phi.copy()
        V_cost = r_cost.copy()

        for _ in range(1, H):
            V_phi = r_phi + gamma * P_pi @ V_phi
            V_cost = r_cost + gamma * P_pi @ V_cost

        # ---------------------------------------------------------
        # Average over all starting states
        # ---------------------------------------------------------
        Phi = np.mean(V_phi)
        Jc = np.mean(V_cost)

        return Phi, Jc
    # def evaluate_policy_exact(self, pi):
    #     """
    #     Exact finite-horizon policy evaluation.

    #     Used only inside the SLSQP proximal optimization.

    #     The evaluation:
    #         1. Enumerates all joint actions.
    #         2. Constructs the policy-induced transition matrix.
    #         3. Computes expected immediate potential and cost.
    #         4. Performs finite-horizon Bellman recursion.
    #         5. Averages the resulting values over all starting states.

    #     This removes Monte-Carlo noise from the SLSQP constraints.
    #     """

    #     M = self.num_players
    #     S = self.S
    #     A = self.A
    #     gamma = self.gamma
    #     H = self.horizon

    #     pi = np.asarray(pi, dtype=float)

    #     # ---------------------------------------------------------
    #     # For the current exact implementation, use only for
    #     # environments with manageable deterministic transitions.
    #     # ---------------------------------------------------------
    #     # if not isinstance(self.env, PollutionTaxEnv):
    #     #     raise NotImplementedError(
    #     #         "Exact evaluation is currently implemented "
    #     #         "only for PollutionTaxEnv."
    #     #     )

    #     P_pi = np.zeros((S, S))
    #     r_phi = np.zeros(S)
    #     r_cost = np.zeros(S)

    #     joint_actions = list(product(range(A), repeat=M))

    #     # ---------------------------------------------------------
    #     # Construct policy-induced transition matrix and
    #     # expected one-step potential/cost.
    #     # ---------------------------------------------------------
    #     for s in range(S):

    #         for joint_action in joint_actions:

    #             # Probability of joint action under independent
    #             # player policies.
    #             prob = 1.0

    #             for i in range(M):
    #                 prob *= pi[i, s, joint_action[i]]

    #             if prob <= 0.0:
    #                 continue

    #             # Environment transition from state s
    #             self.env.reset(state=s)

    #             (
    #                 next_state,
    #                 rewards,
    #                 cost,
    #                 potential,
    #                 done
    #             ) = self.env.step(list(joint_action))

    #             P_pi[s, next_state] += prob
    #             r_phi[s] += prob * potential
    #             r_cost[s] += prob * cost

    #     # ---------------------------------------------------------
    #     # Finite-horizon Bellman recursion
    #     #
    #     # V_0(s) = immediate expected reward
    #     #
    #     # V_h(s) = r(s) + gamma * sum_s' P(s,s') V_{h-1}(s')
    #     # ---------------------------------------------------------

    #     V_phi = r_phi.copy()
    #     V_cost = r_cost.copy()

    #     for _ in range(1, H):
    #         V_phi = r_phi + gamma * P_pi @ V_phi
    #         V_cost = r_cost + gamma * P_pi @ V_cost

    #     # ---------------------------------------------------------
    #     # Same state averaging convention as evaluate_policy()
    #     # ---------------------------------------------------------

    #     Phi = np.mean(V_phi)
    #     Jc = np.mean(V_cost)

    #     return Phi, Jc
    def evaluate_policy(self, pi):
        """
        iProx-CMPG-compatible policy evaluation.

        Evaluates the policy starting from EVERY state, using Monte Carlo
        trajectories of length T, and averages the resulting values across
        states.

        Returns:
            Phi: average discounted potential over starting states
            Jc:  average discounted cost over starting states
        """

        pi = np.asarray(pi, dtype=float).copy()

        # Numerical protection for SLSQP
        pi = np.clip(pi, 0.0, 1.0)

        row_sums = np.sum(pi, axis=2, keepdims=True)
        pi /= np.maximum(row_sums, 1e-15)

        potential_values = {}
        cost_values = {}

        for state in range(self.S):

            potential_total = 0.0
            cost_total = 0.0

            for _ in range(self.num_samples):

                current_state = self.env.reset(state=state)

                episode_potential = 0.0
                episode_cost = 0.0

                for t in range(self.horizon):

                    # Sample one action for every player
                    joint_action = [
                        np.random.choice(
                            self.A,
                            p=pi[i, current_state]
                        )
                        for i in range(self.num_players)
                    ]

                    (
                        new_state,
                        rewards,
                        cost,
                        potential,
                        done
                    ) = self.env.step(joint_action)

                    episode_potential += (self.gamma ** t) * potential
                    episode_cost += (self.gamma ** t) * cost

                    current_state = new_state

                    if done:
                        break

                potential_total += episode_potential
                cost_total += episode_cost

            # Value function for this starting state
            potential_values[state] = (
                potential_total / self.num_samples
            )

            cost_values[state] = (
                cost_total / self.num_samples
            )

        # This is important:
        # iProx averages the value functions over all states.
        Phi = np.mean(list(potential_values.values()))
        Jc = np.mean(list(cost_values.values()))

        return Phi, Jc
    # def evaluate_policy(self, pi):

    #     if not isinstance(self.env, PollutionTaxEnv):
    #         raise NotImplementedError("Exact evaluation is currently implemented only for PollutionTaxEnv.")

    #     M = self.num_players
    #     S = self.S
    #     A = self.A
    #     gamma = self.gamma

    #     # ---------------------------------------------------------
    #     # 1. Make sure policy has the correct shape
    #     # ---------------------------------------------------------
    #     pi = np.asarray(pi, dtype=float).copy()

    #     # Numerical protection for SLSQP finite-difference points
    #     pi = np.clip(pi, 0.0, 1.0)

    #     row_sums = np.sum(pi, axis=2, keepdims=True)
    #     pi /= np.maximum(row_sums, 1e-15)

    #     # ---------------------------------------------------------
    #     # 2. Enumerate all joint actions
    #     # ---------------------------------------------------------
    #     from itertools import product

    #     joint_actions = list(product(range(A), repeat=M))

    #     # ---------------------------------------------------------
    #     # 3. Construct P_pi, r_pi, c_pi
    #     # ---------------------------------------------------------
    #     P_pi = np.zeros((S, S))
    #     r_pi = np.zeros(S)
    #     c_pi = np.zeros(S)

    #     for s in range(S):

    #         for joint_action in joint_actions:

    #             # Probability of this joint action
    #             prob = 1.0

    #             for i in range(M):
    #                 prob *= pi[i, s, joint_action[i]]

    #             if prob == 0:
    #                 continue

    #             # -------------------------------------------------
    #             # Let the environment define reward/cost/transition
    #             # -------------------------------------------------
    #             self.env.reset(state=s)

    #             next_state, rewards, cost, potential, done = \
    #                 self.env.step(list(joint_action))

    #             # Transition probability
    #             P_pi[s, next_state] += prob

    #             # Expected potential
    #             r_pi[s] += prob * potential

    #             # Expected cost
    #             c_pi[s] += prob * cost

    #     # ---------------------------------------------------------
    #     # 4. Solve Bellman equations
    #     #
    #     # V_phi = r_pi + gamma P_pi V_phi
    #     #
    #     # => (I - gamma P_pi)V_phi = r_pi
    #     # ---------------------------------------------------------
    #     I = np.eye(S)

    #     V_phi = np.linalg.solve(
    #         I - gamma * P_pi,
    #         r_pi
    #     )

    #     V_cost = np.linalg.solve(
    #         I - gamma * P_pi,
    #         c_pi
    #     )

    #     # ---------------------------------------------------------
    #     # 5. Initial state
    #     #
    #     # PollutionTaxEnv.reset() starts at state 0
    #     # ---------------------------------------------------------
    #     d0 = np.zeros(S)
    #     d0[POLLUTION_FREE] = 1.0

    #     Phi = d0 @ V_phi
    #     Jc = d0 @ V_cost

    #     return Phi, Jc
        
    
    # def evaluate_policy(self, pi):
    #     pi_eval = np.asarray(pi, dtype=float).copy()
    #     pi_eval = np.clip(pi_eval, 0.0, 1.0)

    #     row_sums = np.sum(pi_eval, axis=2, keepdims=True)
    #     pi_eval /= np.maximum(row_sums, 1e-15)
    #     potential_total = 0.0
    #     cost_total = 0.0
    #     #print("Pi:",pi_eval)
    #     for _ in range(self.num_samples):
    #         state = self.env.reset()
    #         episode_potential = 0.0
    #         episode_cost = 0.0
    #         for ti in range(self.T):
    #             # Sample one action for each player according to pi
    #             joint_action = [
    #                 np.random.choice(
    #                     self.A,
    #                     p=pi_eval[i, state]
    #                 )
    #                 for i in range(self.num_players)
    #             ]
    #             # Let the environment define everything:
    #             # transition, reward, cost, potential, termination
    #             new_state, rewards, cost, potential, done = \
    #                 self.env.step(joint_action)
    #             discount = self.gamma ** ti
    #             episode_potential += discount * potential
    #             episode_cost += discount * cost
    #             state = new_state
    #             if done:
    #                 break
    #         potential_total += episode_potential
    #         cost_total += episode_cost
    #     Phi = potential_total / self.num_samples
    #     Jc = cost_total / self.num_samples
    #     return Phi, Jc

    def solve_proximal_problem(self, pi):
        mu = self.mu
        lam = self.lam
        n_policy = self.num_players * self.S * self.A
        def unpack(x):
            pi_prime = x[:-1].reshape(
                self.num_players,
                self.S,
                self.A
            )
            t = x[-1]
            return pi_prime, t
        def objective(x):
            pi_prime, t = unpack(x)
            proximal_penalty = (1.0 / (2.0 * mu)) * np.sum((pi_prime - pi) ** 2)
            return -(t - proximal_penalty)

        def potential_constraint(x):
            pi_prime, t = unpack(x)
            Phi_prime, _ = self.evaluate_policy_exact(pi_prime)
            return Phi_prime / lam - t

        def cost_constraint(x):
            pi_prime, t = unpack(x)
            _, Jc_prime = self.evaluate_policy_exact(pi_prime)
            return self.b - Jc_prime - t

        def probability_constraints(x):
            pi_prime, _ = unpack(x)
            return (np.sum(pi_prime, axis=2) - 1.0).ravel()

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
        # Evaluate current policy
        phi_current, jc_current = self.evaluate_policy(pi)

        print("INITIAL POLICY:")
        print(pi)

        print(f"MC Initial Phi = {phi_current}")
        print(f"MC Initial Jc  = {jc_current}")

        # Exact evaluation only for constructing the SLSQP
        # feasible starting value.
        phi_exact, jc_exact = self.evaluate_policy_exact(pi)

        print(f"Exact Initial Phi = {phi_exact}")
        print(f"Exact Initial Jc  = {jc_exact}")

        t0 = min(
            phi_exact / lam,
            self.b - jc_exact
        )
        x0 = np.concatenate([pi.flatten(),[t0]])
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-8,
                "disp": False
            }
        )

        if not result.success:
            print("SLSQP warning:", result.message)

        pi_hat, t_hat = unpack(result.x)

        # Exact evaluation must be used to diagnose the SLSQP solution
        # because the SLSQP constraints were defined using it.
        phi_hat, jc_hat = self.evaluate_policy_exact(pi_hat)

        prob_residual = np.sum(pi_hat, axis=2) - 1.0
        potential_residual = phi_hat / lam - t_hat
        cost_residual = self.b - jc_hat - t_hat

        print(
            f"Diagnosis---SLSQP: success={result.success}, "
            f"message='{result.message}', "
            f"t={t_hat:.6f}, "
            f"Phi={phi_hat:.6f}, "
            f"Jc={jc_hat:.6f}, "
            f"prob_max={np.max(np.abs(prob_residual)):.3e}, "
            f"pot_res={potential_residual:.6f}, "
            f"cost_res={cost_residual:.6f}"
        )

        return pi_hat, t_hat, result

    def run_algo_tabular(self):
        pi = np.array(self.args.init_pi, dtype=float)
        for ti in tqdm(range(self.args.T)):
            # Evaluate current policy
            Phi, Jc = self.evaluate_policy(pi)

            self.store_values["potential"].append(Phi)
            self.store_values["Cost"].append(Jc)

            # print("Current pi:")
            # print(pi)

            pi_hat, t_hat, result = self.solve_proximal_problem(pi)

            # print("Pi_hat:")
            # print(pi_hat)

            alpha = self.eta / self.mu
            pi = (1 - alpha) * pi + alpha * pi_hat

            # print("Updated pi:")
            # print(pi)

            # # Proximal update
            # alpha = self.eta / self.mu

            # pi = (1 - alpha) * pi + alpha * pi_hat

            # Numerical cleanup
            pi = np.clip(pi, 0.0, 1.0)

            # Re-normalize probabilities
            pi /= np.sum(pi, axis=2, keepdims=True)

            # if ti % 1 == 0:
            #     print(
            #         f"Iteration {ti}: "
            #         f"Phi={Phi:.6f}, "
            #         f"Jc={Jc:.6f}, "
            #         f"t={t_hat:.6f}, "
            #         f"SLSQP success={result.success}"
            #     )

        data = pd.DataFrame(self.store_values)
        print(data.head())
        try:
            data.to_excel(
                "Multi_agent_" + self.args.env_nm + ".xlsx",
                index=False
            )
        except Exception:
            if self.args.env_nm == 'pollution':
                if self.args.M==2:
                    with open("storage_data_2_agents_pollution.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                elif self.args.M==4:
                    with open("storage_data_4_agents_pollution.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                elif self.args.M==8:
                    with open("storage_data_8_agents_pollution.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                else:
                    print("Invalid choice")
            elif self.args.env_nm == 'energy':
                if self.args.M==2:
                    with open("storage_data_2_agents_energy.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                elif self.args.M==4:
                    with open("storage_data_4_agents_energy.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                elif self.args.M==8:
                    with open("storage_data_8_agents_energy.pkl", "wb") as f:
                        pickle.dump(self.store_values, f)
                    f.close()
                else:
                    print("Invalid choice")
        print("number of agents:",self.args.M)
        return pi