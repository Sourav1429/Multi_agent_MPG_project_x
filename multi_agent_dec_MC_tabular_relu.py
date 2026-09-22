import pickle
import numpy as np
import pandas as pd
from scipy.special import expit

from envs import DemandResponseMarketEnv, PollutionTaxEnv

def project_to_simplex(x, z=1.0):
    """Euclidean projection of a vector onto {p >= 0, sum(p)=z}."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("project_to_simplex expects a 1-D vector")
    if x.size == 0:
        return x.copy()

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, x.size + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        # This should not happen for finite x, but keeps the function safe.
        return np.ones_like(x) * (z / x.size)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(x - theta, 0.0)

class Player:
    """One tabular policy pi_i of shape (nS, nA)."""

    def __init__(self, env, initial_policy=None):
        self.num_states = env.num_states
        self.num_actions = env.num_actions

        if initial_policy is None:
            self.policy = np.ones((self.num_states, self.num_actions), dtype=float) / self.num_actions
        else:
            self.policy = np.asarray(initial_policy, dtype=float).copy()
            if self.policy.shape != (self.num_states, self.num_actions):
                raise ValueError(
                    f"Initial policy has shape {self.policy.shape}; "
                    f"expected {(self.num_states, self.num_actions)}."
                )
            init_epsilon = float(getattr(getattr(env, "_algorithm_args", None), "init_epsilon", 0.0))
            # A score-function policy gradient requires pi(a|s) > 0 for actions
            # that we want to explore.  Blend with uniform only when requested.
            if init_epsilon > 0.0:
                self.policy = (1.0 - init_epsilon) * self.policy + (init_epsilon / self.num_actions)
            for s in range(self.num_states):
                self.policy[s] = project_to_simplex(self.policy[s])
    def get_action(self, state):
        return np.random.choice(self.num_actions, p=self.policy[state])

    def get_policy(self):
        return self.policy

class decentralized_relu_Prject_x:
  def __init__(self,args):
    self.args = args
    if args.env_nm.lower() == "energy":
            self.env = DemandResponseMarketEnv(num_players=self.M,num_states=self.nS,num_actions=self.nA,)
    elif args.env_nm.lower() == "pollution":
        self.env = PollutionTaxEnv(num_players=self.M,num_states=self.nS,num_actions=self.nA,)
    else:
        raise ValueError(f"Unknown environment '{args.env_nm}'. "
            "Use 'energy' or 'pollution'."
        )
    supplied_init = getattr(args, "init_pi", None)
    self.agent_list = []

    if supplied_init is not None:
        supplied_init = np.asarray(supplied_init, dtype=float)
        expected = (self.M, self.nS, self.nA)
        if supplied_init.shape != expected:
            raise ValueError(
                f"init_pi has shape {supplied_init.shape}; expected {expected}."
            )

    # Player uses this only to optionally make a deterministic initial policy
    # slightly exploratory.  Default is 0, preserving the supplied policy.
    self.env._algorithm_args = args

    for i in range(self.M):
        init_i = None if supplied_init is None else supplied_init[i]
        self.agent_list.append(Player(self.env, init_i))

    self.individual_vf = {}
    self.Cf = []
    self.history = []
  def get_joint_policy(self):
        """Return Pi = [pi_1, ..., pi_M], shape (M, nS, nA)."""
        return np.stack([p.get_policy() for p in self.agent_list], axis=0)
        #pass

  def project_pol(self, pol, z=1.0):
      """Compatibility wrapper for the original project_pol method."""
      return project_to_simplex(pol, z)
      #pass
  def exact_evaluate_pollution(self, Pi):

        M = self.M
        gamma = self.gamma
        H = self.horizon

        J_rewards = np.zeros(M)
        J_cost = 0.0

        # Start pollution-free
        state_distribution = np.array([1.0, 0.0])

        tax = TAX_BY_NUM_PLAYERS[M]

        for t in range(H):

            next_state_distribution = np.zeros(2)

            for state in range(2):

                state_prob = state_distribution[state]

                if state_prob == 0:
                    continue

                for a0 in range(2):
                    for a1 in range(2):

                        joint_action = [a0, a1]

                        prob = (
                            Pi[0, state, a0]
                            * Pi[1, state, a1]
                        )

                        if prob == 0:
                            continue

                        # ---------------------------
                        # Rewards
                        # ---------------------------

                        rewards = []

                        for a in joint_action:

                            base_reward = (
                                PROFIT_PER_ITEM
                                if a == CLEAN
                                else 2 * PROFIT_PER_ITEM
                            )

                            pollution_tax = (
                                tax if state == POLLUTED
                                else 0
                            )

                            rewards.append(
                                base_reward - pollution_tax
                            )

                        # ---------------------------
                        # Cost
                        # ---------------------------

                        cost = (
                            2.0
                            * (M - sum(joint_action))
                            / M
                        )

                        weight = (
                            (gamma ** t)
                            * state_prob
                            * prob
                        )

                        for i in range(M):
                            J_rewards[i] += (
                                weight * rewards[i]
                            )

                        J_cost += weight * cost

                        # ---------------------------
                        # Transition
                        # ---------------------------

                        if a0 == CLEAN and a1 == CLEAN:
                            next_state = POLLUTION_FREE
                        else:
                            next_state = POLLUTED

                        next_state_distribution[next_state] += (
                            state_prob * prob
                        )

            state_distribution = next_state_distribution

        return J_rewards, J_cost

        def policy_evaluator(self, Pi=None):
          """
          Estimate J_r1,...,J_rM and J_c under the joint policy Pi.

          This is the Monte-Carlo implementation of Algorithm 2, lines 5-6.
          Each trajectory starts from a uniformly selected state.
          """
          if Pi is None:
              Pi = self.get_joint_policy()
          Pi = np.asarray(Pi, dtype=float)

          J_rewards = np.zeros(self.M, dtype=float)
          J_cost = 0.0
          J_pot = 0.0

          for _ in range(self.n_samples):
              state = 0
              self.env.reset(state)

              for t in range(self.horizon):
                  actions = [
                      np.random.choice(self.nA, p=Pi[i, state])
                      for i in range(self.M)
                  ]

                  next_state, rewards, cost, potential, done = self.env.step(actions)
                  discount = self.gamma ** t

                  J_rewards += discount * np.asarray(rewards, dtype=float)
                  J_cost += discount * float(cost)
                  J_pot += discount*float(potential)

                  state = int(next_state)
                  if done:
                      break

          J_rewards /= self.n_samples
          J_cost /= self.n_samples
          J_pot /= self.n_samples
          return J_rewards, J_cost,J_pot
        def policy_gradient(self, Pi, agent_idx, objective="reward"):
          """
          Estimate the score-function gradient in Algorithm 2, lines 7-8:

              E[grad log pi_i(a_i|s) * reward_i]
              E[grad log pi_i(a_i|s) * cost]

          The result has shape (nS, nA).
          """
          if objective not in ("reward", "cost"):
              raise ValueError("objective must be 'reward' or 'cost'")

          grad = np.zeros((self.nS, self.nA), dtype=float)

          for _ in range(self.n_samples):
              state = np.random.randint(self.nS)
              self.env.reset(state)

              for t in range(self.horizon):
                  actions = []
                  for i in range(self.M):
                      actions.append(
                          np.random.choice(self.nA, p=Pi[i, state])
                      )

                  next_state, rewards, cost, _, done = self.env.step(actions)
                  a_i = int(actions[agent_idx])
                  p = float(Pi[agent_idx, state, a_i])

                  # grad log pi(a|s) with respect to the tabular probability
                  # coordinates is 1/pi(a|s) for the sampled action.
                  score = 1.0 / max(p, 1e-12)
                  signal = (
                      float(rewards[agent_idx])
                      if objective == "reward"
                      else float(cost)
                  )

                  # Discounted policy-gradient estimator.
                  grad[state, a_i] += (self.gamma ** t) * score * signal

                  state = int(next_state)
                  if done:
                      break

          grad /= self.n_samples
          return grad
        def policy_gradient(self, Pi, agent_idx, objective="reward"):
          """
          Estimate the score-function gradient in Algorithm 2, lines 7-8:

              E[grad log pi_i(a_i|s) * reward_i]
              E[grad log pi_i(a_i|s) * cost]

          The result has shape (nS, nA).
          """
          if objective not in ("reward", "cost"):
              raise ValueError("objective must be 'reward' or 'cost'")

          grad = np.zeros((self.nS, self.nA), dtype=float)

          for _ in range(self.n_samples):
              state = np.random.randint(self.nS)
              self.env.reset(state)

              for t in range(self.horizon):
                  actions = []
                  for i in range(self.M):
                      actions.append(
                          np.random.choice(self.nA, p=Pi[i, state])
                      )

                  next_state, rewards, cost, _, done = self.env.step(actions)
                  a_i = int(actions[agent_idx])
                  p = float(Pi[agent_idx, state, a_i])

                  # grad log pi(a|s) with respect to the tabular probability
                  # coordinates is 1/pi(a|s) for the sampled action.
                  score = 1.0 / max(p, 1e-12)
                  signal = (
                      float(rewards[agent_idx])
                      if objective == "reward"
                      else float(cost)
                  )

                  # Discounted policy-gradient estimator.
                  grad[state, a_i] += (self.gamma ** t) * score * signal

                  state = int(next_state)
                  if done:
                      break

          grad /= self.n_samples
          return grad
        def run_algo(self):
          """Run Algorithm 2 for args.T iterations."""
          # Line 2: initialize {pi_i} and form Pi.
          Pi = self.get_joint_policy().copy()

          for t in range(self.args.T):
              # Lines 5-8: J_ri, J_c, g_(1,i), g_hat_(2,i).
              # if (t + 1) % 2== 0:
              #     print("\nPolicies:")
              #     for i in range(self.args.M):
              #         print(f"Agent {i}:")
              #         print(Pi[i])
                      
              J_rewards, J_cost, g_reward, g_cost,J_pot = self.estimate_all(Pi)

              # Line 9: g_(2,i) = g_(1,i) - lambda*sigma(...)*g_hat_(2,i).
              # sigmoid_value = float(
              #     expit((-self.b + J_cost) / self.tau)
              # )
              rel_check = float((-self.b + J_cost)<=0)

              g = np.empty_like(g_reward)
              for i in range(self.M):
                  g[i] = (
                      g_reward[i]
                      - self.lam * rel_check*g_cost[i]
                  )

              # Line 10: Pi^{t+1} = Pi^t + alpha * [g_(2,1),...,g_(2,m)].
              Pi_1 = Pi + self.alpha * g
              Pi_2 = Pi + self.alpha * g_reward

              # Keep every pi_i(.|s) on the probability simplex.
              for i in range(self.M):
                  for s in range(self.nS):
                      Pi_1[i, s] = project_to_simplex(Pi_1[i, s])
              J_reward1,J_cost1,_,_,J_pot1 = self.estimate_all(Pi_1)
              if J_cost > self.b:
                Pi = Pi_1
                J_reward = J_reward1
                J_cost = J_cost1
                J_pot = J_pot1
              else:
                J_reward2,J_cost2,_,_,J_pot2 = self.estimate_all(Pi_2)
                for i in range(self.M):
                  for s in range(self.nS):
                      Pi_2[i, s] = project_to_simplex(Pi_2[i, s])
                if J_cost2 <b:
                  Pi = Pi_2
                  J_reward = J_reward2
                  J_cost = J_cost2
                  J_pot = J_pot2
                else:
                  l=0,u=1
                  while l<u:
                    mid = (l+u)/2
                    Pi_temp = Pi + self.alpha * (mid*g + (1-mid)*g_reward)
                    for i in range(self.M):
                      for s in range(self.nS):
                        Pi_temp[i, s] = project_to_simplex(Pi_temp[i, s])
                    J_reward_temp,J_cost_temp,_,_,J_pot_temp = self.estimate_all(Pi_2)
                    if b-self.args.eps_tol <=J_cost_temp <= b+self.args.eps_tol:
                      Pi = Pi_temp
                      J_reward = J_reward_temp
                      J_cost = J_cost_temp
                      J_pot = J_pot_temp
                      break
                    elif J_cost_temp < b-self.args.eps_tol:
                      l = mid
                    else:
                      u = mid
              # Synchronize Player objects.
              for i in range(self.M):
                  self.agent_list[i].policy = Pi[i].copy()

              self.Cf.append(J_cost)
              self.pot.append(J_pot)
              self.history.append({
                  "iteration": t + 1,
                  "J_cost": J_cost,
                  "J_pot": J_pot,
                  "sigmoid": sigmoid_value,
                  **{
                      f"J_reward_{i}": J_rewards[i]
                      for i in range(self.M)
                  },
              })

              print(
                  f"Iteration {t + 1:4d}/{self.args.T} | "
                  f"Jc = {J_cost: .6f} | "
                  f"Jpot = {J_pot: .6f} | "
                  f"sigma = {sigmoid_value: .6f}"
                  f"J_reward_1 = {J_rewards[0]}: .6f"
                  f"J_reward_2 = {J_rewards[1]:.6f}"
              )

          self.individual_vf["reward"] = [
              [row[f"J_reward_{i}"] for row in self.history]
              for i in range(self.M)
          ]
          self.individual_vf["cost"] = self.Cf
          self.individual_vf["pot"] = self.pot

          self.save_results()
          return Pi
        def save_results(self):
          """Save iteration statistics; failure falls back to pickle."""
          if not self.history:
              return

          data = pd.DataFrame(self.history)

          try:
              data.to_excel(self.args.save_file_nm, index=False)
              print(f"Results saved to {self.args.save_file_nm}")
          except Exception as exc:
              print(f"Excel save failed ({exc}); saving pickle instead.")
              with open(self.args.pickle_save, "wb") as f:
                  pickle.dump(data, f)
              print(f"Results saved to {self.args.pickle_save}")