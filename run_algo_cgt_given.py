import argparse
import numpy as np

from multi_agent_dec_cgt_given import project_x_dec_tabular


def make_initial_policy(env_nm, m, nS, nA):
    """Match the initialization in the user's original launcher."""
    init_pi = np.zeros((m, nS, nA), dtype=float)

    if env_nm == "pollution":
        if nA != 2:
            raise ValueError("Pollution environment requires nA=2.")
        init_pi[:, :, 0] = 0.7
        init_pi[:, :, 1] = 0.3
    elif env_nm == "energy":
        init_pi = np.zeros((m, nS, nA), dtype=float)/nA
    else:
        raise ValueError("env_nm must be 'energy' or 'pollution'.")

    return init_pi


def main():
    parser = argparse.ArgumentParser(description="De-centralized softplus Prox-CPMG")

    parser.add_argument("--env_nm", "--env-nm", default="energy", type=str,choices=["energy", "pollution"])
    parser.add_argument("--T", default=100, type=int,help="Number of algorithm iterations")
    parser.add_argument("--horizon", default=100, type=int,help="Trajectory/evaluation horizon")
    parser.add_argument("--M", type=int, default=2,help="Number of players")
    parser.add_argument("--nS", type=int, default=None,help="Number of states")
    parser.add_argument("--nA", type=int, default=None,help="Number of actions")
    parser.add_argument("--gamma", type=float, default=0.9,help="Discount factor")
    parser.add_argument("--alpha", type=float, default=1e-3,help="Policy update step size")
    parser.add_argument("--b", type=float, default=12,help="Constraint threshold")
    parser.add_argument("--lamdba", "--lambda", dest="lamdba",type=float, default=0,help="Lagrangian coefficient")
    parser.add_argument("--tau", type=float, default=0.02,help="Sigmoid temperature")
    parser.add_argument("--n_samples", type=int, default=500,help="Monte-Carlo trajectories per iteration")
    parser.add_argument("--init_epsilon", type=float, default=0.0,help=("Blend initial policy with uniform exploration; use e.g. 0.05 for Energy, whose original policy is deterministic."))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_file_nm", default=None,help="Excel output filename")
    parser.add_argument("--pickle_save", default="storage_decentralised_pollution_2_agents.pkl",help="Pickle fallback filename")

    args = parser.parse_args()

    # Environment dimensions from the original code.
    if args.nS is None:
        args.nS = 5 if args.env_nm == "energy" else 2
    if args.nA is None:
        args.nA = 5 if args.env_nm == "energy" else 2

    if not (0.0 <= args.init_epsilon < 1.0):
        raise ValueError("init_epsilon must be in [0, 1).")

    if args.save_file_nm is None:
        args.save_file_nm = (f"storage_{args.M}_players_{args.nS}_states_"
        f"tabular_{args.env_nm}.xlsx")

    np.random.seed(args.seed)
    args.init_pi = make_initial_policy(args.env_nm, args.M, args.nS, args.nA)
    print("Number of agents:", args.M)
    print("Environment:", args.env_nm)
    print("States:", args.nS)
    print("Actions:", args.nA)
    print("Algorithm iterations:", args.T)
    print("Horizon:", args.horizon)
    print("Samples/iteration:", args.n_samples)
    agent = project_x_dec_tabular(args)
    final_Pi = agent.run_algo()
    print("\nFinished.")
    print("Final policy shape:", final_Pi.shape)
    print("Final policies:")
    for i in range(args.M):
        print(f"Agent {i}:\n{final_Pi[i]}")
if __name__ == "__main__":
    main()