import sac
import argparse
import gym

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--algorithm", default="SAC")
    parser.add_argument("--env", default="hopper-random-v0")
    parser.add_argument("--exp_name", default="dump")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    env_fn = lambda:gym.make(args.env)

    sac_agent = sac.SAC(env_fn, logger_kwargs={'output_dir':args.exp_name}, seed=args.seed) 
    sac_agent.populate_replay_buffer()

    sac_agent.run()



