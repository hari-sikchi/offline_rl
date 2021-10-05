import sac
from CQL.cql import CQL
from AWAC.awac import AWAC
from AWAC.awac_finetuning import AWAC_online 
from EMAQ.emaq import EMAQ
from EMAQ.learn_behavior_policy import EMAQ_LP
import argparse
import gym

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--algorithm", default="SAC")
    parser.add_argument("--env", default="hopper-random-v0")
    parser.add_argument("--exp_name", default="data/dump")
    parser.add_argument("--num_expert_trajs", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    
    # MOPO parameters
    parser.add_argument("--lamda_pessimism", default=1, type=int)
    parser.add_argument("--rollout_length", default=1, type=int)

    
    args = parser.parse_args()
    env_fn = lambda:gym.make(args.env)

    if 'CQL' in args.algorithm:
        agent = CQL(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}, seed=args.seed,batch_size=256, algo=args.algorithm, automatic_alpha_tuning = True) 
    elif 'EMAQ' in args.algorithm:
        if 'learn-behavior-policy' in args.algorithm:
            agent = EMAQ_LP(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm) 
        else:
            agent = EMAQ(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm) 
    elif 'AWAC' in args.algorithm:
        agent = AWAC(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=1024,  seed=args.seed, algo=args.algorithm)
    elif 'AWAC_online' in args.algorithm:
        agent = AWAC_online(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=1024,  seed=args.seed, algo=args.algorithm)
    else:
        agent = sac.SAC(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256, seed=args.seed, algo=args.algorithm) 
    
    agent.populate_replay_buffer()
    agent.run()



