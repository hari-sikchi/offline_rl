import sac
from CQL.cql import CQL
from CQL_IL.cql_il import CQL_IL
from CQL_IL.cql_batch_il import CQL_batchIL
from CWR.cwr import CWR
from EMAQ.emaq import EMAQ
from EMAQ.learn_behavior_policy import EMAQ_LP
from DT.dt import DT
from DT.learn_marginal import DT_LM
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
    args = parser.parse_args()
    env_fn = lambda:gym.make(args.env)


    if 'CQL_IL' in args.algorithm:
        agent = CQL_IL(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm, num_expert_trajs=args.num_expert_trajs)
    elif 'CQL_batchIL' in args.algorithm:
        agent = CQL_batchIL(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm, num_expert_trajs=args.num_expert_trajs)
    elif 'CQL' in args.algorithm:
        agent = CQL(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}, seed=args.seed,batch_size=256, algo=args.algorithm, automatic_alpha_tuning = True) 
    elif 'EMAQ' in args.algorithm:
        if 'learn-behavior-policy' in args.algorithm:
            agent = EMAQ_LP(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm) 
        else:
            agent = EMAQ(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm) 
    elif 'DT' in args.algorithm:
        if 'learn-marginal' in args.algorithm:
            agent = DT_LM(env_fn, env_name= args.env, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}, batch_size=256,  seed=args.seed, algo=args.algorithm) 
        else:
            agent = DT(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256,  seed=args.seed, algo=args.algorithm) 
    elif 'CWR' in args.algorithm:
        agent = CWR(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=1024,  seed=args.seed, algo=args.algorithm)
    else:
        agent = sac.SAC(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name},batch_size=256, seed=args.seed, algo=args.algorithm) 
    
    agent.populate_replay_buffer()
    agent.run()



