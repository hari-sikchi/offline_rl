
# @author: Harshit Sikchi

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import d4rl
import gym
import time
import core as core
from utils.logx import EpochLogger
import torch.nn.functional as F
from MOPO.model import MBPO_PENN as MBPO_PENN_reward
from MOPO.model import PENN as PENN_reward
import os
import static

# TODO: Batch size for MOPO RL training?


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.state = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.cost = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost=None):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        if cost is not None:
            self.cost[self.ptr] = cost
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def reset(self):
        self.ptr = 0
        self.size = 0

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     cost=self.cost[idxs],
                     done=self.done[idxs])
        return_dict = {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}
        return return_dict


class MOPO:

    def __init__(self, env_fn,env_name=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=200, epochs=50000, replay_size=int(2e6), gamma=0.99, 
        polyak=0.995, lr=3e-4, p_lr=3e-4, alpha=0.1, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=30, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, automatic_alpha_tuning= True,algo='SAC',rollout_length=1, lamda_pessimism=1):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            """

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.output_dir = logger_kwargs['output_dir']
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.gamma  = gamma


        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        
        # Dynamics model
        self.dynamics = MBPO_PENN_reward(
            5, # Number of model ensembles
            self.obs_dim[0],
            self.act_dim,
            0.001, # Learning rate
            replay_buffer=self.replay_buffer,
            hidden_units=[200,200,200,200],
            train_iters= 100,
            epochs=5)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        self.algo = algo
        self.start_steps = start_steps

        # Algorithm specific hyperparams

        self.alpha = alpha # CWR does not require entropy in Q evaluation
        self.target_update_freq = 1
        self.p_lr = 3e-4
        self.lr = 3e-4

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        
        self.use_cyclic_scheduler = False


        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs= epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        # Set up model saving
        # self.logger.setup_pytorch_saver(self.ac)
        self.logger.setup_pytorch_saver(self.dynamics)
        self.env_name = env_name
        self.domain = self.env_name.split('-')[0]
        self.static_fns=static[self.domain.lower()]

        self.rollout_length = rollout_length
        self.rollout_batch_size = int(50e3)
        self.model_train_freq = 1000
        self.lamda = lamda_pessimism
        self.model_retain_epochs = 5
        self.model_replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=int(self.rollout_length*50e3*self.model_retain_epochs))


        self.automatic_alpha_tuning = automatic_alpha_tuning
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -3
            # self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        print("Running Offline RL algorithm: {}".format(self.algo))


    def populate_replay_buffer(self):
        # dataset = d4rl.qlearning_dataset(self.env)
        # self.replay_buffer.state[:dataset['observations'].shape[0],:] = dataset['observations']
        # self.replay_buffer.action[:dataset['actions'].shape[0],:] = dataset['actions']
        # self.replay_buffer.next_state[:dataset['next_observations'].shape[0],:] = dataset['next_observations']
        # self.replay_buffer.reward[:dataset['rewards'].shape[0]] = dataset['rewards']
        # self.replay_buffer.done[:dataset['terminals'].shape[0]] = dataset['terminals']
        # self.replay_buffer.size = dataset['observations'].shape[0]
        # self.replay_buffer.ptr = (self.replay_buffer.size+1)%(self.replay_buffer.max_size)
        buffer_folder = '/home/hsikchi/work/safePDDM/sac/results2/MBRLHopper-v0_LOOP_SAC_save_buffer_0/replay_buffer_159999.npy'

        dataset = (np.load(buffer_folder,allow_pickle=True)).item()
        # import ipdb; ipdb.set_trace()
        self.replay_buffer.state[:dataset['state'].shape[0],:] = dataset['state']
        self.replay_buffer.action[:dataset['action'].shape[0],:] = dataset['action']
        self.replay_buffer.next_state[:dataset['next_state'].shape[0],:] = dataset['next_state']
        self.replay_buffer.reward[:dataset['reward'].shape[0]] = dataset['reward']
        self.replay_buffer.done[:dataset['done'].shape[0]] = dataset['done']
        self.replay_buffer.size = dataset['size']
        self.replay_buffer.ptr = (self.replay_buffer.size+1)%(self.replay_buffer.max_size)
 
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)


        # print(torch.max(backup))

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2


            

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        # Sample recent data for policy update

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)


        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach())

        return loss_pi, pi_info



    def update(self,data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

            # print(self.q_optimizer.param_groups[0]["lr"])
        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False


        # data = self.replay_buffer.sample_batch(self.batch_size)
        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        # if self.use_cyclic_scheduler:
        #     self.pi_scheduler.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), LogPi=pi_info['LogPi'].numpy())

        log_pi = pi_info['LogPi']
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi.to(device) + self.target_entropy)).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()


        # Finally, update target networks by polyak averaging.
        # TODO: Check updating only Q functions
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1.parameters(), self.ac_targ.q1.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q2.parameters(), self.ac_targ.q2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            if(hasattr(self.test_env, 'get_normalized_score')):
                self.logger.store(TestEpRet=100*self.test_env.get_normalized_score(ep_ret), TestEpLen=ep_len)
            else:
                self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def add_model_samples(self,state,action,reward,next_state,done):
        for i in range(state.shape[0]):
            self.model_replay_buffer.store(state[i,:],action[i],reward[i],next_state[i],done[i])


    def rollout_model(self):
        # self.model_replay_buffer.reset()
        batch = self.replay_buffer.sample_batch(self.rollout_batch_size)
        obs = batch['obs']
        for i in range(self.rollout_length):
            act,_ = self.ac.pi(obs)
            next_state_reward = self.dynamics.get_forward_prediction_pessimistic(obs,act,lamda=self.lamda)
            rew = next_state_reward[:,:1]
            obs2 = next_state_reward[:,1:]
            
            obs_np = obs.cpu().detach().numpy()
            act_np = act.cpu().detach().numpy()
            obs2_np = obs2.cpu().detach().numpy()
            rew_np = rew.cpu().detach().numpy()


            done_np = self.static_fns.termination_fn(obs_np,act_np
                                ,obs2_np)

            self.add_model_samples(obs_np,act_np,rew_np, obs2_np,done_np)
            
            nonterm_mask = ~done_np.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break
            obs = obs2[nonterm_mask]



    def combine_batches(self, real_batch, model_batch):
        batch = real_batch.copy()
        for key,value in real_batch.items():
            # import ipdb;ipdb.set_trace()
            batch[key] = torch.cat((batch[key],model_batch[key]),axis=0)
        return batch

    def run(self):



        # Train dynamics model
        max_dynamics_epochs = 1000
        patience = 5
        best_loss = 1e7
        loss_increase = 0
        
        if(os.path.exists(self.output_dir+'/pyt_save/model.pt')):
            print("Loading existing model from: {}".format(self.output_dir+'/pyt_save/model.pt'))
            self.dynamics = torch.load(self.output_dir+'/pyt_save/model.pt')
        else:
            dynamics_trainloss, dynamics_valloss = self.dynamics.train()
            self.logger.save_state({'env': self.env}, None)

        # self.dynamics = torch.load('/home/hsikchi/work/offline_rl/learned_dynamics/'+self.env_name+'/_s0/pyt_save/model.pt')
        # self.dynamics = torch.load('/home/hsikchi/work/offline_rl/data/dump_walker-med-rep_s0/pyt_save/model.pt')
       # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            train_steps = 0

            # # Update handling
            M= 1000
            p = 0.05
            for m in range(M):

                if(m%self.model_train_freq==0):
                    self.rollout_model()     
                # with prob p
                
                if(train_steps>5*m):
                    pass
                
                real_batch_size = int(p*self.batch_size)
                model_batch_size = self.batch_size-real_batch_size
                real_batch = self.replay_buffer.sample_batch(real_batch_size)
                model_batch = self.model_replay_buffer.sample_batch(model_batch_size)
                batch = self.combine_batches(real_batch,model_batch)

                train_steps+=1
                    
                self.update(data=batch,update_timestep=t)


            # End of epoch handling

            epoch = t

            # Save model
            # if (epoch % self.save_freq == 0) or (epoch == self.epochs):
            #     self.logger.save_state({'env': self.env}, None)

            # Test the performance of the deterministic version of the agent.
            self.test_agent()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('TestEpRet', with_min_and_max=True)
            self.logger.log_tabular('TestEpLen', average_only=True)
            self.logger.log_tabular('TotalUpdates', t*M)
            self.logger.log_tabular('Q1Vals', with_min_and_max=True)
            self.logger.log_tabular('Q2Vals', with_min_and_max=True)
            self.logger.log_tabular('LogPi', with_min_and_max=True)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossQ', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()



