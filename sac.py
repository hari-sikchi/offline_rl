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
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



class SAC:

    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, algo='SAC'):
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
        self.alpha = alpha

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        self.algo = algo
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)
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
        self.logger.setup_pytorch_saver(self.ac)



    def populate_replay_buffer(self):
        dataset = d4rl.qlearning_dataset(self.env)
        self.replay_buffer.obs_buf[:dataset['observations'].shape[0],:] = dataset['observations']
        self.replay_buffer.act_buf[:dataset['actions'].shape[0],:] = dataset['actions']
        self.replay_buffer.obs2_buf[:dataset['next_observations'].shape[0],:] = dataset['next_observations']
        self.replay_buffer.rew_buf[:dataset['rewards'].shape[0]] = dataset['rewards']
        self.replay_buffer.done_buf[:dataset['terminals'].shape[0]] = dataset['terminals']
        self.replay_buffer.size = dataset['observations'].shape[0]
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

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2


        current_action, _ = self.ac.pi(o)
        cql_alpha = 5
        # import ipdb; ipdb.set_trace()
        if self.algo == 'CQL':
            # print("CQL update")
            
            samples = 10 
            # Sample from previous policy (10 samples)
            cql_loss_q1 = None
            for sample in range(samples):
                sample_action, _ = self.ac.pi(o)
                
                if cql_loss_q1 is None:
                    cql_loss_q1 = self.ac.q1(o,sample_action).view(-1,1)
                    cql_loss_q2 = self.ac.q2(o,sample_action).view(-1,1)
                    
                else:
                    cql_loss_q1 = torch.cat((cql_loss_q1,self.ac.q1(o,sample_action).view(-1,1) ),dim=1)
                    cql_loss_q2 = torch.cat((cql_loss_q2,self.ac.q2(o,sample_action).view(-1,1) ),dim=1)

            cql_loss_q1 = cql_loss_q1-np.log(samples)
            cql_loss_q2 = cql_loss_q2-np.log(samples)

            cql_loss_q1 = torch.logsumexp(cql_loss_q1,dim=1).mean()
            cql_loss_q2 = torch.logsumexp(cql_loss_q2,dim=1).mean()

            # import ipdb; ipdb.set_trace()
            # cql_losses_q2 = [self.ac.q2(o,self.ac.pi(o)[0]) for sample in range(samples)]

            # cql_loss_q1 = self.ac_targ.q1(o, current_action)
            # cql_loss_q2 = self.ac_targ.q2(o, current_action)
            # Sample from dataset
            cql_loss_q1 -= self.ac.q1(o, a).mean()
            cql_loss_q2 -= self.ac.q2(o, a).mean()


            # import ipdb;ipdb.set_trace()
            loss_q += cql_alpha*(cql_loss_q1.mean() + cql_loss_q2.mean())
            # loss_q -= cql_alpha*(q1.mean() + q2.mean())
            

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info



    def update(self,data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
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
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def run(self):
        # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # # Update handling
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.update(data=batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalUpdates', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()


# def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#         polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
#         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#         logger_kwargs=dict(), save_freq=1, algo='SAC'):
#     """
#     Soft Actor-Critic (SAC)


#     Args:
#         env_fn : A function which creates a copy of the environment.
#             The environment must satisfy the OpenAI Gym API.

#         actor_critic: The constructor method for a PyTorch Module with an ``act`` 
#             method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
#             The ``act`` method and ``pi`` module should accept batches of 
#             observations as inputs, and ``q1`` and ``q2`` should accept a batch 
#             of observations and a batch of actions as inputs. When called, 
#             ``act``, ``q1``, and ``q2`` should return:

#             ===========  ================  ======================================
#             Call         Output Shape      Description
#             ===========  ================  ======================================
#             ``act``      (batch, act_dim)  | Numpy array of actions for each 
#                                            | observation.
#             ``q1``       (batch,)          | Tensor containing one current estimate
#                                            | of Q* for the provided observations
#                                            | and actions. (Critical: make sure to
#                                            | flatten this!)
#             ``q2``       (batch,)          | Tensor containing the other current 
#                                            | estimate of Q* for the provided observations
#                                            | and actions. (Critical: make sure to
#                                            | flatten this!)
#             ===========  ================  ======================================

#             Calling ``pi`` should return:

#             ===========  ================  ======================================
#             Symbol       Shape             Description
#             ===========  ================  ======================================
#             ``a``        (batch, act_dim)  | Tensor containing actions from policy
#                                            | given observations.
#             ``logp_pi``  (batch,)          | Tensor containing log probabilities of
#                                            | actions in ``a``. Importantly: gradients
#                                            | should be able to flow back into ``a``.
#             ===========  ================  ======================================

#         ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
#             you provided to SAC.

#         seed (int): Seed for random number generators.

#         steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
#             for the agent and the environment in each epoch.

#         epochs (int): Number of epochs to run and train agent.

#         replay_size (int): Maximum length of replay buffer.

#         gamma (float): Discount factor. (Always between 0 and 1.)

#         polyak (float): Interpolation factor in polyak averaging for target 
#             networks. Target networks are updated towards main networks 
#             according to:

#             .. math:: \\theta_{\\text{targ}} \\leftarrow 
#                 \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

#             where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
#             close to 1.)

#         lr (float): Learning rate (used for both policy and value learning).

#         alpha (float): Entropy regularization coefficient. (Equivalent to 
#             inverse of reward scale in the original SAC paper.)

#         batch_size (int): Minibatch size for SGD.

#         start_steps (int): Number of steps for uniform-random action selection,
#             before running real policy. Helps exploration.

#         update_after (int): Number of env interactions to collect before
#             starting to do gradient descent updates. Ensures replay buffer
#             is full enough for useful updates.

#         update_every (int): Number of env interactions that should elapse
#             between gradient descent updates. Note: Regardless of how long 
#             you wait between updates, the ratio of env steps to gradient steps 
#             is locked to 1.

#         num_test_episodes (int): Number of episodes to test the deterministic
#             policy at the end of each epoch.

#         max_ep_len (int): Maximum length of trajectory / episode / rollout.

#         logger_kwargs (dict): Keyword args for EpochLogger.

#         save_freq (int): How often (in terms of gap between epochs) to save
#             the current policy and value function.

#     """

#     logger = EpochLogger(**logger_kwargs)
#     logger.save_config(locals())

#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     env, test_env = env_fn(), env_fn()
#     obs_dim = env.observation_space.shape
#     act_dim = env.action_space.shape[0]

#     # Action limit for clamping: critically, assumes all dimensions share the same bound!
#     act_limit = env.action_space.high[0]

#     # Create actor-critic module and target networks
#     ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
#     ac_targ = deepcopy(ac)

#     # Freeze target networks with respect to optimizers (only update via polyak averaging)
#     for p in ac_targ.parameters():
#         p.requires_grad = False
        
#     # List of parameters for both Q-networks (save this for convenience)
#     q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

#     # Experience buffer
#     replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

#     # Count variables (protip: try to get a feel for how different size networks behave!)
#     var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
#     logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
#     self.algo = algo



#     # Set up function for computing SAC Q-losses
#     def compute_loss_q(data):
#         o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

#         q1 = ac.q1(o,a)
#         q2 = ac.q2(o,a)

#         # Bellman backup for Q functions
#         with torch.no_grad():
#             # Target actions come from *current* policy
#             a2, logp_a2 = ac.pi(o2)

#             # Target Q-values
#             q1_pi_targ = ac_targ.q1(o2, a2)
#             q2_pi_targ = ac_targ.q2(o2, a2)
#             q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
#             backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

#         # MSE loss against Bellman backup
#         loss_q1 = ((q1 - backup)**2).mean()
#         loss_q2 = ((q2 - backup)**2).mean()
#         loss_q = loss_q1 + loss_q2


#         current_action, _ = self.ac.pi(o)
#         cql_alpha = 5
#         if self.CQL:
#             # Sample from previous policy
#             cql_loss_q1 = self.ac_targ.q1(o, current_action)
#             cql_loss_q2 = self.ac_targ.q2(o, current_action)
#             # Sample from dataset
#             cql_loss_q1 -= self.ac_targ.q1(o, a)
#             cql_loss_q2 -= self.ac_targ.q2(o, a)


#             # import ipdb;ipdb.set_trace()
#             loss_q += cql_alpha*(cql_loss_q1.mean() + cql_loss_q2.mean())
#             loss_q -= cql_alpha*(q1.mean() + q2.mean())
            

#         # Useful info for logging
#         q_info = dict(Q1Vals=q1.detach().numpy(),
#                       Q2Vals=q2.detach().numpy())

#         return loss_q, q_info

#     # Set up function for computing SAC pi loss
#     def compute_loss_pi(data):
#         o = data['obs']
#         pi, logp_pi = ac.pi(o)
#         q1_pi = ac.q1(o, pi)
#         q2_pi = ac.q2(o, pi)
#         q_pi = torch.min(q1_pi, q2_pi)

#         # Entropy-regularized policy loss
#         loss_pi = (alpha * logp_pi - q_pi).mean()

#         # Useful info for logging
#         pi_info = dict(LogPi=logp_pi.detach().numpy())

#         return loss_pi, pi_info

#     # Set up optimizers for policy and q-function
#     pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
#     q_optimizer = Adam(q_params, lr=lr)

#     # Set up model saving
#     logger.setup_pytorch_saver(ac)

#     def update(data):
#         # First run one gradient descent step for Q1 and Q2
#         q_optimizer.zero_grad()
#         loss_q, q_info = compute_loss_q(data)
#         loss_q.backward()
#         q_optimizer.step()

#         # Record things
#         logger.store(LossQ=loss_q.item(), **q_info)

#         # Freeze Q-networks so you don't waste computational effort 
#         # computing gradients for them during the policy learning step.
#         for p in q_params:
#             p.requires_grad = False

#         # Next run one gradient descent step for pi.
#         pi_optimizer.zero_grad()
#         loss_pi, pi_info = compute_loss_pi(data)
#         loss_pi.backward()
#         pi_optimizer.step()

#         # Unfreeze Q-networks so you can optimize it at next DDPG step.
#         for p in q_params:
#             p.requires_grad = True

#         # Record things
#         logger.store(LossPi=loss_pi.item(), **pi_info)

#         # Finally, update target networks by polyak averaging.
#         with torch.no_grad():
#             for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
#                 # NB: We use an in-place operations "mul_", "add_" to update target
#                 # params, as opposed to "mul" and "add", which would make new tensors.
#                 p_targ.data.mul_(polyak)
#                 p_targ.data.add_((1 - polyak) * p.data)

#     def get_action(o, deterministic=False):
#         return ac.act(torch.as_tensor(o, dtype=torch.float32), 
#                       deterministic)

#     def test_agent():
#         for j in range(num_test_episodes):
#             o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
#             while not(d or (ep_len == max_ep_len)):
#                 # Take deterministic actions at test time 
#                 o, r, d, _ = test_env.step(get_action(o, True))
#                 ep_ret += r
#                 ep_len += 1
#             logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

#     # Prepare for interaction with environment
#     total_steps = epochs * steps_per_epoch
#     start_time = time.time()
#     o, ep_ret, ep_len = env.reset(), 0, 0

#     # Main loop: collect experience in env and update/log each epoch
#     for t in range(total_steps):

#         # Update handling
#         if t >= update_after and t % update_every == 0:
#             for j in range(update_every):
#                 batch = replay_buffer.sample_batch(batch_size)
#                 update(data=batch)

#         # End of epoch handling
#         if (t+1) % steps_per_epoch == 0:
#             epoch = (t+1) // steps_per_epoch

#             # Save model
#             if (epoch % save_freq == 0) or (epoch == epochs):
#                 logger.save_state({'env': env}, None)

#             # Test the performance of the deterministic version of the agent.
#             test_agent()

#             # Log info about epoch
#             logger.log_tabular('Epoch', epoch)
#             logger.log_tabular('TestEpRet', with_min_and_max=True)
#             logger.log_tabular('TestEpLen', average_only=True)
#             logger.log_tabular('TotalUpdates', t)
#             logger.log_tabular('Q1Vals', with_min_and_max=True)
#             logger.log_tabular('Q2Vals', with_min_and_max=True)
#             logger.log_tabular('LogPi', with_min_and_max=True)
#             logger.log_tabular('LossPi', average_only=True)
#             logger.log_tabular('LossQ', average_only=True)
#             logger.log_tabular('Time', time.time()-start_time)
#             logger.dump_tabular()

# class SAC:

#     def __init__(self,env_fn, models,replay_buffer, mpc_replay_buffer,termination_function,actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#             polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000, 
#             update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#             save_freq=1,AWAC = False, CQL=False):
#         """
#         Soft Actor-Critic (SAC)


#         Args:
#             env_fn : A function which creates a copy of the environment.
#                 The environment must satisfy the OpenAI Gym API.

#             actor_critic: The constructor method for a PyTorch Module with an ``act`` 
#                 method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
#                 The ``act`` method and ``pi`` module should accept batches of 
#                 observations as inputs, and ``q1`` and ``q2`` should accept a batch 
#                 of observations and a batch of actions as inputs. When called, 
#                 ``act``, ``q1``, and ``q2`` should return:

#                 ===========  ================  ======================================
#                 Call         Output Shape      Description
#                 ===========  ================  ======================================
#                 ``act``      (batch, act_dim)  | Numpy array of actions for each 
#                                             | observation.
#                 ``q1``       (batch,)          | Tensor containing one current estimate
#                                             | of Q* for the provided observations
#                                             | and actions. (Critical: make sure to
#                                             | flatten this!)
#                 ``q2``       (batch,)          | Tensor containing the other current 
#                                             | estimate of Q* for the provided observations
#                                             | and actions. (Critical: make sure to
#                                             | flatten this!)
#                 ===========  ================  ======================================

#                 Calling ``pi`` should return:

#                 ===========  ================  ======================================
#                 Symbol       Shape             Description
#                 ===========  ================  ======================================
#                 ``a``        (batch, act_dim)  | Tensor containing actions from policy
#                                             | given observations.
#                 ``logp_pi``  (batch,)          | Tensor containing log probabilities of
#                                             | actions in ``a``. Importantly: gradients
#                                             | should be able to flow back into ``a``.
#                 ===========  ================  ======================================

#             ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
#                 you provided to SAC.

#             seed (int): Seed for random number generators.

#             steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
#                 for the agent and the environment in each epoch.

#             epochs (int): Number of epochs to run and train agent.

#             replay_size (int): Maximum length of replay buffer.

#             gamma (float): Discount factor. (Always between 0 and 1.)

#             polyak (float): Interpolation factor in polyak averaging for target 
#                 networks. Target networks are updated towards main networks 
#                 according to:

#                 .. math:: \\theta_{\\text{targ}} \\leftarrow 
#                     \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

#                 where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
#                 close to 1.)

#             lr (float): Learning rate (used for both policy and value learning).

#             alpha (float): Entropy regularization coefficient. (Equivalent to 
#                 inverse of reward scale in the original SAC paper.)

#             batch_size (int): Minibatch size for SGD.

#             start_steps (int): Number of steps for uniform-random action selection,
#                 before running real policy. Helps exploration.

#             update_after (int): Number of env interactions to collect before
#                 starting to do gradient descent updates. Ensures replay buffer
#                 is full enough for useful updates.

#             update_every (int): Number of env interactions that should elapse
#                 between gradient descent updates. Note: Regardless of how long 
#                 you wait between updates, the ratio of env steps to gradient steps 
#                 is locked to 1.

#             num_test_episodes (int): Number of episodes to test the deterministic
#                 policy at the end of each epoch.

#             max_ep_len (int): Maximum length of trajectory / episode / rollout.

#             logger_kwargs (dict): Keyword args for EpochLogger.

#             save_freq (int): How often (in terms of gap between epochs) to save
#                 the current policy and value function.

#         """


#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         self.env, self.test_env = env_fn(), env_fn()
#         self.obs_dim = self.env.observation_space.shape
#         self.act_dim = self.env.action_space.shape[0]
#         self.max_ep_len=max_ep_len
#         self.start_steps=start_steps
#         self.batch_size=batch_size
#         self.gamma=gamma
#         self.alpha=alpha
#         self.polyak=polyak
#         # Action limit for clamping: critically, assumes all dimensions share the same bound!
#         self.act_limit = self.env.action_space.high[0]
#         self.steps_per_epoch=steps_per_epoch
#         self.update_after=update_after
#         self.update_every=update_every
#         self.num_test_episodes=num_test_episodes
#         self.epochs = epochs
#         # Create actor-critic module and target networks
#         self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
#         self.ac_targ = deepcopy(self.ac)
#         self.termination_func = termination_function
#         self.models = models
#         # Freeze target networks with respect to optimizers (only update via polyak averaging)
#         for p in self.ac_targ.parameters():
#             p.requires_grad = False
            
#         # List of parameters for both Q-networks (save this for convenience)
#         self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

#         # Experience buffer
#         self.replay_buffer = replay_buffer
#         self.mpc_replay_buffer = mpc_replay_buffer

#         # Count variables (protip: try to get a feel for how different size networks behave!)
#         self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
#         # Set up optimizers for policy and q-function
#         self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
#         self.q_optimizer = Adam(self.q_params, lr=lr)
#         self.v_optimizer = Adam(self.ac.v.parameters(), lr=lr)
#         self.awac = AWAC
#         if self.awac:
#             assert self.alpha==0
#         self.CQL = CQL

#     # Set up function for computing SAC Q-losses
#     def compute_loss_q(self,data):
#         o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

#         q1 = self.ac.q1(o,a)
#         q2 = self.ac.q2(o,a)


#         # Bellman backup for Q functions
#         with torch.no_grad():
#             # Target actions come from *current* policy
#             a2, logp_a2 = self.ac.pi(o2)

#             # Target Q-values
#             q1_pi_targ = self.ac_targ.q1(o2, a2)
#             q2_pi_targ = self.ac_targ.q2(o2, a2)
#             q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
#             backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)


        


#         # MSE loss against Bellman backup
#         loss_q1 = ((q1 - backup)**2).mean()
#         loss_q2 = ((q2 - backup)**2).mean()
#         loss_q = loss_q1 + loss_q2


#         current_action, _ = self.ac.pi(o)
#         cql_alpha = 0.1
#         if self.CQL:
#             # Sample from previous policy
#             cql_loss_q1 = self.ac_targ.q1(o, current_action)
#             cql_loss_q2 = self.ac_targ.q2(o, current_action)
#             # Sample from dataset
#             cql_loss_q1 -= self.ac_targ.q1(o, a)
#             cql_loss_q2 -= self.ac_targ.q2(o, a)


#             # import ipdb;ipdb.set_trace()
#             loss_q += cql_alpha*(cql_loss_q1.mean() + cql_loss_q2.mean())

#         # Useful info for logging
#         q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
#                     Q2Vals=q2.cpu().detach().numpy())

#         return loss_q, q_info

#     # Loss for AWAC, set alpha = 0
#     def compute_loss_v(self,data):
#         o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
#         v =  self.ac.v(o)
#         with torch.no_grad():
#             a2, logp_a2 = self.ac.pi(o)
#             q1_pi_targ = self.ac_targ.q1(o, a2)
#             q2_pi_targ = self.ac_targ.q2(o, a2)
#             q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
#             backup = q_pi_targ
        

#         loss_v = ((v - backup)**2).mean()
#         v_info = dict(VVals=v.cpu().detach().numpy())
#         return loss_v, v_info




#     def getQ(self, state,action):
#         state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         action = torch.FloatTensor(action.reshape(1, -1)).to(device)
#         Q1,Q2 = self.ac.q1(state,action),self.ac.q2(state,action)
#         return Q1.cpu().data.numpy(),Q2.cpu().data.numpy()

#     def get_value(self,o,reduce_='mean',samples=10):
#         v = None
#         for i in range(samples):
#             pi, logp_pi = self.ac.pi(o)
#             q1_pi = self.ac.q1(o, pi)
#             q2_pi = self.ac.q2(o, pi)
#             q_pi = torch.min(q1_pi, q2_pi).detach()
#             if v is None:
#                 v= q_pi
#             else:
#                 if reduce_=='mean':
#                     v+=q_pi
#                 elif reduce_ =='max':
#                     v=torch.max(v, q_pi)
#         if reduce_=='mean':
#             return v/samples
#         else:
#             return v



#     # Set up function for computing SAC pi loss
#     def compute_loss_pi(self,data):
#         o = data['obs']
#         a = data['act']
#         pi, logp_pi = self.ac.pi(o)
#         q1_pi = self.ac.q1(o, pi)
#         q2_pi = self.ac.q2(o, pi)
#         q_pi = torch.min(q1_pi, q2_pi)
#         # q_pi = torch.max(q1_pi, q2_pi)
#         # q_pi = q1_pi
#         # Entropy-regularized policy loss

#         if self.awac:
#             q_pi = q_pi.detach()
#             v_pi = self.get_value(o, reduce_='mean')
#             # CWR exp
#             # loss_pi = -(logp_pi*torch.clamp(torch.exp(q_pi-v_pi),0,20)).mean()
#             # CWR binary
#             a_pi = q_pi-v_pi
#             loss_pi = -(logp_pi*(a_pi>0).float()).mean()
#         else:
#             loss_pi = (self.alpha * logp_pi - q_pi).mean()
#         # Behavior Cloning loss
#         # loss_pi+= 10*((pi - a)**2).mean()
#         # Useful info for logging
#         pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

#         return loss_pi, pi_info


#     # Set up model saving

#     def update(self,data):
#         # First run one gradient descent step for Q1 and Q2
#         self.q_optimizer.zero_grad()
#         loss_q, q_info =  self.compute_loss_q(data)
#         loss_q.backward()
#         # print("Q loss: {}".format(loss_q))
#         self.q_optimizer.step()

#         # Learn a value function if doing AWAC
#         if self.awac:
#             self.v_optimizer.zero_grad()
#             loss_v , v_info = self.compute_loss_v(data)
#             loss_v.backward()
#             self.v_optimizer.step()

#         # Record things
#         # logger.store(LossQ=loss_q.item(), **q_info)

#         # Freeze Q-networks so you don't waste computational effort 
#         # computing gradients for them during the policy learning step.
#         for p in self.q_params:
#             p.requires_grad = False

#         # Next run one gradient descent step for pi.
#         self.pi_optimizer.zero_grad()
#         loss_pi, pi_info = self.compute_loss_pi(data)
#         loss_pi.backward()
#         self.pi_optimizer.step()

#         # Unfreeze Q-networks so you can optimize it at next DDPG step.
#         for p in self.q_params:
#             p.requires_grad = True

#         # Record things
#         # logger.store(LossPi=loss_pi.item(), **pi_info)

#         # Finally, update target networks by polyak averaging.
#         with torch.no_grad():
#             for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
#                 # NB: We use an in-place operations "mul_", "add_" to update target
#                 # params, as opposed to "mul" and "add", which would make new tensors.
#                 p_targ.data.mul_(self.polyak)
#                 p_targ.data.add_((1 - self.polyak) * p.data)

#     def get_action(self,o, deterministic=False):



#         return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
#                     deterministic)

#     def test_agent(self):
#         for j in range(self.num_test_episodes):
#             o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
#             while not(d or (ep_len == self.max_ep_len)):
#                 # Take deterministic actions at test time 
#                 o, r, d, _ = self.test_env.step(self.get_action(o, True))
#                 ep_ret += r
#                 ep_len += 1
#             return ep_ret,ep_len
#             # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

#     def reset(self):
#         pass
    
#     def train(self):
#         use_MBPO = True
#         if use_MBPO:
#             M= 20
#             for i in range(M):
#                 for j in range(self.update_every):
#                     if np.random.uniform()<1.0:
#                         batch = self.replay_buffer.sample_batch(self.batch_size)
#                     else:
#                         batch = self.mpc_replay_buffer.sample_batch(self.batch_size)

#                     # Imagine data using the model and train on that
#                     imagined_batch = batch.copy()
#                     next_state_reward = self.models.get_forward_prediction_random_ensemble(imagined_batch['obs'],imagined_batch['act'])
#                     imagined_batch['obs2'] = next_state_reward[:,1:]
#                     imagined_batch['reward'] = next_state_reward[:,:1]
                    
#                     # done = self.termination_func(imagined_batch['obs'],imagined_batch['act'],imagined_batch['obs2']).float().to(device)
#                     # done = torch.FloatTensor(done)
#                     # imagined_batch['done'] = done            
#                     imagined_batch['done'] = batch['done']
#                     self.update(data=imagined_batch)
#         else:
#             # m = 8  initially
#             M= 1
#             for i in range(M):
#                 for j in range(self.update_every):
#                     if np.random.uniform()<1.0:
#                         batch = self.replay_buffer.sample_batch(self.batch_size)
#                     else:
#                         batch = self.mpc_replay_buffer.sample_batch(self.batch_size)
#                     self.update(data=batch)


#     def run(self):
#         log_data = {
#             'evaluation': [],
#             'evaluation_training': [],
#             'cost': [],
#             'actor_evaluation': [],
#             'model_error': [],
#             'timesteps': [],
#             'dynamics_trainloss':[],
#             'dynamics_valloss':[],
#             'actor_divergence':[]
#             }
    
#         # Prepare for interaction with environment
#         total_steps = self.steps_per_epoch * self.epochs
#         start_time = time.time()
#         o, ep_ret, ep_len = self.env.reset(), 0, 0

#         # Main loop: collect experience in env and update/log each epoch
#         for t in range(total_steps):
            
#             # Until start_steps have elapsed, randomly sample actions
#             # from a uniform distribution for better exploration. Afterwards, 
#             # use the learned policy. 
#             if t > self.start_steps:
#                 a = self.get_action(o)
#             else:
#                 a = self.env.action_space.sample()

#             # Step the env
#             o2, r, d, _ = self.env.step(a)
#             ep_ret += r
#             ep_len += 1

#             # Ignore the "done" signal if it comes from hitting the time
#             # horizon (that is, when it's an artificial terminal signal
#             # that isn't based on the agent's state)
#             d = False if ep_len==self.max_ep_len else d

#             # Store experience to replay buffer
#             self.replay_buffer.store(o, a, r, o2, d)

#             # Super critical, easy to overlook step: make sure to update 
#             # most recent observation!
#             o = o2

#             # End of trajectory handling
#             if d or (ep_len == self.max_ep_len):
#                 o, ep_ret, ep_len = self.env.reset(), 0, 0

#             # Update handling
#             if t >= self.update_after and t % self.update_every == 0:
#                 for j in range(self.update_every):
#                     batch = self.replay_buffer.sample_batch(self.batch_size)
#                     self.update(data=batch)

#             # End of epoch handling
#             if (t+1) % self.steps_per_epoch == 0:
#                 epoch = (t+1) // self.steps_per_epoch

#                 # Save model
#                 # if (epoch % save_freq == 0) or (epoch == epochs):
#                 #     logger.save_state({'env': env}, None)

#                 # Test the performance of the deterministic version of the agent.
#                 testepret,testeplen = self.test_agent()
#                 print("Evaluation: {} Timestep: {} ".format(testepret,t))
#                 log_data['actor_evaluation'].append(testepret)
#                 log_data['timesteps'].append(t)



# class SAC_ENSEMBLE:

#     def __init__(self,env_fn, models,replay_buffer, mpc_replay_buffer,termination_function,actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#             polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000, 
#             update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#             save_freq=1,AWAC = False, ensemble_size = 5):

#         self.sac_ensemble = [SAC(env_fn, models,replay_buffer, mpc_replay_buffer,termination_function,actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=np.random.randint(1,5), 
#         steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=replay_size, gamma=gamma, 
#         polyak=polyak, lr=lr, alpha=alpha, batch_size=batch_size, start_steps=start_steps, 
#         update_after=update_after, update_every=update_every, num_test_episodes=num_test_episodes, max_ep_len=max_ep_len, 
#         save_freq=save_freq,AWAC = AWAC) for i in range(ensemble_size)]
#         self.env, self.test_env = env_fn(), env_fn()
#         self.obs_dim = self.env.observation_space.shape
#         self.act_dim = self.env.action_space.shape[0]
#         self.max_ep_len=max_ep_len
#         self.start_steps=start_steps
#         self.batch_size=batch_size
#         self.gamma=gamma
#         self.alpha=alpha
#         self.polyak=polyak

#     def get_action(self,o, deterministic=False):
#         return self.sac_ensemble[0].ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
#                     deterministic)
    
    
#     def getQ(self, state,action):
#         state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         action = torch.FloatTensor(action.reshape(1, -1)).to(device)
#         # Q1,Q2 = self.ac.q1(state,action),self.ac.q2(state,action)
#         Q1,Q2 =0,0
#         for i in range(len(self.sac_ensemble)):
#             Q1_,Q2_ = self.sac_ensemble[i].ac.q1(state,action),self.sac_ensemble[i].ac.q2(state,action)
#             Q1+=Q1_
#             Q2+=Q2_
#         Q1/=len(self.sac_ensemble)
#         Q2/=len(self.sac_ensemble)
#         return Q1.cpu().data.numpy(),Q2.cpu().data.numpy()

#     def getQ1(self, state, action):
#         # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         # action = torch.FloatTensor(action.reshape(1, -1)).to(device)
#         # Q1,Q2 = self.ac.q1(state,action),self.ac.q2(state,action)
#         Q1,Q2 =0,0
#         for i in range(len(self.sac_ensemble)):
#             Q1_,Q2_ = self.sac_ensemble[i].ac.q1(state,action),self.sac_ensemble[i].ac.q2(state,action)
#             Q1 += Q1_
#             Q2+= Q2_
#         Q1/=len(self.sac_ensemble)
#         Q2/=len(self.sac_ensemble)
#         return Q1.cpu().data.numpy(),Q2.cpu().data.numpy()

#     def train(self):
#         for i in range(len(self.sac_ensemble)):
#             self.sac_ensemble[i].train()




# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='sac')
#     args = parser.parse_args()

#     # from spinup.utils.run_utils import setup_logger_kwargs
#     # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     torch.set_num_threads(torch.get_num_threads())

#     sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         )