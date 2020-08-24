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
from torch.autograd import Variable

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



'''
CQL variants
cql-rho-fixedAlpha
cql-rho-lagrange
cql-H-fixedAlpha
cql-H-lagrange
'''
class EMAQ:

    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=100, epochs=10000, replay_size=int(1500000), gamma=0.99, 
        polyak=0.995, lr=3e-4, p_lr=3e-5, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, algo='CQL'):
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

        self.lagrange_threshold = 10
        self.penalty_lr = 5e-2
        self.lamda = Variable(torch.log(torch.exp(torch.Tensor([5]))-1), requires_grad=True)
        self.lamda_optimizer = torch.optim.Adam([self.lamda],lr=self.penalty_lr)
        self.tune_lambda = True if 'lagrange' in self.algo else False


        self.alpha = alpha # CWR does not require entropy in Q evaluation
        self.target_update_freq = 1
        self.p_lr = 3e-5
        self.lr = 3e-4
        self.n_samples = 100


        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
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
        print("Running Offline RL algorithm: {}".format(self.algo))

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

        sampled_actions_q1 = None
        sampled_actions_q2 = None
        for i in range(self.n_samples):
            z = np.random.randn(a.shape[0],a.shape[1])
            z = torch.FloatTensor(z)
            actions, _ = self.sampling_policy.inverse(z,y=o2)
            if sampled_actions_q1 is None:
                sampled_actions_q1 = self.ac_targ.q1(o2,actions).view(-1,1)
                sampled_actions_q2 = self.ac_targ.q2(o2,actions).view(-1,1)            
            else:
                sampled_actions_q1 = torch.cat((sampled_actions_q1,self.ac_targ.q1(o2,actions).view(-1,1)),dim=1)
                sampled_actions_q2 = torch.cat((sampled_actions_q2,self.ac_targ.q2(o2,actions).view(-1,1)),dim=1)

        
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)
            # Target Q-values
            q1_pi_targ = torch.max(sampled_actions_q1,dim=1).values
            q2_pi_targ = torch.max(sampled_actions_q2,dim=1).values
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info



    def update(self,data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)


        # Finally, update target networks by polyak averaging.
        if update_timestep%self.target_update_freq==0:
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        sampled_actions_q1 = None
        sampled_actions_q2 = None
        sampled_actions = []
        o = torch.FloatTensor(o).view(1,-1)
        for i in range(self.n_samples):
            z = np.random.randn(1,self.act_dim)
            z = torch.FloatTensor(z)
            actions, _ = self.sampling_policy.inverse(z,y=o)
            sampled_actions.append(actions)
            if sampled_actions_q1 is None:
                sampled_actions_q1 = self.ac_targ.q1(o,actions).view(-1,1)
                sampled_actions_q2 = self.ac_targ.q2(o,actions).view(-1,1)            
            else:
                sampled_actions_q1 = torch.cat((sampled_actions_q1,self.ac_targ.q1(o,actions).view(-1,1)),dim=1)
                sampled_actions_q2 = torch.cat((sampled_actions_q2,self.ac_targ.q2(o,actions).view(-1,1)),dim=1)

        q_values = torch.min(sampled_actions_q1,sampled_actions_q2)
        max_idx = torch.argmax(q_values.view(-1))
        return sampled_actions[max_idx].detach().cpu().numpy()



    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=100*self.test_env.get_normalized_score(ep_ret), TestEpLen=ep_len)

    def run(self):

        # Learn a generative model for data
        density_epochs = 50
        self.sampling_policy = core.MADE(self.act_dim, 256, 2 , cond_label_size = self.obs_dim[0])
        density_optimizer = torch.optim.Adam(self.sampling_policy.parameters(), lr=1e-4, weight_decay=1e-6)
        for i in range(density_epochs):
            sample_indices = np.random.choice(
                    self.replay_buffer.size, self.replay_buffer.size)
            np.random.shuffle(sample_indices)
            ctr = 0
            total_loss = 0
            for j in range(0, self.replay_buffer.size, self.batch_size):
                actions = self.replay_buffer.act_buf[sample_indices[ctr * self.batch_size:(
                        ctr + 1) * self.batch_size],:]
                actions = torch.FloatTensor(actions)
                obs = self.replay_buffer.obs_buf[sample_indices[ctr * self.batch_size:(
                        ctr + 1) * self.batch_size],:]
                obs = torch.FloatTensor(obs)
                density_optimizer.zero_grad()
                loss = -self.sampling_policy.log_prob(actions,y=obs).mean()
                loss.backward()
                total_loss+=loss.data * self.batch_size
                density_optimizer.step()
                ctr+=1
                

            print("Density training loss: {}".format(total_loss/self.replay_buffer.size))

        # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # # Update handling
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.update(data=batch, update_timestep = t)

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
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

