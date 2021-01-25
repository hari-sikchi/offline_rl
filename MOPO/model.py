

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import collections
from torch.autograd import Variable
import torch.nn.functional as F
import time
from tqdm import tqdm



class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


class dynamicsNetwork(nn.Module):
    def __init__(
            self,
            nb_states,
            nb_actions,
            hidden_units=[
                64,
                64],
            init_w=3e-3):
        super(dynamicsNetwork, self).__init__()
        # # Define the network | MLP + Relu
        self.fcs = nn.ModuleList(
            [nn.Linear(nb_states + nb_actions, hidden_units[0])])
        for i in range(len(hidden_units) - 1):
            self.fcs.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        self.fcs.append(nn.Linear(hidden_units[-1], 2 * (nb_states+1)))
        self.relu = nn.ReLU()

    def forward(self, x):

        out = x
        for i in range(len(self.fcs) - 1):
            out = self.fcs[i](out)
            out = self.relu(out)
        out = self.fcs[-1](out)

        return out



class dynamicsNetwork_mbpo(nn.Module):
    def __init__(
            self,
            nb_states,
            nb_actions,
            hidden_units=[
                64,
                64],
            init_w=3e-3):
        super(dynamicsNetwork_mbpo, self).__init__()
        # # Define the network | MLP + Relu
        self.fcs = nn.ModuleList(
            [nn.Linear(nb_states + nb_actions, hidden_units[0])])
        for i in range(len(hidden_units) - 1):
            self.fcs.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        self.fcs.append(nn.Linear(hidden_units[-1], 2 * (nb_states+1)))
        self.swish = Swish()

    def forward(self, x):

        out = x
        for i in range(len(self.fcs) - 1):
            out = self.fcs[i](out)
            out = self.swish(out)
        out = self.fcs[-1](out)

        return out







class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(
        self,
        num_nets,
        state_dim,
        action_dim,
        learning_rate,
        replay_buffer=None,
        device=None,
        hidden_units=[
            64,
            64],
        epochs = 10,
        train_iters=20):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #torch.device( "cpu")
        
        self.hidden_units = hidden_units

        # Model also outputs the reward so dimension is state_dim + 1
        self.max_logvar = nn.Parameter(torch.ones(1, self.state_dim+1, dtype=torch.float32) / 2.0).to(self.device)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.state_dim+1, dtype=torch.float32) * 10.0).to(self.device)

        # Create and initialize your model
        self.models = [self.create_network().to(self.device)
                       for _ in range(self.num_nets)]
        self.epochs = epochs
        self.train_iters=train_iters
        self.optimizers = [
            torch.optim.Adam(
                list(
                    self.models[i].parameters()),
                lr=learning_rate) for i in range(
                self.num_nets)]

        params = []
        for m in self.models:
            params = params + list(m.parameters())

        self.replay_buffer = replay_buffer
        print("++++++++++Using PETS style model+++++++++++++")
        print("++++++++++Device: {}+++++++++++++".format(self.device))

    def load(self, filename):
        for i, model in enumerate(self.models):
            self.models[i].load_state_dict(torch.load(filename + "_" + str(i)))

    def save(self, filename):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), filename + "_" + str(i))

    def proc_state(self,state,output):
        next_state_reward = output
        next_state_reward[:,1:]+=state[:,:self.state_dim]
        return next_state_reward


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim+1]
        logvar = output[:, self.state_dim+1:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # Define the network size for dynamics network
    def create_network(self):
        model = dynamicsNetwork(
            self.state_dim, self.action_dim, hidden_units=self.hidden_units)
        return model

    def gaussian_loglikelihood(self, x, mu, log_var):

        inv_var = torch.exp(-log_var)
        loss = ((x - mu)**2) * inv_var + log_var
        loss = loss.mean()

        return loss

    def model_loss(self, predicted_mean_var, targets, model):
        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
        loss = None
        loss = self.gaussian_loglikelihood(targets,predicted_mean,predicted_logvar)

        loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        # Removing regularization for speed concerns. Might change later TODO

        l2_regularization = 0
        # all_fcs_params = [torch.cat(
        #     [x.view(-1) for x in fc.parameters()]) for fc in model.fcs]

        # for params in all_fcs_params:
        #     l2_regularization += 0.00005 * torch.norm(params, 2)

        return loss + l2_regularization


    def uncertainty_pessimism_loss(self,m, state_t, state_action_concat):
        alpha = 1
        next_state_t = m(state_action_concat)
        nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

        sample_action = np.random.uniform(low=-1.0,high=1.0,size=(state_t.shape[0],self.action_dim))
        sample_action = torch.FloatTensor(sample_action).to(self.device)
        state_random_action_concat =torch.cat((state_t, sample_action), axis=1) 

        _ = m(state_random_action_concat)
        nxt_mean_t, nxt_random_log_var_t = self.get_output(next_state_t)
        
        nxt_var_t = torch.exp(nxt_log_var_t)
        nxt_random_var_t = torch.exp(nxt_random_log_var_t)

        loss = alpha*((nxt_var_t[:,0]-nxt_random_var_t[:,0]).mean())

        return loss


    def get_forward_prediction(self, state, action,mode='ts_inf'):
        next_states = None

        if mode=='ts_inf':
            for i, model in enumerate(self.models):
                model.eval()
                # Convert the state matrix into tensor if it is not
                if not torch.is_tensor(state[i]):
                    state_t = torch.from_numpy(state[i]).float().to(self.device)
                else:
                    state_t = state[i].float().to(self.device)

                action_t = torch.from_numpy(action).float().to(self.device)
                # Concatenate the state and action
                state_action_concat = torch.cat((state_t, action_t), axis=1)

                next_state_t = model(state_action_concat)
                nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

                # Exponentiate the variance as log variance is obtained

                nxt_var_t = torch.exp(nxt_log_var_t)

                next_state_t = nxt_mean_t + \
                    torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
                next_state=self.proc_state(state_t,next_state_t)

                if next_states is None:
                    next_states = next_state.unsqueeze(0)
                else:
                    next_states = torch.cat((next_states,next_state.unsqueeze(0)),0)
        # The next states contains the predicted reward at next_states[:,0]
        return next_states

    # Pessimitic reward prediction for MOPO
    def get_forward_prediction_pessimistic(self, state, action, lamda=1):

        idx = np.random.randint(low=0, high=self.num_nets)
        next_state_t = None
        state_action_concat = torch.cat((state, action), axis=1)
        max_std = None
        mean_reward = None

        for i,m in enumerate(self.models):
            m.eval()
            next_state_t_ = m(state_action_concat)
            nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
            nxt_std_t = torch.exp(nxt_log_var_t).sqrt()
            if max_std is None:
                max_std = nxt_std_t[:,:1]
                # max_std = torch.norm(nxt_std_t, dim=1)
                mean_reward = nxt_mean_t[:,:1]
            else:
                mean_reward+= nxt_mean_t[:,:1]
                max_std = torch.max(max_std,nxt_std_t[:,:1])
                # max_std = torch.max(max_std,torch.norm(nxt_std_t, dim=1))
            if(i==idx):
                # nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
                next_state_t = nxt_mean_t + \
                torch.randn_like(nxt_mean_t) *  nxt_std_t
        
        # import ipdb;ipdb.set_trace()
        # penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

        next_state = self.proc_state(state,next_state_t)
        next_state[:,:1]=(mean_reward/self.num_nets)-lamda*torch.norm(max_std, dim=1).view((-1,1))

        return next_state

    def get_forward_prediction_random_ensemble(self, state, action):
        # print("Using the model with reward")
        model = self.models[np.random.randint(low=0, high=self.num_nets)]

        model.eval()
        # state_t = torch.from_numpy(state).float().to(self.device)
        # action_t = torch.from_numpy(action).float().to(self.device)
        state_action_concat = torch.cat((state, action), axis=1)

        next_state_t = model(state_action_concat)
        nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

        # Exponentiate the variance as log variance is obtained

        nxt_var_t = torch.exp(nxt_log_var_t)

        next_state_t = nxt_mean_t + \
            torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
        # next_state = next_state_t.detach().cpu().numpy()
        
        next_state = self.proc_state(state,next_state_t)

        # next_state = next_state_t
        # return np.array(next_state)

        # The next states contains the predicted reward at next_states[:,0]
        return next_state



    # def get_next_state(self, state, action):

    #     model = self.models[np.random.randint(low=0, high=self.num_nets)]
    #     model.eval()

    #     state_t = torch.from_numpy(np.array(state).reshape(
    #         1, -1)).float().to(self.device)[:, :self.state_dim]
    #     action_t = torch.from_numpy(
    #         action.reshape(1, -1)).float().to(self.device)
    #     state_action_concat = torch.cat((state_t, action_t), axis=1)
    #     next_state_t = model(state_action_concat)
    #     nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)
    #     # Exponentiate the variance as log variance is obtained

    #     nxt_var_t = torch.exp(nxt_log_var_t)

    #     next_state_t = nxt_mean_t + \
    #         torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
    #     next_state = self.proc_state(state_t,next_state_t)
    #     next_state = next_state_t.detach().cpu().numpy()
    #     # The next states contains the predicted reward at next_states[:,0]
    #     return next_state

    def train(self, batch_size=256):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        train_time_start=  time.time()
        loss_total=0
        val_loss = 0
        training_steps = 0
        validation_steps =0
        max_iter = 200
        for epoch in range(self.epochs):
            # Train each model for single epoch
            for i, m in enumerate(self.models):
                start = time.time()
                m.train()

                
                sample_indices =  np.random.permutation(self.replay_buffer.size)
                
                # Split the data into training and test
                holdout_ratio = 0.1
                train_indices = sample_indices[:int((1-holdout_ratio)*self.replay_buffer.size)]
                val_indices = sample_indices[int((1-holdout_ratio)*self.replay_buffer.size):]
                ctr = 0
                for j in range(0, train_indices.shape[0], batch_size):
                    self.optimizers[i].zero_grad()



                    batch_input = np.concatenate((self.replay_buffer.state[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]], self.replay_buffer.action[train_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
                    batch_input = torch.from_numpy(batch_input).float()
                    batch_input = batch_input.to(self.device)
                    
                    batch_target = np.concatenate((self.replay_buffer.reward[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]]),axis=1)
                    batch_target = torch.from_numpy(batch_target).float()
                    batch_target = batch_target.to(self.device)

                    batch_target[:,1:]-=batch_input[:,:self.state_dim]
                    predicted_mean_var = m(batch_input)
                    loss = self.model_loss(predicted_mean_var, batch_target, m)

                    # loss+= self.uncertainty_pessimism_loss(m,torch.from_numpy(self.replay_buffer.state[train_indices[ctr * batch_size:(
                    #     ctr + 1) * batch_size]]).float().to(self.device),batch_input)

                    predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                    train_err= ((predicted_mean-batch_target)**2).mean()
                    loss.backward()
                    loss_total += train_err.item()
                    ctr += 1
                    training_steps+=1
                    self.optimizers[i].step()
                    if(ctr>=max_iter):
                        break
                ctr = 0
                for j in range(0, val_indices.shape[0], batch_size):
                    with torch.no_grad():
                        batch_input = np.concatenate((self.replay_buffer.state[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]], self.replay_buffer.action[val_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
                        batch_input = torch.from_numpy(batch_input).float()
                        batch_input =batch_input.to(self.device)
                        batch_target = np.concatenate((self.replay_buffer.reward[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]]),axis=1)

                        batch_target = torch.from_numpy(batch_target).float()
                        batch_target = batch_target.to(self.device)

                        batch_target[:,1:]-=batch_input[:,:self.state_dim]
                        predicted_mean_var = m(batch_input)
                        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                        val_err = ((predicted_mean-batch_target)**2).mean()
                        # loss = self.model_loss(predicted_mean_var, batch_target, m)                        
                        val_loss+=val_err.item()
                        validation_steps+=1
                        ctr+=1
                    if(ctr>=max_iter):
                        break
                end = time.time()
                # print("One iteration took: {} seconds".format(end-start))
        # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )
        train_time_end= time.time()
        print("Training time: {}".format(train_time_end-train_time_start))
        return loss_total/(training_steps),val_loss/(validation_steps)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class MBPO_PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(
        self,
        num_nets,
        state_dim,
        action_dim,
        learning_rate,
        replay_buffer=None,
        device=None,
        hidden_units=[
            64,
            64],
        epochs = 10,
        train_iters=20):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.device = torch.device( "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_units = hidden_units

        # Model also outputs the reward so dimension is state_dim + 1
        self.max_logvar = nn.Parameter(torch.ones(1, self.state_dim+1, dtype=torch.float32) / 2.0).to(self.device)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.state_dim+1, dtype=torch.float32) * 10.0).to(self.device)

        # Create and initialize your model
        self.models = [self.create_network().to(self.device)
                       for _ in range(self.num_nets)]
        self.epochs = epochs
        self.train_iters=train_iters
        self.optimizers = [
            torch.optim.Adam(
                list(
                    self.models[i].parameters()),
                lr=learning_rate) for i in range(
                self.num_nets)]

        params = []
        for m in self.models:
            params = params + list(m.parameters())

        self.replay_buffer = replay_buffer
        print("++++++++++Using PETS style model+++++++++++++")
        print("++++++++++Device: {}+++++++++++++".format(self.device))

    def load(self, filename):
        for i, model in enumerate(self.models):
            self.models[i].load_state_dict(torch.load(filename + "_" + str(i)))

    def save(self, filename):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), filename + "_" + str(i))

    def proc_state(self,state,output):
        next_state_reward = output
        next_state_reward[:,1:]+=state[:,:self.state_dim]
        return next_state_reward


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim+1]
        logvar = output[:, self.state_dim+1:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # Define the network size for dynamics network
    def create_network(self):
        model = dynamicsNetwork_mbpo(
            self.state_dim, self.action_dim, hidden_units=self.hidden_units)
        return model

    def gaussian_loglikelihood(self, x, mu, log_var):

        inv_var = torch.exp(-log_var)
        loss = ((x - mu)**2) * inv_var + log_var
        loss = loss.mean()

        return loss

    def model_loss(self, predicted_mean_var, targets, model):
        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
        loss = None
        loss = self.gaussian_loglikelihood(targets,predicted_mean,predicted_logvar)

        loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        # Removing regularization for speed concerns. Might change later TODO

        l2_regularization = 0
        all_fcs_params = [torch.cat(
            [x.view(-1) for x in fc.parameters()]) for fc in model.fcs]


        
        for params in all_fcs_params:
            l2_regularization += 0.00005 * torch.norm(params, 2)

        return loss + l2_regularization


    def uncertainty_pessimism_loss(self,m, state_t, state_action_concat):
        alpha = 1
        next_state_t = m(state_action_concat)
        nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

        sample_action = np.random.uniform(low=-1.0,high=1.0,size=(state_t.shape[0],self.action_dim))
        sample_action = torch.FloatTensor(sample_action).to(self.device)
        state_random_action_concat =torch.cat((state_t, sample_action), axis=1) 

        _ = m(state_random_action_concat)
        nxt_mean_t, nxt_random_log_var_t = self.get_output(next_state_t)
        
        nxt_var_t = torch.exp(nxt_log_var_t)
        nxt_random_var_t = torch.exp(nxt_random_log_var_t)

        loss = alpha*((nxt_var_t[:,0]-nxt_random_var_t[:,0]).mean())

        return loss


    def get_forward_prediction(self, state, action,mode='ts_inf'):
        next_states = None

        if mode=='ts_inf':
            for i, model in enumerate(self.models):
                model.eval()
                # Convert the state matrix into tensor if it is not
                if not torch.is_tensor(state[i]):
                    state_t = torch.from_numpy(state[i]).float().to(self.device)
                else:
                    state_t = state[i].float().to(self.device)

                action_t = torch.from_numpy(action).float().to(self.device)
                # Concatenate the state and action
                state_action_concat = torch.cat((state_t, action_t), axis=1)

                next_state_t = model(state_action_concat)
                nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

                # Exponentiate the variance as log variance is obtained

                nxt_var_t = torch.exp(nxt_log_var_t)

                next_state_t = nxt_mean_t + \
                    torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
                next_state=self.proc_state(state_t,next_state_t)

                if next_states is None:
                    next_states = next_state.unsqueeze(0)
                else:
                    next_states = torch.cat((next_states,next_state.unsqueeze(0)),0)
        # The next states contains the predicted reward at next_states[:,0]
        return next_states

    # Pessimitic reward prediction for MOPO
    def get_forward_prediction_pessimistic(self, state, action, lamda=1):

        idx = np.random.randint(low=0, high=self.num_nets)
        next_state_t = None
        state_action_concat = torch.cat((state, action), axis=1)
        max_std = None
        mean_reward = None

        for i,m in enumerate(self.models):
            m.eval()
            next_state_t_ = m(state_action_concat)
            nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
            nxt_std_t = torch.exp(nxt_log_var_t).sqrt()
            if max_std is None:
                # max_std = nxt_std_t[:,:1]
                max_std = torch.norm(nxt_std_t, dim=1)
                mean_reward = nxt_mean_t[:,:1]
            else:
                mean_reward+= nxt_mean_t[:,:1]
                # max_std = torch.max(max_std,nxt_std_t[:,:1])
                max_std = torch.max(max_std,torch.norm(nxt_std_t, dim=1))
            if(i==idx):
                # nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
                next_state_t = nxt_mean_t + \
                torch.randn_like(nxt_mean_t) *  nxt_std_t
        
        # import ipdb;ipdb.set_trace()
        # penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

        next_state = self.proc_state(state,next_state_t)
        next_state[:,:1]=(mean_reward/self.num_nets)-lamda*max_std.view((-1,1))

        return next_state

    def get_forward_prediction_random_ensemble(self, state, action):
        # print("Using the model with reward")
        model = self.models[np.random.randint(low=0, high=self.num_nets)]

        model.eval()
        # state_t = torch.from_numpy(state).float().to(self.device)
        # action_t = torch.from_numpy(action).float().to(self.device)
        state_action_concat = torch.cat((state, action), axis=1)

        next_state_t = model(state_action_concat)
        nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

        # Exponentiate the variance as log variance is obtained

        nxt_var_t = torch.exp(nxt_log_var_t)

        next_state_t = nxt_mean_t + \
            torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
        # next_state = next_state_t.detach().cpu().numpy()
        
        next_state = self.proc_state(state,next_state_t)

        # next_state = next_state_t
        # return np.array(next_state)

        # The next states contains the predicted reward at next_states[:,0]
        return next_state



    # def get_next_state(self, state, action):

    #     model = self.models[np.random.randint(low=0, high=self.num_nets)]
    #     model.eval()

    #     state_t = torch.from_numpy(np.array(state).reshape(
    #         1, -1)).float().to(self.device)[:, :self.state_dim]
    #     action_t = torch.from_numpy(
    #         action.reshape(1, -1)).float().to(self.device)
    #     state_action_concat = torch.cat((state_t, action_t), axis=1)
    #     next_state_t = model(state_action_concat)
    #     nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)
    #     # Exponentiate the variance as log variance is obtained

    #     nxt_var_t = torch.exp(nxt_log_var_t)

    #     next_state_t = nxt_mean_t + \
    #         torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
    #     next_state = self.proc_state(state_t,next_state_t)
    #     next_state = next_state_t.detach().cpu().numpy()
    #     # The next states contains the predicted reward at next_states[:,0]
    #     return next_state

    def train(self, batch_size=256):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        train_time_start= time.time()
        loss_total=0
        val_loss = 0
        training_steps = 0
        validation_steps =0
        max_iter = 200

        max_dynamics_epochs = 1000
        patience = 5
        best_loss = 1e7
        loss_increase = 0

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        num_holdout = int(self.replay_buffer.size * 0.2)
        permutation = np.random.permutation(self.replay_buffer.size)

        all_inputs = np.concatenate((self.replay_buffer.state[:self.replay_buffer.size,:], self.replay_buffer.action[:self.replay_buffer.size,:]), axis=1)
        all_inputs = torch.from_numpy(all_inputs).float()
        all_inputs = all_inputs.to(self.device)

        all_targets = np.concatenate((self.replay_buffer.reward[:self.replay_buffer.size].reshape(-1,1),self.replay_buffer.next_state[:self.replay_buffer.size,:]),axis=1)
        all_targets = torch.from_numpy(all_targets).float()
        all_targets = all_targets.to(self.device)

        inputs, holdout_inputs = all_inputs[permutation[num_holdout:]], all_inputs[permutation[:num_holdout]]
        targets, holdout_targets = all_targets[permutation[num_holdout:]], all_targets[permutation[:num_holdout]]

        for epoch in tqdm(range(max_dynamics_epochs),desc='Training Dynamics Model: '):
            # Train each model for single epoch
            for i, m in enumerate(self.models):
                start = time.time()
                m.train()

                
                sample_indices =  np.random.permutation(inputs.shape[0])
                
                for batch_num in range(int(np.ceil(inputs.shape[0] / batch_size))):
                    batch_idxs = sample_indices[ batch_num * batch_size:(batch_num + 1) * batch_size]

                    batch_input = inputs[batch_idxs]
                    batch_target = targets[batch_idxs]
                    batch_target[:,1:]-=batch_input[:,:self.state_dim]
                    self.optimizers[i].zero_grad()
                    predicted_mean_var = m(batch_input)
                    loss = self.model_loss(predicted_mean_var, batch_target, m)
                    # loss+= self.uncertainty_pessimism_loss(m,torch.from_numpy(self.replay_buffer.state[train_indices[ctr * batch_size:(
                    #     ctr + 1) * batch_size]]).float().to(self.device),batch_input)
                    predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                    train_err= ((predicted_mean-batch_target)**2).mean()
                    loss.backward()
                    loss_total += train_err.item()
                    training_steps+=1
                    self.optimizers[i].step()
                
                sample_indices =  np.random.permutation(holdout_inputs.shape[0])
                for batch_num in range(int(np.ceil(holdout_inputs.shape[0] / batch_size))):
                    with torch.no_grad():
                        batch_idxs = sample_indices[ batch_num * batch_size:(batch_num + 1) * batch_size]

                        batch_input = holdout_inputs[batch_idxs]
                        batch_target = holdout_targets[batch_idxs]
                        batch_target[:,1:]-=batch_input[:,:self.state_dim]
                        predicted_mean_var = m(batch_input)
                        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                        val_err = ((predicted_mean-batch_target)**2).mean()                   
                        val_loss+=val_err.item()
                        validation_steps+=1
                # # Split the data into training and test
                # holdout_ratio = 0.1
                # train_indices = sample_indices[:int((1-holdout_ratio)*self.replay_buffer.size)]
                # val_indices = sample_indices[int((1-holdout_ratio)*self.replay_buffer.size):]
                # ctr = 0
                # for j in range(0, train_indices.shape[0], batch_size):
                #     self.optimizers[i].zero_grad()



                #     batch_input = np.concatenate((self.replay_buffer.state[train_indices[ctr * batch_size:(
                #         ctr + 1) * batch_size]], self.replay_buffer.action[train_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
                #     batch_input = torch.from_numpy(batch_input).float()
                #     batch_input = batch_input.to(self.device)
                    
                #     batch_target = np.concatenate((self.replay_buffer.reward[train_indices[ctr * batch_size:(
                #         ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[train_indices[ctr * batch_size:(
                #         ctr + 1) * batch_size]]),axis=1)
                #     batch_target = torch.from_numpy(batch_target).float()
                #     batch_target = batch_target.to(self.device)

                #     batch_target[:,1:]-=batch_input[:,:self.state_dim]
                #     predicted_mean_var = m(batch_input)
                #     loss = self.model_loss(predicted_mean_var, batch_target, m)

                #     # loss+= self.uncertainty_pessimism_loss(m,torch.from_numpy(self.replay_buffer.state[train_indices[ctr * batch_size:(
                #     #     ctr + 1) * batch_size]]).float().to(self.device),batch_input)

                #     predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                #     train_err= ((predicted_mean-batch_target)**2).mean()
                #     loss.backward()
                #     loss_total += train_err.item()
                #     ctr += 1
                #     training_steps+=1
                #     self.optimizers[i].step()
                #     if(ctr>=max_iter):
                #         break
                # ctr = 0
                # for j in range(0, val_indices.shape[0], batch_size):
                #     with torch.no_grad():
                #         batch_input = np.concatenate((self.replay_buffer.state[val_indices[ctr * batch_size:(
                #             ctr + 1) * batch_size]], self.replay_buffer.action[val_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
                #         batch_input = torch.from_numpy(batch_input).float()
                #         batch_input =batch_input.to(self.device)
                #         batch_target = np.concatenate((self.replay_buffer.reward[val_indices[ctr * batch_size:(
                #             ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[val_indices[ctr * batch_size:(
                #             ctr + 1) * batch_size]]),axis=1)

                #         batch_target = torch.from_numpy(batch_target).float()
                #         batch_target = batch_target.to(self.device)

                #         batch_target[:,1:]-=batch_input[:,:self.state_dim]
                #         predicted_mean_var = m(batch_input)
                #         predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                #         val_err = ((predicted_mean-batch_target)**2).mean()
                #         # loss = self.model_loss(predicted_mean_var, batch_target, m)                        
                #         val_loss+=val_err.item()
                #         validation_steps+=1
                #         ctr+=1
                #     if(ctr>=max_iter):
                #         break
                # end = time.time()
                # print("One iteration took: {} seconds".format(end-start))
            print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )
            if((best_loss-(val_loss/(validation_steps)))/best_loss>0.005):
                loss_increase = 0
                best_loss =val_loss/(validation_steps)
                
            else:
                loss_increase+=1
            
            if(loss_increase>patience):
                break
            else:
                training_steps = 0
                validation_steps =0
                val_loss=0
                loss_total = 0
        
        train_time_end= time.time()
        # print("Training time: {}".format(train_time_end-train_time_start))
        return loss_total/(training_steps),val_loss/(validation_steps)



# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# import random
# import collections
# from torch.autograd import Variable
# import torch.nn.functional as F
# import time
# import copy
# from tqdm import tqdm


# class Swish(nn.Module):
#     def forward(self, input_tensor):
#         return input_tensor * torch.sigmoid(input_tensor)

# class dynamicsNetworkCost(nn.Module):
#     def __init__(
#             self,
#             nb_states,
#             nb_actions,
#             hidden_units=[
#                 64,
#                 64],
#             init_w=3e-3):
#         super(dynamicsNetworkCost, self).__init__()
#         # # Define the network | MLP + Relu
#         self.fcs = nn.ModuleList(
#             [nn.Linear(nb_states + nb_actions, hidden_units[0])])
#         for i in range(len(hidden_units) - 1):
#             self.fcs.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
#         self.fcs.append(nn.Linear(hidden_units[-1], 2 * (nb_states+2)))
#         self.relu = nn.ReLU()

#     def forward(self, x):

#         out = x
#         for i in range(len(self.fcs) - 1):
#             out = self.fcs[i](out)
#             out = self.relu(out)
#         out = self.fcs[-1](out)

#         return out
# class dynamicsNetwork(nn.Module):
#     def __init__(
#             self,
#             nb_states,
#             nb_actions,
#             hidden_units=[
#                 64,
#                 64],
#             init_w=3e-3):
#         super(dynamicsNetwork, self).__init__()
#         # # Define the network | MLP + Relu
#         self.fcs = nn.ModuleList(
#             [nn.Linear(nb_states + nb_actions, hidden_units[0])])
#         for i in range(len(hidden_units) - 1):
#             self.fcs.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
#         self.fcs.append(nn.Linear(hidden_units[-1], 2 * (nb_states+1)))
#         self.relu = nn.ReLU()

#     def forward(self, x):

#         out = x
#         for i in range(len(self.fcs) - 1):
#             out = self.fcs[i](out)
#             out = self.relu(out)
#         out = self.fcs[-1](out)

#         return out


# class costNetwork(nn.Module):
#     def __init__(
#             self,
#             nb_states,
#             hidden_units=[
#                 64,
#                 64],
#             init_w=3e-3):
#         super(costNetwork, self).__init__()
#         # # Define the network | MLP + Relu
#         self.fcs = nn.ModuleList(
#             [nn.Linear(nb_states, hidden_units[0])])
#         for i in range(len(hidden_units) - 1):
#             self.fcs.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
#         self.fcs.append(nn.Linear(hidden_units[-1], 2))
#         self.relu = nn.ReLU()

#     def forward(self, x):

#         out = x
#         for i in range(len(self.fcs) - 1):
#             out = self.fcs[i](out)
#             out = self.relu(out)
#         out = self.fcs[-1](out)
#         return out    

# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1)).reshape(-1,2)
#         pt = torch.exp(-BCE_loss)
#         # import ipdb;ipdb.set_trace()
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()

# class costModelNN:
#     def __init__(
#         self,
#         num_nets,
#         state_dim,
#         action_dim,
#         learning_rate,
#         replay_buffer=None,
#         unsafe_replay_buffer = None,
#         safe_replay_buffer = None,
#         device=None,
#         hidden_units=[
#             64,
#             64],
#         epochs = 70,
#         train_iters=20,
#         loss_type='cross_entropy',
#         use_seperate_buffer=False):
#         """
#         :param num_nets: number of networks in the ensemble
#         :param state_dim: state dimension
#         :param action_dim: action dimension
#         :param learning_rate:
#         """

#         self.num_nets = num_nets
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.hidden_units = hidden_units

#         # Model also outputs the reward so dimension is state_dim + 1
#         self.max_logvar = nn.Parameter(torch.ones(1, 1, dtype=torch.float32) / 2.0).to(self.device)
#         self.min_logvar = nn.Parameter(- torch.ones(1, 1, dtype=torch.float32) * 10.0).to(self.device)

#         # Create and initialize your model
#         self.models = [self.create_network().to(self.device)
#                        for _ in range(self.num_nets)]
#         self.epochs = epochs
#         self.train_iters=train_iters
#         self.optimizers = [
#             torch.optim.Adam(
#                 list(
#                     self.models[i].parameters()),
#                 lr=learning_rate) for i in range(
#                 self.num_nets)]

#         params = []
#         for m in self.models:
#             params = params + list(m.parameters())

#         self.replay_buffer = replay_buffer
#         self.loss_type = loss_type # focal/cross_entropy
#         self.safe_replay_buffer = safe_replay_buffer
#         self.unsafe_replay_buffer = unsafe_replay_buffer
#         self.use_seperate_buffer = use_seperate_buffer
#         print("++++++++++Using Cost Neural Network style model+++++++++++++")


#     def load(self, filename):
#         for i, model in enumerate(self.models):
#             self.models[i].load_state_dict(torch.load(filename + "_" + str(i)))

#     def save(self, filename):
#         for i, model in enumerate(self.models):
#             torch.save(model.state_dict(), filename + "_" + str(i))

#     def proc_state(self,state,output):
#         next_state_reward = output
#         next_state_reward[:,1:]+=state[:,:self.state_dim]
#         return next_state_reward


#     # Define the network size for dynamics network
#     def create_network(self):
#         model = costNetwork(
#             self.state_dim,  hidden_units=self.hidden_units)
#         return model

#     def model_loss(self, predicted_values, targets, model):
#         if(self.loss_type=='cross_entropy'):
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(predicted_values,targets.long().view(-1))
#         elif(self.loss_type=='focal'):
#             loss_fn = WeightedFocalLoss()
#             target_one_hot = torch.zeros((targets.shape[0],2))
#             target_one_hot = np.eye(2)[targets.long()[:].cpu().detach().numpy().reshape(-1)]
#             target_one_hot = torch.FloatTensor(target_one_hot).to(self.device)
#             # target_one_hot[targets.long()[:].cpu().detach().numpy().reshape(-1)]=1
#             # import ipdb;ipdb.set_trace()
#             loss = loss_fn(predicted_values,target_one_hot.to(self.device))
            
#         return loss 


#     def get_forward_prediction(self, state, action,mode='ts_inf'):
#         costs = None

#         if mode=='ts_inf':
#             for i, model in enumerate(self.models):
#                 model.eval()
#                 # Convert the state matrix into tensor if it is not
#                 if not torch.is_tensor(state[i]):
#                     state_t = torch.from_numpy(state[i]).float().to(self.device)
#                 else:
#                     state_t = state[i].float().to(self.device)



#                 state_action_concat = state_t
#                 cost_t = model(state_action_concat)

#                 if costs is None:
#                     costs = cost_t.unsqueeze(0)
#                 else:
#                     costs = torch.cat((costs,cost_t.unsqueeze(0)),0)
#         # The next states contains the predicted reward at next_states[:,0]
#         return next_states



#     def get_forward_prediction_random_ensemble(self, state):
#         # print("Using the model with reward")
#         model = self.models[np.random.randint(low=0, high=self.num_nets)]
#         model.eval()

#         if not torch.is_tensor(state):
#             state_t = torch.from_numpy(state).float().to(self.device)
#         else:
#             state_t = state.float().to(self.device)

#         state_action_concat = state_t
#         cost_t = torch.argmax(model(state_action_concat),axis=1)

#         return cost_t



#     def copy_batch_to_replay(self,train_replay_buffer, replay_buffer, replay_size):
#         curr_size = train_replay_buffer.size
#         train_replay_buffer.state[curr_size:curr_size+replay_size,:]=replay_buffer.state[max(replay_buffer.size-replay_size,0):replay_buffer.size,:]
#         train_replay_buffer.next_state[curr_size:curr_size+replay_size,:]=replay_buffer.next_state[max(replay_buffer.size-replay_size,0):replay_buffer.size,:]
#         train_replay_buffer.action[curr_size:curr_size+replay_size,:]=replay_buffer.action[max(replay_buffer.size-replay_size,0):replay_buffer.size,:]
#         train_replay_buffer.reward[curr_size:curr_size+replay_size]=replay_buffer.reward[max(replay_buffer.size-replay_size,0):replay_buffer.size]
#         train_replay_buffer.cost[curr_size:curr_size+replay_size]=replay_buffer.cost[max(replay_buffer.size-replay_size,0):replay_buffer.size]
#         train_replay_buffer.done[curr_size:curr_size+replay_size]=replay_buffer.done[max(replay_buffer.size-replay_size,0):replay_buffer.size]
#         train_replay_buffer.ptr+=replay_size
#         train_replay_buffer.size+=replay_size


#     def train(self, batch_size=256):
#         """
#         Arguments:
#           inputs: state and action inputs.  Assumes that inputs are standardized.
#           targets: resulting states
#         """
#         # patience = 5
#         # best_loss = 1e7
#         # loss_increase = 0

#         # loss_total=0
#         # val_loss = 0
#         # training_steps = 0
#         # validation_steps =0
#         # max_iter = 200
#         # max_ratio = 3
#         # train_replay_buffer = copy.deepcopy(self.replay_buffer)
#         # train_replay_buffer.size = 0
#         # train_replay_buffer.ptr = 0


#         # if self.use_seperate_buffer:
#         #     unsafe_size, safe_size = self.unsafe_replay_buffer.size,self.safe_replay_buffer.size
#         #     # safe_size = min(unsafe_size*max_ratio+1,safe_size)
#         #     # self.copy_batch_to_replay(train_replay_buffer, self.unsafe_replay_buffer, unsafe_size)
#         #     # self.copy_batch_to_replay(train_replay_buffer, self.safe_replay_buffer,safe_size )
#         #     self.copy_batch_to_replay(train_replay_buffer, self.safe_replay_buffer,safe_size )
#         #     self.copy_batch_to_replay(train_replay_buffer, self.unsafe_replay_buffer, unsafe_size)
#         #     for k in range(safe_size//unsafe_size):
#         #         self.copy_batch_to_replay(train_replay_buffer, self.unsafe_replay_buffer, unsafe_size)
#         # else:
#         #     train_replay_buffer = self.replay_buffer

#         # # import ipdb;ipdb.set_trace()

#         # for epoch in tqdm(range(self.epochs),desc='Training Cost Model: '):
#         #     # Train each model for single epoch
#         #     for i, m in enumerate(self.models):
#         #         start = time.time()
#         #         m.train()

#         #         sample_indices =  np.random.permutation(train_replay_buffer.size)
                
#         #         # Split the data into training and test
#         #         holdout_ratio = 0.1
#         #         train_indices = sample_indices[:int((1-holdout_ratio)*train_replay_buffer.size)]
#         #         val_indices = sample_indices[int((1-holdout_ratio)*train_replay_buffer.size):]
#         #         ctr = 0
#         #         for j in range(0, train_indices.shape[0], batch_size):

#         #             self.optimizers[i].zero_grad()
#         #             batch_input = train_replay_buffer.next_state[train_indices[ctr * batch_size:(
#         #                 ctr + 1) * batch_size]]

#         #             batch_input = torch.from_numpy(batch_input).float()
#         #             batch_input = batch_input.to(self.device)
#         #             batch_target = train_replay_buffer.cost[train_indices[ctr * batch_size:(
#         #                 ctr + 1) * batch_size]].reshape(-1,1)
#         #             batch_target = torch.from_numpy(batch_target).float()
#         #             batch_target = batch_target.to(self.device)


#         #             predicted_values = m(batch_input)
#         #             loss = self.model_loss(predicted_values, batch_target, m)

#         #             train_err = ((torch.argmax(predicted_values,dim=1)-batch_target.view(-1))**2).mean().item()
#         #             loss.backward()
#         #             loss_total += train_err
#         #             ctr += 1
#         #             training_steps+=1
#         #             self.optimizers[i].step()
#         #             if(ctr>=max_iter):
#         #                 break
#         #         ctr = 0
#         #         for j in range(0, val_indices.shape[0], batch_size):
#         #             with torch.no_grad():
#         #                 batch_input = train_replay_buffer.next_state[val_indices[ctr * batch_size:(
#         #                     ctr + 1) * batch_size]]

#         #                 batch_input = torch.from_numpy(batch_input).float()
#         #                 batch_input = batch_input.to(self.device)
#         #                 batch_target = train_replay_buffer.cost[val_indices[ctr * batch_size:(
#         #                     ctr + 1) * batch_size]].reshape(-1,1)
#         #                 batch_target = torch.from_numpy(batch_target).float()
#         #                 batch_target = batch_target.to(self.device)
#         #                 predicted_values = m(batch_input)
#         #                 # import ipdb;ipdb.set_trace()
#         #                 val_err = ((torch.argmax(predicted_values,dim=1)-batch_target.view(-1))**2).mean().item()                 
#         #                 val_loss+=val_err
#         #                 validation_steps+=1
#         #                 ctr+=1
#         #             if(ctr>=max_iter):
#         #                 break
#         #         end = time.time()

#         #     mean_val_loss = val_loss/validation_steps
#         #     if(mean_val_loss<best_loss):
#         #         loss_increase = 0
#         #         best_loss = mean_val_loss
#         #     else:
#         #         loss_increase+=1
            
#         #     if(loss_increase>patience):
#         #         break


#         #         # print("One iteration took: {} seconds".format(end-start))
#         # # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )

#         # return loss_total/(training_steps),val_loss/(validation_steps)



#         loss_increase = 0
#         loss_total=0
#         val_loss = 0
#         training_steps = 0
#         validation_steps =0
#         max_iter = 200
#         max_ratio = 3
#         train_replay_buffer = copy.deepcopy(self.replay_buffer)
#         train_replay_buffer.size = 0
#         train_replay_buffer.ptr = 0


#         if self.use_seperate_buffer:
#             unsafe_size, safe_size = self.unsafe_replay_buffer.size,self.safe_replay_buffer.size
#             # safe_size = min(unsafe_size*max_ratio+1,safe_size)
#             # self.copy_batch_to_replay(train_replay_buffer, self.unsafe_replay_buffer, unsafe_size)
#             # self.copy_batch_to_replay(train_replay_buffer, self.safe_replay_buffer,safe_size )
#             self.copy_batch_to_replay(train_replay_buffer, self.safe_replay_buffer,safe_size )
#             for k in range(safe_size//unsafe_size):
#                 self.copy_batch_to_replay(train_replay_buffer, self.unsafe_replay_buffer, unsafe_size)


#         else:
#             train_replay_buffer = self.replay_buffer

#         for epoch in range(self.epochs):
#             # Train each model for single epoch
#             for i, m in enumerate(self.models):
#                 start = time.time()
#                 m.train()

                
#                 sample_indices =  np.random.permutation(train_replay_buffer.size)
                
#                 # Split the data into training and test
#                 holdout_ratio = 0.1
#                 train_indices = sample_indices[:int((1-holdout_ratio)*train_replay_buffer.size)]
#                 val_indices = sample_indices[int((1-holdout_ratio)*train_replay_buffer.size):]
#                 ctr = 0
#                 for j in range(0, train_indices.shape[0], batch_size):

#                     self.optimizers[i].zero_grad()
#                     batch_input = train_replay_buffer.next_state[train_indices[ctr * batch_size:(
#                         ctr + 1) * batch_size]]

#                     batch_input = torch.from_numpy(batch_input).float()
#                     batch_input = batch_input.to(self.device)
#                     batch_target = train_replay_buffer.cost[train_indices[ctr * batch_size:(
#                         ctr + 1) * batch_size]].reshape(-1,1)
#                     batch_target = torch.from_numpy(batch_target).float()
#                     batch_target = batch_target.to(self.device)


#                     predicted_values = m(batch_input)
#                     loss = self.model_loss(predicted_values, batch_target, m)

#                     train_err = ((torch.argmax(predicted_values,dim=1)-batch_target)**2).mean().item()
#                     loss.backward()
#                     loss_total += train_err
#                     ctr += 1
#                     training_steps+=1
#                     self.optimizers[i].step()
#                     if(ctr>=max_iter):
#                         break
#                 ctr = 0
#                 for j in range(0, val_indices.shape[0], batch_size):
#                     # print(j)
#                     with torch.no_grad():
                        
#                         batch_input = train_replay_buffer.next_state[train_indices[ctr * batch_size:(
#                             ctr + 1) * batch_size]]

#                         batch_input = torch.from_numpy(batch_input).float()
#                         batch_input = batch_input.to(self.device)
#                         batch_target = train_replay_buffer.cost[train_indices[ctr * batch_size:(
#                             ctr + 1) * batch_size]].reshape(-1,1)
#                         batch_target = torch.from_numpy(batch_target).float()
#                         batch_target = batch_target.to(self.device)


#                         predicted_values = m(batch_input)
#                         val_err = ((torch.argmax(predicted_values,dim=1)-batch_target)**2).mean().item()                 
#                         val_loss+=val_err
#                         validation_steps+=1
#                         ctr+=1
#                     if(ctr>=max_iter):
#                         break
#                 end = time.time()
#                 # print("One iteration took: {} seconds".format(end-start))
#         # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )

#         return loss_total/(training_steps),val_loss/(validation_steps)




# class PENN:
#     """
#     (P)robabilistic (E)nsemble of (N)eural (N)etworks
#     """

#     def __init__(
#         self,
#         num_nets,
#         state_dim,
#         action_dim,
#         learning_rate,
#         replay_buffer=None,
#         device=None,
#         hidden_units=[
#             64,
#             64],
#         epochs = 5,
#         train_iters=20):
#         """
#         :param num_nets: number of networks in the ensemble
#         :param state_dim: state dimension
#         :param action_dim: action dimension
#         :param learning_rate:
#         """

#         self.num_nets = num_nets
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.hidden_units = hidden_units

#         # Model also outputs the reward so dimension is state_dim + 1
#         self.max_logvar = nn.Parameter(torch.ones(1, self.state_dim+1, dtype=torch.float32) / 2.0).to(self.device)
#         self.min_logvar = nn.Parameter(- torch.ones(1, self.state_dim+1, dtype=torch.float32) * 10.0).to(self.device)

#         # Create and initialize your model
#         self.models = [self.create_network().to(self.device)
#                        for _ in range(self.num_nets)]
#         self.epochs = epochs
#         self.train_iters=train_iters
#         self.optimizers = [
#             torch.optim.Adam(
#                 list(
#                     self.models[i].parameters()),
#                 lr=learning_rate) for i in range(
#                 self.num_nets)]

#         params = []
#         for m in self.models:
#             params = params + list(m.parameters())

#         self.replay_buffer = replay_buffer
#         print("++++++++++Using PETS style model+++++++++++++")
#         print("++++++++++Device: {}+++++++++++++".format(self.device))

#     def load(self, filename):
#         for i, model in enumerate(self.models):
#             self.models[i].load_state_dict(torch.load(filename + "_" + str(i)))

#     def save(self, filename):
#         for i, model in enumerate(self.models):
#             torch.save(model.state_dict(), filename + "_" + str(i))

#     def proc_state(self,state,output):
#         next_state_reward = output
#         next_state_reward[:,1:]+=state[:,:self.state_dim]
#         return next_state_reward


#     def get_output(self, output):
#         """
#         Argument:
#           output: tf variable representing the output of the keras models, i.e., model.output
#         Return:
#           mean and log variance tf tensors
#         Note that you will still have to call sess.run on these tensors in order to get the
#         actual output.
#         """
#         mean = output[:, 0:self.state_dim+1]
#         logvar = output[:, self.state_dim+1:]
#         logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
#         logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
#         return mean, logvar

#     # Define the network size for dynamics network
#     def create_network(self):
#         model = dynamicsNetwork(
#             self.state_dim, self.action_dim, hidden_units=self.hidden_units)
#         return model

#     def gaussian_loglikelihood(self, x, mu, log_var):

#         inv_var = torch.exp(-log_var)
#         loss = ((x - mu)**2) * inv_var + log_var
#         loss = loss.mean()

#         return loss

#     def model_loss(self, predicted_mean_var, targets, model):
#         predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
#         loss = None
#         loss = self.gaussian_loglikelihood(targets,predicted_mean,predicted_logvar)

#         loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
#         # Removing regularization for speed concerns. Might change later TODO
#         l2_regularization = 0
#         # all_fcs_params = [torch.cat(
#         #     [x.view(-1) for x in fc.parameters()]) for fc in model.fcs]

#         # for params in all_fcs_params:
#         #     l2_regularization += 0.00005 * torch.norm(params, 2)

#         return loss + l2_regularization


#     def get_forward_prediction(self, state, action,mode='ts_inf'):
#         next_states = None

#         if mode=='ts_inf':
#             for i, model in enumerate(self.models):
#                 model.eval()
#                 # Convert the state matrix into tensor if it is not
#                 if not torch.is_tensor(state[i]):
#                     state_t = torch.from_numpy(state[i]).float().to(self.device)
#                 else:
#                     state_t = state[i].float().to(self.device)

#                 action_t = torch.from_numpy(action).float().to(self.device)
#                 # Concatenate the state and action
#                 state_action_concat = torch.cat((state_t, action_t), axis=1)

#                 next_state_t = model(state_action_concat)
#                 nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

#                 # Exponentiate the variance as log variance is obtained

#                 nxt_var_t = torch.exp(nxt_log_var_t)

#                 next_state_t = nxt_mean_t + \
#                     torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
#                 next_state=self.proc_state(state_t,next_state_t)

#                 if next_states is None:
#                     next_states = next_state.unsqueeze(0)
#                 else:
#                     next_states = torch.cat((next_states,next_state.unsqueeze(0)),0)
#         # The next states contains the predicted reward at next_states[:,0]
#         return next_states

#     # Pessimitic reward prediction for MOPO
#     def get_forward_prediction_pessimistic(self, state, action):

#         idx = np.random.randint(low=0, high=self.num_nets)
#         next_state_t = None
#         state_action_concat = torch.cat((state, action), axis=1)
#         max_std = None
#         mean_reward = None
#         lamda = 1
#         for i,m in enumerate(self.models):
#             m.eval()
#             next_state_t_ = m(state_action_concat)
#             nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
#             nxt_std_t = torch.exp(nxt_log_var_t).sqrt()
#             # import ipdb;ipdb.set_trace()
#             if max_std is None:
#                 max_std = nxt_std_t[:,:1]
#                 mean_reward = nxt_mean_t[:,:1]
#             else:
#                 mean_reward+= nxt_mean_t[:,:1]
#                 max_std = torch.max(max_std,nxt_std_t[:,:1])

#             if(i==idx):
#                 # nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t_)
#                 next_state_t = nxt_mean_t + \
#                 torch.randn_like(nxt_mean_t) *  nxt_std_t
        
#         # import ipdb;ipdb.set_trace()

#         next_state = self.proc_state(state,next_state_t)
#         next_state[:,:1]=(mean_reward/self.num_nets)-lamda*max_std

#         return next_state

#     def get_forward_prediction_random_ensemble(self, state, action):
#         # print("Using the model with reward")
#         model = self.models[np.random.randint(low=0, high=self.num_nets)]

#         model.eval()
#         # state_t = torch.from_numpy(state).float().to(self.device)
#         # action_t = torch.from_numpy(action).float().to(self.device)
#         state_action_concat = torch.cat((state, action), axis=1)

#         next_state_t = model(state_action_concat)
#         nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

#         # Exponentiate the variance as log variance is obtained

#         nxt_var_t = torch.exp(nxt_log_var_t)

#         next_state_t = nxt_mean_t + \
#             torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
#         # next_state = next_state_t.detach().cpu().numpy()
        
#         next_state = self.proc_state(state,next_state_t)

#         # next_state = next_state_t
#         # return np.array(next_state)

#         # The next states contains the predicted reward at next_states[:,0]
#         return next_state




#     def train(self, batch_size=256):
#         """
#         Arguments:
#           inputs: state and action inputs.  Assumes that inputs are standardized.
#           targets: resulting states
#         """
#         # patience = 5
#         # best_loss = 1e7
#         # loss_increase = 0
#         # loss_total=0
#         # val_loss = 0
#         # training_steps = 0
#         # validation_steps =0
#         # max_iter = 200
#         # for epoch in tqdm(range(self.epochs),desc='Training dynamics Model: '):
#         #     # Train each model for single epoch
#         #     for i, m in enumerate(self.models):
#         #         start = time.time()
#         #         m.train()

                
#         #         sample_indices =  np.random.permutation(self.replay_buffer.size)
                
#         #         # Split the data into training and test
#         #         holdout_ratio = 0.1
#         #         train_indices = sample_indices[:int((1-holdout_ratio)*self.replay_buffer.size)]
#         #         val_indices = sample_indices[int((1-holdout_ratio)*self.replay_buffer.size):]
#         #         ctr = 0
#         #         for j in range(0, train_indices.shape[0], batch_size):
#         #             self.optimizers[i].zero_grad()



#         #             batch_input = np.concatenate((self.replay_buffer.state[train_indices[ctr * batch_size:(
#         #                 ctr + 1) * batch_size]], self.replay_buffer.action[train_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
#         #             batch_input = torch.from_numpy(batch_input).float()
#         #             batch_input = batch_input.to(self.device)
                    
#         #             batch_target = np.concatenate((self.replay_buffer.reward[train_indices[ctr * batch_size:(
#         #                 ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[train_indices[ctr * batch_size:(
#         #                 ctr + 1) * batch_size]]),axis=1)
#         #             batch_target = torch.from_numpy(batch_target).float()
#         #             batch_target = batch_target.to(self.device)

#         #             batch_target[:,1:]-=batch_input[:,:self.state_dim]
#         #             predicted_mean_var = m(batch_input)
#         #             loss = self.model_loss(predicted_mean_var, batch_target, m)
#         #             predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
#         #             train_err= ((predicted_mean-batch_target)**2).mean()
#         #             loss.backward()
#         #             loss_total += train_err.item()
#         #             ctr += 1
#         #             training_steps+=1
#         #             self.optimizers[i].step()
#         #             if(ctr>=max_iter):
#         #                 break
#         #         ctr = 0
#         #         for j in range(0, val_indices.shape[0], batch_size):
#         #             with torch.no_grad():
#         #                 batch_input = np.concatenate((self.replay_buffer.state[val_indices[ctr * batch_size:(
#         #                     ctr + 1) * batch_size]], self.replay_buffer.action[val_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
#         #                 batch_input = torch.from_numpy(batch_input).float()
#         #                 batch_input =batch_input.to(self.device)
#         #                 batch_target = np.concatenate((self.replay_buffer.reward[val_indices[ctr * batch_size:(
#         #                     ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[val_indices[ctr * batch_size:(
#         #                     ctr + 1) * batch_size]]),axis=1)

#         #                 batch_target = torch.from_numpy(batch_target).float()
#         #                 batch_target = batch_target.to(self.device)

#         #                 batch_target[:,1:]-=batch_input[:,:self.state_dim]
#         #                 predicted_mean_var = m(batch_input)
#         #                 predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
#         #                 # import ipdb;ipdb.set_trace()
#         #                 val_err = ((predicted_mean-batch_target)**2).mean()
#         #                 # loss = self.model_loss(predicted_mean_var, batch_target, m)                        
#         #                 val_loss+=val_err.item()
#         #                 validation_steps+=1
#         #                 ctr+=1
#         #             if(ctr>=max_iter):
#         #                 break
#         #         end = time.time()
#         #     mean_val_loss = val_loss/validation_steps
#         #     if(mean_val_loss<best_loss):
#         #         loss_increase = 0
#         #         best_loss = mean_val_loss
#         #     else:
#         #         loss_increase+=1
            
#         #     if(loss_increase>patience):
#         #         break

#         #         # print("One iteration took: {} seconds".format(end-start))
#         # # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )

#         # return loss_total/(training_steps),val_loss/(validation_steps)
#         loss_total=0
#         val_loss = 0
#         training_steps = 0
#         validation_steps =0
#         max_iter = 200
#         for epoch in tqdm(range(self.epochs),desc='Training dynamics model: '):
#             # Train each model for single epoch
#             for i, m in enumerate(self.models):
#                 start = time.time()
#                 m.train()

                
#                 sample_indices =  np.random.permutation(self.replay_buffer.size)
#                 # print(self.replay_buffer.size)
#                 # Split the data into training and test
#                 holdout_ratio = 0.1
#                 train_indices = sample_indices[:int((1-holdout_ratio)*self.replay_buffer.size)]
#                 val_indices = sample_indices[int((1-holdout_ratio)*self.replay_buffer.size):]
#                 ctr = 0
#                 for j in range(0, train_indices.shape[0], batch_size):
#                     self.optimizers[i].zero_grad()



#                     batch_input = np.concatenate((self.replay_buffer.state[train_indices[ctr * batch_size:(
#                         ctr + 1) * batch_size]], self.replay_buffer.action[train_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
#                     batch_input = torch.from_numpy(batch_input).float()
#                     batch_input = batch_input.to(self.device)
                    
#                     batch_target = np.concatenate((self.replay_buffer.reward[train_indices[ctr * batch_size:(
#                         ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[train_indices[ctr * batch_size:(
#                         ctr + 1) * batch_size]]),axis=1)
#                     batch_target = torch.from_numpy(batch_target).float()
#                     batch_target = batch_target.to(self.device)

#                     batch_target[:,1:]-=batch_input[:,:self.state_dim]
#                     predicted_mean_var = m(batch_input)
#                     loss = self.model_loss(predicted_mean_var, batch_target, m)
#                     predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
#                     train_err= ((predicted_mean-batch_target)**2).mean()
#                     loss.backward()
#                     loss_total += train_err.item()
#                     ctr += 1
#                     training_steps+=1
#                     self.optimizers[i].step()
#                     if(ctr>=max_iter):
#                         break
#                 ctr = 0
#                 for j in range(0, val_indices.shape[0], batch_size):
#                     with torch.no_grad():
#                         batch_input = np.concatenate((self.replay_buffer.state[val_indices[ctr * batch_size:(
#                             ctr + 1) * batch_size]], self.replay_buffer.action[val_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1)
#                         batch_input = torch.from_numpy(batch_input).float()
#                         batch_input =batch_input.to(self.device)
#                         batch_target = np.concatenate((self.replay_buffer.reward[val_indices[ctr * batch_size:(
#                             ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[val_indices[ctr * batch_size:(
#                             ctr + 1) * batch_size]]),axis=1)

#                         batch_target = torch.from_numpy(batch_target).float()
#                         batch_target = batch_target.to(self.device)

#                         batch_target[:,1:]-=batch_input[:,:self.state_dim]
#                         predicted_mean_var = m(batch_input)
#                         predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
#                         val_err = ((predicted_mean-batch_target)**2).mean()
#                         # loss = self.model_loss(predicted_mean_var, batch_target, m)                        
#                         val_loss+=val_err.item()
#                         validation_steps+=1
#                         ctr+=1
#                     if(ctr>=max_iter):
#                         break
#                 end = time.time()
#                 # print("One iteration took: {} seconds".format(end-start))
#         # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps),val_loss/(validation_steps)) )

#         return loss_total/(training_steps),val_loss/(validation_steps)


class PENN_COST:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(
        self,
        num_nets,
        state_dim,
        action_dim,
        learning_rate,
        replay_buffer=None,
        device=None,
        hidden_units=[
            64,
            64],
        epochs = 10,
        train_iters=20):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        self.hidden_units = hidden_units

        # Model also outputs the reward so dimension is state_dim + 1
        self.max_logvar = nn.Parameter(torch.ones(1, self.state_dim+2, dtype=torch.float32) / 2.0).to(self.device)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.state_dim+2, dtype=torch.float32) * 10.0).to(self.device)

        # Create and initialize your model
        self.models = [self.create_network().to(self.device)
                       for _ in range(self.num_nets)]
        self.epochs = epochs
        self.train_iters=train_iters
        self.optimizers = [
            torch.optim.Adam(
                list(
                    self.models[i].parameters()),
                lr=learning_rate) for i in range(
                self.num_nets)]

        params = []
        for m in self.models:
            params = params + list(m.parameters())

        self.replay_buffer = replay_buffer
        print("++++++++++Using PETS style model+++++++++++++")

    def load(self, filename):
        for i, model in enumerate(self.models):
            self.models[i].load_state_dict(torch.load(filename + "_" + str(i)))

    def save(self, filename):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), filename + "_" + str(i))

    def proc_state(self,state,output):
        next_state_reward = output
        next_state_reward[:,2:]+=state[:,:self.state_dim]
        return next_state_reward


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim+2]
        logvar = output[:, self.state_dim+2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # Define the network size for dynamics network
    def create_network(self):
        model = dynamicsNetworkCost(
            self.state_dim, self.action_dim, hidden_units=self.hidden_units)
        return model

    def gaussian_loglikelihood(self, x, mu, log_var):

        inv_var = torch.exp(-log_var)
        loss = ((x - mu)**2) * inv_var + log_var
        loss = loss.mean()

        return loss

    def model_loss(self, predicted_mean_var, targets, model):
        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
        loss = None
        loss = self.gaussian_loglikelihood(targets,predicted_mean,predicted_logvar)

        loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        # Removing regularization for speed concerns. Might change later TODO
        l2_regularization = 0
        # all_fcs_params = [torch.cat(
        #     [x.view(-1) for x in fc.parameters()]) for fc in model.fcs]

        # for params in all_fcs_params:
        #     l2_regularization += 0.00005 * torch.norm(params, 2)

        return loss + l2_regularization


    def get_forward_prediction(self, state, action,mode='ts_inf'):
        next_states = None

        if mode=='ts_inf':
            for i, model in enumerate(self.models):
                model.eval()
                # Convert the state matrix into tensor if it is not
                if not torch.is_tensor(state[i]):
                    state_t = torch.from_numpy(state[i]).float().to(self.device)
                else:
                    state_t = state[i].float().to(self.device)

                action_t = torch.from_numpy(action).float().to(self.device)
                # Concatenate the state and action
                state_action_concat = torch.cat((state_t, action_t), axis=1)

                next_state_t = model(state_action_concat)
                nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

                # Exponentiate the variance as log variance is obtained

                nxt_var_t = torch.exp(nxt_log_var_t)

                next_state_t = nxt_mean_t + \
                    torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
                next_state=self.proc_state(state_t,next_state_t)

                if next_states is None:
                    next_states = next_state.unsqueeze(0)
                else:
                    next_states = torch.cat((next_states,next_state.unsqueeze(0)),0)
        # The next states contains the predicted reward at next_states[:,0]
        return next_states

    def get_forward_prediction_random_ensemble(self, state, action):
        # print("Using the model with reward")
        model = self.models[np.random.randint(low=0, high=self.num_nets)]

        model.eval()
        # state_t = torch.from_numpy(state).float().to(self.device)
        # action_t = torch.from_numpy(action).float().to(self.device)
        state_action_concat = torch.cat((state, action), axis=1)

        next_state_t = model(state_action_concat)
        nxt_mean_t, nxt_log_var_t = self.get_output(next_state_t)

        # Exponentiate the variance as log variance is obtained

        nxt_var_t = torch.exp(nxt_log_var_t)

        next_state_t = nxt_mean_t + \
            torch.randn_like(nxt_mean_t) * nxt_var_t.sqrt()
        # next_state = next_state_t.detach().cpu().numpy()
        
        next_state = self.proc_state(state,next_state_t)

        # next_state = next_state_t
        # return np.array(next_state)

        # The next states contains the predicted reward at next_states[:,0]
        return next_state




    def train(self, batch_size=256):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """

        loss_total=0
        val_loss = 0
        training_steps = 0
        validation_steps =1
        max_iter = 200
        for epoch in range(self.epochs):
            # Train each model for single epoch
            for i, m in enumerate(self.models):
                m.train()

                
                sample_indices =  np.random.permutation(self.replay_buffer.size)
                
                # Split the data into training and test
                holdout_ratio = 0.1
                train_indices = sample_indices[:int((1-holdout_ratio)*self.replay_buffer.size)]
                val_indices = sample_indices[int((1-holdout_ratio)*self.replay_buffer.size):]
                ctr = 0
                for j in range(0, train_indices.shape[0], batch_size):
                    self.optimizers[i].zero_grad()
                    # import ipdb; ipdb.set_trace()
                    batch_input = np.concatenate((self.replay_buffer.state[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]], self.replay_buffer.action[train_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1).copy()
                    batch_input = torch.from_numpy(batch_input).float()
                    batch_input = batch_input.to(self.device)

                    # import ipdb; ipdb.set_trace()
                    batch_target = np.concatenate((self.replay_buffer.cost[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]]),axis=1)

                    batch_target = np.concatenate((self.replay_buffer.reward[train_indices[ctr * batch_size:(
                        ctr + 1) * batch_size]].reshape(-1,1),batch_target),axis=1).copy()
                    # BATCH [reward, cost, state]
                    batch_target = torch.from_numpy(batch_target).float()
                    batch_target = batch_target.to(self.device)

                    batch_target[:,2:]-=batch_input[:,:self.state_dim]
                    predicted_mean_var = m(batch_input)
                    loss = self.model_loss(predicted_mean_var, batch_target, m)
                    predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                    train_err= ((predicted_mean-batch_target)**2).mean()
                    loss_total += ((predicted_mean-batch_target)**2).mean().item()
                    loss.backward()
                    # loss_total += train_err.item()
                    ctr += 1
                    training_steps+=1
                    self.optimizers[i].step()
                    if(ctr>=max_iter):
                        break

                ctr = 0
                for j in range(0, val_indices.shape[0], batch_size):
                    with torch.no_grad():
                        batch_input = np.concatenate((self.replay_buffer.state[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]], self.replay_buffer.action[val_indices[ctr * batch_size:(ctr + 1) * batch_size]]), axis=1).copy()
                        batch_input = torch.from_numpy(batch_input).float()
                        batch_input =batch_input.to(self.device)

                        batch_target = np.concatenate((self.replay_buffer.cost[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]].reshape(-1,1),self.replay_buffer.next_state[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]]),axis=1)

                        batch_target = np.concatenate((self.replay_buffer.reward[val_indices[ctr * batch_size:(
                            ctr + 1) * batch_size]].reshape(-1,1),batch_target),axis=1).copy()

                        batch_target = torch.from_numpy(batch_target).float()
                        batch_target = batch_target.to(self.device)

                        batch_target[:,2:]-=batch_input[:,:self.state_dim]
                        predicted_mean_var = m(batch_input)
                        predicted_mean, predicted_logvar = self.get_output(predicted_mean_var)
                        # val_err = ((predicted_mean-batch_target)**2).mean()
                        val_loss+=((predicted_mean-batch_target)**2).mean().item()
                        # loss = self.model_loss(predicted_mean_var, batch_target, m)                        
                        # val_loss+=val_err.item()
                        validation_steps+=1
                        ctr+=1
                    if(ctr>=max_iter):
                        break
        # print("Training loss: {}, Validation loss: {}".format(loss_total/(training_steps*batch_size),val_loss/(validation_steps*batch_size)) )

        return loss_total/(training_steps),val_loss/(validation_steps)


