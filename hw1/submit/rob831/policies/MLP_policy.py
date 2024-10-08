import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy
import pdb

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO return the action that the policy prescribes
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = torch.Tensor(observation).to(ptu.device)
        if self.discrete:
            action_logits = self.logits_na(observation)
            probs = torch.softmax(action_logits, dim=-1)
            dist = distributions.Categorical(probs)
            action = dist.sample()
        else:
            std = torch.exp(self.logstd)  # Exponentiate logstd to get std
            dist = torch.distributions.Normal(self.mean_net(observation), std)
            action = dist.sample()
        return action.cpu().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            action_logits = self.logits_na(observation)
            probs = torch.softmax(action_logits, dim=-1)
            dist = distributions.Categorical(probs)
            action = dist.sample()
        else:
            std = torch.exp(self.logstd)  # Exponentiate logstd to get std
            dist = torch.distributions.Normal(self.mean_net(observation), std)
            action = dist.sample()
        return action


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        ob_tensor = torch.tensor(observations).to(ptu.device)
        ac_tensor = torch.tensor(actions).to(ptu.device)
        # TODO: update the policy and return the loss
        if self.discrete:
            pred_action = self.logits_na(ob_tensor)
            # pred_action = torch.exp(pred_action)
        else:
            mean = self.mean_net(ob_tensor)
            std = torch.exp(self.logstd) 
            dist = torch.distributions.Normal(mean, std)
            pred_action = dist.rsample()
        loss = self.loss(pred_action, ac_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
