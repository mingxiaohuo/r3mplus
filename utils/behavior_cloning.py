# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from r3meval.utils.logger import DataLog
from tqdm import tqdm


class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 set_transforms = False,
                 finetune = False,
                 proprio = 0,
                 encoder_params = [],
                 **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.finetune = finetune
        self.proprio = proprio
        self.steps = 0

        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(list(self.policy.trainable_params) + list(encoder_params), lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

        # make logger
        if self.save_logs:
            self.logger = DataLog()

    def compute_transformations(self, e):
        # get transformations
        if self.expert_paths == [] or self.expert_paths is None:
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            # observations = np.concatenate([path["observations"] for path in self.expert_paths])
            observations = np.concatenate([path["images"] for path in self.expert_paths])
            observations = self.encodefn(observations, finetune=self.finetune)
            actions = np.concatenate([path["actions"] for path in self.expert_paths])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

    def loss(self, data, idx=None, traj_idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx, traj_idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None, traj_idx=None):
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        # obs = self.encodefn(obs,finetune=self.finetune)  #not voltron
        obs_first=data['observations_first'][traj_idx]
        ## Encode images with environments encode function
        obs = self.encodefn(obs, obs_first,finetune=self.finetune)
        act_expert = data['expert_actions'][idx]
        if type(obs) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False).cuda()

        ## Concatenate proprioceptive data
        if self.proprio:
            proprio= data['proprio'][idx]
            if type(proprio) is not torch.Tensor:
                proprio = Variable(torch.from_numpy(proprio).float(), requires_grad=False).cuda()
            obs = torch.cat([obs, proprio], -1)
        if type(act_expert) is not torch.Tensor:
            act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
        act_pi = self.policy.model(obs)
        return self.loss_criterion(act_pi, act_expert.detach())

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        num_samples = data["observations"].shape[0]
        num_traj = data["observations_first"].shape[0]
        traj_length = num_samples/num_traj
        # log stats before
        if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)

        # train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                traj_id=[int(x/traj_length) for x in rand_idx]
                loss = self.loss(data, idx=rand_idx,traj_idx=traj_id)
                loss.backward()
                self.optimizer.step()
                self.steps += 1
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        # log stats after
        if self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))

    def train(self, pixel=True, **kwargs):
        ## If using proprioception, select the first N elements from the state observation
        ## Assumes proprioceptive features are at the front of the state observation
        if self.proprio:
            proprio = np.concatenate([path["observations"] for path in self.expert_paths])
            proprio = proprio[:, :self.proprio]
        else:
            proprio = None

        ## Extract images
        if pixel:
            observations = np.concatenate([path["images"] for path in self.expert_paths])
            observations_first = np.concatenate([np.expand_dims(path["images"][0], axis=0) for path in self.expert_paths])
        else:
            observations = np.concatenate([path["observations"] for path in self.expert_paths])

        ## Extract actions
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        data = dict(observations=observations, observations_first=observations_first, proprio=proprio, expert_actions=expert_actions)
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)