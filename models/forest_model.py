# Author: Guillaume VIDOT (AIRBUS SAS)
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np


class DecisionTree(torch.nn.Module):
    '''
    This is the class ``DecisionTree''.

    This source code is based on [1].
    The reader may refer to [1] for more details.

    Parameters
    ----------
    device:
        The device you work on (e.g. cuda or cpu).
    depth: int
        The depth of the decision tree
    mask: float
        The percentage of the input feature that we mask.
    seed: int
        The seed for the random draw
    '''
    def __init__(self, device, depth=2, mask=0.5, seed=None):
        super(DecisionTree, self).__init__()
        self.__device = device
        self.depth = depth
        self.mask = mask

        self.seed = seed

        self.f_mask = None
        self.f_linear = torch.nn.Parameter(None)
        self.f_bias = torch.nn.Parameter(None)
        self.pi = torch.nn.Parameter(None)

    def init_f(self, n_features):
        '''
           Initialize the module

           Parameters
            ----------
            n_feature: int
                The number of features
        '''
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        mask = torch.rand((2**self.depth-1, n_features))
        mask_min = mask.min(dim=1)[0].unsqueeze(1)
        mask = (mask-mask_min)
        mask_max = mask.max(dim=1)[0].unsqueeze(1)
        mask = mask/mask_max
        self.f_mask = (mask > self.mask).float()

        self.f_linear.data = torch.empty(2**self.depth-1,
                                         n_features, device=self.__device)
        torch.nn.init.xavier_normal_(self.f_linear.data)
        self.f_bias.data = torch.zeros(2**self.depth-1, device=self.__device)
        self.pi.data = torch.empty(1, 2**self.depth, device=self.__device)
        torch.nn.init.xavier_normal_(self.pi.data)

    def reinitialize(self):
        pass

    def set_mu(self, mu, d, d_, i, depth):
        '''
           Recursive computation of mu

           Parameters
           ----------
            mu:
                The probability mu of taking a path (in the tree)
            d:
                The current probability to take a path at a particular node
            d_:
                The probability to arrive at the current node
            i:
                The current node number
            depth:
                The current depth

           Return
           ------
           t: array(float)
                Return the mu for each path of the tree
        '''
        if(self.depth == depth):
            return d_.unsqueeze(1)
        else:
            t = torch.cat((
                self.set_mu(mu, d, d_*d[:, i], 2*i+1, depth+1),
                self.set_mu(mu, d, d_*(1.0-d[:, i]), 2*i+2, depth+1)
            ), axis=1)
            return t

    def forward(self, batch):
        '''
           Take a batch and return the associated predictions to it by
           forwarding the input to the tree.

           Note: store the predictions in a class variable.

           Parameters
           ----------
            batch:
                batch of inputs
        '''
        x = batch["x"]

        if(self.f_linear.shape[0] == 0
           and self.f_bias.shape[0] == 0
           and self.f_mask is None
           and self.pi.shape[0] == 0
           ):
            self.init_f(x.shape[1])

        f = torch.nn.functional.linear(
            x, self.f_linear, bias=self.f_bias)
        d = torch.sigmoid(f)

        mu = torch.ones(x.shape[0], 2**self.depth)
        l = self.set_mu(mu, d, 1.0, 0, 0)
        label = torch.tanh(self.pi)

        r = l@label.T
        self.out = r
        self.pred = torch.sign(r).int()


class Module(torch.nn.Module):
    '''
    This is the class ``Module''.

    This source code is based on [1].
    The reader may refer to [1] for more details.

    Parameters
    ----------
    device:
        The device you work on (e.g. cuda or cpu).
    sample_size: int
        The number of trees
    depth: int
        The depth of the decision tree
    mask: float
        The percentage of the input feature that we mask.
    init_prior: Boolean
        Boolean to know if we are learning the prior or the posterior.
        If True, we load a prior then we are learning the posterior.
        If False, we are learning the prior.
    signed: Boolean
        Either we take the signed of each tree prediction (true)
        or their logits (false) to make the majority vote.
    learn_post: Boolean
        Boolean to know if we are learning the posterior.
        If True, we are learning the posterior.
        If False, we are learning the prior.
    '''
    def __init__(self, device, sample_size=25, depth=2, mask=0.5,
                 deterministic=False, init_prior=False, signed=True,
                 learn_post=False
                 ):
        super(Module, self).__init__()
        self.deterministic = deterministic
        self.__device = device
        self.depth = depth
        self.mask = mask

        self.init_prior = init_prior
        self.signed = signed
        self.learn_post = learn_post

        if(sample_size == -1 or sample_size is None):
            sample_size = 25
        self.sample_size = sample_size

        self.prior = torch.nn.Parameter(
            torch.ones(
                self.sample_size, 1,
                device=self.__device)/self.sample_size, requires_grad=False)
        self.post_ = torch.nn.Parameter(
            torch.ones(self.sample_size, 1, device=self.__device))

        self.tree_list = torch.nn.ModuleList([])
        for i in range(self.sample_size):
            self.tree_list.append(
                DecisionTree(device, depth=self.depth, mask=self.mask, seed=i))

    def reinitialize(self):
        pass

    def forward(self, batch):
        '''
           Take a batch and return the associated predictions to it by
           forwarding the input to the tree.

           Note: store the predictions in a class variable.

           Parameters
           ----------
            batch:
                batch of inputs
        '''
        x = batch["x"]

        out_list = None
        pred_list = None

        if(self.sample_size == -1 or self.sample_size is None):
            self.sample_size = 25
        for i in range(self.sample_size):
            self.tree_list[i](batch)
            if(out_list is None and pred_list is None):
                out_list = self.tree_list[i].out
                pred_list = self.tree_list[i].pred.float()
            else:
                out_list = torch.cat(
                    (out_list, self.tree_list[i].out), axis=1)
                pred_list = torch.cat(
                    (pred_list, self.tree_list[i].pred.float()), axis=1)

        # We take h instead of sign(h)
        if not self.signed:
            pred_list = out_list.detach()

        # When learning posterior, we do not learn the weights of the tree
        # if non signed : we get "out_list" without gradients
        # if signed : we get "pred_list" (which has no gradients)
        if((self.learn_post or self.init_prior) and not(x.requires_grad)):
            out_list = pred_list

        self.post = torch.nn.functional.softmax(self.post_, dim=0)
        self.div = torch.sum(self.post*torch.log(self.post/self.prior))

        self.out = out_list@self.post
        self.out_list = out_list

        self.pred = torch.sign(pred_list@self.post).int()

    def load_state_dict(self, state_dict, beginning=False):
        '''
           Load an existing state_dict in the decision tree model.
           Store the prior data only of we are at the beginning of the learning
           of the posterior.

           Parameters
           ----------
            state_dict:
                The dict containing the parameters that will be loaded
            beginning: Boolean
                Boolean that inform whether we are at the beginning
                of the learning
        '''
        for i in range(self.sample_size):
            if(self.tree_list[i].f_linear.shape[0] == 0
               and self.tree_list[i].f_bias.shape[0] == 0
               and self.tree_list[i].f_mask is None
               and self.tree_list[i].pi.shape[0] == 0):
                self.tree_list[i].init_f(
                    state_dict['tree_list.0.f_linear'].shape[1])

        super(Module, self).load_state_dict(state_dict)

        if(self.learn_post and beginning):
            self.prior.data = torch.nn.functional.softmax(
                self.post_.data.detach(), dim=0)

###############################################################################

# References:
# [1] Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, and Samuel Rota
#     Bulò. Deep Neural Decision Forests. In IJCAI, 2016
