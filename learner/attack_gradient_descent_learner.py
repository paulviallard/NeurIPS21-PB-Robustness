# Author: Guillaume VIDOT (AIRBUS SAS)
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from learner.gradient_descent_learner import GradientDescentLearner


###############################################################################

class AttackGradientDescentLearner(GradientDescentLearner):

    def __init__(
        self, model, loss, metric, attack, optim, device,
        epoch=10, batch_size=None
    ):
        '''
        This class instantiates a ''learner''

        Parameters
        ----------
        model: module
            The model to optimize
        loss:
            The loss to use in the optimization
        attack:
            The attack to use in the optimization
        optim:
            The optimizer
        device:
            The device where the model is
        epoch: int
            The number of epochs
        batch_size: int
            The number of examples in the batch
        '''
        super().__init__(
            model, loss, metric, optim, device,
            epoch=epoch, batch_size=batch_size)
        self.attack = attack

    def _optimize(self, batch):
        '''
        Compute a gradient descent step

        Parameters
        ----------
        batch: dict
            The batch containing the example
        '''
        # We attack the examples and forward in the model
        # (=> adversarial training)
        x_, y_ = self.attack.fit(batch["x"], batch["y"])
        batch["x"] = x_
        batch["y"] = y_
        batch["x"] = batch["x"].detach()
        self.model(batch)

        if(self.attack.nb_noise != -1):
            out_full = torch.reshape(
                self.model.out,
                (-1, self.attack.nb_noise, self.model.out.shape[1]))
            y = torch.reshape(batch["y"], (-1, self.attack.nb_noise))
            y = y[:, 0:1]
        else:
            out_full = self.model.out.unsqueeze(1)
            y = batch["y"]

        # We compute the (adversarial) loss, risk and the KL divergence
        self._loss = self.loss(out_full, y)
        self._tS = self.metric(
            self.model.pred.cpu(), batch["y"].cpu().detach().numpy())
        self._kl = self.model.div

        # We perform the gradient step
        self.optim.zero_grad()
        self._loss.backward()
        self.optim.step()

###############################################################################
