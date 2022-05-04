# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import logging
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from core.numpy_dataset import NumpyDataset


###############################################################################

class GradientDescentLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, model, loss, metric, optim, device,
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
        metric:
            The metric to use in the optimization
        optim:
            The optimizer
        device:
            The device where the model is
        epoch: int
            The number of epochs
        batch_size: int
            The number of examples in the batch
        '''
        self.model = model
        self.loss = loss
        self.metric = metric
        self.optim = optim
        self.device = device

        self.epoch = epoch
        self.batch_size = batch_size

        self.list_loss = list()

    def fit(self, X, y):
        '''
        The ''learner'' performs early stopping

        Parameters
        ----------
        X: np.ndarray
            The training inputs
        y: np.ndarray
            The training labels
        '''
        # We initialize the dataset
        data = NumpyDataset({
            "x_train": X,
            "y_train": y})
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        # We compute batch size
        num_batch = int(len(data)/self.batch_size)
        if(len(data) % self.batch_size != 0):
            num_batch += 1

        # For each epoch,
        for epoch in range(self.epoch):

            logging.info(("Running epoch [{}/{}] ...\n").format(
                epoch+1, self.epoch))

            loss_sum = 0.0
            tS_sum = 0.0

            # for each batch,
            for i, batch in enumerate(loader):

                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long).unsqueeze(0).T
                self._mS = float(batch["size"][0])

                # we optimize the model with the current batch
                self._optimize(batch)

                # we compute mean loss
                loss_sum += self._loss
                loss_mean = loss_sum/(i+1)

                tS_sum += self._tS.item()
                tS_mean = tS_sum/(i+1)

                # we print loss and error
                logging.info((
                    "[{}/{}] - KL {:.4f}" + " - loss {:.4f} - tS {:.4f}\r"
                ).format(i+1, num_batch, self._kl, loss_mean, tS_mean))

                if i+1 == num_batch:
                    logging.info("\n")
            self.list_loss.append(loss_mean.cpu().detach().item())

        return self

    def output(self, X, y=None, out_type=None):
        '''
        Output the predictions of the model

        Parameters
        ----------
        X: np.ndarray
            The inputs that require the predictions
        y: np.array
            The associated labels
        out_type: str
            The ''type'' of the output
        '''

        # We construct the dataset
        data = NumpyDataset({"x_test": X})
        if(out_type == "attack" and y is not None):
            data = NumpyDataset({"x_test": X, "y_test": y})

        # We construct the batches
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        # For each batch,
        output = None
        for i, batch in enumerate(loader):

            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)
            if(out_type == "attack" and y is not None):
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.float32)
                x_, y_ = self.attack.fit(batch["x"], batch["y"])
                batch["x"] = x_
                batch["y"] = y_
                batch["x"] = batch["x"].detach()

            # we forward the examples in the model
            # (the examples can be adversarial)
            self.model(batch)

            # and we concatenate the examples
            out = self.model.out.cpu().detach().numpy()
            if(out_type == "list"):
                out = self.model.out_list.cpu().detach().numpy()

            if(output is None):
                output = out
            else:
                output = np.concatenate((output, out))

        return output

    def save(self):
        '''
        We save the parameters of the model
        '''
        return copy.deepcopy(self.model.state_dict())

    def load(self, state_dict, beginning=False):
        '''
        We load the parameters in the state_dict

        state_dict: dict
            The parameters of the model to be loaded
        beginning: bool
            Boolean to check that the loading is the first one
        '''
        return self.model.load_state_dict(state_dict, beginning)

    def _optimize(self):
        '''
        Compute a gradient descent step
        '''
        raise NotImplementedError

###############################################################################
