# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import numpy as np


###############################################################################

class MetaEarlyStoppingLearner(type):

    def __call__(cls, *args, **kwargs):
        bases = (cls, args[0].__class__, )
        new_cls = type(cls.__name__, bases, {})
        return super(MetaEarlyStoppingLearner, new_cls
                     ).__call__(*args, **kwargs)


class EarlyStoppingLearner(metaclass=MetaEarlyStoppingLearner):

    def __init__(self, learner, criteria, val_epoch=10):
        '''
        The class instantiates the ''learner'' that perform early stopping

        Parameters
        ----------
        learner:
            The learner used during the optimization
        criteria:
            The criteria used to perform the early stopping
        val_epoch: int
            The number of epochs
        '''
        self.__dict__ = learner.__dict__
        self.val_epoch = val_epoch
        self.criteria = criteria

    def fit(self, x_train, y_train, x_val, y_val):
        '''
        The ''learner'' performs early stopping

        Parameters
        ----------
        x_train: np.ndarray
            The training inputs
        y_train: np.ndarray
            The training labels
        x_val: np.ndarray
            The validation inputs
        y_val: np.ndarray
            The validation labels
        '''
        crit_val_best = math.inf
        y_val = np.expand_dims(y_val, 1)

        # For each validation epoch,
        for epoch in range(self.val_epoch):

            logging.info(("Running validation epoch [{}/{}] ...\n").format(
                epoch+1, self.val_epoch))

            # We fit the training examples
            super().fit(x_train, y_train)

            # We compute the predictions associated to the validation set
            predict_val = super().output(x_val, y_val, out_type="attack")
            # (and compute the criteria)
            crit_val = self.criteria(predict_val, y_val)

            # We save the model accordingly
            logging.info(("critval info: {:.4f} \n").format(crit_val))
            if(crit_val_best > crit_val):
                logging.info(("{:.4f} > {:.4f} -> saving ...\n").format(
                    crit_val_best, crit_val))
                crit_val_best = crit_val
                b = self.save()

        # and load the best at the end
        self.load(b)

        return self

###############################################################################
