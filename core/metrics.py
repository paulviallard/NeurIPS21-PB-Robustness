# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Note: The structure of this file is based on [1]
import torch
import importlib
import inspect
import numpy as np
from core.kl_inv import klInvFunction
import math


###############################################################################

class MetaMetrics(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.metrics"), inspect.isclass
        ):
            if(class_name != "MetaMetrics" and class_name != "Metrics"):
                class_name = class_name.lower()
                class_name = class_name.replace("metrics", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the module
        if("name" not in kwargs):
            class_name = args[0]
        else:
            class_name = kwargs["name"]

        # Getting the module dictionnary
        class_dict = cls.__get_class_dict()

        # Checking that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # Adding the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaMetrics, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Metrics(metaclass=MetaMetrics):

    def __init__(self, name, model):
        '''
        This class instantiates a metric

        Parameters
        ----------
        name: string
            The name of the metric.
        model: module
            The model used in the metric
        '''
        super().__init__()
        self.model = model
        self.param = None

    def numpy_to_torch(self, x, y):
        '''
        Convert a numpy array to a torch tensor

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        if(isinstance(x, np.ndarray)):
            x = torch.tensor(x)
        if(isinstance(y, np.ndarray)):
            y = torch.tensor(y)
        return x, y

    def torch_to_numpy(self, x, y, m):
        '''
        Convert a torch tensor to a numpy array

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        m: np.ndarray
            The tensor to convert
        '''
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = m.detach().numpy()
        return m

    def float_to_numpy_torch(self, x, y, m):
        '''
        Convert a float into a torch tensor or a numpy array
        depending on the type of x and y

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        m: np.ndarray
            The tensor to convert
        '''
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = np.array(m)
        elif(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            m = torch.tensor(m)
        return m

    def load(self, param):
        '''
        Load the parameters

        Parameters
        ----------
        param: torch.Tensor
            The parameters to load
        '''
        if(isinstance(param, torch.Tensor)):
            self.param.data = param.data

    def save(self):
        '''
        Save the parameters
        '''
        return self.param

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        raise NotImplementedError


class GibbsMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)

        # Need the condition when using bound as a metric
        # for validation on non-perturbed sample
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Reduction of the network's sampling
        # (if only one network, it stays the same)
        x = torch.mean(x, dim=2)
        m = 0.5*(1.0-x*y)
        m = torch.mean(m)

        return self.torch_to_numpy(x, y, m)


class AttackGibbsMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)

        m = 0.5*(1.0-x*y)
        m = torch.mean(m)
        return self.torch_to_numpy(x, y, m)


class GibbsMaxMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        # E(h~Q) 1/m \sum 1/2(1 - min_{epsilon}(y*h(x+epsilon)))

        x, y = self.numpy_to_torch(x, y)

        y_ = y.unsqueeze(axis=1)

        # take the noise that minimise the margin of each hypothesis
        m, _ = torch.min(x*y_, axis=1)

        # Apply the linear loss
        m = 0.5*(1.0-m)

        # Average on all the data
        m = torch.mean(m, axis=0)

        # Apply the majority vote
        post = self.model.post.squeeze(axis=1)

        m = torch.sum(post*m)

        return self.torch_to_numpy(x, y, m)


class GibbsMaxTVMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)
        m, _ = torch.min(x*y, axis=1)
        m = 0.5*(1.0-m)
        m = torch.mean(m)
        return self.torch_to_numpy(x, y, m)


class MajorityVoteMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)

        # Reduction of the network's sampling
        # (if only one network, it stays the same)
        x = torch.mean(x, dim=2)
        m = 0.5*(1.0-torch.sign(x)*y)
        m = torch.mean(m)

        return self.torch_to_numpy(x, y, m)


class MajorityVoteMaxMetrics():

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)
        # Reduction of the network's sampling
        # (if only one network, it stays the same)
        x = torch.mean(x, dim=2)

        m = 0.5*(1.0-torch.sign(x)*y)

        m, _ = torch.max(m, dim=1)

        m = torch.mean(m)
        return self.torch_to_numpy(x, y, m)


class BoundTh7Metrics():

    def __init__(self, name, model, m, delta, t):
        super().__init__(name, model)
        self.m = float(m)
        self.delta = delta
        self._risk = Metrics("gibbsmax", model)
        self.t = float(t)
        self.tv = torch.tensor(-1.0)

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Check what it is doing during training !
        # if (x.shape[2] != 1) -> we are not training
        if(x.shape[2] != 1):
            x_ = x
        else:
            x_ = torch.mean(x, axis=2)

        r = self._risk.fit(x_, y)

        m = self.m
        t = self.t
        div = self.model.div
        delta = self.delta

        self.div = div
        self.r = r
        b = r + torch.sqrt((1.0/m)*(
            div+math.log((2.0*t*math.sqrt(m))/delta)))
        b = 2.0*b

        return self.torch_to_numpy(x, y, b)


class BoundTh7TVMetrics():

    def __init__(self, name, model, m, delta, t):
        '''
        Parameters
        ----------
        name: string
            The name of the metric.
        model: module
            The model used in the metric
        m: int
            The number of examples
        delta: float
            The delta parameter in the bounds
        t: int
            The T parameter in the bounds
        '''
        super().__init__(name, model)
        self.m = float(m)
        self.delta = delta
        self._risk = Metrics("gibbsmaxtv", model)
        self.t = float(t)

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        x, y = self.numpy_to_torch(x, y)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # if (x.shape[2] != 1) -> we are not training
        if(x.shape[2] != 1):
            post = self.model.post.T.unsqueeze(0)
            x_ = torch.sum(post*x, axis=2)
        else:
            x_ = torch.mean(x, axis=2)

        r = self._risk.fit(x_, y)

        m = self.m
        t = self.t
        div = self.model.div
        delta = self.delta

        # if (x.shape[2] != 1) -> we are not training
        if(x.shape[2] != 1):
            y_ = y.unsqueeze(1)
            y_ = y_.repeat((1, x.shape[1], x.shape[2]))

            _, dis_rho = torch.min(x*y_, axis=1)
            dis_rho = torch.zeros_like(
                y_.permute(0, 2, 1)).scatter_(2, dis_rho.unsqueeze(2), 1)
            dis_rho = dis_rho.float()
            dis_pi = torch.mean(dis_rho, dim=1).unsqueeze(1)

            # For numerical stability
            dis_pi[dis_pi == 0.0] = -1.0
            tv = dis_rho/dis_pi
            tv[tv <= 0.0] = 0.0
            dis_pi[dis_pi == -1.0] = 0.0

            tv = 0.5*torch.abs(tv-1.0)
            tv = torch.sum(dis_pi*tv, dim=2)
            tv = tv@self.model.post
            tv = torch.mean(tv)
        else:
            tv = torch.tensor(0.0)
        self.tv = tv
        self.div = div
        self.r = r
        b = r + tv + torch.sqrt((1.0/m)*(
            div+math.log((2.0*t*math.sqrt(m))/delta)))
        b = 2.0*b
        return self.torch_to_numpy(x, y, b)


class BoundTh6Metrics():

    def __init__(self, name, model, m, delta, t):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        super().__init__(name, model)
        self.param = torch.nn.Parameter(torch.ones(1)*0.05)
        self.param.requires_grad = True
        self.m = float(m)
        self.delta = delta
        self._risk = Metrics("gibbs", model)
        self.t = float(t)
        self.tv = torch.tensor(-1.0)

    def fit(self, x, y):
        '''
        Compute the metric

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The x tensor
        y: np.ndarray or torch.Tensor
            The y tensor
        '''
        t = self.t
        r = self._risk.fit(x, y)
        m = self.m
        d = self.delta

        # NOTE: We assume that we already have the good divergence
        div = self.model.div

        self.r = r
        self.div = div

        b = (1.0/m)*(div+math.log(((m+1.0)*t)/d))
        b = klInvFunction.apply(r, b, "MAX")

        return 2.0*b

###############################################################################

# References:
#  [1] https://github.com/paulviallard/ECML21-PB-CBound/ (under MIT license)
