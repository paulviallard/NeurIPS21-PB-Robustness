# Author: Guillaume VIDOT (AIRBUS SAS)
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from core.numpy_dataset import NumpyDataset
import torch
import importlib
import inspect
import logging
import numpy as np


###############################################################################

class MetaAttack(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.attack"), inspect.isclass
        ):
            if(class_name != "MetaAttack" and class_name != "Attack"):
                class_name = class_name.lower()
                class_name = class_name.replace("attack", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the module
        class_name = args[0]

        # Getting the module dictionnary
        class_dict = cls.__get_class_dict()

        # Checking that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # Adding the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaAttack, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Attack(metaclass=MetaAttack):
    '''
    This is the class ``Attack''.

    This is the class that you call to instanciate any attack
    (e.g. PGD attack or IFGSM attack) by specifying the right name.

    Parameters
    ----------
    name: string
        The name of the attack. Could be one among the following list
        -> (nothing, uniform, pgd, fgsm, iterativefgsm, uniformnoisepgd,
            uniformnoiseiterativefgsm)
    model: module
        The model that meant to be fooled.
    device: device
        The device you work on (e.g. cuda or cpu).
    loss: metrics
        The loss function
    '''
    def __init__(self, name, model, device, loss):
        super().__init__()
        self.name = name
        self.model = model
        self.device = device
        self.loss = loss
        self.nb_noise = -1

    def fit(self, x, y):
        '''
            Raise an exception if not implemented
        '''
        raise NotImplementedError


class NothingAttack():
    '''
        NothingAttack corresponds to no attack.
    '''
    def fit(self, x, y):
        '''
            Parameters
            ----------
            x: tensor
                The input data
            y: tensor
                The label data

            Return
            ------
            x: tensor
                The same input data
            y: tensor
                The same label data
        '''
        return x, y


class UniformNoiseAttack():
    '''
        UniformNoiseAttack samples uniform noises and generate a perturbed
        dataset from the orginal one.
        Note1: if the size of the original dataset is "m" and you choose to
        samples 100 noises (parameter "n"), the perturbed dataset will have a
        size of m*n.
        Note2: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        eps_max: float
            The epsilon max that delimit the neighborhood
            around an orignal example.
            Default: 1.0
        min_clip: float
            The minimum value that an example can reach (usually
            corresponds the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach (usually
            corresponds the maximum value of the input domain).
            Default: None
        n: integer
            Number of noises to sample per example.
            Default: -1
        batch_size: integer
            The size of the batch.
            Default: 64
        lp_norm: string
            The norm used to compute the distance between the
            orginal example and the perturbed example.
            Default: linf
    '''
    def __init__(self, name, model, device, loss,
                 eps_max=1.0, min_clip=None, max_clip=None, n=-1,
                 batch_size=64, lp_norm="linf"):
        super().__init__(name, model, device, loss)
        self.eps_max = eps_max
        self.nb_noise = n
        self.batch_size = batch_size
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.lp_norm = lp_norm

    def _attack(self, x, y):
        '''
           Private function that take as input x and y and
           return their perturbed counterparts.
           It handles the different norms (either linf or l2),
           the sampling and the clip.

           Parameters
            ----------
            x: tensor/ndarray
                The input data
            y: tensor/ndarray
                The label data

           Return
            ------
            x: ndarray
                The perturbed input data (uniform noise attack)
            y: ndarray
                The same label data
        '''
        x_ = x
        y_ = y

        if(isinstance(x, np.ndarray)):
            x_ = torch.tensor(x)
        if(isinstance(y, np.ndarray)):
            y_ = torch.tensor(y)

        x_shape = x_.shape
        if(self.nb_noise > 1):
            x_shape = list(x_.shape)
            x_shape.append(self.nb_noise)

        if self.lp_norm == "linf":
            eps = self.eps_max*(2.0*(torch.rand(
                x_shape,
                device=x_.device
            )-0.5))
        elif self.lp_norm == "l2":
            eps = torch.tensor(np.random.uniform(
                self.min_clip, self.max_clip, x_shape),  device=self.device)
            x__ = x_
            if(self.nb_noise > 1):
                x__ = x_.unsqueeze(2)
            eps = eps - x__

            if(self.nb_noise > 1):
                norm = torch.reshape(eps, (eps.shape[0], -1, self.nb_noise))
                norm = torch.norm(norm, p=2, dim=1, keepdim=True)
            else:
                norm = torch.reshape(eps, (eps.shape[0], -1))
                norm = torch.norm(norm, p=2, dim=1, keepdim=True)
            factor = torch.min(self.eps_max / norm, torch.ones_like(norm))
            eps = factor * eps
            eps = eps.type(torch.float32)

        if(isinstance(x_, np.ndarray)):
            eps = eps.numpy()

        if(self.nb_noise > 1):
            x_ = x_.unsqueeze(2)

        x_ = x_ + eps

        if(self.min_clip is not None and self.max_clip is not None):
            x_ = torch.clamp(
                x_, self.min_clip, self.max_clip)

        if(self.nb_noise > 1):
            y_ = y_.unsqueeze(1)
            y_ = y_.repeat(1, 1, x_.shape[2])

            x_y = torch.cat((x_, y_.float()), axis=1)

            x_y = x_y.permute(0, 2, 1)

            x_y = x_y.reshape((x_y.shape[0]*x_y.shape[1], x_y.shape[2]))

            x_ = x_y[:, :-1]
            y_ = x_y[:, -1]
            y_ = y_.long()

        if(isinstance(x, np.ndarray)):
            x_ = x_.detach().numpy()
        if(isinstance(y, np.ndarray)):
            y_ = y_.detach().numpy()

        return x_, y_

    def fit(self, x, y):
        '''
            A wrapper of the _attack function that handle
            the different type of x and y.
            If it is a Tensor we are at training time.
            If it is an ndarray we are at testing time.
            Parameters
            ----------
            x: tensor/ndarray
                The input data
            y: tensor/ndarray
                The label data

            Return
            ------
            x: ndarray
                The perturbed input data
            y: ndarray
                The same label data
        '''

        if(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            x, y = self._attack(x, y)

            return x, y

        elif(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            data = NumpyDataset({
                "x_train": x,
                "y_train": y})
            loader = torch.utils.data.DataLoader(
                data, batch_size=self.batch_size)

            # Computing batch size
            num_batch = int(len(data)/self.batch_size)
            if(len(data) % self.batch_size != 0):
                num_batch += 1

            x = None
            y = None

            for i, batch in enumerate(loader):

                logging.info(("Attacking batch [{}/{}] ...\r").format(
                    i+1, num_batch))

                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long)

                x_, y_ = self._attack(batch["x"], batch["y"])
                batch["x"] = x_
                batch["y"] = y_

                if(x is None):
                    x = batch["x"]
                    y = batch["y"]
                else:
                    x = torch.cat((x, batch["x"]))
                    y = torch.cat((y, batch["y"]))

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            logging.info("\n")
            return x, y


class PGDAttack():
    '''
        PGDAttack finds an adversarial example using the
        Projected Gradient Descent technique.
        It maximize the loss of the model until fooling it.

        Note1: it handles the linf norm and l2 norm.
        Note2: To get more details on this attack
        see https://arxiv.org/pdf/1706.06083.pdf
        Note3: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        k: integer
            The number of step to perform for the PGD.
            Default: 30
        step: float
            The step to apply for the update of a PGD iteration.
            Default: 0.01
        eps_max: float
            The epsilon max that delimit the neighborhood
            around an orignal example.
            Default: 0.3
        min_clip: float
            The minimum value that an example can reach
            (usually corresponds the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach
            (usually corresponds the maximum value of the input domain).
            Default: None
        batch_size: integer
            The size of the batch.
            Default: 64
        random_init: Boolean
            Whether or not we apply a random initialization
            to x using uniform noise.
        lp_norm: string
            The norm used to compute the distance between
            the orginal example and the perturbed example.
            Default: linf
    '''
    def __init__(self, name, model, device, loss, k=30, step=0.01, eps_max=0.3,
                 min_clip=None, max_clip=None, batch_size=64, random_init=True,
                 lp_norm="linf"):
        super().__init__(name, model, device, loss)
        self.loss = loss
        self._k = k
        self._step = step
        self._eps_max = eps_max
        self._min_clip = min_clip
        self._max_clip = max_clip
        self._batch_size = batch_size
        self._random_init = random_init
        self._lp_norm = lp_norm

        self.x_orig = None
        self.adversarial = None

        logging.info(("\nDEBUG: init k {}\n").format(self._k))
        logging.info(("\nDEBUG: init lp_norm {}\n").format(self._lp_norm))

    def _attack(self, batch):
        '''
            Private function that take as input a batch and return its
            perturbed counterparts.
            It handles the different norms (either linf or l2), the sampling
            and the clip.

            Note: The batch is modify with the perturbed batch, so this
            function does not need to return it.
            Note2: Because of our heuristic that stop perturbing a sample when
            it become adversarial the batch is shuffled

            Parameters
            ----------
            batch: dict
                The input data

            Return
            ------
            x_orig: ndarray
                The original input data (with the new data order)
            adversarial: ndarray(Boolean)
                The boolean array that keep track of which example
                became adversarial after the PGD attack.
        '''
        x_orig = copy.deepcopy(batch["x"].data)
        x_orig_tracker = None
        adversarial = None
        if self._random_init:
            # random initialization
            if self._lp_norm == "linf":
                batch["x"] = batch["x"] + torch.tensor(
                    np.random.uniform(-self._eps_max, self._eps_max,
                                      batch["x"].shape), device=self.device)

            elif self._lp_norm == "l2":
                init_pert = torch.tensor(np.random.uniform(
                    self._min_clip, self._max_clip, batch["x"].shape),
                    device=self.device)
                init_pert = init_pert - batch["x"]

                norm = torch.reshape(init_pert, (init_pert.shape[0], -1))
                norm = torch.norm(norm, p=2, dim=1, keepdim=True)
                factor = torch.min(self._eps_max / norm, torch.ones_like(norm))
                init_pert = factor * init_pert

                batch["x"] = batch["x"] + init_pert

            if(self._min_clip is not None and self._max_clip is not None):
                batch["x"] = torch.clamp(
                    batch["x"], self._min_clip, self._max_clip)
            batch["x"] = batch["x"].type(torch.float32)
        batch["x"].requires_grad_()

        # run k steps
        batch_inv = {"x": None, "y": None}
        for k in range(self._k):
            old_sample_size = self.model.sample_size
            self.model.sample_size = -1
            self.model(batch)
            self.model.sample_size = old_sample_size

            self._loss = self.loss(self.model.out, batch["y"])
            self._loss.backward()

            x_grad = batch["x"].grad

            # Take away the ones that already are adversarial
            mask = batch["y"] != self.model.pred
            mask = torch.squeeze(mask, 1)
            if batch_inv["x"] is None:
                batch_inv["x"] = batch["x"][mask]
                batch_inv["y"] = batch["y"][mask]
                x_orig_tracker = x_orig[mask]
            else:
                batch_inv["x"] = torch.cat((batch_inv["x"], batch["x"][mask]))
                batch_inv["y"] = torch.cat((batch_inv["y"], batch["y"][mask]))
                x_orig_tracker = torch.cat((x_orig_tracker, x_orig[mask]))

            # Keep only the ones that are not adversarial
            mask = batch["y"] == self.model.pred
            mask = torch.squeeze(mask, 1)
            batch["x"] = batch["x"][mask].detach().requires_grad_()
            batch["y"] = batch["y"][mask]

            x_grad = x_grad.data[mask]
            x_orig = x_orig[mask]

            if(batch["x"].shape[0] == 0):
                # All example are adversarial, we can stop
                break

            batch_size_tmp = x_grad.shape[0]
            if self._lp_norm == "linf":
                batch["x"].data = batch["x"].data + self._step * x_grad.sign()
            elif self._lp_norm == "l2":
                x_grad_norm = torch.reshape(x_grad, (batch_size_tmp, -1))
                x_grad_norm = torch.norm(x_grad_norm, p=2, dim=1, keepdim=True)
                x_grad_norm = torch.max(
                    x_grad_norm, torch.ones_like(x_grad_norm) * 1e-6)
                perturbation = x_grad / x_grad_norm
                batch["x"].data = batch["x"].data + self._step * perturbation

            # Check if we are in the interval x-eps; x+eps
            batch["x"].data = torch.max(batch["x"].data, x_orig-self._eps_max)
            batch["x"].data = torch.min(batch["x"].data, x_orig+self._eps_max)
            if(self._min_clip is not None and self._max_clip is not None):
                batch["x"].data = torch.clamp(
                    batch["x"].data, self._min_clip, self._max_clip)

            batch["x"].grad = None

        batch["x"] = torch.cat((batch["x"], batch_inv["x"]))
        batch["y"] = torch.cat((batch["y"], batch_inv["y"]))
        x_orig = torch.cat((x_orig, x_orig_tracker))

        # Generate the mask to know which examples are adversarial
        self.model(batch)
        adversarial = batch["y"] != self.model.pred

        return (x_orig, adversarial)

    def fit(self, x, y):
        '''
            A wrapper of the _attack function that
            handle the different type of x and y.
            if it is a Tensor we are at training time.
            If it is an ndarray we are at testing time.
            Parameters
            ----------
            x: tensor/ndarray
                The input data
            y: tensor/ndarray
                The label data

            Return
            ------
            x: ndarray
                The perturbed input data
            y: ndarray
                The same label data
        '''
        if(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            batch = {"x": x, "y": y}
            self._attack(batch)
            return batch["x"], batch["y"]

        elif(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            data = NumpyDataset({
                "x_train": x,
                "y_train": y})
            loader = torch.utils.data.DataLoader(
                data, batch_size=self._batch_size)

            # Computing batch size
            num_batch = int(len(data)/self._batch_size)
            if(len(data) % self._batch_size != 0):
                num_batch += 1

            print(y.shape)

            x = None
            y = None
            x_orig = None
            batch = None

            for i, batch in enumerate(loader):

                logging.info(("Attacking batch [{}/{}] ...\r").format(
                    i+1, num_batch))

                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long)

                # Attack return the orignal x. Need that because the heuristic
                # in _attack change the arrangement of the batch
                (x_orig_batch, adversarial_batch) = self._attack(batch)

                if(x is None):
                    x = batch["x"]
                    y = batch["y"]

                    x_orig = x_orig_batch
                    adversarial = adversarial_batch

                else:
                    x = torch.cat((x, batch["x"]))
                    y = torch.cat((y, batch["y"]))

                    x_orig = torch.cat((x_orig, x_orig_batch))
                    adversarial = torch.cat((adversarial, adversarial_batch))

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            x_orig = x_orig.cpu().detach().numpy()
            adversarial = adversarial.cpu().detach().numpy()
            self.x_orig = x_orig
            self.adversarial = adversarial

            logging.info("\n")
            return x, y


class FGSMAttack(PGDAttack):
    '''
        FGSMAttack finds an adversarial example using the FGSM technique.
        It is a particular instance of the PGD attack.
        We do only one iteration : k=1
        We do not do a random initialization of
        the example x: random_init=False

        Note1: it handles the linf norm and l2 norm.
        Note2: To get more details on this attack
        see https://arxiv.org/pdf/1412.6572.pdf
        Note3: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        step: float
            The step to apply for the update of a PGD iteration.
            Default: 0.01
        eps_max: float
            The epsilon max that delimit the neighborhood
            around an orignal example.
            Default: 0.3
        min_clip: float
            The minimum value that an example can reach (usually
            corresponds the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach (usually
            corresponds the maximum value of the input domain).
            Default: None
        batch_size: integer
            The size of the batch.
            Default: 64
        lp_norm: string
            The norm used to compute the distance between the
            orginal example and the perturbed example.
            Default: linf
    '''
    def __init__(self, name, model, device, loss, step=0.01, eps_max=0.3,
                 min_clip=None, max_clip=None, batch_size=64, lp_norm="linf"):
        k = 1
        random_init = False
        super(FGSMAttack, self).__init__(name, model, device, loss, k, step,
                                         eps_max, min_clip, max_clip,
                                         batch_size, random_init, lp_norm)


class IterativeFGSMAttack(PGDAttack):
    '''
        IterativeFGSMAttack finds an adversarial example using the
        IterativeFGSM technique.
        It is a particular instance of the PGD attack.
        We do not do a random initialization of the
        example x: random_init=False

        Note1: it handles the linf norm and l2 norm.
        Note2: To get more details on this attack
        see https://arxiv.org/pdf/1611.01236.pdf
        Note3: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        k: integer
            The number of step to perform for the PGD.
            Default: 30
        step: float
            The step to apply for the update of a PGD iteration.
            Default: 0.01
        eps_max: float
            The epsilon max that delimit the neighborhood
            around an orignal example.
            Default: 0.3
        min_clip: float
            The minimum value that an example can reach (usually
            corresponds the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach (usually
            corresponds the maximum value of the input domain).
            Default: None
        batch_size: integer
            The size of the batch.
            Default: 64
        lp_norm: string
            The norm used to compute the distance between the orginal example
            and the perturbed example.
            Default: linf
    '''
    def __init__(self, name, model, device, loss, k=30, step=0.01, eps_max=0.3,
                 min_clip=None, max_clip=None, batch_size=64, lp_norm="linf"):
        random_init = False
        super(IterativeFGSMAttack, self).__init__(name, model, device, loss, k,
                                                  step, eps_max, min_clip,
                                                  max_clip, batch_size,
                                                  random_init, lp_norm)


class UniformNoiseIterativeFGSMAttack():
    '''
        UniformNoiseIterativeFGSMAttack applies in sequence the IterativeFGSM
        attack and then the UniformNoise attack.

        Note1: if the size of the original dataset is "m" and you choose to
        samples 100 noises (parameter "n"), the perturbed dataset will have a
        size of m*n.
        Note2: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        k: integer
            The number of step to perform for the PGD.
            Default: 30
        step: float
            The step to apply for the update of a PGD iteration.
            Default: 0.01
        it_fgsm_eps_max: float
            The epsilon max that delimit the neighborhood around an orignal
            example for the IterativeFGSM attack.
            Default: 0.3
        min_clip: float
            The minimum value that an example can reach (usually corresponds
            the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach (usually
            corresponds the maximum value of the input domain).
            Default: None
        noise_eps_max: float
            The epsilon max that delimit the neighborhood around an orignal
            example for the UniformNoise attack.
            Default: 1.0
        n: integer
            Number of noises to sample per example.
            Default: -1
        batch_size: integer
            The size of the batch.
            Default: 64
        lp_norm: string
            The norm used to compute the distance between the orginal
            example and the perturbed example.
            Default: linf
    '''
    def __init__(self, name, model, device, loss,
                 k=30, step=0.01, it_fgsm_eps_max=0.3, min_clip=None,
                 max_clip=None, noise_eps_max=1.0, n=-1,
                 batch_size=64, lp_norm="linf"):
        super().__init__(name, model, device, loss)
        self.__iterativeFGSM = Attack(
            "iterativefgsm", model, device, loss,
            k=k, step=step, eps_max=it_fgsm_eps_max,
            min_clip=min_clip, max_clip=max_clip,
            batch_size=batch_size, lp_norm=lp_norm)
        self.__noise = Attack(
            "uniformnoise", model, device, loss,
            eps_max=noise_eps_max, min_clip=min_clip, max_clip=max_clip,
            n=n, batch_size=batch_size)
        self.nb_noise = n
        logging.info(("\nDEBUG: uniformnoiseiterativeFGSM nb_noise {}\n"
                      ).format(self.nb_noise))

    def fit(self, x, y):
        '''
            Fit sequentially x and y to IterativeFGSM attack
            and UniformNoise attack

            Parameters
            ----------
            x: tensor/ndarray
                The input data
            y: tensor/ndarray
                The label data

            Return
            ------
            x: ndarray
                The perturbed input data
            y: ndarray
                The same label data
        '''
        x, y = self.__iterativeFGSM.fit(x, y)
        x, y = self.__noise.fit(x, y)
        return x, y


class UniformNoisePGDAttack():
    '''
        UniformNoisePGDAttack applies in sequence the PGD attack and
        then the UniformNoise attack.

        Note1: if the size of the original dataset is "m" and you choose to
        samples 100 noises (parameter "n"), the perturbed dataset will have a
        size of m*n.
        Note2: See the class ``Attack'' for the parameters not explained.

        Parameters
        ----------
        k: integer
            The number of step to perform for the PGD.
            Default: 30
        step: float
            The step to apply for the update of a PGD iteration.
            Default: 0.01
        pgd_eps_max: float
            The epsilon max that delimit the neighborhood around an orignal
            example for the PGD attack.
            Default: 0.3
        min_clip: float
            The minimum value that an example can reach (usually
            corresponds the minimum value of the input domain).
            Default: None
        max_clip: float
            The maximum value that an example can reach (usually
            corresponds the maximum value of the input domain).
            Default: None
        lp_norm: string
            The norm used to compute the distance between the orginal example
            and the perturbed example.
            Default: linf
        random_init: Boolean
            Whether or not we apply a random initialization to x
            using uniform noise.
        noise_eps_max: float
            The epsilon max that delimit the neighborhood around an
            orignal example for the UniformNoise attack.
            Default: 1.0
        n: integer
            Number of noises to sample per example.
            Default: -1
        batch_size: integer
            The size of the batch.
            Default: 64
    '''
    def __init__(self, name, model, device, loss,
                 k=30, step=0.01, pgd_eps_max=0.3, min_clip=None,
                 max_clip=None, lp_norm="linf", random_init=True,
                 noise_eps_max=1.0, n=-1, batch_size=64):
        super().__init__(name, model, device, loss)
        self.__pgd = Attack(
            "pgd", model, device, loss,
            k=k, step=step, eps_max=pgd_eps_max, lp_norm=lp_norm,
            min_clip=min_clip, max_clip=max_clip, batch_size=batch_size,
            random_init=random_init)
        self.__noise = Attack(
            "uniformnoise", model, device, loss,
            eps_max=noise_eps_max, min_clip=min_clip, max_clip=max_clip,
            n=n, batch_size=batch_size)
        self.nb_noise = n
        self.x_orig = None
        self.adversarial = None
        logging.info(("\nDEBUG: uniformnoisepgd nb_noise {}\n").format(
            self.nb_noise))

    def fit(self, x, y):
        '''
            Fit sequentially x and y to PGD attack and UniformNoise attack

            Parameters
            ----------
            x: tensor/ndarray
                The input data
            y: tensor/ndarray
                The label data

            Return
            ------
            x: ndarray
                The perturbed input data
            y: ndarray
                The same label data
        '''
        x, y = self.__pgd.fit(x, y)
        self.x_orig = self.__pgd.x_orig
        self.adversarial = self.__pgd.adversarial

        x, y = self.__noise.fit(x, y)
        return x, y
