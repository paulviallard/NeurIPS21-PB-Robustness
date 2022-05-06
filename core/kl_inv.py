# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Note: The source code of this file is based on [1]
import torch


def kl(q, p):
    """
    Compute the KL divergence between two Bernoulli distribution
    (with parameters q and p)

    Parameters
    ----------
    q: Tensor
        Parameter of the first Bernoulli distribution
    p: Tensor
        Parameter of the second Bernoulli distribution

    Return
    ------
    Tensor
    Value of the KL divergence
    """
    return q * torch.log(q/p) + (1-q) * torch.log((1-q)/(1-p))


def kl_inv(q, epsilon, mode, nb_iter_max=1000):
    """
    Compute kl^{-1} with the bisection optimization method

    Parameters
    ----------
    q: Tensor
        The empirical risk (parameter of the first Bernoulli distribution
    epsilon: Tensor
        The value of the bound
    mode: str
        The mode for the function kl_inv (either "MAX" or "MIN")
    nb_iter_max: int
        The maximum number of iterations for the bisection (1000 by default)

    Return
    ------
    Tensor
    The upper-bound of the true risk computed by kl^{-1}
    (parameter of the second Bernoulli distribution)
    """

    # Verify that mode \in {"MIN", "MAX"} and epislon > 0
    assert mode in ["MIN", "MAX"]
    assert epsilon >= 0

    # If q is zero, we clamp to avoid numerical problems
    if(q <= 0.0):
        q.data = torch.tensor(1e-9)

    # We initialize the bounds of p
    if(mode == "MAX"):
        p_max, p_min = 1., q
    else:
        p_max, p_min = q, 1e-9

    # For each iteration,
    for _ in range(nb_iter_max):

        # We get the parameter p
        p = (p_min+p_max)/2.
        # and we compute the KL divergence
        p_kl = kl(q, p)

        # If the KL is close to epsilon or if the bounds on p are close
        # we return p
        if torch.isclose(p_kl, epsilon) or (p_max-p_min)/2. < 1e-9:
            return p

        # Otherwise, we update the bounds
        if mode == "MAX":
            if p_kl > epsilon:
                p_max = p
            elif p_kl < epsilon:
                p_min = p

        elif p_kl > epsilon:
            p_min = p
        elif p_kl < epsilon:
            p_max = p

    return p


class klInvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, epsilon, mode="MAX"):
        """
        Compute kl^{-1}(q|epsilon) thanks to the function kl_inv

        Parameters
        ----------
        q: Tensor
            The empirical risk (parameter of the first Bernoulli distribution)
        epsilon: Tensor
            The value of the bound
        mode: str
            The mode for the function kl_inv (either "MAX" or "MIN")

        Return
        ------
        Tensor
        Value of the KL divergence
        """
        ctx.save_for_backward(q, epsilon)
        # We compute the KL divergence
        out = kl_inv(q, epsilon, mode)
        # We clamp the values to avoid numerical problems
        out = torch.clamp(out, 1e-9, 1-1e-4)

        # We save kl^{-1} and the mode for the gradient
        ctx.out = out
        ctx.mode = mode

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of kl^{-1} using [2]

        Parameters
        ----------
        grad_output: Tensor
            Gradient of the modules (that are already computed)

        Return
        ------
        (Tensor, Tensor, None)
        Gradient of kl^{-1} w.r.t q and epsilon
        """
        q, epsilon = ctx.saved_tensors

        term_1 = (1. - q)/(1. - ctx.out)
        term_2 = q / ctx.out

        grad_q = torch.log(term_1/term_2) / (term_1-term_2)
        grad_epsilon = 1. / (term_1-term_2)

        return grad_output * grad_q, grad_output * grad_epsilon, None

###############################################################################

# References:
#  [1] https://github.com/paulviallard/ECML21-PB-CBound/ (under MIT license)
#  [2] Learning Gaussian Processes by Minimizing PAC-Bayesian
#      Generalization Bounds
#      David Reeb Andreas Doerr Sebastian Gerwinn Barbara Rakitsch, 2018
