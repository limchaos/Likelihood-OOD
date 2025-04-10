import torch
import numpy as np
from torchdiffeq import odeint as odeint_torch
import project.sde_lib as sde_lib


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t, p):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model(x, labels, p)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model(x, labels, p)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t, p):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model(x, labels, p)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps, p):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t, p) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(
    sde,
    hutchinson_type="Rademacher",
    rtol=1e-5,
    atol=1e-5,
    method="fehlberg2",
    eps=1e-5,
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.
    # Solve the ODE
    # 'dopri8' 7s
    # 'dopri5' 1.9s - good same as scipy.solve_ivp rk45
    # 'bosh3' 2.5s
    # 'fehlberg2' 1.4s - is scipy.solve_ivp rkf45
    # 'adaptive_heun' 4s
    # 'euler' nan
    # 'midpoint' nan
    # 'rk4' 1s inaccurate
    # 'explicit_adams' 1s inaccurate
    # 'implicit_adams' 1s inaccurate
    # 'fixed_adams' 1s inaccurate
    # 'scipy_solver'

    Args:
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      inverse_scaler: The inverse data normalizer.
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t, p):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, p)[0]

    def div_fn(model, x, t, noise,p):
        return get_div_fn(lambda xx, tt, pp: drift_fn(model, xx, tt, pp))(x, t, noise, p)

    def likelihood_fn(model, x, p):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          model: A score model.
          x: A PyTorch tensor.

        Returns:
          bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
          z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        device = x.device
        with torch.no_grad():
            shape = x.shape
            if hutchinson_type == "Gaussian":
                epsilon = torch.randn_like(x)
            elif hutchinson_type == "Rademacher":
                epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, data):
                sample = data[: shape.numel()].clone().reshape(shape).float()

                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = drift_fn(model, sample, vec_t, p)
                div = div_fn(model, sample, vec_t, epsilon, p)
                return torch.cat([drift.reshape(-1), div], 0)

            init_state = torch.cat(
                [x.reshape(-1), torch.zeros(shape[0], device=device)], 0
            )
            timesteps = torch.tensor([eps, sde.T], device=device)

            # Solving the ODE
            res = odeint_torch(
                ode_func, init_state, timesteps, rtol=rtol, atol=atol, method=method
            )
            zp = res[-1]

            z = zp[: -shape[0]].reshape(shape)
            prior_logp = sde.prior_logp(z)

            delta_logp = zp[-shape[0] :].reshape(shape[0])

            # bpd = -(prior_logp + delta_logp) / np.log(2)
            # N = torch.prod(torch.tensor(shape[1:], device=device)).item()
            # bpd = bpd / N + 8  # Convert log-likelihood to bits/dim

            N = torch.prod(torch.tensor(shape[1:], device=device)).item()
            bpd = -(prior_logp + delta_logp) / N
            return bpd

    return likelihood_fn
