import torch
import utils.pytorch_util as ptu
from trainer.policies import TanhNormal
import math
import numpy as np

def get_optimistic_exploration_action(ob_np, policy=None, qfs=None, trainer=None, hyper_params=None, deterministic=False):
    if deterministic:
        return get_optimistic_exploration_action_deterministic(ob_np, policy, qfs, hyper_params, trainer=trainer)
    else:
        return get_optimistic_exploration_action_stochastic(ob_np, policy, qfs, hyper_params, trainer=trainer)


def get_optimistic_exploration_action_stochastic(ob_np, policy=None, qfs=None, hyper_params=None, trainer=None):

    assert ob_np.ndim == 1

    beta_UB = hyper_params['beta_UB']
    delta = hyper_params['delta']
    share_layers = hyper_params['share_layers']

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    # Ensure that pretanh_mu_T is not batched
    assert len(list(pre_tanh_mu_T.shape)) == 1, pre_tanh_mu_T
    assert len(list(std.shape)) == 1

    pre_tanh_mu_T.requires_grad_()
    tanh_mu_T = torch.tanh(pre_tanh_mu_T)

    # Get the upper bound of the Q estimate

    if trainer is not None:
        Q_UB = trainer.predict(ob.unsqueeze(0), tanh_mu_T.unsqueeze(0), upper_bound=True, beta_UB=beta_UB)
    else:
        args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T))
        try:
            Q1 = qfs[0](*args)
            Q2 = qfs[1](*args)
            mu_Q = (Q1 + Q2) / 2.0
            sigma_Q = torch.abs(Q1 - Q2) / 2.0
        except:

            q_preds = []
            for i in range(len(qfs)):
                q_preds.append(qfs[i](*args))

            qs = torch.stack(q_preds, dim=0)

            if share_layers:
                qs = qs.permute(2, 1, 0)
            mu_Q = torch.mean(qs, dim=0)
            sigma_Q = torch.std(qs, dim=0)

        Q_UB = mu_Q + beta_UB * sigma_Q

    # Obtain the gradient of Q_UB wrt to a
    # with a evaluated at mu_t
    grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
    grad = grad[0]

    assert grad is not None
    assert pre_tanh_mu_T.shape == grad.shape

    # Obtain Sigma_T (the covariance of the normal distribution)
    Sigma_T = torch.pow(std, 2)

    # The dividor is (g^T Sigma g) ** 0.5
    # Sigma is diagonal, so this works out to be
    # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
    denom = torch.sqrt(
        torch.sum(
            torch.mul(torch.pow(grad, 2), Sigma_T)
        )
    ) + 10e-6

    # Obtain the change in mu
    mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom

    assert mu_C.shape == pre_tanh_mu_T.shape

    mu_E = pre_tanh_mu_T + mu_C

    # Construct the tanh normal distribution and sample the exploratory action from it
    assert mu_E.shape == std.shape

    dist = TanhNormal(mu_E, std)

    ac = dist.sample()

    ac_np = ptu.get_numpy(ac)

    # mu_T_np = ptu.get_numpy(pre_tanh_mu_T)
    # mu_C_np = ptu.get_numpy(mu_C)
    # mu_E_np = ptu.get_numpy(mu_E)
    # dict(
    #     mu_T=mu_T_np,
    #     mu_C=mu_C_np,
    #     mu_E=mu_E_np
    # )

    # Return an empty dict, and do not log
    # stats for now
    return ac_np, {}

def get_optimistic_exploration_action_deterministic(ob_np, policy=None, qfs=None, hyper_params=None, trainer=None):

    assert ob_np.ndim == 1

    beta_UB = hyper_params['beta_UB']
    delta = hyper_params['delta']
    share_layers = hyper_params['share_layers']

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    # Ensure that pretanh_mu_T is not batched
    assert len(list(pre_tanh_mu_T.shape)) == 1, pre_tanh_mu_T
    assert len(list(std.shape)) == 1

    pre_tanh_mu_T.requires_grad_()
    tanh_mu_T = torch.tanh(pre_tanh_mu_T)

    # Get the upper bound of the Q estimate
    if trainer is not None:
        Q_UB = trainer.predict(ob, tanh_mu_T, upper_bound=True, beta_UB=beta_UB)
    else:
        args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T))
        # Q1 = qfs[0](*args)
        # Q2 = qfs[1](*args)
        # mu_Q = (Q1 + Q2) / 2.0
        # sigma_Q = torch.abs(Q1 - Q2) / 2.0
        q_preds = []
        for i in range(len(qfs)):
            q_preds.append(qfs[i](*args))

        qs = torch.stack(q_preds, dim=0)
        if share_layers:
            qs = qs.permute(2, 1, 0)
        mu_Q = torch.mean(qs, dim=0)
        sigma_Q = torch.std(qs, dim=0)

        Q_UB = mu_Q + beta_UB * sigma_Q

    # Obtain the gradient of Q_UB wrt to a
    # with a evaluated at mu_t
    grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
    grad = grad[0]

    assert grad is not None
    assert pre_tanh_mu_T.shape == grad.shape

    # The dividor is (g^T) ** 0.5
    # Sigma is diagonal, so this works out to be
    # ( sum_{i=1}^k (g^(i))^2
    denom = torch.sqrt(
        torch.sum(
            torch.pow(grad, 2)
        )
    ) + 10e-6

    # Obtain the change in mu
    mu_C = math.sqrt(2.0 * delta) * grad / denom

    assert mu_C.shape == pre_tanh_mu_T.shape

    mu_E = pre_tanh_mu_T + mu_C

    # Construct the tanh normal distribution and sample the exploratory action from it
    assert mu_E.shape == std.shape

    ac = mu_E

    ac_np = ptu.get_numpy(ac)

    # mu_T_np = ptu.get_numpy(pre_tanh_mu_T)
    # mu_C_np = ptu.get_numpy(mu_C)
    # mu_E_np = ptu.get_numpy(mu_E)
    # dict(
    #     mu_T=mu_T_np,
    #     mu_C=mu_C_np,
    #     mu_E=mu_E_np
    # )

    # Return an empty dict, and do not log
    # stats for now
    return ac_np, {}
