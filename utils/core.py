import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import utils.pytorch_util as ptu

def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    if isinstance(outputs, tuple):
        return tuple(np_ify(x) for x in outputs)
    else:
        return np_ify(outputs)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def optimize_policy(policy,  policy_optimizer, buffer, init_policy, action_space, obj_func,
                    batch_size=128, num_actions=10, upper_bound=False, iterations=150, out_dir='', epoch=0,
                    save_fig=False):
    dataset = np.copy(buffer.get_dataset())
    ptu.copy_model_params_from_to(init_policy, policy)
    zero_tensor = torch.tensor(0.)
    losses = []
    norms = []
    best_loss = -np.inf
    for it in range(iterations):
        random.shuffle(dataset)
        start = 0
        losses_ = []
        norms_ = []
        batch_size = dataset.shape[0]
        while start < dataset.shape[0]:
            states = torch_ify(dataset[start:start + batch_size])
            iters = 1
            prev_actions, policy_mean, policy_log_std, log_pi, *_ = policy(
                    obs=states, reparameterize=True, return_log_prob=True, deterministic=True
                )
            for i in range(iters):
                target_actions, policy_mean, policy_log_std, log_pi, *_ = policy(
                    obs=states, reparameterize=True, return_log_prob=True, deterministic=True
                )
                if torch.isclose(torch.norm(prev_actions - target_actions), zero_tensor, atol=1e-3) and i != 0:
                    #print("Actions are the same, Stoping")
                    break

                obj = obj_func(states, target_actions, upper_bound)
                ##upper_bound (in some way)
                policy_loss = (-obj).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                norm = grad_norm(policy)
                losses_.append(np.asscalar(ptu.get_numpy(policy_loss)))
                norms_.append(np.asscalar(norm))
                #print("Gradient Norm:", norm)
                if torch.isclose(norm, zero_tensor, atol=1e-3):
                    #print("Gradient Norm is zero, Stopping")
                    break

                prev_actions = target_actions
            start += batch_size
        curr_loss = -np.mean(losses_)
        losses.append(curr_loss)
        norms.append(np.mean(norms_))
        if curr_loss > best_loss:
            best_loss = curr_loss
            best_params = copy.deepcopy(policy.state_dict())
    # if curr_loss != best_loss:
    #     policy.load_state_dict(best_params)
    if save_fig:
        fig, ax = plt.subplots()
        # make a plot
        ax.plot(losses, color="red", label='Q')
        # set x-axis label
        # set y-axis label
        ax.set_ylabel("Q", color="red", fontsize=14)
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(norms, color="blue", label='grad norm')
        ax2.set_ylabel("Grad Norm", color="blue", fontsize=14)
        #plt.show()
        # save the plot as a file
        fig.savefig(out_dir + '/' + ('upper_bound_' if upper_bound else '') + 'policy_opt_' + str(epoch) + '.jpg',
                    format='jpeg',
                    dpi=100,
                    bbox_inches='tight')
        plt.close(fig)
    print("Optimized")
    return policy


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
        except:
            pass
    total_norm = total_norm ** (1. / 2)
    return total_norm