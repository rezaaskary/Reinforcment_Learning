import numpy as np
from numba import njit, jit
import matplotlib.pylab as plt

np.random.seed(seed=42)


@jit(nopython=True)
def soft_max(H: np.ndarray = None) -> np.ndarray:
    """

    :param H:
    :return:
    """
    exp_logits = np.exp(H)
    return exp_logits / exp_logits.sum()


@jit(nopython=True)
def reward(actions: float = None,
           scale: float = 0) -> float:
    """
    :param actions:
    :param scale:
    :return:
    """
    return np.random.normal(loc=actions, scale=scale)


@jit(forceobj=True)
def main(n_iter: int = None,
         Q: np.ndarray = None,
         R: np.ndarray = None,
         a: np.ndarray = None,
         H: np.ndarray = None,
         alpha: float = 0.01,
         action_list: np.ndarray = None
         ):
    """

    :param action_list:
    :param alpha:
    :param n_iter:
    :param counter:
    :param Q:
    :param R:
    :param a:
    :param H:
    :return:
    """
    action_indexess = np.arange(start=0, stop=action_list.shape[0], step=1, dtype=np.int64)
    for i in range(n_iter - 1):
        # getting probablity of each action
        pi = soft_max(H=H[i, :])

        # select an action index
        action_ind = np.random.choice(a=action_indexess, p=pi)

        # save the action
        a[i] = action_list[action_ind]

        # define mask variable for updating cumolative reward function
        mask = action_list == a[i]

        # getting the current reward value
        reward_value = reward(actions=R[i, action_ind])

        # the cumolative reward of the selected action
        Q[i + 1] = Q[i] + (1 / (i + 1)) * (reward_value - Q[i])

        # updating logits:
        H[i + 1, mask] = H[i, mask] + alpha * (R[i, mask] - Q[i + 1]) * (1 - pi[mask])

        # updating logits:
        H[i + 1, ~mask] = H[i, ~mask] - alpha * (R[i, ~mask] - Q[i + 1]) * pi[~mask]

    return a, Q, H


n_actions = 4
n_iter = 10000
alpha = 0.5
Q = np.zeros(shape=(n_iter,))
H = np.zeros(shape=(n_iter, n_actions))

a = np.zeros(shape=(n_iter,), dtype=int)
counter = np.zeros(shape=(n_actions,), dtype=int)

# define a non stationary reward function
R = np.zeros(shape=(n_iter, n_actions))
R[:, 0] = np.exp(-np.arange(n_iter) / 1000) * 2 * 0 + 2
R[:, 1] = np.exp(-np.arange(n_iter) / 1000) * -2 * 0 - 2
R[:, 2] = np.exp(-np.arange(n_iter) / 1000) * -1 * 0 - 1
R[:, 3] = np.exp(-np.arange(n_iter) / 1000) * 1 * 0 + 1

action_list = np.array([0, 1, 2, 3], dtype=int)

if __name__ == '__main__':
    a, Q, H = main(n_iter=n_iter,
                   Q=Q,
                   R=R,
                   H=H,
                   a=a,
                   alpha=alpha,
                   action_list=action_list
                   )
# plt.plot(a, '.')
# plt.plot(Q[:,1])
plt.plot(np.cumsum(a == 0) / np.arange(1, n_iter + 1) * 100)
plt.show()
