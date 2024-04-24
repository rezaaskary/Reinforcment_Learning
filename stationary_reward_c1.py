import numpy as np
import numba
from numba import njit, jit
import matplotlib.pylab as plt

np.random.seed(seed=42)


@jit(nopython=True)
def reward(actions: float = None) -> float:
    return np.random.normal(loc=actions, scale=1)


@jit(nopython=True)
def random_epsilon(idx: int = None,
                   eps: float = 0.01,
                   Q: np.ndarray = None,
                   actions: np.ndarray = None,
                   action_list: np.ndarray = None):
    if np.random.rand() < eps:
        return np.random.choice(action_list)
    else:
        return np.argmax(Q)


n_actions = 4
n_iter = 10000
Q = np.zeros(shape=(n_iter, n_actions))
a = np.zeros(shape=(n_iter,), dtype=int)
counter = np.zeros(shape=(n_actions,), dtype=int)
rewards = np.array([-1, 1, -2, 2], dtype=int)
action_list = np.array([0, 1, 2, 3], dtype=int)

if __name__ == '__main__':
    for i in range(n_iter - 1):
        index_action = random_epsilon(idx=i, eps=0.1, Q=Q[i, :], action_list=action_list)
        a[i] = index_action
        counter[index_action] += 1
        Q[i + 1, action_list != index_action] = Q[i, action_list != index_action]
        rw = reward(actions=rewards[index_action])
        Q[i + 1, index_action] = Q[i, index_action] + (1 / counter[index_action]) * (rw - Q[i, index_action])

    plt.plot(Q[:, -1])
    plt.show()
