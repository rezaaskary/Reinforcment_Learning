import numpy as np
from numba import njit, jit
import matplotlib.pylab as plt

np.random.seed(seed=42)


@jit(nopython=True)
def reward(actions: float = None,
           scale: float = 0.3) -> float:
    return np.random.normal(loc=actions, scale=scale)


@jit(nopython=True)
def random_epsilon(eps: float = 0.01,
                   Q: np.ndarray = None,
                   action_list: np.ndarray = None):
    if np.random.rand() < eps:
        return np.random.choice(action_list)
    else:
        return np.argmax(Q)


n_actions = 4
n_iter = 10000
alpha = 0.1
Q = np.zeros(shape=(n_iter, n_actions))
Q[0,:] = np.random.normal(loc=0, scale=5,size=(n_actions,))

a = np.zeros(shape=(n_iter,), dtype=int)
counter = np.zeros(shape=(n_actions,), dtype=int)

# define a non stationary reward function
R = np.zeros(shape=(n_iter, n_actions))
R[:, 0] = np.exp(-np.arange(n_iter) / 1000) * 2 + 2
R[:, 1] = np.exp(-np.arange(n_iter) / 1000) * -2 - 2
R[:, 2] = np.exp(-np.arange(n_iter) / 1000) * -1 - 1
R[:, 3] = np.exp(-np.arange(n_iter) / 1000) * 1 + 1

action_list = np.array([0, 1, 2, 3], dtype=int)

if __name__ == '__main__':
    for i in range(n_iter - 1):
        # select an action index
        action_ind = random_epsilon(eps=0.1,
                                    Q=Q[i, :],
                                    action_list=action_list
                                    )
        # take the action value
        a[i] = action_list[action_ind]

        # define mask variable for updating cumolative reward function
        mask = action_list == action_list[action_ind]

        # cumolative reward function of all other action remain unchanged
        Q[i + 1, ~mask] = Q[i, ~mask]

        # get the reward/punishment value
        reward_value = reward(actions=R[i, action_ind])

        # update the reward cumolative function
        Q[i + 1, mask] = Q[i, mask] + alpha * (reward_value - Q[i, mask])

plt.plot(Q[:, 0], label='first action cumolative reward')
plt.plot(Q[:, 1], label='second action cumolative reward')
plt.plot(Q[:, 2], label='third action cumolative reward')
plt.plot(Q[:, 3], label='fourth action cumolative reward')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.cumsum(a == 0) / np.arange(1,n_iter+1) * 100)
plt.ylabel('% of correct actions')
plt.grid()
plt.show()
