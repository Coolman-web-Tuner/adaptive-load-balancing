"""
dqn_agent.py
------------
A Deep Q-Network implemented from scratch on top of NumPy: a real
multi-layer neural network trained with backprop + Adam, an experience
replay buffer, a target network, and epsilon-greedy exploration.

Note: this sandboxed container doesn't have enough free disk to install
GPU-bundled PyTorch/TensorFlow wheels, so the network itself is written
directly in NumPy instead of importing torch/tf. The math (forward pass,
gradients, Adam update, target-network sync) is exactly what those
libraries would do under the hood, so this drops in as a real DQN.
If you run this on your own machine, `agent_torch.py` (see README) shows
the equivalent ~30-line PyTorch version you can swap in unchanged.
"""

import numpy as np
from collections import deque
import random


def _init_layer(n_in, n_out, rng):
    limit = np.sqrt(6 / (n_in + n_out))
    W = rng.uniform(-limit, limit, size=(n_in, n_out)).astype(np.float32)
    b = np.zeros(n_out, dtype=np.float32)
    return W, b


class MLP:
    """Two hidden-layer fully-connected network with ReLU activations,
    trained with Adam. Used both as the online and target Q-network."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1, self.b1 = _init_layer(in_dim, hidden, rng)
        self.W2, self.b2 = _init_layer(hidden, hidden, rng)
        self.W3, self.b3 = _init_layer(hidden, out_dim, rng)
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t_step = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, x, cache=False):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        out = a2 @ self.W3 + self.b3
        if cache:
            return out, (x, z1, a1, z2, a2)
        return out

    def backward(self, cache, d_out):
        x, z1, a1, z2, a2 = cache
        n = x.shape[0]
        dW3 = a2.T @ d_out / n
        db3 = d_out.mean(axis=0)

        da2 = d_out @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1 / n
        db1 = dz1.mean(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t_step += 1
        p = self.params()
        for k in p:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g * g)
            m_hat = self.m[k] / (1 - beta1 ** self.t_step)
            v_hat = self.v[k] / (1 - beta2 ** self.t_step)
            p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other):
        for k, v in other.params().items():
            self.params()[k][...] = v


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=0):
        self.buf = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden=64, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=3000,
                 buffer_size=20000, batch_size=64, target_update_every=200, seed=0):
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.online = MLP(state_dim, hidden, action_dim, seed=seed)
        self.target = MLP(state_dim, hidden, action_dim, seed=seed)
        self.target.copy_from(self.online)

        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

    def epsilon(self):
        frac = min(self.steps / self.eps_decay, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state, greedy=False):
        eps = 0.0 if greedy else self.epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.action_dim))
        q = self.online.forward(state[None, :])[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        self.steps += 1
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        q_next = self.target.forward(s2)
        max_q_next = q_next.max(axis=1)
        target = r + self.gamma * max_q_next * (1 - d)

        q_pred, cache = self.online.forward(s, cache=True)
        d_out = np.zeros_like(q_pred)
        # gradient of 0.5*(Q(s,a) - target)^2 w.r.t. Q(s,a) only, for the taken action
        td_error = (q_pred[np.arange(len(a)), a] - target)
        d_out[np.arange(len(a)), a] = td_error

        grads = self.online.backward(cache, d_out)
        self.online.adam_step(grads, lr=self.lr)

        if self.steps % self.target_update_every == 0:
            self.target.copy_from(self.online)

        return float(np.mean(td_error ** 2))
