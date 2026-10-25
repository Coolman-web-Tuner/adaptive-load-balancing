"""
A compact DQN implementation using only numpy (no torch/tensorflow needed).

Network: state -> Dense(hidden, relu) -> Dense(action_size, linear)
Trained with mean-squared TD error via manual backprop and Adam.
"""
import numpy as np
from collections import deque
import random


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class QNetwork:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        limit1 = np.sqrt(6 / (state_size + hidden_size))
        limit2 = np.sqrt(6 / (hidden_size + action_size))
        self.W1 = rng.uniform(-limit1, limit1, (state_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.uniform(-limit2, limit2, (hidden_size, action_size)).astype(np.float32)
        self.b2 = np.zeros(action_size, dtype=np.float32)
        self.lr = lr

        # Adam optimizer state
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2, (x, z1, a1)

    def predict(self, x):
        q, _ = self.forward(x)
        return q

    def train_batch(self, states, targets):
        """
        states: (B, state_size)
        targets: (B, action_size) - full target Q vector (only chosen-action entries differ)
        """
        q, (x, z1, a1) = self.forward(states)
        B = states.shape[0]
        dz2 = (q - targets) / B  # dL/dz2 for MSE loss

        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        self._adam_step(grads)

    def _adam_step(self, grads):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self._m[k] / (1 - self.beta1 ** self._t)
            v_hat = self._v[k] / (1 - self.beta2 ** self._t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, k, getattr(self, k) - update)

    def copy_weights_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3,
                 gamma=0.95, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.997,
                 buffer_size=20000, batch_size=64, target_update_every=200, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.q_network = QNetwork(state_size, action_size, hidden_size, lr, seed)
        self.target_network = QNetwork(state_size, action_size, hidden_size, lr, seed + 1)
        self.target_network.copy_weights_from(self.q_network)

        self.memory = ReplayBuffer(buffer_size, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        q = self.q_network.predict(state[None, :])[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_current = self.q_network.predict(states)
        q_next = self.target_network.predict(next_states)
        max_next_q = q_next.max(axis=1)

        targets = q_current.copy()
        for i in range(len(actions)):
            td_target = rewards[i] + (1 - dones[i]) * self.gamma * max_next_q[i]
            targets[i, actions[i]] = td_target

        self.q_network.train_batch(states, targets)

        self.steps += 1
        if self.steps % self.target_update_every == 0:
            self.target_network.copy_weights_from(self.q_network)

        loss = float(np.mean((q_current - targets) ** 2))
        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
