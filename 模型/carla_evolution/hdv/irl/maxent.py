from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class MaxEntIRL:
    """Small MaxEnt IRL optimizer for linear reward weights."""

    def __init__(
        self,
        feature_dim: int,
        n_iters: int = 200,
        lr: float = 0.05,
        lam: float = 0.01,
        seed: int = 0,
    ):
        self.feature_dim = int(feature_dim)
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.lam = float(lam)
        self.rng = np.random.RandomState(seed)
        self.theta = self.rng.normal(0.0, 0.05, size=self.feature_dim)
        self.pm = np.zeros(self.feature_dim, dtype=float)
        self.pv = np.zeros(self.feature_dim, dtype=float)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.log = {
            "iteration": [],
            "feature_difference": [],
            "log_likelihood": [],
            "theta": [],
        }

    def fit(
        self,
        expert_features: Sequence[np.ndarray],
        candidate_scenes: Sequence[Sequence[np.ndarray]],
    ) -> Dict[str, List]:
        if not expert_features:
            raise ValueError("No expert features were provided.")
        if not candidate_scenes:
            raise ValueError("No candidate scenes were generated.")

        experts = [np.asarray(item, dtype=float) for item in expert_features]
        scenes = [
            [np.asarray(item, dtype=float) for item in scene if item is not None]
            for scene in candidate_scenes
        ]
        scenes = [scene for scene in scenes if scene]
        if not scenes:
            raise ValueError("No non-empty candidate scenes were generated.")

        for iteration in range(self.n_iters):
            expert_exp = np.zeros(self.feature_dim, dtype=float)
            model_exp = np.zeros(self.feature_dim, dtype=float)
            log_likes = []

            for scene_idx, scene in enumerate(scenes):
                expert = experts[scene_idx % len(experts)]
                scene_features = np.asarray(scene, dtype=float)
                rewards = scene_features.dot(self.theta)
                max_reward = float(np.max(rewards))
                probs = np.exp(rewards - max_reward)
                probs = probs / max(float(np.sum(probs)), 1e-12)

                model_exp += probs.dot(scene_features)
                expert_exp += expert

                expert_reward = float(np.dot(self.theta, expert))
                log_partition = max_reward + np.log(np.sum(np.exp(rewards - max_reward)))
                log_likes.append(expert_reward - float(log_partition))

            n = float(len(scenes))
            grad = expert_exp / n - model_exp / n - 2.0 * self.lam * self.theta
            self._adam_update(grad, iteration)

            self.log["iteration"].append(iteration + 1)
            self.log["feature_difference"].append(float(np.linalg.norm(expert_exp / n - model_exp / n)))
            self.log["log_likelihood"].append(float(np.mean(log_likes)))
            self.log["theta"].append(self.theta.copy().tolist())

        return self.log

    def _adam_update(self, grad: np.ndarray, iteration: int):
        self.pm = self.beta1 * self.pm + (1.0 - self.beta1) * grad
        self.pv = self.beta2 * self.pv + (1.0 - self.beta2) * (grad * grad)
        mhat = self.pm / (1.0 - self.beta1 ** (iteration + 1))
        vhat = self.pv / (1.0 - self.beta2 ** (iteration + 1))
        self.theta += self.lr * mhat / (np.sqrt(vhat) + self.eps)
