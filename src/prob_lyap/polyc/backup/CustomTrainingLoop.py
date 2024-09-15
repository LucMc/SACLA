from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import jax
import numpy as np
import torch as th
from gymnasium import spaces
# from stable_baselines3.common.buffers import RolloutBuffer
from prob_lyap.polyc.CustomRollout import CustomRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from sbx.common.on_policy_algorithm import OnPolicyAlgorithmJax
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv

# from sbx.ppo.policies import Actor, Critic, PPOPolicy

class CustomOnPolicyAlgorithmJax(OnPolicyAlgorithmJax):

    def _setup_model(self) -> None:
        super()._setup_model()
        # jax.debug.breakpoint()

        self.rollout_buffer = CustomRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            device="cpu",  # force cpu device to easy torch -> numpy conversion
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"  # type: ignore[has-type]
        # Switch to eval mode (this affects batch norm / dropout)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise()

            if not self.use_sde or isinstance(self.action_space, gym.spaces.Discrete):
                # Always sample new stochastic action
                self.policy.reset_noise()

            obs_tensor, _ = self.policy.prepare_obs(self._last_obs)  # type: ignore[has-type]
            actions, log_probs, values = self.policy.predict_all(obs_tensor, self.policy.noise_key)

            actions = np.array(actions)
            log_probs = np.array(log_probs)
            values = np.array(values)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.prepare_obs(infos[idx]["terminal_observation"])[0]
                    terminal_value = np.array(
                        self.vf.apply(  # type: ignore[union-attr]
                            self.policy.vf_state.params,
                            terminal_obs,
                        ).flatten()
                    )

                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore
                new_obs, # Observation_
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore
                th.as_tensor(values),
                th.as_tensor(log_probs),
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        values = np.array(
            self.vf.apply(  # type: ignore[union-attr]
                self.policy.vf_state.params,
                self.policy.prepare_obs(new_obs)[0],  # type: ignore[arg-type]
            ).flatten()
        )

        rollout_buffer.compute_returns_and_advantage(last_values=th.as_tensor(values), dones=dones)

        callback.on_rollout_end()

        return True