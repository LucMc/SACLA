from functools import partial
import click
import numpy as np
import sbx
from stable_baselines3.common.utils import explained_variance
import torch as th
import jax
# from jax._src import prng
import jax.numpy as jnp
import jax.random as random
# from jax.experimental.checkify import check

import flax
from flax.training.train_state import TrainState
import orbax.checkpoint
from flax.training import orbax_utils
from flax.core import FrozenDict

from prob_lyap.utils.type_aliases import ReplayBufferSamplesNp, RLTrainState, CustomTrainState, LyapConf
from prob_lyap.utils import utils
from prob_lyap.lyap_func import Lyap_net
from prob_lyap.world_model import WorldModel
from prob_lyap.objectives import get_objective
from prob_lyap.polyc.CustomRollout import CustomRolloutBuffer

from copy import deepcopy
from typing import Callable

from gymnasium import spaces
from gymnasium.wrappers.flatten_observation import FlattenObservation
import optax
from typing import Literal, Dict
import copy

# from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback


class POLYC(sbx.PPO): # Make an abstract class and inherit from that?
    def __init__(self, lyap_config: LyapConf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_fn = get_objective(lyap_config.objective)
        self.learning_rate = lyap_config.actor_lr
        self.lyap_config = lyap_config
        self.rng = random.PRNGKey(lyap_config.seed)# Take in as arg or random.PRNGKey(1)
        self.debug = lyap_config.debug
        lyap = Lyap_net(lyap_config.n_hidden, lyap_config.n_layers) 

        lyap_key, self.rng = random.split(self.rng)
        self.lyap_state = utils.create_train_state("lyapunov", lyap_key, lyap, lyap_config.lyap_lr, self.env)
        self.ckpt_dir = lyap_config.ckpt_dir
        # self.objective = lyap_config.objective # Which learning objective to use [adverserial, standard, LSO-objective]
        self.beta = lyap_config.beta
        # Checkpoints
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=20, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.ckpt_dir, orbax_checkpointer, options
        )
        self.n_saves = 0


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

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage"])
    def _one_update(
        actor_state: TrainState,
        vf_state: TrainState,
        lyap_state: TrainState,
        observations: np.ndarray,
        observations_: np.ndarray, # next obs for lyap
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_prob: np.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        beta: float,
        normalize_advantage: bool = True
    ):
        """
        To avoid making obs L2 we can just use the diff from lyapsac of ag and dg        
        """
        # Modified to use samples from memory rather than wm
        def lyap_loss(params):
            # ag - dg
            # l2_obs = observations[:, :3] - observations[:, 3:6] 
            # l2_next_obs = observations_[:, :3] - observations_[:, 3:6] 

            lyap_values = lyap_state.apply_fn(params, observations)
            next_lyap_values = jax.lax.stop_gradient(lyap_state.apply_fn(params, observations_))
            # assert not jnp.array_equal(l2_obs, l2_next_obs), "obs and next obs are identical" # If debug? Use checkify
            # checkify.check()
            dvdxs = (next_lyap_values - lyap_values) #* jnp.sum(l2_next_obs - l2_obs) # All these are the same?
            return dvdxs.mean(), dvdxs.flatten()

        (v_loss, dvdxs), grads = jax.value_and_grad(lyap_loss, has_aux=True)(lyap_state.params)
        lyap_state = lyap_state.apply_gradients(grads=grads)

        # Normalize advantage
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean() / (advantages.std() + 1e-8)) # Arround 0 w/ std 1+e
        
        def actor_loss(params):
            dist = actor_state.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob) # exp to undo the log
            policy_loss_1 = advantages * ratio # Advantage of that action * the difference in probability of taking it
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean() # best of new policy and clipped policy
# Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss
        
        pg_loss_value, grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        polyc_advantage = (1-beta)*returns + beta*jnp.minimum(0, dvdxs) # Why are we clipping here?
        returns = polyc_advantage # ?
        def critic_loss(params):
            # Value loss using the TD(gae_lambda) target
            vf_values = vf_state.apply_fn(params, observations).flatten()
            return ((returns - vf_values) ** 2).mean() # MSE predicted value vs returns
        
        vf_loss_value, grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        # loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        return (actor_state, vf_state, lyap_state), (pg_loss_value, vf_loss_value, v_loss)
    
    def train(self) -> None:
        """
        Update policy and Lyapunov funciton using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip_range
        clip_range = self.clip_range_schedule(self._current_progress_remaining)

        # train for n_epochs
        for _ in range(self.n_epochs):
            # JIT only one update
            for rollout_data in  self.rollout_buffer.get(self.batch_size):
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().numpy().astype(np.int32)
                else:
                    actions = rollout_data.actions.numpy()
                (self.policy.actor_state, self.policy.vf_state, self.lyap_state), (pg_loss, value_loss, lyap_loss) = self._one_update(
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    lyap_state=self.lyap_state,
                    observations=rollout_data.observations.numpy(),
                    observations_=rollout_data.observations_.numpy(),
                    actions=actions,
                    advantages=rollout_data.advantages.numpy(),
                    returns=rollout_data.old_log_prob.numpy(),
                    old_log_prob=rollout_data.old_log_prob.numpy(),
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    beta=self.beta,
                    normalize_advantage=self.normalize_advantage
                )
                self._n_updates += self.n_epochs
                explained_var = explained_variance(
                    self.rollout_buffer.values.flatten(), # type: ignore[attr-defined]
                    self.rollout_buffer.returns.flatten() # type: ignore[attr-defined]
                )

                # Logs
                # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
                # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
                # TODO: use mean instead of one point
                self.logger.record("train/value_loss", value_loss.item())
                self.logger.record("train/lyap_loss", lyap_loss.item())
                # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
                # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
                self.logger.record("train/pg_loss", pg_loss.item())
                self.logger.record("train/explained_variance", explained_var)
                # if hasattr(self.policy, "log_std"):
                #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/clip_range", clip_range)
                # if self.clip_range_vf is not None:
                #     self.logger.record("train/clip_range_vf", clip_range_vf)
                # print(self.num_timesteps)
                # print(self.lyap_config.ckpt_every)
                if self.lyap_config.logging != "none" and not self.lyap_config.debug and (self.num_timesteps > self.lyap_config.ckpt_every*self.n_saves):
                    self.save_state()
                    self.n_saves += 1

    def save_state(self) -> None:
        # frozen_conf = FrozenDict(jax.tree_map(lambda x: str(x), self.lyap_config.__dict__))
        ckpt = {'lyap_state': self.lyap_state,
                 'config': self.lyap_config,
                 'actor_state': self.policy.actor_state}

        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save((self.num_timesteps//10000)*10000, ckpt, save_kwargs={'save_args': save_args}) # round down to nearest thousand
        # self.checkpoint_manager.wait_until_finished()


    def restore_ckpt(self, step:int) -> TrainState:
        if not step: # Get latest if not specified
            step = self.checkpoint_manager.latest_step()
        assert step, f"Step should be an integer. Step: {step}"

        # if self.debug:
        prev_actor_params = copy.deepcopy(self.policy.actor_state.params)
        prev_lyap_params = copy.deepcopy(self.lyap_state.params)
        # prev_wm_params = copy.deepcopy(self.wm_state.params)

        click.secho(f"Loading step: {step}", fg="green")
        restored_ckpt = self.checkpoint_manager.restore(step)
        # print(restored_ckpt["config"])

        self.lyap_state = CustomTrainState(**restored_ckpt['lyap_state'], apply_fn=self.lyap_state.apply_fn, tx=None)
        self.lyap_config = restored_ckpt['config']
        self.policy.actor_state = TrainState(**restored_ckpt["actor_state"], apply_fn=self.policy.actor_state.apply_fn, tx=None)
        self.lyap_config = restored_ckpt['config']
        # self.wm_state = CustomTrainState(**restored_ckpt['wm_state'], apply_fn=self.wm_state.apply_fn, tx=None)

        # Assert actor params are now equal (Default vs Loaded)
        assert not utils.params_equal(prev_actor_params['params'], self.policy.actor_state.params['params']), "Updated params same as default"
        assert utils.params_equal(restored_ckpt['actor_state']['params']['params'], self.policy.actor_state.params['params']), "Updated params different to loaded params"

        # Assert lyap params are now equal (Default vs Loaded)
        # assert not utils.params_equal(prev_lyap_params['params'], self.lyap_state.params['params']), "Updated params same as default"
        # assert utils.params_equal(restored_ckpt['lyap_state']['params']['params'], self.lyap_state.params['params']), "Updated params different to loaded params"

        # Assert wm params are now equal (Default vs Loaded)
        # assert not utils.params_equal(prev_wm_params['params'], self.wm_state.params['params']), "Updated params same as default"
        # assert utils.params_equal(restored_ckpt['wm_state']['params']['params'], self.wm_state.params['params']), "Updated params different to loaded params"

        return restored_ckpt

# # if __name__ == "__main__":
# #     # test PPO
# #     import gymnasium as gym
# #     env = FlattenObservation(gym.make("FetchSlide-v2"))
# #     model = POLYC({}, "MlpPolicy", env, verbose=1)
# #     # model = PPO("MultiInputPolicy", env, verbose=1)
# #     model.learn(10_000)