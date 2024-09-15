
import numpy as np
import gymnasium as gym
import jax.numpy as jnp
import distrax

from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.world_model import WorldModel
from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils import utils
from prob_lyap.utils.type_aliases import LyapConf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax
import jax.numpy as jnp
from pathlib import Path
import sbx

from typing import Callable

'''
TODO:
 - Plot wm distribution at first frame
 - Plot distribution over time
'''
from scipy import interpolate

def one_step_wm(seed: int,
            env:gym.Env,
            policy:Callable,
            model:Lyap_SAC,
            label: str):

    obs, _, env = utils.seed_modules(env, seed)
    obs = utils.flatten_obs(obs)
    action = jnp.zeros_like(env.action_space.sample())
    print(f"Observation: {obs}\n\n")
    mu, sigma = model.wm_state.apply_fn(model.wm_state.params, obs, action.reshape(1,-1))
    print(f"Mus: {mu}\n\n")
    print(f"Difference: {mu + sigma}")

    done = False
    mus = []
    sigmas = []
    obss = []

    while not done:
        action = policy(obs)
        if isinstance(obs, dict):
            obs = jnp.concatenate([x for x in obs.values()]).reshape(1, -1)

        obss.append(obs)
        mu, sigma = model.wm_state.apply_fn(model.wm_state.params, obs, action.reshape(1,-1))
        mus.append(mu)
        sigmas.append(sigma)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        
    difference = np.mean(abs(np.array(mus) - np.array(obss)))
    print(f"Average difference over episode: {difference}")
    # Pretty much the same difference for random model!!
    # The problem I was having with the actor state the same for these?

@click.command()
@click.option("-r", "--render", is_flag=True, default=False, help="Render environment", show_default=True)
def main(render: bool):
    use_pretrained = True
    delay_type = NoneWrapper
    test_conf = LyapConf(
        seed=np.random.randint(100),
        env_id="FetchReach-v2",
        # env_id="InvertedPendulum-v4",
        delay_type=delay_type, #  AugmentedRandomDelayWrapper OR UnseenRandomDelayWrapper OR NoneWrapper
        act_delay=range(0,1),
        obs_delay=range(0,1),
        ckpt_dir=f"models/{delay_type.__name__}" # Copy the checkpoint you want to investigate from ~/.prob_lyap
    )
    render_mode = "human" if render else None
    env = delay_type(gym.make(test_conf.env_id, render_mode=render_mode),
                                act_delay_range=test_conf.act_delay,
                                obs_delay_range=test_conf.obs_delay)

    pre_rl_model_path = Path(__file__).parent / ".." / "pretrained/pretrained_models/NoneWrapper/FetchReach-v2/SAC_0"
    step = 80_000
    model = Lyap_SAC(test_conf, "MultiInputPolicy", env)
    model = utils.get_from_checkpoint(step=step, model=model) # replaces config anyway
    pre_rl = sbx.SAC.load(pre_rl_model_path)
    sbx.SAC.set_parameters(pre_rl, pre_rl_model_path, exact_match=True)
    
    ###
    pre_rl_actor = pre_rl.policy.actor_state
    key = jax.random.PRNGKey(test_conf.seed)
    ###
    pre_trained_policy = lambda obs: pre_rl_actor.apply_fn(pre_rl_actor.params, obs).sample(seed=key)[0]
    # pre_trained_policy = lambda obs: pre_rl.predict(obs, deterministic=True)[0]
    ###

    SAC_policy = lambda obs: model.predict(obs, deterministic=True)[0]
    random_policy = lambda _: env.action_space.sample()

    one_step_wm(test_conf.seed, env, pre_trained_policy, model, label=r"SAC Controller $(V(x_2)/t)$")

if __name__ == "__main__":
    main()
    # test_rollout(seed, env, random_policy, model, label="Random Controller (V/t)")


# def flatten_obs(obs: dict) -> jnp.ndarray:
#     return jnp.concatenate([x for x in obs.values()]).reshape(1,-1)