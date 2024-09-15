import gymnasium as gym

from sbx import SAC
import jax
import jax.numpy as jnp

from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.world_model import WorldModel
from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils import utils
from prob_lyap.utils.type_aliases import LyapConf

from pathlib import Path

pre_rl_model_path = Path(__file__).parent / ".." / "pretrained/pretrained_models/NoneWrapper/FetchReach-v2/SAC_0"

delay_type = NoneWrapper
test_conf = LyapConf(
    seed=0,
    env_id="FetchReach-v2",
    # env_id="InvertedPendulum-v4",
    delay_type=delay_type, #  AugmentedRandomDelayWrapper OR UnseenRandomDelayWrapper OR NoneWrapper
    # act_delay=range(0,1),
    # obs_delay=range(0,1),
    ckpt_dir=f"models/{delay_type.__name__}" # Copy the checkpoint you want to investigate from ~/.prob_lyap
)

env = gym.make("FetchReach-v2") #test_conf.delay_type(gym.make(test_conf.env_id), act_delay_range=test_conf.act_delay, obs_delay_range=test_conf.obs_delay)

model = SAC("MultiInputPolicy", env, verbose=0)
print(f"PARAMS BEFORE TRAINING:\n{model.policy.actor_state.params['params']['Dense_0']}")
model.learn(total_timesteps=1_000, progress_bar=True)
model.save("sac_test")
print(f"PARAMS AFTER TRAINING:\n{model.policy.actor_state.params['params']['Dense_0']}")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_test")
print(f"PARAMS AFTER LOADING:\n{model.policy.actor_state.params['params']['Dense_0']}")

SAC.set_parameters(model, "sac_test", exact_match=True)
print(f"PARAMS AFTER SETTING:\n{model.policy.actor_state.params['params']['Dense_0']}")

actor = model.policy.actor_state

pre_rl = SAC.load(pre_rl_model_path)
SAC.set_parameters(pre_rl, pre_rl_model_path, exact_match=True)
pre_rl_actor = pre_rl.policy.actor_state

key = jax.random.PRNGKey(test_conf.seed)

policies = {
    "test_state_predict" : lambda obs: actor.apply_fn(actor.params, utils.flatten_obs(obs)).sample(seed=key)[0],
    "test_predict" : lambda obs: model.predict(obs)[0],
    "pre_rl_predict" : lambda obs: pre_rl.predict(obs)[0],
    "pre_rl_policy_predict" : lambda obs: pre_rl.policy.predict(obs)[0],
    "pre_rl_state" : lambda obs: pre_rl_actor.apply_fn(actor.params, utils.flatten_obs(obs)).sample(seed=key)[0],
    # "pre_rl_state_gets" : lambda obs: pre_rl.policy.actor.get_distribution(utils.flatten_obs(obs)).get_actions(deterministic=True) #.sample(seed=key)[0],
    "pre_rl_sample" : lambda obs: pre_rl.policy.sample_action(pre_rl_actor, pre_rl.policy.prepare_obs(obs)[0], key)[0]
    }

obs, _ = env.reset()
breakpoint()
for name, policy in policies.items():
    print(f"{name} test episode")
    utils.test_episode(policy, env, 0)


env.close()



'''
Test reward of loading in the weights vs using the saved model
'''