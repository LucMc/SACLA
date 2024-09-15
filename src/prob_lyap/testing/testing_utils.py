import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import random
from flax.training.train_state import TrainState
import optax
import distrax

import numpy as np
import gymnasium as gym

import prob_lyap.utils.wrappers_rd as wrd
from prob_lyap.utils.wrappers_rd import UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.world_model import WorldModel
from prob_lyap.utils.utils import seed_modules, params_equal
from prob_lyap.utils import utils
from copy import deepcopy


def test_actor_state(model, seed, policy): # UNTESTED
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

def test_objectives():
   pass 

def test_seeding(env, seed):
    # Only using dict observation spaces

    # First seeding
    def get_values():
        rnd_int = np.random.randint(0,100)
        rnd_obs, _ = env.reset()
        rnd_action = env.action_space.sample()

        for i in range(10):
            rnd_transition = env.step(rnd_action) # Step a few times

        return rnd_int, rnd_obs, rnd_action, rnd_transition

    # Second seeding
    seed_modules(env, seed)
    first_int, first_obs, first_action, first_transition = get_values()
    seed_modules(env, seed)
    second_int, second_obs, second_action, second_transition = get_values()

    # Assertions
    assert first_int == second_int, "Numpy seeding failed"
    
    assert np.array_equal(np.concatenate(list(first_obs.values())),
                           np.concatenate(list(second_obs.values()))),\
                           "Environment seeding failed (reset function)"
    
    assert np.array_equal(first_action, second_action)
    assert len(first_transition) == len(second_transition)

    assert jnp.array_equal(np.concatenate(list(first_transition[0].values())),
                        np.concatenate(list(second_transition[0].values()))),\
                        "Environment seeding failed (obs step function)" # obs
    
    assert first_transition[1] == second_transition[1] # reward
    assert first_transition[2] == second_transition[2] # terminated
    assert first_transition[3] == second_transition[3] # truncated

    return True

def check_run_all():
    for delay_type in [None, AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper]:
        LYAP_CONFIG["delay_type"] = delay_type

        if LYAP_CONFIG["delay_type"]: # If not None
            env = LYAP_CONFIG["delay_type"]( gym.make(LYAP_CONFIG["env_id"]), **LYAP_CONFIG["delay_config"] )
        else:
            env = gym.make(LYAP_CONFIG["env_id"])
    
    return True
    ## Incomplete

def test_delayed_wrappers(env, seed=0):
    # Test we can propperly seed each wrapper
    for wrapper in [lambda x:x, wrd.UnseenRandomDelayWrapper, wrd.AugmentedRandomDelayWrapper]:
        test_seeding(wrapper(env), seed)
    
    return True

def test_world_model(env, seed=0):
    rng = random.PRNGKey(0)
    obs, _, env = seed_modules(env, seed)
    # obs_size = sum([x.shape[0] for x in env.observation_space.values()]) # total state size
    wm_key, rng = random.split(rng)
    learning_rate = 0.0001

    sample_obs = jnp.hstack([x for x in obs.values()]).reshape(1,-1)
    sample_act = jnp.array(env.action_space.sample()).reshape(1,-1)
    obs_size = sample_obs.shape[1]
    wm  = WorldModel()

    wm_state = TrainState.create(
        apply_fn=wm.apply,
        params=wm.init(wm_key, sample_obs, sample_act),
        tx=optax.adam(learning_rate=learning_rate)
    )

    # Testing
    mu, sigma = wm_state.apply_fn(wm_state.params, sample_obs, sample_act)
    lyap_dist = distrax.Normal(mu, sigma)
    # print(f"Mu: {mu}, Sigma: {sigma}")
    assert sigma.item() > 0
    # assert len(mus.flatten()) == obs_size
    # assert len(sigmas.flatten()) == obs_size
    # assert jnp.all(sigmas.flatten() > 0)
    return True

def test_delayed_obs(env, seed=0):
    no_delay_range = range(0,1)
    
    # Original env
    env = deepcopy(env)
    obs, _, env = seed_modules(env, seed)

    # Aug delay
    aug_env = AugmentedRandomDelayWrapper(deepcopy(env),
                                          obs_delay_range=no_delay_range, 
                                          act_delay_range=no_delay_range) # Specify no delay here, then try with delay later
    aug_obs, _, aug_env = seed_modules(aug_env, seed)
    
    # Unseen delay
    uns_env = UnseenRandomDelayWrapper(deepcopy(env),
                                       obs_delay_range=no_delay_range, 
                                       act_delay_range=no_delay_range)
    
    uns_obs, _, uns_env = seed_modules(uns_env, seed)
    
    assert np.all([x==y for x,y in zip(obs, uns_obs)]) # Initial states w/ no delay should be identical
    assert np.all([x==y for x,y in zip(obs, aug_obs)]) # Initial states w/ no delay should be identical

    for i in range(10):
        env.step(env.action_space.sample())
        uns_env.step(uns_env.action_space.sample())
        aug_env.step(uns_env.action_space.sample())

    assert np.all([x==y for x,y in zip(obs, uns_obs)]) # Zip checks doesn't check elements there isn't a match for
    assert np.all([x==y for x,y in zip(obs, aug_obs)]) # Zip checks doesn't check elements there isn't a match for
    # ONLY CHECKING DICTIONARY KEYYS

    delay_range = range(4,8) # Increased range
    
    # Original env
    env = deepcopy(env)
    obs, _, env = seed_modules(env, seed)
    
    # Aug delay
    # print(env.observation)
    aug_env = AugmentedRandomDelayWrapper(deepcopy(env),
                                          obs_delay_range=delay_range, 
                                          act_delay_range=delay_range) # Specify no delay here, then try with delay later
    aug_obs, _, aug_env = seed_modules(aug_env, seed)

    # Unseen delay
    uns_env = UnseenRandomDelayWrapper(deepcopy(env), # deepcopy otherwise obs space gets wild
                                       obs_delay_range=delay_range, 
                                       act_delay_range=delay_range)
    uns_obs, _, uns_env = seed_modules(uns_env, seed)

    assert np.all([x==y for x,y in zip(obs, uns_obs)]) # Initial states w/ delay should be identical
    assert np.all([x==y for x,y in zip(obs, aug_obs)]) # Zip checks doesn't check elements there isn't a match for
    assert len(aug_obs['observation']) == env.observation_space['observation'].shape[0] + \
                                          aug_env.action_space.shape[0]*(max(delay_range)*2+1) # Just for act del +1 for RLRD
    assert env.observation_space['observation'].shape == (10,)
    assert uns_env.observation_space['observation'].shape == (10,)
    assert aug_env.observation_space['observation'].shape == (38,)

    for i in range(10):
        uns_env.step(uns_env.action_space.sample())
        aug_env.step(uns_env.action_space.sample())

    assert np.all([x==y for x,y in zip(uns_obs, aug_obs)]) # Zip checks doesn't check elements there isn't a match for


if __name__ == "__main__":
    SEED = 0
    env = gym.make("FetchReach-v2")
    test_passed = True
    tests = [
        test_world_model,
        test_delayed_wrappers,
        test_delayed_obs,
             ]
    
    for test in tests:
        test_passed = test(env, SEED)
        print("---- Test Sucessful ----")
    print("xxxx All Tests Complete xxxx")