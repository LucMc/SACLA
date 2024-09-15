from typing import Optional
import atexit

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import jax.numpy as jnp

import sbx
from pathlib import Path
from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.polyc.POLYC import POLYC
from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils.utils import get_ckpt_dir
from prob_lyap.testing.testing_utils import params_equal, seed_modules#, check_error
# from prob_lyap.main import LYAP_CONFIG # Maybe have a json somewhere instead
from prob_lyap.utils.type_aliases import LyapConf
from prob_lyap.objectives import OBJ_TYPES
from prob_lyap.main import ALGOS
from prob_lyap.utils import utils

import matplotlib.pyplot as plt
import os
import click
from copy import deepcopy

POLICIES = ["standard", "adverserial", "polyc", "zero", "random", "all"]

def test_rollout(seed, env, policy, model, label, axs):
    obs, _, env = seed_modules(env, seed)
    done = False
    lyap_values = []
    errors = [] # Since reward is sparse
    actions = []

    while not done:
        action = policy(obs)
        flattened_obs = np.concatenate(list(obs.values()))
        lyap_v = model.lyap_state.apply_fn(model.lyap_state.params, flattened_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        lyap_values.append( lyap_v.item()) # Negative \(._.)/
        error = abs(sum(obs["desired_goal"] - obs["achieved_goal"]))
        errors.append(error)
        actions.append(action)

    #fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(list(range(len(lyap_values))), lyap_values, label="Lyapunov Values")
    axs[0].set_title(label + " Lyapunov Values")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Lyapunov Value")
    axs[0].legend()

    axs[1].plot(list(range(len(errors))), errors, label="Errors", color='orange')
    axs[1].set_title(label + " Errors")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Error")
    axs[1].legend()
    return axs[0], axs[1]

@click.command()
@click.option("-s", "--seed", default=np.random.randint(100), type=int, help="Random seed")
@click.option("-ts", "--step", default=-1, type=int, help="Training step (if -1 just use the latest)")
@click.option("-id", "--run-id", default=-1, type=int, help="Run id from logs/ckpts (if -1 just use the latest)")
@click.option("-g", "--graph", default="all", type=str, help=f"Which policies to plot ({'|'.join(POLICIES)})")
@click.option("-o", "--objective", type=str, default="adverserial", help=f"Learning objective function for RL ({' | '.join(OBJ_TYPES)})", show_default=True)
@click.option("--use-test-dir", type=bool, default=False, help=f"Use the default location for modles (~/.prob_lyap/... or ./models)", show_default=True)
@click.option("-nh", "--n-hidden", type=int, default=16, help="Number of hidden neurons in Lyapunov neural network", show_default=True)
@click.option("--env-id", type=str, default="FetchReach-v2", help="registered gym environment id", show_default=True)
@click.option("-a", "--algorithm", type=str, default="LSAC", help=f"Which algorithm to train from ({' | '.join(ALGOS)})", show_default=True)
@click.option("-fn", "--file-name", type=str, default="", help=f"Save the figure with provided file name", show_default=True)
def main(seed: int, step: Optional[int], run_id: Optional[int], graph: str, objective: str, use_test_dir: bool, n_hidden: int, env_id: str, algorithm: str, file_name: str):
    assert objective in OBJ_TYPES,f"please specifiy objective from ({'|'.join(OBJ_TYPES)})" 
    assert graph in POLICIES, f"please specifiy from ({'|'.join(POLICIES)})"
    # use_pretrained = True
    delay_type = NoneWrapper
    
    # Validation
    algorithm = algorithm.lower()
    objective = objective.lower()

    test_conf = LyapConf(
        seed=seed,
        env_id=env_id,
        # env_id="InvertedPendulum-v4",
        delay_type=delay_type, #  AugmentedRandomDelayWrapper OR UnseenRandomDelayWrapper OR NoneWrapper
        act_delay=range(0,1),
        obs_delay=range(0,1),
        objective=objective,
        n_hidden=n_hidden
        #ckpt_dir=ckpt_dir # Copy the checkpoint you want to investigate from ~/.prob_lyap
    )

    ckpt_dir = utils.get_model_dir(use_test_dir, test_conf, delay_type, objective, run_id, algorithm=algorithm)
    test_conf.ckpt_dir = ckpt_dir

    try:
        if step<0: step=np.array(os.listdir(ckpt_dir), dtype=int).max() 
    except ValueError:
        raise ValueError(f"checkpoint dir {ckpt_dir} has items {os.listdir(ckpt_dir)}, you may have stopped a run before a checkpoint was made!")
    
    click.secho(f"Loading step: {step} from model from: {ckpt_dir} using seed: {seed}", fg="green")
    click.secho(f"{'-'*24} Config {'-'*24}" + f"\n{test_conf}\n" + f"{'-'*50}",fg="yellow")

    pre_rl = sbx.SAC.load(Path(__file__).parent / ".." / f"pretrained/pretrained_models/NoneWrapper/{env_id}/SAC_0")
    env = test_conf.delay_type(gym.make(test_conf.env_id), act_delay_range=test_conf.act_delay, obs_delay_range=test_conf.obs_delay)

    # Chosen model
    if algorithm == "lsac":
        model = Lyap_SAC(test_conf, "MultiInputPolicy", env, verbose=1) # Params in correct order?
    elif algorithm == "polyc":
        model = POLYC(test_conf, "MlpPolicy", FlattenObservation(env), verbose=1)

    model.restore_ckpt(step) #utils.get_from_checkpoint(step=step, model=model) # replaces config anyway

    # Adverserial Model


    test_conf_adv = deepcopy(test_conf)
    test_conf_adv.objective = "adverserial"
    adv_ckpt_dir = utils.get_model_dir(use_test_dir, test_conf_adv, delay_type, "adverserial", run_id=-1, algorithm="lsac") # Just take the latest adv policy run
    test_conf_adv.ckpt_dir = adv_ckpt_dir
    adv_model = Lyap_SAC(test_conf_adv, "MultiInputPolicy", env, verbose=1) # Params in correct order?
    adv_step = np.array(os.listdir(adv_ckpt_dir), dtype=int).max() 
    adv_model.restore_ckpt(adv_step)# Use 60k as that's when the control policy seems to be converged #utils.get_from_checkpoint(step=step, model=model) # replaces config anyway

    adv_policy = lambda obs: adv_model.predict(obs, deterministic=True)[0]
    pre_trained_policy = lambda obs: pre_rl.predict(obs, deterministic=True)[0]
    random_policy = lambda _: env.action_space.sample()
    zero_policy = lambda _ : np.zeros_like(env.action_space.sample())
    
    if graph == "all": graph=POLICIES # For now
    if "polyc" in graph: graph.remove("polyc")
    if "all" in graph: graph.remove("all")

    fig, axs = plt.subplots(len(graph),2, figsize=(10,6))
    plt.suptitle(algorithm)
    # print(graph)
    if "standard" in graph:
        idx = graph.index("standard")
        #plt_lyap, plt_errors = test_rollout(test_conf.seed, env, pre_trained_policy, model, label="Standard SAC Controller (V/t)")
        # plot_config(plt_lyap, plt_errors)
        test_rollout(test_conf.seed, env, pre_trained_policy, model, label="Standard SAC Controller (V/t)", axs=[axs[idx, 0], axs[idx, 1]])

    if "adverserial" in graph:
        idx = graph.index("adverserial")
        test_rollout(test_conf.seed, env, adv_policy, model, label="Adversarial SAC Controller (V/t)", axs=[axs[idx, 0], axs[idx, 1]])

    if "polyc" in graph:
        pass
        # idx = graph.index("polyc")
        # axs[idx], axs[idx+1] = test_rollout(test_conf.seed, env, pre_trained_policy, model, label="Standard SAC Controller (V/t)")

    if "zero" in graph:
        idx = graph.index("zero")
        test_rollout(test_conf.seed, env, zero_policy, model, label="Zero Controller (V/t)", axs=[axs[idx, 0], axs[idx, 1]])

    if "random" in graph:
        idx = graph.index("random")
        test_rollout(test_conf.seed, env, random_policy, model, label="Random Controller (V/t)", axs=[axs[idx, 0], axs[idx, 1]])
    
    plt.tight_layout()
    if file_name == "":
        plt.waitforbuttonpress(0)
    else:
        plt.savefig(f"./plots/{file_name}.png")

@atexit.register
def close():
    plt.close()

if __name__ == "__main__":
    main()
