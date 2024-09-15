import os
import pandas as pd
from copy import deepcopy
import sbx
import jax
import jax.numpy as jnp

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

import matplotlib.pyplot as plt
import scienceplots
import atexit
import click
from pathlib import Path
from collections import namedtuple, OrderedDict
 
from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.lyap_func import Lyap_net
from prob_lyap.polyc.POLYC import POLYC
from prob_lyap.pretrained.PID import PID
from prob_lyap.utils.wrappers_rd import NoneWrapper#, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils.type_aliases import LyapConf
from prob_lyap.utils import utils
from prob_lyap.objectives import OBJ_TYPES
from prob_lyap.main import ALGOS # Should make a new file with constants in it, shouldn't import from main

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  # Use 'Times New Roman' as the serif font
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Fallback to a sans-serif font

# import time
# start = time.time()

'''
Note the PID controller is essentially a proportional controller
since we only evalue the Lie derivative over a single transition.
'''
Colours = namedtuple("Colours", ["light", "medium", "dark"])
REDS = Colours('lightcoral', 'indianred', 'firebrick') # Indexes are ambiguous, used namedT or something
BLUES = Colours('cornflowerblue', 'royalblue', 'slateblue')

def choose_colour(c: np.ndarray, elem: float):

    std = c.std()
    mean = c.mean()

    cs = REDS if elem > 0 else BLUES
    if abs(elem) > mean + std:
        return cs.dark
    elif abs(elem) < std:
        return cs.medium
    else:
        return cs.light

def pprint_config(**kwargs):
    click.secho(f"{'-'*25} CONFIG {'-'*26}", fg="yellow")
    for name, value in kwargs.items():
        click.secho(f"{name}: {value}", fg="bright_yellow")
    click.secho('-'*60, fg="yellow")

def get_models(env_id: int, seed: int, step: int, objective: str, n_hidden: int, n_layers: int, algorithm: str, use_test_dir: bool, run_id: int):
    delay_type = NoneWrapper
    test_conf = LyapConf(
        seed=seed,
        env_id=env_id,
        # env_id="InvertedPendulum-v4",
        delay_type=delay_type, #  AugmentedRandomDelayWrapper OR UnseenRandomDelayWrapper OR NoneWrapper
        act_delay=range(0,1),
        obs_delay=range(0,1),
        objective=objective,
        n_hidden=n_hidden,
        n_layers=n_layers
    )

    if use_test_dir:
        ckpt_dir = Path(__file__).parent / "models" / delay_type.__name__ / objective 
    else:
        ckpt_dir = utils.get_ckpt_dir(test_conf, algorithm=algorithm, create=False)[0].parent

    # handle default cases
    if run_id == -1:
        run_id = np.array(os.listdir(ckpt_dir), dtype=int).max() 
    ckpt_dir = ckpt_dir / str(run_id)
    if step<0: step=np.array(os.listdir(ckpt_dir), dtype=int).max() 

    assert os.path.isdir(ckpt_dir), f"ckpt dir {ckpt_dir} doesnt exist."
    test_conf.ckpt_dir = ckpt_dir
    click.secho(f"Loading model from: {ckpt_dir} using seed: {seed}", fg="green")

    env = test_conf.delay_type(gym.make(test_conf.env_id), act_delay_range=test_conf.act_delay, obs_delay_range=test_conf.obs_delay)
    if algorithm == "lsac": model = Lyap_SAC(test_conf, "MultiInputPolicy", deepcopy(env))
    elif algorithm == "polyc": model = POLYC(test_conf, "MlpPolicy", FlattenObservation(deepcopy(env)))
    else: raise(f"Invalid algorithm selection, please choice from ({'|'.join(ALGOS)})")
    # model = deepcopy(model)
    # breakpoint()
    model.restore_ckpt(step=step) # replaces config anyway
    pid_controller = PID(P=0.5,I=0,D=0) # P=2.5; I=0.112; D=0.0
    pre_rl = sbx.SAC.load(Path(__file__).parent / ".." / "pretrained/pretrained_models/NoneWrapper/FetchReach-v2/SAC_0") # Could also take in as param
    
    pre_trained_policy = lambda obs: pre_rl.predict(obs, deterministic=True)[0]
    adverserial_policy = lambda obs: model.predict(obs, deterministic=True)[0]
    PID_policy = lambda x: jnp.append(pid_controller.act(x), jnp.array([0]))

    pre_rl_actor = pre_rl.policy.actor_state
    key = jax.random.PRNGKey(0) # Maybe changeable?
    pre_trained_policy = lambda obs: pre_rl_actor.apply_fn(pre_rl_actor.params, obs).sample(seed=key)[0].reshape(1,-1)

    adverserial_actor = model.policy.actor_state
    adverserial_policy = lambda obs: adverserial_actor.apply_fn(adverserial_actor.params, obs).sample(seed=key)[0].reshape(1,-1)

    random_policy = lambda obs: env.action_space.sample().reshape(1,-1) # NOT RANDOM !?

    zero_policy = lambda obs: np.zeros_like(env.action_space.sample()).reshape(1,-1)
    return env, model, PID_policy, pre_trained_policy, adverserial_policy, random_policy, zero_policy


def get_p_name(policies: dict):
    names = list(policies.keys())
    index = list(policies.values()).index(True)
    return names[index]
    
# python plot_vector_field.py -l 200 -vr 0.5
@click.command()
@click.option("--env-id", type=str, default="FetchReach-v2", help="registered gym environment id", show_default=True)
@click.option("-a", "--algorithm", type=str, default="lsac", help=f"Which algorithm to train from ({' | '.join(ALGOS)})", show_default=True)
@click.option("-vr", "--value-range", default=2, type=float, help="Range to plot around goal location", show_default=True)
@click.option("-r", "--resolution", default=8, type=int, help="The number of vectors to plot per axis", show_default=True)
@click.option("-f", "--file-name", default="", type=str, help="File name to save the plot if specified", show_default=False)
@click.option("-ts", "--step", default=-1, type=int, help="Training step (if -1 just use the latest)")
@click.option("-l", "--length", default=2, type=float, help="Plotted arrow length", show_default=True)
@click.option("--plot-actions", default=False, type=bool, is_flag=True, help="Plots actions rather than Lie derivatives", show_default=True)
@click.option("-s", "--seed", default=np.random.randint(1000), type=int, help="Random seed", show_default=True)
@click.option("--pid", is_flag=True, default=False, type=bool, help="Use a PID controller instead of RL", show_default=True)
@click.option("--adverserial", is_flag=True, default=False, type=bool, help="Use the adverserial policy instead of ~optimal policy", show_default=True)
@click.option("--random", is_flag=True, default=False, type=bool, help="Use a random policy instead of ~optimal policy", show_default=True)
@click.option("--zero", is_flag=True, default=False, type=bool, help="Use zero-action policy instead of ~optimal policy", show_default=True)
@click.option("-id", "--run-id", default=-1, type=int, help="Run id from logs/ckpts (if -1 just use the latest)")
@click.option("-o", "--objective", type=str, default="adverserial", help=f"Learning objective function for RL ({' | '.join(OBJ_TYPES)})", show_default=True)
@click.option("--use-test-dir", type=bool, default=False, help=f"Use the default location for modles (~/.prob_lyap/... or ./models)", show_default=True)
@click.option("-nh", "--n-hidden", type=int, default=16, help="Number of hidden neurons in Lyapunov neural network", show_default=True)
@click.option("-nl", "--n-layers", type=int, default=1, help="Number of mlp layers in Lyapunov neural network", show_default=True)
@click.option("--run-all-objectives", type=bool, is_flag=True, default=False, help=f"run all objectives one after the other", show_default=True)
@click.option("--verbose", type=bool, is_flag=True, default=False, help=f"output extra config information", show_default=True)
@click.option("--hide-plots", type=bool, is_flag=True, default=False, help="Hide plots so they don't show you just get the stats", show_default=True)
@click.option("--full-experiment", type=bool, is_flag=True, default=False, help="runs full experiment for all seeds", show_default=True)
@click.option("--df-file", type=str, default="\0", help="Save dataframe from run as csv with given filename")
def create_vectorfield(env_id: str,
                       algorithm: str,
                       value_range: int,
                       resolution: int,
                       file_name: str,
                       step: int,
                       length: int,
                       plot_actions: bool,
                       seed: int,
                       pid: bool,
                       adverserial: bool,
                       random: bool,
                       zero: bool,
                       run_id: int,
                       objective: str,
                       use_test_dir: bool,
                       n_hidden: int,
                       n_layers: int,
                       run_all_objectives: bool,
                       verbose: bool,
                       hide_plots: bool,
                       full_experiment: bool,
                       df_file: str):
    
    algorithm=algorithm.lower(); objective=objective.lower()
    # if algorithm == "polyc":
    #     objective="polyc"

    df = pd.DataFrame(columns=["seed", "Objective", "step", "percent", "+ive", "-ive"])
    if file_name!="":
        # plt.style.use(['ieee'])
        hide_plots=True
        

    if full_experiment:
        seeds = list(range(5))
        steps = np.arange(20_000,240_001,20_000) # start, end, step
        run_all_objectives = True
        hide_plots = True
    else:
        seeds = [seed]
        steps = [step]

    if run_all_objectives and not algorithm=="polyc":
        objectives = list(OBJ_TYPES.keys())
    else:
        objectives = [objective]

    for step in steps:
        for seed in seeds:
            for objective in objectives:
                if verbose: pprint_config(value_range=value_range,
                                resolution=resolution,
                                file_name=file_name,
                                step=step,
                                length=length,
                                plot_actions=plot_actions,
                                seed=seed,
                                pid=pid,
                                adverserial=adverserial,
                                random=random,
                                zero=zero,
                                run_id=run_id,
                                objective=objective,
                                use_test_dir=use_test_dir,
                                n_hidden=n_hidden,
                                run_all_objectives=run_all_objectives,
                                hide_plots=hide_plots
                                )

                
                length = length/1000 if plot_actions else length


                (env,
                model,
                PID_policy,
                pre_trained_policy, 
                adverserial_policy,
                random_policy,
                zero_policy) = get_models(env_id, seed, step, objective, n_hidden, n_layers, algorithm, use_test_dir, run_id)

                policies_bool = [pid, adverserial, random, zero]
                if True in policies_bool:
                    p_index = policies_bool.index(True)
                    policies = OrderedDict()
                    policies["PID"] = PID_policy
                    policies["Adverserial"] = adverserial_policy
                    policies["Random"] = random_policy
                    policies["Zero"] = zero_policy

                    policy_labels = list(policies.keys())
                    policy_fns = list(policies.values())

                    policy = policy_fns[p_index]
                    label = policy_labels[p_index] 
                else:
                    policy = pre_trained_policy
                    label = "Pre-Trained SAC"

                # label += " Policy Actions" if plot_actions else r' Lie Derivative Vector field $\vec{\delta V}$'
                label="" # REMOVED TITLE
                # Create a blank region arround the goal location
                obs, _ = env.reset(seed=seed)
                xg, yg, zg = obs['desired_goal'] # Random goal location

                # Specify range around goal
                X = np.linspace(-value_range, value_range, resolution) + xg
                Y = np.linspace(-value_range, value_range, resolution) + yg
                Z = np.linspace(-value_range, value_range, resolution) + zg

                lyap_state = model.lyap_state
                prev_lyap_state = deepcopy(lyap_state)
                if algorithm == "lsac":
                    wm_state = model.wm_state
                else:
                    _, wm_model, _, _, _, _, _ =  get_models(env_id, seed, -1, "mixed_adv", n_hidden, n_layers, "lsac", use_test_dir, run_id) # Should add a way of handling lsac and polyc with different params
                    wm_state = wm_model.wm_state
                # from prob_lyap.utils.utils import params_equal
                # params_equal(prev_lyap_state.params, model.lyap_state.params)

                (env,
                model,
                PID_policy,
                pre_trained_policy, 
                adverserial_policy,
                random_policy,
                zero_policy) = get_models(env_id, seed, step, objective, n_hidden, n_layers, algorithm, use_test_dir, run_id)
                lyap_state = model.lyap_state

                # Experiment with velocity

                lie_vs = []
                X, Y, Z = np.meshgrid(X, Y, Z)
                print(r"[ ] Generating data", end="\r")
                test_obs = jnp.zeros_like(obs['observation']) # random values?

                @jax.jit
                def get_vec(test_obs, XYZ):
                    x, y, z = XYZ
                    # Prepare_obs -> ag, dg, obs

                    test_obs_ = test_obs.at[:3].set(jnp.array([x, y, z])) # Add position
                    test_obs = jnp.concatenate((jnp.array([x, y, z, xg, yg, zg]), test_obs_)).reshape(1, -1) # Create dict and do with prepare obs?

                    # new_elements = jnp.array([[x, y, z], [xg, yg, zg]])
                    # test_obs = jnp.concatenate([test_obs_updated.flatten(), new_elements.flatten()]).reshape(1, -1)
                    if pid: # Make this a select/cond too
                        action = -policy(jnp.array([xg, yg, zg]) - jnp.array([x, y, z])).reshape(1, -1)
                    # elif adverserial:
                    #     action = adverserial_policy(test_obs)#.reshape(1, -1)
                    else:
                        action = policy(test_obs)#.reshape(1, -1)
                        # print(action)
                    
                    v_candidate = lyap_state.apply_fn(lyap_state.params, test_obs)  
                    lie_v = -Lyap_net.lie_derivative(lyap_state, wm_state, test_obs, action, v_candidate)


                    return jax.lax.select(plot_actions, action[0][:3], lie_v[0])
                    # return action[0][:4] if plot_actions else lie_v[0]#-lie_v[0]

                XYZ = jnp.stack((jnp.array(X.flat), jnp.array(Y.flat), jnp.array(Z.flat)), axis=-1)
                v_get_vec = jax.vmap(get_vec, in_axes=(None, 0))
                lie_vs = v_get_vec(test_obs, XYZ)
                if algorithm=="polyc": lie_vs=-lie_vs # Because state is negative in POLYC implementation
                # print(action[0][:3].reshape(lie_v[0].shape))
                # lie_vs.append(action[0][:3].reshape(lie_v[0].shape))
                
                print(r"[X] Generating data")

                fig = plt.figure(figsize=(18,10))
                ax = fig.add_subplot(111, projection='3d')

                u=np.array(lie_vs[:, 0]).reshape(X.shape)
                v=np.array(lie_vs[:, 1]).reshape(Y.shape)
                w=np.array(lie_vs[:, 2]).reshape(Z.shape)
                c = np.mean(np.array(lie_vs), axis=1)
                signs = np.sign(c).flatten()

                positive = len(signs[signs>0])
                negative = len(signs[signs<0])
                percent = (negative / (positive+negative)) * 100
                click.echo(f"Objective: {objective} Percentage positive: {percent}, +ive {positive} -ive {negative}")
                c = np.array([choose_colour(c,c[i]) for i in range(len(c))])
                #c = np.concatenate([c, np.repeat(c, 2)])
                # c = np.repeat(c,2)
                ax.quiver(X, Y, Z, u, v, w, colors=c, edgecolor=c, length=length)
                # ax.quiver(X, Y, Z, -X, -Y, -Z, colors=c, length=0.4)

                ax.set_xlabel('X Axis')
                ax.set_ylabel('Z Axis')
                ax.set_zlabel('Y Axis')
                ax.scatter(xg, yg, zg, color='lime', s=100, label='Desired Goal')  # s is the size of the point
                
                ax.set_title(label)

                if file_name != "":
                    print(r"[ ] Saving", end="\r")
                    if run_all_objectives:
                        fig_path = Path(__file__).parent / "plots" / f'{file_name}_{objective}.png'
                        click.secho(fig_path, fg="blue")
                    else:
                        fig_path = Path(__file__).parent / "plots" / f'{file_name}.png'
                    fig.savefig(fig_path, dpi=650)
                    print(r"[X] Saved{}".format(fig_path)) # Why a random g after?
                if not hide_plots: plt.show()
                plt.close()
    
                df.loc[len(df)] = [seed, objective, step, float(percent), float(positive), float(negative)]

    if df_file != "\0":
        click.secho(f"Saving dataset as {df_file}", fg="blue")
        df.to_csv(f'./data/{df_file}.csv', index=False)

    elif full_experiment:
        click.secho("Saving dataset ...", fg="blue")
        df.to_csv('./data/obj_data.csv', index=False)


if __name__ == "__main__":
    create_vectorfield()
