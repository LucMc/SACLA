import numpy as np
import gymnasium as gym
import jax.numpy as jnp
import distrax

from prob_lyap.lyap_SAC import Lyap_SAC
from prob_lyap.world_model import WorldModel
from prob_lyap.utils.wrappers_rd import NoneWrapper, UnseenRandomDelayWrapper, AugmentedRandomDelayWrapper
from prob_lyap.utils import utils
from prob_lyap.utils.type_aliases import LyapConf
from prob_lyap.objectives import OBJ_TYPES

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax
import jax.numpy as jnp
from pathlib import Path
import sbx

import click
from typing import Callable

import os
import imageio

from pathlib import Path
from tqdm import tqdm
from flax.training.train_state import TrainState
import scienceplots

# plt.style.use(['ieee'])
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  # Use 'Times New Roman' as the serif font
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Fallback to a sans-serif font

# plt.style.use('ieee')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']  # Use 'Times New Roman' as the serif font
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Fallback to a sans-serif font

# plt.gcf().subplots_adjust(bottom=0.15)

'''
TODO:
 - Plot wm distribution at first frame
 - Plot distribution over time
'''
# from scipy import interpolate

def plot_dist_top(T, Z, pdf_values, label, f_name):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, Z, pdf_values, cmap='viridis')
    ax.set_xlabel('Time step')  # , fontsize=30, labelpad=15)
    ax.set_ylabel('Lyapunov Value')  # , fontsize=30, labelpad=15)
    # ax.set_zlabel('Probability Density')  # , fontsize=30, labelpad=15)
    plt.xticks()  # Adjust x-axis tick labels font size
    plt.yticks()  # Adjust y-axis tick labels font size
    ax.set_zticklabels([])

    # ax.set_title(label)
    ax.view_init(elev=90, azim=-90) # for 90 degree
    plt.subplots_adjust(bottom=0.2)

    # Add a colorbar which maps values to colors
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    colorbar.set_label('Probability Density')
    colorbar.ax.invert_yaxis()  # Invert the colorbar so high values are on top
    
    if f_name != "\0":
        plt.savefig(f"{f_name}.png")
    else:
        plt.show()
    # plt.waitforbuttonpress(0)
    plt.close()

def plot_dist_top_2D(T, Z, pdf_values, label, f_name):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(T, Z, pdf_values, cmap='viridis', shading='auto')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Lyapunov Value')

    # Add a colorbar to indicate the probability density
    cbar = plt.colorbar(c)
    cbar.set_label('Probability Density')
    ax.set_xlim(T.min(), T.max())
    ax.set_ylim(Z.min(), Z.max())
    plt.xticks()  # Adjust x-axis tick labels font size
    plt.yticks()  # Adjust y-axis tick labels font size

    # ax.set_title(label)
    # ax.view_init(elev=90, azim=-90) # for 90 degree
    # plt.subplots_adjust(bottom=0.2)

    # Add a colorbar which maps values to colors
    # colorbar.ax.invert_yaxis()  # Invert the colorbar so high values are on top
    
    if f_name != "\0":
        plt.show()
        # plt.savefig(f"{f_name}.png")
    else:
        plt.show()
    # plt.waitforbuttonpress(0)
    plt.close()


def plot_dist(T, Z, pdf_values, label, f_name):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, Z, pdf_values, cmap='viridis')
    ax.set_xlabel('Time step')  # , fontsize=30, labelpad=15)
    ax.set_ylabel('Lyapunov Value')  # , fontsize=30, labelpad=15)
    ax.set_zlabel('Probability Density')  # , fontsize=30, labelpad=15)
    plt.xticks()  # Adjust x-axis tick labels font size
    plt.yticks()  # Adjust y-axis tick labels font size

    # ax.set_title(label)
    plt.subplots_adjust(bottom=0.2)

    # Add a colorbar which maps values to colors
    # colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # optional
    # colorbar.set_label('Probability Density')
    # colorbar.ax.invert_yaxis()  # Invert the colorbar so high values are on top
    
    if f_name != "\0":
        plt.savefig(f"{f_name}.png")
    else:
        plt.show()
    # plt.waitforbuttonpress(0)
    plt.close()
    
def calc_error_distribution(g_size:int, mu:jnp.ndarray, sigma: jnp.ndarray): #use_lyap:bool, lyap_trainstate: TrainState):
    # When adding two normal dists simply add means and sigmas
    ag = mu[0][:g_size] # First part of obs
    dg = mu[0][g_size:g_size*2] # second part of obs
    ag_sig = sigma[0][:g_size]
    dg_sig = sigma[0][g_size:g_size*2]    
    
    # if use_lyap:
    #     erro_mu = lyap_trainstate.apply_fn(lyap_trainstate.params, obs)
    # else:
    error_mu = abs(dg-ag)
    assert error_mu.shape[0] == g_size
    error_sig = abs(dg_sig-ag_sig)
    assert error_mu.shape[0] == g_size    
    # print(f"Error sum {error_mu} Error sig {error_sig} sum mu: {error_mu.sum()} sum sigma: {error_sig.sum()}")
    
    return float(error_mu.sum()), float(error_sig.sum()) # Shouldn't sum sigmas?

def plot_wm(seed: int,
            env:gym.Env,
            policy:Callable,
            model:Lyap_SAC,
            label: str,
            view_steps: int,
            smoothing_factor: int,
            video_file: str,
            plot_file: str,
            USE_ERR: bool,
            DEBUG: bool,
            top: bool):

    view_steps *= smoothing_factor

    obs, _, env = utils.seed_modules(env, seed)
    assert env.observation_space['desired_goal'].shape[0] == env.observation_space['achieved_goal'].shape[0]
    g_size = env.observation_space['desired_goal'].shape[0]
    done = False
    error_mus = []
    lyap_mus = []

    sigmas = []
    actions = []

    images = []
    score = 0

    while not done:
        action = policy(obs)
        if isinstance(obs, dict):
            obs = model.policy.prepare_obs(obs)[0].reshape(1,-1)

        mu, sigma = model.wm_state.apply_fn(model.wm_state.params, obs, action.reshape(1,-1))
        error_mu, error_sig = calc_error_distribution(g_size, mu, sigma)

        # Change the mean error to lyap value, std should remain the same?
        if DEBUG: click.secho(f"DEBUG: pre-scaling Sigma {error_sig}", fg="green")

        if not USE_ERR: 
            lyap_mu = abs(model.lyap_state.apply_fn(model.lyap_state.params, mu).item())
            # error_sig = error_sig * (lyap_mu/error_mu) # Scale wm sigma values
            assert type(lyap_mu) == type(error_mu), f"Types of lyap mus ({type(lyap_mu)}) and error mu ({type(error_mu)}) differ: "
            lyap_mus.append(lyap_mu)
            
            if DEBUG: 
                click.secho(f"DEBUG: post-scaling Sigma {error_sig}", fg="green")
                click.secho(f"DEBUG: Guassian ratio {(lyap_mu/error_mu)} Error Value: {error_mu} Lyap Value: {lyap_mu}", fg="green")

        error_mus.append(error_mu)
        sigmas.append(error_sig)

        obs, reward, terminated, truncated, info = env.step(action)

        if video_file != "\0" and env.render_mode == "rgb_array":
            images.append(env.render())

        score += reward
        done = terminated | truncated
        actions.append(action)

        if video_file != "\0":
            imageio.mimsave(f"{video_file}.gif", [np.array(img) for img in images][:view_steps], duration=200)
    # Assuming mus and sigmas are already populated lists of mu and sigma over time
    # print(f"Mus: {mus[:10]} shape: {len(mus)}")
    # print(f"Sigmas: {sigmas[:10]} shape: {len(sigmas)}")
    error_mus = np.array(error_mus).flatten()
    sigmas = np.array(sigmas).flatten()
    time_steps = np.arange(len(error_mus))

    if not USE_ERR:
        lyap_mus = np.array(lyap_mus).flatten()
        error_range = max(error_mus) - min(error_mus)
        lyap_range = max(lyap_mus) - min(lyap_mus)
        sigmas *= lyap_range / error_range

    mus = error_mus if USE_ERR else lyap_mus
    time = np.linspace(0, len(mus), len(mus)*smoothing_factor)
    mus_smooth = np.interp(time, time_steps, mus)
    sigmas_smooth = np.interp(time, time_steps, sigmas)

    mus = mus_smooth; sigmas=sigmas_smooth; time_steps=time
    if DEBUG: click.secho(f"DEBUG: timesteps: {len(time)} sigmas: {len(sigmas)} mus: {len(mus)}" ,fg="green")
    # Create a grid for time and value space
    T, Z = np.meshgrid(time_steps, np.linspace(mus.min(), mus.max(), num=100))
    pdf_values = np.zeros_like(T, dtype=float)

    # Evaluate the PDF of the normal distribution at each point
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):#, total=len(mus):
        # print(mu,sigma)
        dist = distrax.Normal(loc=mu, scale=sigma)
        pdf_values[:, i] = dist.prob(Z[:, i])
    
    if top:
        plot_file += "_top"
        plot_dist_top_2D(T[:, :view_steps], Z[:, :view_steps], pdf_values[:, :view_steps], label + f" <{view_steps//smoothing_factor} timesteps", plot_file) 
        plot_dist_top(T[:, :view_steps], Z[:, :view_steps], pdf_values[:, :view_steps], label + f" <{view_steps//smoothing_factor} timesteps", plot_file)
        plot_dist_top(T, Z, pdf_values, label + " full episode", plot_file+"_full")
    else:
        plot_dist(T[:, :view_steps], Z[:, :view_steps], pdf_values[:, :view_steps], label + f" <{view_steps//smoothing_factor} timesteps", plot_file)
        plot_dist(T, Z, pdf_values, label + " full episode", plot_file+"_full")
       
    stats = {
          "Seed" : seed,
          "Smoothing Factor" : smoothing_factor,
          "View Steps" : view_steps,
          "Min Mu" : min(mus),
          "Avg mu" : mus.mean(),
          "max mu" : max(mus),
          "Min Sigma" : min(sigmas),
          "Avg sigma" : sigmas.mean(),
          "Max Sigma" : max(sigmas),
          "Score" : score,
          "Episode Length" : len(actions)
        }
    click.secho(f"{'-'*24} STATS {'-'*28}\n" +\
                "\n".join([f"{key}: {value}" for key, value in stats.items()]) +\
                f"\n{'-'*50}", fg="yellow")


@click.command()
@click.option("-vs", "--view-steps", default=8, help="The number of times steps to view for", show_default=True)
@click.option("-sf", "--smoothing-factor", default=4, help="Smoothen the manifold through interpolation", show_default=True)
@click.option("-s", "--seed", default=0, help="Random Seed", show_default=True)
@click.option("-ts", "--step", default=-1, type=int, help="Training step (if -1 just use the latest)")
@click.option("--L2", default=False, is_flag=True, help="Use eclidean distance (L2 norm) instead of Neural Lyapunov function", show_default=True)
@click.option("-vf", "--video-file", default="\0", help="Save video of episode up to <view steps> at specified location", show_default=True) # NULL \0 is invlid file name
@click.option("-pf", "--plot-file", default="\0", help="Save plots of episode up to <view steps> and full length at specified location", show_default=True) # NULL \0 is invlid file name
@click.option("--RENDER", is_flag=True, default=False, help="Render environment", show_default=True)
@click.option("-id", "--run-id", default=-1, type=int, help="Run id from logs/ckpts (if -1 just use the latest)")
@click.option("-o", "--objective", type=str, default="adverserial", help=f"Learning objective function for RL ({' | '.join(OBJ_TYPES)})", show_default=True)
@click.option("--use-test-dir", type=bool, default=False, help=f"Use the default location for modles (~/.prob_lyap/... or ./models)", show_default=True)
@click.option("-nh", "--n-hidden", type=int, default=16, help="Number of hidden neurons in Lyapunov neural network", show_default=True)
@click.option("--DEBUG", default=False, is_flag=True, help="Provides extra debugging information", show_default=True)
@click.option("--env-id", type=str, default="FetchReach-v2", help="registered gym environment id", show_default=True)
@click.option("--top", type=bool, default=False, is_flag=True, help="top view", show_default=True)
def main(view_steps: int,
         smoothing_factor: int,
         seed: int, 
         step: int, 
         video_file: str, 
         plot_file: str,
         l2: bool, 
         render: bool, 
         run_id: int, 
         objective: str, 
         use_test_dir: bool, 
         n_hidden: int, 
         debug: bool, 
         env_id: str,
         top: bool):


    assert objective in OBJ_TYPES,f"please specifiy objective from ({'|'.join(OBJ_TYPES)})" 
    delay_type = NoneWrapper
    pre_rl_model_path = Path(__file__).parent / ".." / f"pretrained/pretrained_models/NoneWrapper/{env_id}/SAC_0"

    test_conf = LyapConf(
        seed=seed,
        env_id=env_id,
        # env_id="InvertedPendulum-v4",
        delay_type=delay_type, #  AugmentedRandomDelayWrapper OR UnseenRandomDelayWrapper OR NoneWrapper
        act_delay=range(0,1),
        obs_delay=range(0,1),
        objective=objective,
        n_hidden=n_hidden
    )

    if use_test_dir:
        ckpt_dir = Path(__file__).parent / "models" / delay_type.__name__ / objective 
    else:
        ckpt_dir = utils.get_ckpt_dir(test_conf, create=False)[0].parent

    # handle default cases
    if run_id == -1:
        run_id = np.array(os.listdir(ckpt_dir), dtype=int).max() 
    ckpt_dir = ckpt_dir / str(run_id)

    if step<0: step=np.array(os.listdir(ckpt_dir), dtype=int).max() 
    assert os.path.isdir(ckpt_dir), f"ckpt dir {ckpt_dir} doesnt exist."
    test_conf.ckpt_dir = ckpt_dir
    click.secho(f"Loading model from: {ckpt_dir} using seed: {seed}", fg="green")

    if video_file != "\0":
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    env = delay_type(gym.make(test_conf.env_id, render_mode=render_mode),
                                act_delay_range=test_conf.act_delay,
                                obs_delay_range=test_conf.obs_delay)


    model = Lyap_SAC(test_conf, "MultiInputPolicy", env)
    model = utils.get_from_checkpoint(step=step, model=model) # replaces config anyway
    pre_rl = sbx.SAC.load(pre_rl_model_path)
    # pre_rl_actor = pre_rl.policy.actor_state
    # key = jax.random.PRNGKey(test_conf.seed)
    # pre_trained_policy = lambda obs: pre_rl_actor.apply_fn(pre_rl_actor.params, flatten_obs(obs)).sample(seed=key)[0]
    pre_trained_policy = lambda obs: pre_rl.predict(obs, deterministic=True)[0]
    # breakpoint()
    SAC_policy = lambda obs: model.predict(obs, deterministic=True)[0]
    random_policy = lambda _: env.action_space.sample()

    plot_wm(test_conf.seed,
            env,
            pre_trained_policy,
            model,
            label = r"Pre-trained SAC Controller $(e/t)$" if l2 else r"Pre-trained SAC Controller $(V(x)/t)$", 
            view_steps=view_steps,
            smoothing_factor=smoothing_factor,
            video_file=video_file,
            plot_file=plot_file,
            USE_ERR=l2,
            DEBUG=debug,
            top=top)

if __name__ == "__main__":
    main()
