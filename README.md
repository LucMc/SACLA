# Efficient Neural Lyapunov Function Approximation with Reinforcement Learning

# NOTE: This repository has been moved to: https://github.com/CAV-Research-Lab/SACLA


Installation instructions
1. clone this github repository
2. Install requirements via `pip install -e .` when in the root folder. This will install all the requirements to build the prob_lyap package from the `pyproject.toml` file.
3. run `main.py --help` to see configuration options.

We recommend using environments from the Gymnasium robotics benchmark or `InvertedPendulum-v4` however you can easily modify the chosen equilibrium state in `lyap_func_InvertedPendulum.py` to extend to other non goal-conditioned environments.

Evaluation is provided in `/src/prob_lyap/eval` where you can generate plots for your models. This automatically reads from the `~/.prob_lyap` directory created during training. You can specify training checkpoints to test using the `-ts [option]`. For more information use `--help` on any of the plotting scripts for a list of parameters and configuration options.

