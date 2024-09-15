
import jax
import jax.numpy as jnp

@jax.jit
def standard(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return rewards

@jax.jit
def adverserial(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return -lyap_risks

@jax.jit
def POLYC_objective(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return (beta * lyap_risks) + ( (1-beta) * rewards ) # Try not flatten

@jax.jit
def mixed_adv_objective(rewards: jnp.ndarray, lyap_risks: jnp.ndarray, beta: float):
    return (beta * -lyap_risks + ((1-beta) * rewards )) 

OBJ_TYPES = {"standard": standard,
            "adverserial": adverserial,
            "polyc": POLYC_objective,
             "mixed_adv": mixed_adv_objective}

def get_objective(objective="standard") -> None:
    return OBJ_TYPES[objective]
