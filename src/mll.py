from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from tqdm import tqdm

from jaxutils.data.pt_preprocess import NumpyLoader
from jaxutils.data.utils import get_agnostic_batch
from jaxutils.utils import flatten_params
from src.utils import hessian_of_loss_fn, scaled_jvp

Array = chex.Array
PyTree = chex.PyTreeDef


def weight_space_Mackay_update(
    init_λ: float,
    diag_cov: jnp.ndarray,
    w_lin: PyTree,
    min_λ: float = 1e-2,
    max_λ: float = 1e10,
) -> float:

    # Take square, and get D
    w2 = jnp.power(flatten_params(w_lin), 2)
    D = flatten_params(w_lin).shape[0]

    eff_dim = D - init_λ * jnp.sum(diag_cov)
    eff_dim = jnp.clip(eff_dim, a_min=1, a_max=D - 1)

    new_λ = eff_dim / jnp.sum(w2)

    return new_λ, eff_dim, D, jnp.sum(diag_cov)


def compute_fn_space_update(
    x: Array,
    w_sample: PyTree,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    H_L_jitter: Optional[float] = None,
    scale_vec=None,
):

    # Define apply_fn for a single datapoint
    def apply_fn(w):
        return jnp.squeeze(
            model.apply(
                {"params": w, **model_state}, x[None, ...], train=False, mutable={}
            )[0]
        )

    # (O,), (O,)
    logits, lin_pred = scaled_jvp(apply_fn, params, w_sample, scale_vec=scale_vec)

    # Use amortized forward pass from VJP to calculate H_L, and Cholesky
    H_L = hessian_of_loss_fn(logits, H_L_jitter=H_L_jitter)  # (O x O)

    per_datapoint_eff_dim = jnp.squeeze(lin_pred[None, ...] @ H_L @ lin_pred[:, None])

    # Scalar value
    return per_datapoint_eff_dim


def compute_fn_space_update_batched(
    batch_x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    w_samples: PyTree,
    **kwargs
):
    """Batched version of compute_fn_space_update."""

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), axis_name="samples")
    @partial(jax.vmap, in_axes=(0, None), axis_name="batch")
    def vmapped_compute_fn_space_update(x, w_sample):
        return compute_fn_space_update(
            x, w_sample, model=model, params=params, model_state=model_state, **kwargs
        )

    # (K x B)
    all_samples_eff_dim = vmapped_compute_fn_space_update(batch_x, w_samples)

    # We need to average over the samples, add over the batch.
    batch_eff_dim = jnp.mean(jnp.sum(all_samples_eff_dim, axis=1), axis=0)

    # Further sum over devices
    return jax.lax.psum(batch_eff_dim, axis_name="device")


def fn_space_Mackay_update(
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    data_iterator: NumpyLoader,
    steps_per_epoch: int,
    init_λ: float,
    w_lin: PyTree,
    w_samples: PyTree,
    min_λ: float = 1e-2,
    max_λ: float = 1e10,
    H_L_jitter: Optional[float] = None,
    scale_vec=None,
    dataset_type="pytorch",
) -> float:

    _compute_fn_space_fn = partial(
        compute_fn_space_update_batched,
        model=model,
        params=params,
        model_state=model_state,
        w_samples=w_samples,
        H_L_jitter=H_L_jitter,
        scale_vec=scale_vec,
    )
    compute_fn_space_fn = jax.pmap(_compute_fn_space_fn, axis_name="device")

    fn_space_eff_dim = 0.0
    print("Calculating Function Space MacKay update")
    for i in tqdm(range(steps_per_epoch)):
        batch = get_agnostic_batch(next(data_iterator), dataset_type=dataset_type)

        fn_space_eff_dim += compute_fn_space_fn(batch[0])[0]

    # Take square, and get D
    w2 = jnp.power(flatten_params(w_lin), 2)
    D = flatten_params(w_lin).shape[0]

    # eff_dim = D - init_λ * jnp.sum(diag_cov)
    fn_space_eff_dim = jnp.clip(fn_space_eff_dim, a_min=1, a_max=D - 1)

    new_λ = fn_space_eff_dim / jnp.sum(w2)

    return new_λ, fn_space_eff_dim, D
