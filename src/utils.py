from functools import partial
from typing import Any, Mapping, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.traverse_util import ModelParamTraversal
from jax import jit, vmap
from tqdm import tqdm

from jaxutils.data.pt_preprocess import NumpyLoader
from jaxutils.data.utils import get_agnostic_batch
from jaxutils.utils import flatten_jacobian, flatten_params

Array = chex.Array
KwArgs = Mapping[str, Any]
PyTree = chex.PyTreeDef

traverse_batchnorm_params = ModelParamTraversal(lambda p, _: "BatchNorm" in p)
traverse_not_last_layer = ModelParamTraversal(lambda p, _: "Dense_0" not in p)


def hessian_of_loss_fn(
    logits: Array, H_L_jitter: Optional[float] = None, loss: str = "crossentropy"
) -> Array:
    """Single point hessian of loss, H_L = diag(y_hat) - y_hat @ y_hat.T

    This function is used to calculate H_L when we wish to avoid a forward pass
    through the neural network, e.g. when the VJP fn returns an amortised
    forward pass."""
    y_hat = jax.nn.softmax(logits, axis=-1)

    if loss == "crossentropy":
        H_L = jnp.diag(y_hat) - y_hat[:, None] @ y_hat[:, None].T
    else:
        raise NotImplementedError("only crossentropy loss is implemented for now.")
    if H_L_jitter is not None:
        H_L += H_L_jitter * jnp.eye(H_L.shape[0])

    return H_L


def compute_hessian_of_loss(
    x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    n_out: int = 10,
    w_lin: Optional[PyTree] = None,
    use_linear_model: bool = False,
    H_L_jitter: Optional[float] = None,
    loss: str = "crossentropy",
    scale_vec: Optional[PyTree] = None,
) -> Array:
    """Compute the hessian of the loss function H_L for a single input."""
    chex.assert_rank(x, 3)

    # Define apply_fn for a single datapoint
    def apply_fn(w):
        return jnp.squeeze(
            model.apply(
                {"params": w, **model_state}, x[None, ...], train=False, mutable={}
            )[0]
        )

    if use_linear_model:
        # We need to zero out batchnorm statistics in w_lin, so that in the
        # linear prediction using the vJP, the batchnorm statistics and params
        # do not have an effect.
        w_lin = zeroed_batchnorm_params(w_lin)

        _, logits = scaled_jvp(apply_fn, params, w_lin, scale_vec=scale_vec)
    else:
        logits = apply_fn(params)

    H_L = hessian_of_loss_fn(logits, H_L_jitter=H_L_jitter, loss=loss)

    chex.assert_shape(H_L, (n_out, n_out))

    return H_L


def compute_hessian_of_loss_batched(
    batch_x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    **kwargs: KwArgs,
) -> PyTree:
    """Batched version of compute_hessian_of_loss across x."""
    chex.assert_rank(batch_x, 4)

    fn = partial(
        compute_hessian_of_loss,
        model=model,
        params=params,
        model_state=model_state,
        **kwargs,
    )

    return jit(vmap(fn))(batch_x)


def compute_ggn_single_input(
    x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    n_out: int = 10,
    H_L_jitter: Optional[float] = None,
    scale_vec=None,
) -> Array:
    """Compute G = J_f^T H_L J_f to approximate Hessian for single example."""
    chex.assert_rank(x, 3)

    J_f = compute_jacobian(x, model, params, model_state)

    # TODO: Figure this out with PyTrees instead of flattening
    J_f = flatten_jacobian(J_f, n_out)  # (O x P)
    if scale_vec is not None:
        scale_vec = flatten_params(scale_vec)  # (P)
        J_f = J_f * scale_vec[None, ...]
    H_L = compute_hessian_of_loss(x, model, params, model_state, H_L_jitter=H_L_jitter)

    G = J_f.T @ H_L @ J_f

    return G


def compute_ggn_batched(
    batch_x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    **kwargs: KwArgs,
) -> PyTree:
    """Batched version of project_targets across x and y."""

    fn = partial(
        compute_ggn_single_input,
        model=model,
        params=params,
        model_state=model_state,
        **kwargs,
    )

    def sum_fn(x):
        return jax.tree_map(lambda w: jnp.sum(w, axis=0), vmap(fn)(x))

    return jax.lax.psum(jit(sum_fn)(batch_x), axis_name="device")


def get_ggn_matrix(
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    data_iterator: NumpyLoader,
    steps_per_epoch: int,
    n_out: int,
    H_L_jitter: Optional[float] = None,
    dataset_type: str = "pytorch",
    scale_vec=None,
) -> Array:
    """Compute GGN = sum G_i = J_f^T H_L J_f over an entire dataset."""

    # Get the length of the params
    D = flatten_params(params).shape[0]
    G = jnp.zeros((D, D))

    # Wrap the fn so that only x is passed to the fn
    compute_ggn_fn = partial(
        compute_ggn_batched,
        model=model,
        params=params,
        model_state=model_state,
        n_out=n_out,
        H_L_jitter=H_L_jitter,
        scale_vec=scale_vec,
    )

    compute_ggn_pmapped = jax.pmap(compute_ggn_fn, axis_name="device")

    print("Computing GGN matrix over entire dataset...")
    for i in tqdm(range(steps_per_epoch)):
        batch = get_agnostic_batch(next(data_iterator), dataset_type=dataset_type)
        G += compute_ggn_pmapped(batch[0])[0]

    return G


def compute_jacobian(
    x: Array, model: nn.Module, params: PyTree, model_state: PyTree
) -> PyTree:
    """Calculate elementwise Jacobian of model.apply(params, x) wrt params."""
    chex.assert_rank(x, 3)

    # Define apply_fn for a single datapoint
    def apply_fn(w):
        return jnp.squeeze(
            model.apply(
                {"params": w, **model_state}, x[None, ...], train=False, mutable={}
            )[0]
        )

    jac = jax.jacrev(apply_fn)(params)

    return jac


def compute_jacobian_batched(
    batch_x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    **kwargs: KwArgs,
) -> PyTree:
    """Batched version of compute_jacobians across x."""
    chex.assert_rank(batch_x, 4)

    fn = partial(
        compute_jacobian, model=model, params=params, model_state=model_state, **kwargs
    )

    return jit(vmap(fn))(batch_x)


def zeroed_batchnorm_params(params: PyTree) -> Tuple[PyTree, PyTree]:
    """For a given Flax model params, return zeroed bn params.

    Args:
        params: Flax model params.
        model_state: Flax model_state containing batchnorm stats.

    Returns:
        zeroed_bn_params
    """
    zeroed_params = traverse_batchnorm_params.update(
        lambda x: jnp.zeros_like(x), params
    )

    return zeroed_params


def scaled_jvp(f, primals, tangents, scale_vec=None):
    if scale_vec is not None:
        tangents = jax.tree_map(lambda x, y: jnp.multiply(x, y), tangents, scale_vec)

    tangents = zeroed_batchnorm_params(tangents)
    primals_out, tangents_out = jax.jvp(f, (primals,), (tangents,))

    return primals_out, tangents_out


def scaled_vjp(f, primals, scale_vec=None):
    logits, vjp_fn = jax.vjp(f, primals)

    def scaled_vjp_fn(tangents):
        # tangents = zeroed_batchnorm_params(tangents)
        if scale_vec is not None:
            return jax.tree_map(
                lambda x, y: jnp.multiply(x, y), scale_vec, vjp_fn(tangents)[0]
            )
        else:
            return vjp_fn(tangents)[0]

    return logits, scaled_vjp_fn
