from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import chex
import distrax
import flax
import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from flax import linen as nn
from flax.jax_utils import unreplicate
from jax import lax
from jax.scipy.special import logsumexp
from jax.tree_util import tree_leaves
from tqdm import tqdm

from jaxutils.data.pt_preprocess import NumpyLoader
from jaxutils.data.utils import get_agnostic_batch
from jaxutils.train.utils import TrainState
from jaxutils.utils import flatten_params, get_agg_fn
from src.utils import (
    get_ggn_matrix,
    hessian_of_loss_fn,
    scaled_jvp,
    zeroed_batchnorm_params,
)

Array = chex.Array
PyTree = chex.PyTreeDef


class SamplingTrainState(TrainState):
    """TrainState augmented with w_0s and w_samples, and support for
    Polyak averaging over w_samples, by maintaining an avg_w_samples value.
    Whether to use Polyak averaging is controlled by passing in non-None
    polyak_step_size."""

    w_lin: PyTree
    prior_prec: float
    w0_samples: PyTree
    w_samples: PyTree
    avg_w_samples: Optional[PyTree] = None
    exact_w_samples: Optional[jnp.ndarray] = None
    polyak_step_size: Optional[float] = None
    scale_vec: Optional[PyTree] = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.w_samples)

        new_w_samples = optax.apply_updates(self.w_samples, updates)

        # If using Polyak averaging, update avg_w_samples
        if self.polyak_step_size is not None:
            avg_w_samples = optax.incremental_update(
                new_w_samples, self.avg_w_samples, self.polyak_step_size
            )
        else:
            # If step size is None, avg_w_samples reports the latest w_samples
            avg_w_samples = None

        return self.replace(
            step=self.step + 1,
            w_samples=new_w_samples,
            opt_state=new_opt_state,
            avg_w_samples=avg_w_samples,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        w_lin,
        tx,
        prior_prec,
        w0_samples,
        w_samples,
        avg_w_samples,
        polyak_step_size=None,
        exact_w_samples=None,
        scale_vec=None,
        **kwargs
    ):
        opt_state = tx.init(w_samples)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            w_lin=w_lin,
            prior_prec=prior_prec,
            w0_samples=w0_samples,
            w_samples=w_samples,
            avg_w_samples=avg_w_samples,
            polyak_step_size=polyak_step_size,
            exact_w_samples=exact_w_samples,
            scale_vec=scale_vec,
            **kwargs,
        )


class SamplingPredictState(TrainState):
    """TrainState augmented with w_0s and w_samples, and support for
    Polyak averaging over w_samples, by maintaining an avg_w_samples value.
    Whether to use Polyak averaging is controlled by passing in non-None
    polyak_step_size."""

    w_lin: PyTree
    prior_prec: float
    w_samples: PyTree
    avg_w_samples: PyTree
    scale_vec: Optional[PyTree] = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        w_lin,
        tx,
        prior_prec,
        w_samples,
        avg_w_samples,
        scale_vec=None,
        **kwargs
    ):
        opt_state = None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            w_lin=w_lin,
            prior_prec=prior_prec,
            w_samples=w_samples,
            avg_w_samples=avg_w_samples,
            scale_vec=scale_vec,
            **kwargs,
        )


@partial(jax.jit)
@partial(jax.vmap, in_axes=(0, None, None), axis_name="sample")
def sample_from_prior_dist(rng, prior_prec, w_lin):
    """Vmapped version of sampling weights w ~ N(0, prior_prec^{inv})."""
    w_sample = jax.tree_map(
        lambda x: distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(x), scale_diag=prior_prec**-0.5 * jnp.ones_like(x)
        ).sample(seed=rng),
        w_lin,
    )

    return w_sample


def compute_sto_samples(
    x: Array,
    rng: random.PRNGKey,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    prior_prec: float,
    H_L_jitter: Optional[float] = None,
) -> PyTree:
    """For single point, get both prior and target samples.

    Prior sample is defined as w0_i ~ N(0, Lambda^{-1} J_f^T H_L J_f), which we get by
    using SVD decomposition of H_L, w0_i = Lambda^{-1} J_f^T U @ S^{0.5} eps_i. The
    target sample is defined as y_i ~ N(0, H_L^{-1}), which we get by using the same
    SVD decomposition and same eps_i, y_i = U @ S^{-0.5} eps_i."""

    # Fold in batch ID and samples ID to rng for different rng for each batch and sample
    rng = random.fold_in(
        random.fold_in(rng, lax.axis_index("batch")), lax.axis_index("samples")
    )

    # Define apply_fn for a single datapoint
    def apply_fn(w):
        return jnp.squeeze(
            model.apply(
                {"params": w, **model_state}, x[None, ...], train=False, mutable={}
            )[0]
        )

    logits, vjp_fn = jax.vjp(apply_fn, params)

    # Use amortized forward pass from VJP to calculate H_L, and Cholesky
    H_L = hessian_of_loss_fn(logits, H_L_jitter=H_L_jitter)

    U, S, _ = jnp.linalg.svd(H_L)  # O x O and O
    S_inv = 1 / S

    # Sample from standard normal
    eps = distrax.Normal(0.0, 1.0).sample(seed=rng, sample_shape=(H_L.shape[0],))  # O

    chol_inv_eps = U @ ((S_inv**0.5) * eps)  # O
    chol_eps = U @ ((S**0.5) * eps)  # O

    # Calculate J^T @ chol @ eps (PyTree)
    J_chol_eps = vjp_fn(chol_eps)[0]
    J_chol_eps = zeroed_batchnorm_params(J_chol_eps)  # TODO: This wasn't there before.

    # Calculate Lambda^-1 @ J^T @ chol @ eps
    scaled_J_chol_eps = jax.tree_map(lambda x: prior_prec**-1 * x, J_chol_eps)

    # Calculate (J^T @ chol @ eps)^2
    J_chol_eps_sq = jax.tree_map(lambda x: x**2, J_chol_eps)

    # (PyTree,), (PyTree), (HWC,), (O,)
    return scaled_J_chol_eps, J_chol_eps_sq, chol_inv_eps


def compute_sto_samples_batched(
    batch_x: Array,
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    prior_prec: float,
    rng: jax.random.PRNGKey,
    **kwargs
) -> PyTree:
    """Vmapped across num_samples, summed across vmapped batch_x."""

    # Since we pmap, we do not need to jit, as pmap automatically jit compiles the
    # fn given to it, see https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#pmap-and-jit
    # Pop num_samples, pass the remaining kwargs to compute_prior_sample
    num_samples = kwargs.pop("num_samples", 1)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), axis_name="samples")
    @partial(jax.vmap, in_axes=(0, None), axis_name="batch")
    def vmapped_compute_sto_samples(x, sample_id):
        return compute_sto_samples(
            x,
            rng=rng,
            model=model,
            params=params,
            model_state=model_state,
            prior_prec=prior_prec,
            **kwargs,
        )

    # K x B x PyTree, K x B x PyTree, K x B x O
    (
        all_scaled_w0_data_samples,
        all_w0_data_samples_sq,
        all_y_samples,
    ) = vmapped_compute_sto_samples(batch_x, jnp.arange(num_samples))

    # We need to sum w0 over batch dimension (which is dimension 1).
    summed_scaled_w0_data_samples = jax.tree_map(
        lambda w: jnp.sum(w, axis=1), all_scaled_w0_data_samples
    )

    # We need to sum w0_sq over samples and batch dimension (which is dimension 0 & 1).
    summed_w0_data_samples_sq = jax.tree_map(
        lambda w: jnp.sum(w, axis=(0, 1)), all_w0_data_samples_sq
    )

    # K x PyTree, PyTree, K x B x O
    return (
        jax.lax.psum(summed_scaled_w0_data_samples, axis_name="device"),
        jax.lax.psum(summed_w0_data_samples_sq, axis_name="device"),
        all_y_samples,
    )


def get_sto_samples(
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    data_iterator: NumpyLoader,
    steps_per_epoch: int,
    prior_prec: float,
    num_samples: int,
    rng: jax.random.PRNGKey,
    num_train_points: int,
    n_out: int = 10,
    H_L_jitter: Optional[float] = None,
    loss: str = "crossentropy",
    target_samples_path: str = "./sampled_targets.pkl",
    dataset_type: str = "tf",
):
    """Sample Eps from Eq. (6), and theta_0 and theta_n from Eq. (7).

    This function samples all initial quantities of interest for the Sampled
    Laplace procedure. This includes the following:
        y_samples: Eps from Eq. (6), ~N(0, H_L^{-1}), where H_L is the Hessian
            of the loss function at the MAP parameters. These values are
            calculated batchwise, and saved in an h5py file to avoid
            recalculating each STO step. Shape (N, O).
        w0s_prior: theta_0 from Eq. (6) and (7), ~N(0, prior_prec^{-1} * I).
            Shape (K, PyTree).
        w0s_data: theta_n from Eq. (7), Calculated batchwise from the training
            data using the same samples as y_samples. Each batch is sampled from
            ~N(0, H_L^{-1} J^T J H_L), which is calculated reusing samples as
            sum H_L^{-1} J^T prior_prec * eps. Shape (K, PyTree).
        inv_scale_vec: s from Eq. (17), given as prior_prec^{-1} * theta_N^2.
            Shape (PyTree,).

    Args:
        model: Flax module.
        params: Flax model parameters as a PyTree.
        model_state: Flax model state as a PyTree.
        data_iterator: Iterator over the training data. Should contain only
            unaugmented images.
        steps_per_epoch: Number of steps per epoch.
        prior_prec: Precision of the prior distribution.
        num_samples: Number of samples to draw for the STO procedure.
        rng: Jax random number generator.
        num_train_points: Number of training points.
        n_out: Number of output dimensions.
        H_L_jitter: Jitter to add to the Hessian of the loss function.
        loss: Loss function to use.
        target_samples_path: Path to save the sampled y_samples to.
        dataset_type: Type of dataset to use. Either "pytorch" or "tensorflow".

    Returns:
        w0s_prior, w0s_data, inv_scale_vec
    """
    target_samples_path = Path(target_samples_path).resolve()

    # Create the H5py file to save the sampled targets.
    target_samples_shape = (num_train_points, n_out, num_samples)
    max_shape = (None, n_out, num_samples)

    with h5py.File(target_samples_path, "w") as f:
        dset = f.create_dataset(
            "target_samples", target_samples_shape, maxshape=max_shape, dtype="f"
        )
        dset_targets = f.create_dataset(
            "targets", (num_train_points,), maxshape=(None,), dtype="f"
        )

    # Get rngs for sampling eta_0 ~ N(0, Lambda^{-1})
    prior_sample_rngs = random.split(rng, num_samples)

    # K, PyTree
    w0s_prior = sample_from_prior_dist(prior_sample_rngs, prior_prec, params)
    # K, PyTree containing zeros
    w0s_data = jax.tree_map(lambda w: jnp.zeros_like(w), w0s_prior)
    # PyTree containing zeros
    inv_scale_vec = jax.tree_map(lambda w: jnp.zeros_like(w), params)

    # Wrap all redundant arguments to expose pmapabbale arguments.
    def _compute_sto_samples_fn(x, rng):
        return compute_sto_samples_batched(
            x,
            rng=rng,
            model=model,
            params=params,
            model_state=model_state,
            prior_prec=prior_prec,
            H_L_jitter=H_L_jitter,
            num_samples=num_samples,
        )

    # pmap over the batch dimensions, and the rngs.
    compute_sto_samples_fn = jax.pmap(_compute_sto_samples_fn, axis_name="device")

    # For indexing into the H5 array, maintain a running counter.
    curr_id = 0
    print("Computing w0s, y_samples, and inv_scale_vec over entire dataset...")
    for i in tqdm(range(steps_per_epoch)):
        # Split rng for unique rng per batch.
        rng, sample_rng = random.split(rng)

        # n_devices x B_per_device x HWC
        batch = get_agnostic_batch(next(data_iterator), dataset_type)

        # n_devices x K x PyTree, n_devices x PyTree, n_devices x K x B_prd x O
        (
            scaled_w0s_data_batch,
            w0s_data_sq_batch,
            y_samples_batch,
        ) = compute_sto_samples_fn(
            batch[0], flax.training.common_utils.shard_prng_key(rng)
        )

        # n_devices x K x B_prd x O -> n_devices x B_prd x K x O
        y_samples_batch = jnp.transpose(y_samples_batch, (0, 2, 1, 3))

        # n_devices x B_prd x K x O -> B x K x O
        y_samples_batch = jnp.reshape(
            y_samples_batch,
            (
                y_samples_batch.shape[0] * y_samples_batch.shape[1],
                *y_samples_batch.shape[2:],
            ),
        )
        # B x K x O -> B x O x K
        y_samples_batch = jnp.transpose(y_samples_batch, (0, 2, 1))
        B = y_samples_batch.shape[0]

        # Reshape the labels as well
        new_batch_labels = jnp.reshape(
            batch[1], (batch[1].shape[0] * batch[1].shape[1], *batch[1].shape[2:])
        )

        # n_devices x K x PyTree -> K x PyTree, n_devices x PyTree -> PyTree
        scaled_w0s_data_batch = unreplicate(scaled_w0s_data_batch)
        w0s_data_sq_batch = unreplicate(w0s_data_sq_batch)

        # Accumulate over the entire dataset.
        w0s_data = jax.tree_map(lambda x, y: x + y, w0s_data, scaled_w0s_data_batch)
        inv_scale_vec = jax.tree_map(
            lambda x, y: x + y, inv_scale_vec, w0s_data_sq_batch
        )

        with h5py.File(target_samples_path, "a") as f:
            # Save the target_samples for the current batch
            dset = f["target_samples"]
            dset[curr_id : curr_id + B, :, :] = np.array(y_samples_batch)
            # Save the targets for future reference and correspondence
            dset_targets = f["targets"]
            dset_targets[curr_id : curr_id + B] = np.array(new_batch_labels)

        curr_id += B

    return w0s_prior, w0s_data, inv_scale_vec


def compute_exact_samples(
    model: nn.Module,
    params: PyTree,
    model_state: PyTree,
    data_iterator: NumpyLoader,
    samples_data_iterator: NumpyLoader,
    prior_prec: float,
    steps_per_epoch: int,
    num_samples: int,
    w_0s_prior: PyTree,
    w_0s: PyTree,
    n_out: int = 10,
    H_L_jitter: Optional[float] = None,
    dataset_type: str = "tf",
    recompute_GGN_matrix: bool = False,
    ggn_matrix_path: Optional[str] = None,
    scale_vec=None,
) -> PyTree:
    """Compute w_samples by inverting the GGN matrix exactly.

    Instead of solving Eq. (7) iteratively, we can compute the exact solution to
    the convex objective by inverting the GGN matrix. This is more expensive and
    is only used for comparison purposes.

    Args:
        model: The model to compute the GGN matrix for.
        params: The parameters of the model.
        model_state: The model state.
        data_iterator: The data iterator for training data.
        samples_data_iterator: The data iterator for y_samples.
        prior_prec: The prior precision.
        steps_per_epoch: The number of steps per epoch.
        num_samples: The number of samples to compute.
        w_0s_prior: theta_0 from Eq. (6), of shape (K, PyTree).
        w_0s: theta_n from Eq. (7), of shape (K, PyTree).
        n_out: The number of output dimensions.
        H_L_jitter: The jitter to add to the Hessian of the loss.
        dataset_type: The type of dataset to use.
        recompute_GGN_matrix: Whether to recompute the GGN matrix or load.
        ggn_matrix_path: The path to where the GGN matrix is saved.
        scale_vec: The scale vector from Eq. (17), which is not None if the
            g-prior is used.
    """

    ggn_matrix_path = Path(ggn_matrix_path) if ggn_matrix_path else None
    if ggn_matrix_path.exists() and not recompute_GGN_matrix:
        with open(ggn_matrix_path, "rb") as f:
            H = np.load(f)
    else:
        H = get_ggn_matrix(
            model,
            params,
            model_state,
            data_iterator,
            steps_per_epoch=steps_per_epoch,
            n_out=n_out,
            H_L_jitter=H_L_jitter,
            dataset_type=dataset_type,
            scale_vec=scale_vec,
        )
        with open(ggn_matrix_path, "wb") as f:
            np.save(f, H)

    H_plus_λ_I = H + prior_prec * jnp.eye(H.shape[0])
    # PyTree -> skinny vector (P,)
    w_0s = jax.vmap(flatten_params)(w_0s)
    w_0s_prior = jax.vmap(flatten_params)(w_0s_prior)

    # Solve the matrix inversion as a linear system on CPU, to have more memory.
    @partial(jax.jit, backend="cpu")
    def solve(A, b):
        return jnp.linalg.solve(A, b)

    exact_w_samples = []
    # We solve (H + Λ I)^(-1) (Λ w_0) = (Λ w_0s_prior + Λs * w_0s_data)
    print("Doing linsolve for exact_w_samples over num_samples...")
    for i in tqdm(range(num_samples)):
        exact_w_samples.append(solve(H_plus_λ_I, prior_prec * w_0s[i, ...]))

    exact_w_samples = jnp.stack(exact_w_samples, axis=0)

    return exact_w_samples, H_plus_λ_I


def create_sample_then_optimise_loss(
    model: nn.Module,
    batch_inputs: jnp.ndarray,
    target_samples: jnp.ndarray,
    w0_samples: jnp.ndarray,
    num_classes: int,
    prior_prec: float,
    num_train_points: int,
    num_samples: int,
    diagonal_jitter: Optional[float] = None,
    aggregate: str = "mean",
    exact_w_samples: Optional[jnp.ndarray] = None,
    use_new_objective: bool = False,
    scale_vec=None,
    **kwargs: Any
) -> Callable:
    """Creates the loss fn to calculate samples from posterior of linear model.

    The loss function is defined as the sum over a prior regularisation term
    $||\theta - \theta_0||^2_{prior_prec}, and the fit term which is given by
    $||y_i - J(x_i) \theta ||^2_{H_L(x_i)}.

    Args:
        model: a ``nn.Module`` object defining the flax model
        batch_inputs: size (B, *) with batched inputs
        target_samples: size (B, O, K) with sampled targets.
        w0_samples: size (B, P) with sampled weight priors.
        num_classes: number of classes in the dataset
        prior_prec: precision matrix of the prior over weights.
        train: whether to mutate batchnorm statistics or not.
        aggregate: whether to aggregate using 'mean' or 'sum'
    Returns:
        Scalar containing mean (or sum) of loss over a minibatch.
    """

    def batched_loss_fn(w_samples, params, model_state):
        # Define the model forward pass
        def apply_fn(w):
            return model.apply(
                {"params": w, **model_state}, batch_inputs, train=False, mutable={}
            )[0]

        def per_sample_loss_fn(w_sample, w0_sample, batch_target_sample):
            # Calc (lin_pred - target_sample). H_L . (lin_pred - target_sample)

            # We need to zero out batchnorm statistics in w_lin, so that in the
            # linear prediction using the vJP, the batchnorm statistics/params
            # do not have an effect.
            w_sample = zeroed_batchnorm_params(w_sample)
            w0_sample = zeroed_batchnorm_params(w0_sample)

            batch_logits, batch_lin_pred = scaled_jvp(
                apply_fn, params, w_sample, scale_vec=scale_vec
            )

            # Calculate ||(lin_pred - target_samples)||^2_{H_L} elementwise
            def per_example_fit_term(lin_pred, target_sample, curvature_pred):
                y_hat = nn.softmax(curvature_pred, axis=-1)
                if use_new_objective:
                    v = lin_pred
                else:
                    v = lin_pred - target_sample

                # Calculate H_L.v = H_L.(lin_pred - target_sample) using the
                # explicit formula for H_L = diag(y_hat) - y_hat . y_hat^T
                y_v = y_hat * v
                HvP = y_v - y_hat * jnp.sum(y_v, axis=0, keepdims=True)  # (B, O)
                # Add diagonal jitter to H_L here, as we use it when sampling
                if diagonal_jitter is not None:
                    HvP = HvP + diagonal_jitter * v

                return v @ HvP

            # Vmap the fit term over the batch dimension
            fit_term = jax.vmap(per_example_fit_term, in_axes=(0, 0, 0))(
                batch_lin_pred, batch_target_sample, batch_logits
            )

            # Calculate regularisation term (w_sample - w_0)^2_prior_prec
            w_diff_norm = 0.0
            for a, b in zip(tree_leaves(w_sample), tree_leaves(w0_sample)):
                w_diff_norm += jnp.sum((a - b) ** 2)

            # Divide reg_term by num_train_points here, for loss_val parity.
            reg_term = (prior_prec / num_train_points) * w_diff_norm

            # Divide fit_term by B here, for loss_val parity with blt.
            return jnp.mean(fit_term), reg_term

        # Vmap over all the samples
        all_samples_fit, all_samples_reg = jax.vmap(
            per_sample_loss_fn, in_axes=(0, 0, 2)
        )(w_samples, w0_samples, target_samples)

        # Divide loss_fit by (2 * num_classes) to have gradient parity with blt
        all_samples_loss = all_samples_fit / (2.0 * num_classes)
        # Divide loss_reg by (2 * num_classes) for gradient parity with blt
        all_samples_loss += all_samples_reg / (2.0 * num_classes)

        batch_metrics = {
            "loss": all_samples_fit + all_samples_reg,
            "fit_term": all_samples_fit,
            "reg_term": all_samples_reg,
            "scaled_loss": all_samples_loss,
            "scaled_fit_term": all_samples_fit / (2.0 * num_classes),
            "scaled_reg_term": all_samples_reg / (2.0 * num_classes),
        }

        agg = get_agg_fn(aggregate)

        batch_metrics = jax.tree_map(lambda x: agg(x, axis=0), batch_metrics)

        def w_norm_fn(w):
            return sum((jnp.sum(x**2) for x in tree_leaves(w)))

        all_w_samples_norms = jax.vmap(w_norm_fn)(w_samples)

        global_metrics = {
            "avg_w_samples_norm": jnp.mean(all_w_samples_norms),
        }

        if exact_w_samples is not None:
            # exact_w_samples = (PyTree) -> (BatchNorm, NonBatchNorm)
            flat_w_samples = jax.vmap(flatten_params)(w_samples)
            avg_w_err_norm = jnp.mean(
                jnp.sum((flat_w_samples - exact_w_samples) ** 2, axis=1)
            )
            global_metrics["avg_w_err_norm"] = avg_w_err_norm
        # We always need to sum over samples for gradient parity with blt
        loss = jnp.sum(all_samples_loss, axis=0)

        return loss, (batch_metrics, global_metrics)

    return jax.jit(batched_loss_fn)


def create_sampling_train_step(
    model,
    optimizer,
    num_classes,
    num_samples,
    num_train_points,
    H_L_jitter,
    use_new_objective,
):
    @jax.jit
    def train_step(state, batch_inputs, target_samples):
        # target_samples = (B, O, K), w_0s = (B, PyTree)
        # We aggregate="mean" to get average values over samples. The values are
        # already averaged over batches, which we should correct for when
        # aggregating metrics.

        loss_fn = create_sample_then_optimise_loss(
            model,
            batch_inputs,
            target_samples,
            state.w0_samples,
            num_classes,
            state.prior_prec,
            num_train_points,
            num_samples,
            diagonal_jitter=H_L_jitter,
            aggregate="mean",
            train=False,
            exact_w_samples=state.exact_w_samples,
            use_new_objective=use_new_objective,
            scale_vec=state.scale_vec,
        )

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, (train_metrics, global_metrics)), grads = loss_grad_fn(
            state.w_samples, state.params, state.model_state
        )

        grads = jax.lax.pmean(grads, "device")
        train_metrics = jax.lax.pmean(train_metrics, "device")
        global_metrics = jax.lax.pmean(global_metrics, "device")

        return state.apply_gradients(grads=grads), train_metrics, global_metrics

    return train_step


def create_sampling_eval_step(
    model,
    num_classes,
    num_samples,
    num_train_points,
    H_L_jitter,
    use_new_objective,
):
    @jax.jit
    def eval_step(state, batch_inputs, target_samples):
        # target_samples = (B, K, O), w_0s = (B, PyTree)
        loss_fn = create_sample_then_optimise_loss(
            model,
            batch_inputs,
            target_samples,
            state.w0_samples,
            num_classes,
            state.prior_prec,
            num_train_points,
            num_samples,
            diagonal_jitter=H_L_jitter,
            aggregate="mean",
            train=False,
            exact_w_samples=state.exact_w_samples,
            use_new_objective=use_new_objective,
            scale_vec=state.scale_vec,
        )

        _, (eval_metrics, global_metrics) = loss_fn(
            state.avg_w_samples, state.params, state.model_state
        )

        eval_metrics = jax.lax.pmean(eval_metrics, "device")
        global_metrics = jax.lax.pmean(global_metrics, "device")

        return eval_metrics, global_metrics

    return eval_step


def gibbs_mackay_logprob(batch_labels, f_xμ, all_samples_lin_pred):
    """
    batch_labels  (B,)
    f_xμ  (B, O)
    all_samples_lin_pred  (K, B, O)
    """
    # K -> lin_pred^2 = 1/K (J * w_Samples)^2
    marginal_variances = jnp.mean(jnp.power(all_samples_lin_pred, 2), axis=0)
    κ = 1 / jnp.sqrt(1 + jnp.pi * 0.125 * marginal_variances)  # (B, O)
    py_xs = jax.nn.log_softmax(f_xμ * κ, axis=1)  # (B, O)
    ll = py_xs[jnp.arange(batch_labels.shape[0]), batch_labels]  # (B,)
    return ll


def mc_logprob(batch_labels, f_xμ, all_samples_lin_pred):
    """
    batch_labels  (B,)
    f_xμ  (B, O)
    all_samples_lin_pred  (K, B, O)
    """
    logit_samples = f_xμ[None, :, :] + all_samples_lin_pred  # (K, B, O)
    class_sample_logprobs = jax.nn.log_softmax(logit_samples, axis=2)  # (K, B, O)
    class_logprobs = logsumexp(class_sample_logprobs, axis=0) - jnp.log(
        class_sample_logprobs.shape[0]
    )  # (B, O)
    ll = class_logprobs[jnp.arange(batch_labels.shape[0]), batch_labels]  # (B,)
    return ll


def mc_joint_logprob(batch_labels, f_xμ, all_samples_lin_pred):
    logit_samples = f_xμ[None, :, :] + all_samples_lin_pred  # (K, B, O)
    class_sample_logprobs = jax.nn.log_softmax(logit_samples, axis=2)  # (K, B, O)
    class_sample_lls = class_sample_logprobs[
        :, jnp.arange(batch_labels.shape[0]), batch_labels
    ]  # (K, B)
    class_sample_batch_ll = class_sample_lls.sum(axis=1)  # (K,)
    joint_ll = logsumexp(class_sample_batch_ll, axis=0) - jnp.log(
        class_sample_batch_ll.shape[0]
    )  # scalar
    return joint_ll


def dyadic_mc_joint_logprob(batch_labels, f_xμ, all_samples_lin_pred, rng):
    logit_samples = f_xμ[None, :, :] + all_samples_lin_pred  # (K, B, O)

    # logit_samples = jnp.reshape(logit_samples, (logit_samples.shape[0], 2, logit_samples.shape[1] // 2, logit_samples.shape[2])) # (K, 2, B // 2, O)
    class_sample_logprobs = jax.nn.log_softmax(logit_samples, axis=2)  # (K, B, O)
    class_sample_lls = class_sample_logprobs[
        :, jnp.arange(batch_labels.shape[0]), batch_labels
    ]  # (K, B)

    dyadic_lls = jnp.reshape(
        class_sample_lls, (class_sample_lls.shape[0], 2, class_sample_lls.shape[1] // 2)
    )  # (K, 2, B // 2)

    num_element_samples = jax.random.randint(
        rng, shape=(dyadic_lls.shape[2],), minval=1, maxval=9
    )  # (B // 2)

    num_element_samples = jnp.stack(
        [num_element_samples, 10.0 - num_element_samples], axis=0
    )  # (2, B // 2)

    scaled_dyadic_lls = dyadic_lls * num_element_samples[None, ...]  # (K, 2, B // 2)

    class_sample_batch_ll = scaled_dyadic_lls.sum(axis=1)  # (K, B // 2)
    joint_ll = logsumexp(class_sample_batch_ll, axis=0) - jnp.log(
        class_sample_batch_ll.shape[0]
    )  # (B // 2)

    # ll = jnp.mean(joint_ll)
    joint_ll = jnp.stack([joint_ll, joint_ll], axis=0)  # (B)
    return joint_ll


def create_sampled_laplace_prediction(
    model: nn.Module,
    batch_inputs: jnp.ndarray,
    batch_labels: jnp.ndarray,
    aggregate: str = "mean",
    scale_vec=None,
    method: str = "gibbs",
    rng: Optional[jax.random.PRNGKey] = None,
    **kwargs: Any
) -> Callable:
    """Creates the prediction function for the sampled laplace model.

    The loss function is defined as the sum over a prior regularisation term
    $||\theta - \theta_0||^2_{prior_prec}, and the fit term which is given by
    $||y_i - J(x_i) \theta ||^2_{H_L(x_i)}.

    Args:
        model: a ``nn.Module`` object defining the flax model
        batch_inputs: size (B, *) with batched inputs
        aggregate: whether to aggregate using 'mean' or 'sum'
    Returns:
        Scalar containing mean (or sum) of loss over a minibatch.
    """

    def batched_predict_fn(w_samples, params, model_state):
        # Define the model forward pass
        def apply_fn(w):
            return model.apply(
                {"params": w, **model_state}, batch_inputs, train=False, mutable={}
            )[0]

        def per_sample_forward_pass(w_sample):
            w_sample = zeroed_batchnorm_params(w_sample)

            batch_logits, batch_lin_pred = scaled_jvp(
                apply_fn, params, w_sample, scale_vec=scale_vec
            )

            return batch_logits, batch_lin_pred

        # Vmap over all the samples
        all_samples_logits, all_samples_lin_pred = jax.vmap(
            per_sample_forward_pass, in_axes=(0)
        )(w_samples)

        # all_samples_logits is repeated along the K dimension, take 0th axis
        f_xμ = all_samples_logits[0]

        if method == "gibbs":
            ll = gibbs_mackay_logprob(batch_labels, f_xμ, all_samples_lin_pred)  # B, O
        elif method == "mc":
            ll = mc_logprob(batch_labels, f_xμ, all_samples_lin_pred)
        elif method == "dyadic":
            ll = dyadic_mc_joint_logprob(batch_labels, f_xμ, all_samples_lin_pred, rng)

        batch_metrics = {
            "ll": ll,
        }

        agg = get_agg_fn(aggregate)

        batch_metrics = jax.tree_map(lambda x: agg(x, axis=0), batch_metrics)

        return batch_metrics

    return jax.jit(batched_predict_fn)


def create_sampling_prediction_step(model, prediction_method):
    @jax.jit
    def eval_step(state, batch_inputs, batch_labels, batch_rng):
        # target_samples = (B, K, O), w_0s = (B, PyTree)
        predict_fn = create_sampled_laplace_prediction(
            model,
            batch_inputs,
            batch_labels,
            aggregate="sum",  # We need to sum LL over the dataset.
            scale_vec=state.scale_vec,
            method=prediction_method,
            rng=batch_rng,
        )

        eval_metrics = predict_fn(state.w_samples, state.params, state.model_state)

        eval_metrics = jax.lax.pmean(eval_metrics, "device")

        return eval_metrics

    return eval_step
