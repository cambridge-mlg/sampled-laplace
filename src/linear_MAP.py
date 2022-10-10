from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.tree_util import tree_leaves

from jaxutils.train.utils import TrainState
from jaxutils.utils import get_agg_fn
from src.utils import scaled_jvp, zeroed_batchnorm_params

PyTree = Any


class LinearTrainState(TrainState):
    """TrainState from jaxutils augmented with w_lin and prior_prec."""

    w_lin: PyTree
    prior_prec: float
    scale_vec: Optional[PyTree] = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.w_lin)
        new_w_lin = optax.apply_updates(self.w_lin, updates)
        return self.replace(
            step=self.step + 1,
            w_lin=new_w_lin,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *, apply_fn, params, w_lin, tx, prior_prec, scale_vec=None, **kwargs
    ):
        opt_state = tx.init(w_lin)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            w_lin=w_lin,
            prior_prec=prior_prec,
            scale_vec=scale_vec,
            **kwargs,
        )


def create_linear_mode_loss(
    model: nn.Module,
    batch_inputs: jnp.ndarray,
    batch_labels: jnp.ndarray,
    num_classes: int,
    prior_prec: float,
    num_train_points: int,
    aggregate: str = "mean",
    scale_vec=None,
    **kwargs: Any
) -> Callable:
    """Creates the convex loss fn used to calculate the linear mode of the BNN.

    The loss function is defined as $L(J(.).\theta) + ||\theta||^2_{prior_prec}$
    where L is the regression/classification loss function used for training the
    model, and J(.) is the Jacobian of the NN model w.r.t MAP estimates of the
    params. The prior over the params is Gaussian with zero mean and a precision
    matrix given by `prior_prec`.

    Args:
        model: a ``nn.Module`` object defining the flax model
        batch_inputs: size (B, ...) with batched inputs
        batch_labels: size (B, 1) with batched class labels
        num_classes: number of classes in the dataset
        prior_prec: precision matrix of the prior over weights.
        train: whether to mutate batchnorm statistics or not.
        aggregate: whether to aggregate using 'mean' or 'sum'
    Returns:
        loss: Scalar containing mean (or sum) of loss over a minibatch.
        batch_metrics: Dict containing metrics calculated batchwise for data.
        global_metrics: Dict containing metrics calculated globally for data.
    """

    def batched_loss_fn(w_lin, params, model_state):
        # We need to zero out batchnorm statistics in w_lin, so that in the
        # linear prediction using the vJP, the batchnorm statistics and params
        # do not have an effect.
        w_lin = zeroed_batchnorm_params(w_lin)

        def apply_fn(w):
            return model.apply(
                {"params": w, **model_state}, batch_inputs, train=False, mutable={}
            )[0]

        # Perform JvP to get lin_pred = J( )_{params} . w_lin
        batch_logits, lin_pred = scaled_jvp(
            apply_fn, params, w_lin, scale_vec=scale_vec
        )

        # Calculate norm of w_lin
        w_norm = sum((jnp.sum(w**2) for w in tree_leaves(w_lin)))
        P = sum((jnp.reshape(w, -1).shape[0] for w in tree_leaves(w_lin)))

        w_angle = sum((jnp.sum(w) for w in tree_leaves(w_lin))) / (w_norm * jnp.sqrt(P))

        global_metrics = {"w_norm": w_norm, "w_angle": w_angle}

        batch_metrics = _create_loss_and_metrics(
            batch_logits,
            batch_labels,
            lin_pred,
            w_lin,
            w_norm,
            num_classes,
            prior_prec,
            num_train_points,
        )

        # Get either mean or sum aggregations
        agg = get_agg_fn(aggregate)

        batch_metrics = jax.tree_map(lambda x: agg(x, axis=0), batch_metrics)
        loss = batch_metrics["nll"]

        return loss, (batch_metrics, global_metrics)

    return jax.jit(batched_loss_fn)


def create_linear_train_step(model, optimizer, num_classes, num_train_points):
    @jax.jit
    def train_step(state, batch_inputs, batch_labels):
        loss_fn = create_linear_mode_loss(
            model,
            batch_inputs,
            batch_labels,
            num_classes,
            state.prior_prec,
            num_train_points,
            aggregate="sum",
            train=False,
            scale_vec=state.scale_vec,
        )
        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, (train_metrics, global_metrics)), grads = loss_grad_fn(
            state.w_lin, state.params, state.model_state
        )

        grads = jax.lax.pmean(grads, "device")
        train_metrics = jax.lax.pmean(train_metrics, "device")
        global_metrics = jax.lax.pmean(global_metrics, "device")

        return state.apply_gradients(grads=grads), train_metrics, global_metrics

    return train_step


def create_linear_eval_step(model, num_classes, num_train_points):
    @jax.jit
    def eval_step(state, batch_inputs, batch_labels):
        loss_fn = create_linear_mode_loss(
            model,
            batch_inputs,
            batch_labels,
            num_classes,
            state.prior_prec,
            num_train_points,
            aggregate="sum",
            train=False,
            scale_vec=state.scale_vec,
        )

        _, (eval_metrics, _) = loss_fn(state.w_lin, state.params, state.model_state)

        eval_metrics = jax.lax.pmean(eval_metrics, "device")

        return eval_metrics

    return eval_step


def _create_loss_and_metrics(
    batch_logits,
    batch_labels,
    lin_pred,
    w_lin,
    w_norm,
    num_classes,
    prior_prec,
    num_train_points,
):
    # optax.softmax_cross_entropy takes in one-hot labels
    labels_onehot = jax.nn.one_hot(batch_labels, num_classes=num_classes)

    # We divide by (num_classes) here to have parity in gradients from blt

    fit_term = optax.softmax_cross_entropy(lin_pred, labels_onehot) / num_classes
    # Add the regularisation term, divide by (2*train_points*num_classes) to
    # have parity with the gradients from blt
    reg_term = (
        prior_prec
        * w_norm
        * jnp.ones_like(fit_term)
        / (2 * num_train_points * num_classes)
    )

    loss = fit_term + reg_term
    # loss = reg_term
    accuracy = jnp.argmax(lin_pred, -1) == batch_labels
    nn_accuracy = jnp.argmax(batch_logits, -1) == batch_labels

    return {
        "nll": loss,
        "loss_fit": num_classes * fit_term,
        "loss_reg": (2 * num_train_points * num_classes) * reg_term,
        "accuracy": accuracy,
        "nn_accuracy": nn_accuracy,
    }
