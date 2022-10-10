from typing import Callable, Iterable, Optional

import jax
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard_prng_key

import wandb
from jaxutils.data.utils import get_agnostic_batch
from jaxutils.train.utils import (
    add_prefix_to_dict_keys,
    aggregated_metrics_dict,
    batchwise_metrics_dict,
)
from src.sampling import SamplingTrainState


def train_and_eval_sampling_epoch(
    train_step_fn: Callable,
    eval_step_fn: Callable,
    data_iterator: Iterable,
    steps_per_epoch: int,
    num_points: int,
    state: SamplingTrainState,
    wandb_run: wandb.run,
    log_prefix: str,
    eval_log_prefix: str,
    dataset_type: str = "tf",
    log_global_metrics: bool = False,
    epoch: int = 0,
    num_epochs: int = 1,
    em_step: Optional[int] = None,
    perform_eval: bool = True,
):
    # Define some iterator values to log with wandb, to make logging much easier
    log_prefix = log_prefix + f"/em_{em_step}" if em_step is not None else log_prefix
    eval_log_prefix = (
        eval_log_prefix + f"/em_{em_step}" if em_step is not None else eval_log_prefix
    )
    em_epoch = em_step * num_epochs + epoch if em_step is not None else epoch

    batch_metrics, eval_batch_metrics = [], []
    for i in range(steps_per_epoch):
        batch = get_agnostic_batch(
            next(data_iterator), dataset_type, tfds_keys=["image", "label", "y_samples"]
        )

        n_devices, B = batch[0].shape[:2]

        if log_global_metrics:
            state, metrics, global_metrics = train_step_fn(state, batch[0], batch[2])
            if perform_eval:
                eval_metrics, eval_global_metrics = eval_step_fn(
                    state, batch[0], batch[2]
                )
        else:
            state, metrics = train_step_fn(state, batch[0], batch[2])
            if perform_eval:
                eval_metrics = eval_step_fn(state, batch[0], batch[1])

        ######################## EVERYTHING BELOW IS FOR W&B LOGGING ##############
        train_step = epoch * steps_per_epoch + i
        em_train_step = em_epoch * steps_per_epoch + i

        # NOTE: train_step outputs values averaged over both batches and
        # samples, therefore we need to compensate when aggregating metrics
        if log_global_metrics:
            global_metrics = unreplicate(global_metrics)
            eval_global_metrics = (
                unreplicate(eval_global_metrics) if perform_eval else {}
            )
            if em_step is not None:
                global_metrics = add_prefix_to_dict_keys(global_metrics, log_prefix)
                if perform_eval:
                    eval_global_metrics = add_prefix_to_dict_keys(
                        eval_global_metrics, eval_log_prefix
                    )
            for k, v in global_metrics.items():
                global_metrics[k] = v.item()
            metric = {
                **global_metrics,
                **eval_global_metrics,
                **{
                    "sampling/train_step": train_step,
                    "sampling/em_train_step": em_train_step,
                },
            }
            wandb_run.log(metric)

        # Rescale metrics by batch_size, to account for averaging
        metrics = unreplicate(metrics)
        metrics = {k: v * B for k, v in metrics.items()}
        batch_metrics.append(metrics)

        if perform_eval:
            eval_metrics = unreplicate(eval_metrics)
            eval_metrics = {k: v * B for k, v in eval_metrics.items()}
            eval_batch_metrics.append(eval_metrics)
            wandb_run.log(
                {
                    **batchwise_metrics_dict(
                        eval_metrics, batch[0].shape[1], f"{eval_log_prefix}/batchwise"
                    ),
                    **{
                        "sampling/train_step": train_step,
                        "sampling/em_train_step": em_train_step,
                    },
                }
            )

        # Further divide by sharded batch size, to get average metrics
        metrics = {
            **batchwise_metrics_dict(
                metrics, batch[0].shape[1], f"{log_prefix}/batchwise"
            ),
            **{
                "sampling/train_step": train_step,
                "sampling/em_train_step": em_train_step,
            },
        }

        wandb_run.log(metrics)

    train_metrics = aggregated_metrics_dict(
        batch_metrics, num_points, log_prefix, n_devices=n_devices
    )
    if perform_eval:
        eval_metrics = aggregated_metrics_dict(
            eval_batch_metrics, num_points, eval_log_prefix, n_devices=n_devices
        )
    else:
        eval_metrics = {}

    wandb_run.log(
        {
            **train_metrics,
            **eval_metrics,
            **{"sampling/train_epoch": epoch, "sampling/em_train_epoch": em_epoch},
        }
    )

    if log_global_metrics:
        return state, train_metrics, eval_metrics, global_metrics
    else:
        return state, train_metrics, eval_metrics


def eval_sampled_laplace_epoch(
    predict_step_fn: Callable,
    data_iterator: Iterable,
    steps_per_epoch: int,
    num_points: int,
    state: SamplingTrainState,
    wandb_run: wandb.run,
    eval_log_prefix: str,
    log_global_metrics: bool = False,
    dataset_type: str = "pytorch",
    aux_log_dict: Optional[dict] = None,
    rng: Optional[jax.random.PRNGKey] = None,
):

    eval_batch_metrics = []
    for i in range(steps_per_epoch):
        batch = get_agnostic_batch(next(data_iterator), dataset_type)

        if rng is not None:
            batch_rng, rng = jax.random.split(rng, 2)
        else:
            batch_rng = None

        n_devices, B = batch[0].shape[:2]
        # NOTE: eval_step outputs values summed over batches_per_device but averaged
        # over num_devices, therefore we need to compensate when aggregating metrics
        eval_metrics = predict_step_fn(
            state, batch[0], batch[1], batch_rng=shard_prng_key(batch_rng)
        )
        eval_metrics = unreplicate(eval_metrics)

        eval_batch_metrics.append(eval_metrics)
        wandb_run.log(
            batchwise_metrics_dict(
                eval_metrics, batch[0].shape[0], f"{eval_log_prefix}/batchwise"
            )
        )
    eval_metrics = aggregated_metrics_dict(
        eval_batch_metrics, num_points, eval_log_prefix, n_devices=n_devices
    )
    logging_dict = {**eval_metrics, **aux_log_dict} if aux_log_dict else eval_metrics
    wandb_run.log(logging_dict)

    return state, eval_metrics
