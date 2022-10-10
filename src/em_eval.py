import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections.config_flags
import torch
from absl import app, flags
from flax.training import checkpoints

import jaxutils.models as models
import wandb
from jaxutils.data.pt_ood import load_corrupted_dataset, load_rotated_dataset
from jaxutils.data.utils import get_agnostic_iterator
from jaxutils.train.classification import create_eval_step
from jaxutils.train.utils import eval_epoch
from jaxutils.utils import flatten_nested_dict, setup_training, update_config_dict
from src.sampling import SamplingPredictState, create_sampling_prediction_step
from src.sampling_train_utils import eval_sampled_laplace_epoch

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "./experiments/mnist_gcloud_eval.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        ####################### Refresh Config Dicts #########################
        # Update config file with run.config to update hparam_sweep values
        config.unlock()
        config.update_from_flattened_dict(run.config)
        # Add hparams that need to be computed from sweeped hparams to configs
        computed_configs = {}
        update_config_dict(config, run, computed_configs)
        config.lock()
        # Setup training flags and log to Wandb
        setup_training(run)

        ######################## Set up random seeds #########################
        seed = config.get("global_seed", 0)
        torch.manual_seed(seed)

        model_rng = random.PRNGKey(config.model_seed)

        ################ Create and initialise model ##########################
        # Create and initialise model
        model_cls = getattr(models, config.model_name)
        model = model_cls(**config.model.to_dict())

        dummy_init = jnp.expand_dims(jnp.ones(config.dataset.image_shape), 0)
        variables = model.init(model_rng, dummy_init)
        del variables

        ################# Load from checkpoint ################################
        # checkpoint_dir = checkpoint_dir / "w_samples"
        em_steps = jnp.arange(config.num_em_steps)
        for em_step in em_steps:
            checkpoint_dir = Path(config.checkpoint_dir).resolve()
            checkpoint_dir = checkpoint_dir / f"em_{em_step}" / "w_samples"
            print(f"checkpoint_dir: {checkpoint_dir}")
            checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
            restored_state = checkpoints.restore_checkpoint(
                checkpoint_path, target=None
            )

            state = SamplingPredictState.create(
                apply_fn=model.apply,
                tx=None,
                params=restored_state["params"],
                model_state=restored_state["model_state"],
                w_lin=restored_state["w_lin"],
                prior_prec=restored_state["prior_prec"],
                w_samples=restored_state["w_samples"],
                avg_w_samples=restored_state["avg_w_samples"],
                scale_vec=restored_state["scale_vec"],
            )

            if config.eval_dataset == "rotated":
                datasplits = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
                load_dataset_fn = load_rotated_dataset
            elif config.eval_dataset == "corrupted":
                datasplits = [0, 1, 2, 3, 4, 5]
                load_dataset_fn = load_corrupted_dataset

            prediction_method = config.sampling.get("prediction_method", "gibbs")
            if config.method == "sampled_laplace":
                predict_step = create_sampling_prediction_step(model, prediction_method)
            elif config.method == "map":
                predict_step = create_eval_step(
                    model, num_classes=config.dataset.num_classes
                )

            # Create parallel version of the train and eval step
            p_predict_step = jax.pmap(predict_step, "device")
            state = state.replicate()

            for split in datasplits:
                split_loader, split_dataset = load_dataset_fn(
                    config.dataset.dataset_name,
                    split,
                    config.dataset.data_dir,
                    batch_size=config.sampling.eval_process_batch_size,
                    num_workers=config.dataset.num_workers,
                )

                n_dataset = len(split_dataset)
                steps_per_epoch = len(split_loader)

                if config.method == "sampled_laplace":
                    _, predict_metrics = eval_sampled_laplace_epoch(
                        predict_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        eval_log_prefix=f"split_{split}",
                        dataset_type=config.dataset_type,
                        aux_log_dict={"split": split, "em_step": em_step},
                        rng=jax.random.PRNGKey(split),
                    )
                    print(predict_metrics)
                elif config.method == "map":
                    predict_metrics = eval_epoch(
                        eval_step_fn=p_predict_step,
                        data_iterator=get_agnostic_iterator(
                            split_loader, config.dataset_type
                        ),
                        steps_per_epoch=steps_per_epoch,
                        num_points=n_dataset,
                        state=state,
                        wandb_run=run,
                        log_prefix=f"split_{split}",
                        dataset_type=config.dataset_type,
                    )
                    print(predict_metrics)
            if config.method == "map":
                # We don't need to run for N EM steps, can break after 1st loop.
                break


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
