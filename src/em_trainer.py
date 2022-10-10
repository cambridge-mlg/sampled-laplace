import logging
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections.config_flags
import torch
from absl import app, flags
from flax import linen as nn
from flax.jax_utils import unreplicate
from flax.training import checkpoints
from tqdm import trange

import jaxutils.models as models
import wandb
from jaxutils.data.pt_preprocess import NumpyLoader
from jaxutils.data.tf_image import PYTORCH_TO_TF_NAMES, get_image_dataloader
from jaxutils.data.tf_image import get_image_dataset as get_tf_image_dataset
from jaxutils.data.utils import get_agnostic_iterator
from jaxutils.train.utils import (
    eval_epoch,
    get_lr_and_schedule,
    get_lr_from_opt_state,
    train_epoch,
)
from jaxutils.utils import (
    flatten_nested_dict,
    flatten_params,
    setup_training,
    update_config_dict,
)
from src.data import tf_target_samples_dataset
from src.linear_MAP import (
    LinearTrainState,
    create_linear_eval_step,
    create_linear_train_step,
)
from src.mll import fn_space_Mackay_update, weight_space_Mackay_update
from src.sampling import (
    SamplingTrainState,
    compute_exact_samples,
    create_sampling_eval_step,
    create_sampling_train_step,
    get_sto_samples,
)
from src.sampling_train_utils import train_and_eval_sampling_epoch

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "./experiments/cifar100_gcloud_em.py",
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
        global_rng = random.PRNGKey(seed)
        model_rng = random.PRNGKey(config.model_seed)
        datasplit_rng = config.datasplit_seed

        ################### Load datasets ######################################
        # NOTE: On a single TPU v3 pod, we have 1 process, 8 devices
        linear_batch_size = config.linear.process_batch_size * jax.process_count()
        linear_eval_batch_size = (
            config.linear.eval_process_batch_size * jax.process_count()
        )
        sampling_batch_size = config.sampling.process_batch_size * jax.process_count()

        # Load dataset without augmentations in train_dataset
        datasets, num_examples = get_tf_image_dataset(
            dataset_name=PYTORCH_TO_TF_NAMES[config.dataset.dataset_name],
            process_batch_size=config.linear.process_batch_size,
            eval_process_batch_size=config.linear.eval_process_batch_size,
            shuffle_train_split=config.dataset.shuffle_train_split,
            shuffle_eval_split=config.dataset.shuffle_eval_split,
            drop_remainder=config.dataset.drop_remainder,
            data_dir=config.dataset.data_dir,
            try_gcs=config.dataset.try_gcs,
            val_percent=config.dataset.val_percent,
            datasplit_rng=random.PRNGKey(datasplit_rng),
        )

        train_dataset, val_dataset, test_dataset = datasets
        shuffle_rng, global_rng = random.split(global_rng, 2)
        shuffle_rngs = random.split(shuffle_rng, 3)

        # Create Dataloaders
        map_train_loader = get_image_dataloader(
            train_dataset,
            config.dataset.dataset_name,
            process_batch_size=config.linear.process_batch_size,
            num_epochs=config.num_em_steps * config.linear.n_epochs,
            shuffle=config.dataset.shuffle_train_split,
            shuffle_buffer_size=config.dataset.shuffle_buffer_size,
            rng=shuffle_rngs[0],
            cache=config.dataset.cache,
            repeat_after_batching=config.dataset.repeat_after_batching,
            drop_remainder=config.dataset.drop_remainder,
            prefetch_size=config.dataset.prefetch_size,
            prefetch_on_device=config.dataset.prefetch_on_device,
        )
        map_test_loader = get_image_dataloader(
            test_dataset,
            config.dataset.dataset_name,
            process_batch_size=config.linear.eval_process_batch_size,
            num_epochs=config.num_em_steps * config.linear.n_epochs,
            shuffle=config.dataset.shuffle_eval_split,
            shuffle_buffer_size=config.dataset.shuffle_buffer_size,
            rng=shuffle_rngs[1],
            cache=config.dataset.cache,
            repeat_after_batching=config.dataset.repeat_after_batching,
            drop_remainder=config.dataset.drop_remainder,
            prefetch_size=config.dataset.prefetch_size,
            prefetch_on_device=config.dataset.prefetch_on_device,
        )
        if val_dataset is not None:
            map_val_loader = get_image_dataloader(
                val_dataset,
                config.dataset.dataset_name,
                process_batch_size=config.linear.eval_process_batch_size,
                num_epochs=config.num_em_steps * config.linear.n_epochs,
                shuffle=config.dataset.shuffle_eval_split,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                rng=shuffle_rngs[2],
                cache=config.dataset.cache,
                repeat_after_batching=config.dataset.repeat_after_batching,
                drop_remainder=config.dataset.drop_remainder,
                prefetch_size=config.dataset.prefetch_size,
                prefetch_on_device=config.dataset.prefetch_on_device,
            )
        else:
            map_val_loader = None
        unshuffled_train_loader = get_image_dataloader(
            train_dataset,
            config.dataset.dataset_name,
            process_batch_size=config.sampling.process_batch_size,
            num_epochs=3
            + config.num_em_steps,  # 3 epochs extra for get_sto_samples, compute_exact_samples, fn_space_Mackay_update
            shuffle=False,
            rng=jax.random.PRNGKey(3),
            cache=config.dataset.cache,
            repeat_after_batching=config.dataset.repeat_after_batching,
            drop_remainder=config.dataset.drop_remainder,
            prefetch_size=config.dataset.prefetch_size,
            prefetch_on_device=config.dataset.prefetch_on_device,
            perform_augmentations=False,
        )

        train_config = {
            "n_train": num_examples[0],
            "n_val": num_examples[1],
            "n_test": num_examples[2],
            "linear.batch_size": linear_batch_size,
            "linear.eval_batch_size": linear_eval_batch_size,
            "train_steps_per_epoch": num_examples[0] // linear_batch_size,
            "sampling_steps_per_epoch": num_examples[0] // sampling_batch_size,
            "val_steps_per_epoch": num_examples[1] // linear_eval_batch_size
            if num_examples[1] is not None
            else None,
            "test_steps_per_epoch": num_examples[2] // linear_eval_batch_size,
        }

        # Add all training dataset hparams back to config_dicts
        update_config_dict(config, run, train_config)

        ################ Create and initialise model ##########################
        # Create and initialise model
        model_cls = getattr(models, config.model_name)
        model = model_cls(**config.model.to_dict())

        dummy_init = jnp.expand_dims(jnp.ones(config.dataset.image_shape), 0)
        _ = model.init(model_rng, dummy_init)

        ################# Load from checkpoint ################################
        checkpoint_dir = Path(config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_dir = Path(config.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"checkpoint_dir: {checkpoint_dir}")
        checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
        restored_state = checkpoints.restore_checkpoint(checkpoint_path, target=None)

        ################## Setup EM Step ######################################
        λ = config.prior_prec
        if config.sampling.compute_exact_w_samples:
            exact_λ, exact_sam_λ = config.prior_prec, config.prior_prec
            run.log(
                {
                    "em/exact_λ": exact_λ,
                    "em/exact_sto_mackay_λ": exact_sam_λ,
                    "em/sto_mackay_λ": λ,
                    "em_step": 0,
                }
            )
        else:
            run.log({"em/sto_mackay_λ": λ, "em_step": 0})

        ########################## Set up for w_samples #######################
        # Get Optimizer and Learning Rate Schedule, if any
        opt_w_lin = get_lr_and_schedule(
            config.linear.optim_name,
            config.linear.optim,
            config.linear.get("lr_schedule_name", None),
            config.linear.get("lr_schedule", None),
        )
        opt_w_samples = get_lr_and_schedule(
            config.sampling.optim_name,
            config.sampling.optim,
            config.sampling.get("lr_schedule_name", None),
            config.sampling.get("lr_schedule", None),
        )

        target_samples_path = Path(config.sampling.target_samples_path).resolve()
        target_samples_path.parent.mkdir(parents=True, exist_ok=True)

        # Get rng for sampling
        samples_rng, global_rng = random.split(global_rng)

        w0s_prior, w0s_data, inv_scale_vec = get_sto_samples(
            model=model,
            params=restored_state["params"],
            model_state=restored_state["model_state"],
            data_iterator=get_agnostic_iterator(
                unshuffled_train_loader, config.dataset_type
            ),
            steps_per_epoch=config.sampling_steps_per_epoch,
            prior_prec=λ,
            num_samples=config.sampling.num_samples,
            rng=samples_rng,
            num_train_points=config.n_train,
            n_out=config.dataset.num_classes,
            H_L_jitter=config.sampling.H_L_jitter,
            loss="crossentropy",
            target_samples_path=target_samples_path,
            dataset_type=config.dataset_type,
        )

        ############ Calculate g-prior scale vector ############################
        # Clip inv_scale_vec to 1e-6 to prevent numerical issues on inverting.
        inv_scale_vec = jax.tree_map(lambda x: jnp.clip(x, a_min=1e-6), inv_scale_vec)

        # Define w0 for the STO objective
        if config.sampling.use_g_prior:
            # Define s^-1 = 1 / sqrt(\sum J H J^T)
            scale_vec = jax.tree_map(
                lambda w: jnp.sqrt(config.sampling.num_samples / w), inv_scale_vec
            )
            run.summary["scale_vec"] = flatten_params(scale_vec)
        else:
            scale_vec = None
        del inv_scale_vec

        if scale_vec:
            w0s = jax.tree_map(
                lambda x, y, s: x + s[None, ...] * y, w0s_prior, w0s_data, scale_vec
            )
        else:
            w0s = jax.tree_map(lambda x, y: x + y, w0s_prior, w0s_data)
        run.summary["w0s_prior"] = jax.vmap(flatten_params)(w0s_prior)
        run.summary["w0s_data"] = jax.vmap(flatten_params)(w0s_data)

        ################## DEFINE DATASET FOR LOADING y_samples ###############
        # Define Sampling Dataloaders now that y_samples have been generated
        sampling_batch_size = config.sampling.process_batch_size * jax.process_count()
        sampling_eval_batch_size = (
            config.sampling.eval_process_batch_size * jax.process_count()
        )

        samples_dataset = tf_target_samples_dataset(
            train_dataset,
            config.sampling.target_samples_path,
            config.sampling.num_samples,
        )
        samples_loader = get_image_dataloader(
            samples_dataset,
            config.dataset.dataset_name,
            process_batch_size=config.sampling.process_batch_size,
            num_epochs=3 * config.num_em_steps * config.sampling.n_epochs + 1,
            shuffle=True,
            rng=jax.random.PRNGKey(3),
            cache=config.dataset.cache,
            repeat_after_batching=config.dataset.repeat_after_batching,
            drop_remainder=config.dataset.drop_remainder,
            prefetch_size=config.dataset.prefetch_size,
            prefetch_on_device=config.dataset.prefetch_on_device,
        )
        train_config = {
            "sampling.batch_size": sampling_batch_size,
            "sampling.eval_batch_size": sampling_eval_batch_size,
            "sampling_steps_per_epoch": num_examples[0] // sampling_batch_size,
        }

        # Add all training dataset hparams back to config_dicts
        update_config_dict(config, run, train_config)

        # Initialise w_lin as zeroes
        w_lin = jax.tree_map(lambda x: jnp.zeros_like(x), restored_state["params"])
        # Initialise samples as zeroes
        w_samples = jax.tree_map(lambda x: jnp.zeros_like(x), w0s)  # (K, PyTree)

        if config.sampling.update_step_size is not None:  # Polyak Average
            avg_w_samples = jax.tree_map(lambda x: jnp.zeros_like(x), w0s)
        else:
            avg_w_samples = None

        if config.sampling.compute_exact_w_samples:
            exact_w_samples, H_plus_λ_I = compute_exact_samples(
                model=model,
                params=restored_state["params"],
                model_state=restored_state["model_state"],
                data_iterator=get_agnostic_iterator(
                    unshuffled_train_loader, config.dataset_type
                ),
                samples_data_iterator=get_agnostic_iterator(
                    samples_loader, config.dataset_type
                ),
                prior_prec=λ,
                steps_per_epoch=config.sampling_steps_per_epoch,
                num_samples=config.sampling.num_samples,
                w_0s_prior=w0s_prior,
                w_0s=w0s,
                n_out=config.dataset.num_classes,
                H_L_jitter=config.sampling.H_L_jitter,
                dataset_type=config.dataset_type,
                recompute_GGN_matrix=True,
                ggn_matrix_path=config.sampling.ggn_matrix_path,
                scale_vec=scale_vec,
            )
            H_plus_exact_λ_I = H_plus_λ_I.copy()
        else:
            exact_w_samples = None

        if config.sampling.init_samples_at_prior:
            init_w_samples = w0s_prior
        else:
            # Initialise at zeroes.
            init_w_samples = jax.tree_map(lambda w: jnp.zeros_like(w), w0s_prior)

        ######################### PERFORM EM STEPS #############################
        for em_step in range(config.num_em_steps):

            ############# Perform Linear Mode Evaluation #######################
            # Create Linear Train State
            w_lin_state = LinearTrainState.create(
                apply_fn=model.apply,
                params=restored_state["params"],
                tx=opt_w_lin,
                model_state=restored_state["model_state"],
                w_lin=w_lin,  # Init zero at EM_step=0, otherwise at previous.
                prior_prec=λ,  # this will change during EM
                scale_vec=scale_vec,  # If g_prior, this will be not None.
            )

            w_lin_dir = save_dir / f"em_{em_step}" / "w_lin"
            w_lin_dir.mkdir(parents=True, exist_ok=True)
            w_lin_state = optimise_linear_MAP(
                model=model,
                state=w_lin_state,
                train_loader=map_train_loader,
                val_loader=map_val_loader,
                test_loader=map_test_loader,
                checkpoint_dir=w_lin_dir,
                run=run,
                config=config,
                em_step=em_step,
            )

            ######### Perform STO for posterior samples ########################
            # Decide between using old objective L or new objective L' here.
            if config.sampling.use_new_objective:
                w0_samples = w0s
            else:
                w0_samples = w0s_prior

            w_samples_state = SamplingTrainState.create(
                apply_fn=model.apply,
                tx=opt_w_samples,
                params=restored_state["params"],
                model_state=restored_state["model_state"],
                w_lin=w_lin_state.w_lin,  # Use MAP from current EM step.
                prior_prec=λ,  # this will update each step
                w0_samples=w0_samples,
                w_samples=init_w_samples
                if em_step == 0
                else w_samples,  # init at 0, reinit at previous optima.
                avg_w_samples=avg_w_samples,
                polyak_step_size=config.sampling.update_step_size,
                exact_w_samples=exact_w_samples,  # this should update as well
                scale_vec=scale_vec,
            )

            w_samples_dir = save_dir / f"em_{em_step}"
            w_samples_dir.mkdir(parents=True, exist_ok=True)
            w_samples_state = optimise_samples(
                model=model,
                state=w_samples_state,
                samples_loader=samples_loader,
                checkpoint_dir=w_samples_dir,
                run=run,
                config=config,
                em_step=em_step,
            )

            if config.sampling.update_step_size is not None:
                # Use the polyak average for subsequent calculations.
                w_samples = w_samples_state.avg_w_samples
            else:
                # Use the online samples for subsequent calculations.
                w_samples = w_samples_state.w_samples

            ############# Update Prior Precision using Mackay Update ###########
            # Calculate Mackay update using K samples
            if config.sampling.get("mackay_update", "weight") == "weight":
                # Use weight-space MacKay update for λ using K samples
                diag_cov = jnp.mean(
                    jnp.power(jax.vmap(flatten_params)(w_samples), 2), axis=0
                )
                new_λ, sto_eff_dim, D, sto_mar_var = weight_space_Mackay_update(
                    λ, diag_cov, w_samples_state.w_lin
                )
            else:
                # Do one pass of data to calculate Function Space update.
                new_λ, sto_eff_dim, D = fn_space_Mackay_update(
                    model,
                    w_samples_state.params,
                    w_samples_state.model_state,
                    data_iterator=get_agnostic_iterator(
                        samples_loader, config.dataset_type
                    ),
                    steps_per_epoch=config.sampling_steps_per_epoch,
                    init_λ=λ,
                    w_lin=w_samples_state.w_lin,
                    w_samples=w_samples,
                    H_L_jitter=config.sampling.H_L_jitter,
                    scale_vec=w_samples_state.scale_vec,
                    dataset_type=config.dataset_type,
                )
                # Set the logging value of mar_var = 0
                sto_mar_var = 0

            if config.sampling.compute_exact_w_samples:
                # TODO: Do we leave these ablations here @jantoran?
                # Calculate weight-space λ using the exact samples.
                diag_cov = jnp.mean(jnp.power(exact_w_samples, 2), axis=0)
                (
                    exact_sam_new_λ,
                    exact_sam_eff_dim,
                    _,
                    exact_sto_mar_var,
                ) = weight_space_Mackay_update(
                    λ,
                    diag_cov,
                    w_samples_state.w_lin,
                )

                # Calculate exact λ using the GGN matrix.
                diag_cov = jnp.diag(
                    jax.jit(jnp.linalg.inv, backend="cpu")(H_plus_exact_λ_I)
                )
                (
                    exact_new_λ,
                    exact_eff_dim,
                    _,
                    exact_mar_var,
                ) = weight_space_Mackay_update(
                    exact_λ,
                    diag_cov,
                    w_samples_state.w_lin,
                )
                run.log(
                    {
                        "em/sto_mackay_λ": new_λ,
                        "em/exact_λ": exact_new_λ,
                        "em/exact_sto_mackay_λ": exact_sam_new_λ,
                        "em/sto_eff_dim": sto_eff_dim,
                        "em/exact_eff_dim": exact_eff_dim,
                        "em/exact_sto_eff_dim": exact_sam_eff_dim,
                        "em/D": D,
                        "em/exact_sto_mar_var": exact_sto_mar_var,
                        "em/exact_mar_var": exact_mar_var,
                        "em/sto_mar_var": sto_mar_var,
                        "em_step": em_step + 1,
                    }
                )
            else:
                run.log(
                    {
                        "em/sto_eff_dim": sto_eff_dim,
                        "em/sto_mackay_λ": new_λ,
                        "em/D": D,
                        "em/sto_mar_var": sto_mar_var,
                        "em_step": em_step + 1,
                    }
                )

            # Update the w0 samples by multiplying by old λ, dividing by new λ
            w0s_data = jax.tree_map(lambda w: λ * w / new_λ, w0s_data)
            w0s_prior = jax.tree_map(lambda w: λ**0.5 * w / new_λ**0.5, w0s_prior)
            if scale_vec is not None:
                new_w0s = jax.tree_map(
                    lambda x, y, s: x + s[None, ...] * y,
                    w0s_prior,
                    w0s_data,
                    scale_vec,
                )
            else:
                new_w0s = jax.tree_map(lambda x, y: x + y, w0s_prior, w0s_data)

            if config.sampling.compute_exact_w_samples:
                # TODO: These do EM with Exact Mackay, but the w_lin is still with
                # TODO: STO lambda, do we need these here?
                exact_w_samples, H_plus_λ_I = compute_exact_samples(
                    model,
                    restored_state["params"],
                    restored_state["model_state"],
                    get_agnostic_iterator(unshuffled_train_loader, config.dataset_type),
                    get_agnostic_iterator(samples_loader, config.dataset_type),
                    prior_prec=new_λ,
                    steps_per_epoch=config.sampling_steps_per_epoch,
                    num_samples=config.sampling.num_samples,
                    w_0s_prior=w0s_prior,
                    w_0s=new_w0s,
                    n_out=config.dataset.num_classes,
                    H_L_jitter=config.sampling.H_L_jitter,
                    dataset_type=config.dataset_type,
                    recompute_GGN_matrix=False,  # GGN matrix does not change across EM steps
                    ggn_matrix_path=config.sampling.ggn_matrix_path,
                    scale_vec=scale_vec,
                )
                _, H_plus_exact_λ_I = compute_exact_samples(
                    model,
                    restored_state["params"],
                    restored_state["model_state"],
                    get_agnostic_iterator(unshuffled_train_loader, config.dataset_type),
                    get_agnostic_iterator(samples_loader, config.dataset_type),
                    prior_prec=exact_new_λ,
                    steps_per_epoch=config.sampling_steps_per_epoch,
                    num_samples=config.sampling.num_samples,
                    w_0s_prior=w0s_prior,
                    w_0s=new_w0s,
                    n_out=config.dataset.num_classes,
                    H_L_jitter=config.sampling.H_L_jitter,
                    dataset_type=config.dataset_type,
                    recompute_GGN_matrix=False,  # GGN matrix does not change across EM steps
                    ggn_matrix_path=config.sampling.ggn_matrix_path,
                    scale_vec=scale_vec,
                )

            # Reinitialise the next step with the optima of the previous step.
            if config.sampling.update_step_size is not None:
                w_samples = w_samples_state.avg_w_samples
                w_lin = w_samples_state.w_lin
            else:
                w_samples = w_samples_state.w_samples
                w_lin = w_samples_state.w_lin

            # Update λ and w0s
            λ = new_λ
            if config.sampling.compute_exact_w_samples:
                exact_λ = exact_new_λ
                exact_sam_λ = exact_sam_new_λ

            w0s = new_w0s

            del new_w0s, w_lin_state, w_samples_state


def optimise_linear_MAP(
    model: nn.Module,
    state: LinearTrainState,
    train_loader: NumpyLoader,
    val_loader: Optional[NumpyLoader],
    test_loader: NumpyLoader,
    checkpoint_dir: str,
    run: wandb.run,
    config: ml_collections.ConfigDict,
    em_step: int,
):
    """Calculate w_lin for a given prior_prec."""
    # Create training and evaluation functions
    train_step = create_linear_train_step(
        model, state.tx, config.dataset.num_classes, config.n_train
    )

    eval_step = create_linear_eval_step(
        model, config.dataset.num_classes, config.n_train
    )

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "device")
    p_eval_step = jax.pmap(eval_step, "device")

    # Replicate the train state on each device
    state = state.replicate()

    # Perform Training
    val_losses = []
    epochs = trange(config.linear.n_epochs)
    for epoch in epochs:
        state, train_metrics, global_metrics = train_epoch(
            train_step_fn=p_train_step,
            data_iterator=get_agnostic_iterator(train_loader, config.dataset_type),
            steps_per_epoch=config.train_steps_per_epoch,
            num_points=config.n_train,
            state=state,
            wandb_run=run,
            log_prefix="linear/train",
            dataset_type=config.dataset_type,
            log_global_metrics=True,
            epoch=epoch,
            num_epochs=config.linear.n_epochs,
            em_step=em_step,
        )

        epochs.set_postfix({**train_metrics, **global_metrics})

        # Optionally evaluate on val dataset
        if val_loader is not None:
            val_metrics = eval_epoch(
                eval_step_fn=p_eval_step,
                data_iterator=get_agnostic_iterator(val_loader, config.dataset_type),
                steps_per_epoch=config.val_steps_per_epoch,
                num_points=config.n_val,
                state=state,
                wandb_run=run,
                log_prefix="linear/val",
                dataset_type=config.dataset_type,
            )

            val_losses.append(val_metrics["linear/val/nll"])

            # Save best validation loss/epoch over training
            if val_metrics["linear/val/nll"] <= min(val_losses):
                run.summary["best_val_loss"] = val_metrics["linear/val/nll"]
                run.summary["best_epoch"] = epoch

                checkpoints.save_checkpoint(
                    checkpoint_dir / "w_lin" / "best",
                    state,
                    epoch + 1,
                    keep=1,
                    overwrite=True,
                )

        # Now eval on test dataset every few intervals if perform_eval=True
        if (
            epoch + 1
        ) % config.linear.eval_interval == 0 and config.linear.perform_eval:
            _ = eval_epoch(
                eval_step_fn=p_eval_step,
                data_iterator=get_agnostic_iterator(test_loader, config.dataset_type),
                steps_per_epoch=config.test_steps_per_epoch,
                num_points=config.n_test,
                state=state,
                wandb_run=run,
                log_prefix="test",
                dataset_type=config.dataset_type,
            )

        # Find the part of opt_state that contains InjectHyperparamsState
        lr_to_log = get_lr_from_opt_state(unreplicate(state).opt_state)

        run.log({"lr/linear": lr_to_log})
        # Save Model Checkpoints
        if (epoch + 1) % config.linear.save_interval == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir,
                unreplicate(state),
                epoch + 1,
                keep=5,
                overwrite=True,
            )

    return unreplicate(state)


def optimise_samples(
    model: nn.Module,
    state: LinearTrainState,
    samples_loader: NumpyLoader,
    checkpoint_dir: str,
    run: wandb.run,
    config: ml_collections.ConfigDict,
    em_step: int,
):

    train_step = create_sampling_train_step(
        model,
        state.tx,
        config.dataset.num_classes,
        config.sampling.num_samples,
        config.n_train,
        config.sampling.H_L_jitter,
        config.sampling.use_new_objective,
    )
    eval_step = create_sampling_eval_step(
        model,
        config.dataset.num_classes,
        config.sampling.num_samples,
        config.n_train,
        config.sampling.H_L_jitter,
        config.sampling.use_new_objective,
    )

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "device")
    p_eval_step = jax.pmap(eval_step, "device")

    # Replicate the train state on each device
    state = state.replicate()

    # Perform Training
    epochs = trange(config.sampling.n_epochs)
    for epoch in epochs:
        (
            state,
            train_metrics,
            eval_metrics,
            global_metrics,
        ) = train_and_eval_sampling_epoch(
            p_train_step,
            p_eval_step,
            data_iterator=get_agnostic_iterator(samples_loader, config.dataset_type),
            steps_per_epoch=config.sampling_steps_per_epoch,
            num_points=config.n_train,
            state=state,
            wandb_run=run,
            log_prefix="sampling/train",
            eval_log_prefix="sampling/avg",
            dataset_type=config.dataset_type,
            log_global_metrics=True,
            epoch=epoch,
            num_epochs=config.sampling.n_epochs,
            em_step=em_step,
            perform_eval=config.sampling.update_step_size is not None,
        )

        epochs.set_postfix({**train_metrics, **global_metrics})

        # Find the part of opt_state that contains InjectHyperparamsState
        lr_to_log = get_lr_from_opt_state(unreplicate(state).opt_state)
        run.log({"lr/sampling": lr_to_log})

    # Save Model Checkpoints
    checkpoints.save_checkpoint(
        checkpoint_dir / "w_samples",
        unreplicate(state),
        epoch + 1,
        keep=1,
        overwrite=True,
    )

    return unreplicate(state)


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
