"""Training a ResNet18 model on CIFAR100."""

import ml_collections

from jaxutils.data.pt_image import METADATA


def get_config():
    """Config for training ResNet18 on CIFAR100."""
    config = ml_collections.ConfigDict()

    config.use_tpu = True
    config.global_seed = 0
    config.model_seed = 0
    config.datasplit_seed = 0

    # Dataset Configs
    config.dataset_type = "tf"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "CIFAR100"
    config.dataset.try_gcs = False
    if config.dataset_type == "tf" and config.dataset.try_gcs:
        config.dataset.data_dir = None
    else:
        config.dataset.data_dir = "/mnt/disks/storage/raw_data"

    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.0
    config.dataset.perform_augmentations = False
    config.dataset.num_workers = 16

    config.dataset.cache = False
    config.dataset.repeat_after_batching = False
    config.dataset.shuffle_train_split = True
    config.dataset.shuffle_eval_split = False
    config.dataset.shuffle_buffer_size = 10_000
    config.dataset.prefetch_size = 4
    config.dataset.prefetch_on_device = None
    config.dataset.drop_remainder = True

    # Add METADATA information from jaxutils
    for key in METADATA:
        config.dataset[key] = METADATA[key][config.dataset.dataset_name]

    # Model Configs
    config.model_name = "ResNet18"
    config.model = ml_collections.ConfigDict()
    config.model.num_classes = config.dataset.num_classes
    config.model.initial_conv = "1x3"

    config.checkpoint_dir = "/mnt/disks/storage/converted_models/cifar100/0"
    config.save_dir = "/mnt/disks/storage/CIFAR100/g1000_fn_8samples"

    ##################### EM Step Configs #####################
    config.num_em_steps = 10

    ###################### Linear Mode Evaluation Configs ####################
    config.linear = ml_collections.ConfigDict()

    # Training Configs
    config.linear.process_batch_size = 2000
    config.linear.eval_process_batch_size = 2000  # 10000/125

    config.linear.n_epochs = 25
    config.linear.perform_eval = True
    config.linear.eval_interval = 5
    config.linear.save_interval = 25

    # Optimizer Configs
    config.linear.optim_name = "sgd"
    config.linear.optim = ml_collections.ConfigDict()
    config.linear.optim.lr = 5e-3
    config.linear.optim.nesterov = True
    config.linear.optim.momentum = 0.9

    config.linear.optim.absolute_clipping = 0.1  # None to avoid clipping
    linear_num_steps = int(
        config.linear.n_epochs
        * config.dataset.num_train
        * (1 - config.dataset.val_percent)
        / config.linear.process_batch_size
    )

    config.linear.lr_schedule_name = "linear_schedule"
    config.linear.lr_schedule = ml_collections.ConfigDict()

    if config.linear.lr_schedule_name == "linear_schedule":
        config.linear.lr_schedule.decay_rate = 1 / 33
        config.linear.lr_schedule.transition_steps = int(
            linear_num_steps * 0.95
        )  # I set this to N steps * 0.75
        config.linear.lr_schedule.end_value = config.linear.optim.lr / 33

    ######################## Sample-then-Optimise Configs #####################
    config.sampling = ml_collections.ConfigDict()

    config.sampling.compute_exact_w_samples = False
    config.sampling.init_samples_at_prior = True
    config.sampling.recompute_ggn_matrix = False

    config.sampling.mackay_update = "function"

    config.sampling.use_g_prior = True
    config.sampling.use_new_objective = True

    if config.sampling.use_g_prior:
        config.prior_prec = 1000.0
    else:
        config.prior_prec = 10000.0

    config.sampling.target_samples_path = (
        "/home/shreyaspadhy_gmail_com/flax_models/CIFAR100/0/target_samples.h5"
    )
    config.sampling.ggn_matrix_path = (
        "/home/shreyaspadhy_gmail_com/flax_models/CIFAR100/0/H.npy"
    )

    config.sampling.num_samples = 6
    config.sampling.H_L_jitter = 1e-6

    # Training Configs
    config.sampling.process_batch_size = 100
    config.sampling.eval_process_batch_size = 200  # 10000/125
    config.sampling.n_epochs = 10
    config.sampling.perform_eval = True
    config.sampling.eval_interval = 5
    config.sampling.save_interval = 20

    # Optimizer Configs
    config.sampling.optim_name = "sgd"
    config.sampling.optim = ml_collections.ConfigDict()
    if config.sampling.use_g_prior:
        config.sampling.optim.lr = 1e-2
    else:
        config.sampling.optim.lr = 1e-3

    config.sampling.optim.nesterov = True
    config.sampling.optim.momentum = 0.9

    config.sampling.update_scheme = "polyak"  # polyak
    config.sampling.update_step_size = None
    config.sampling.lr_schedule_name = "linear_schedule"
    config.sampling.lr_schedule = ml_collections.ConfigDict()

    config.sampling.optim.absolute_clipping = 1  # None to avoid clipping

    sampling_num_steps = int(
        config.sampling.n_epochs
        * config.dataset.num_train
        / config.sampling.process_batch_size
    )

    if config.sampling.lr_schedule_name == "linear_schedule":
        config.sampling.lr_schedule.decay_rate = 1 / 33
        config.sampling.lr_schedule.transition_steps = int(
            sampling_num_steps * 0.95
        )  # I set this to N steps * 0.75
        config.sampling.lr_schedule.end_value = config.sampling.optim.lr / 330

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "linearised-NNs"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/linearised-NNs"
    return config
