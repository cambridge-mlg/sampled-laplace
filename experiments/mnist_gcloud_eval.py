"""Training a LeNetSmall model on MNIST."""

import ml_collections

from jaxutils.data.pt_image import METADATA


def get_config():
    """Config for training LeNetSmall on MNIST."""
    config = ml_collections.ConfigDict()

    config.use_tpu = True

    config.global_seed = 0
    config.model_seed = 0
    config.datasplit_seed = 0

    config.eval_dataset = "rotated"  # "corrupted"
    # Dataset Configs
    config.dataset_type = "pytorch"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "MNIST"
    config.dataset.try_gcs = True
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
    config.model = ml_collections.ConfigDict()
    config.model.n_out = config.dataset.num_classes
    # config.checkpoint_dir = "/scratch3/gosset/sp2058/flax_models/MNIST"
    # config.checkpoint_dir += f"{config.model_seed}/best"
    # config.checkpoint_dir = "/home/shreyaspadhy_gmail_com/flax_models/torch/MNIST/"
    # config.checkpoint_dir = "/home/shreyaspadhy_gmail_com/flax_models/MNIST/0"
    config.checkpoint_dir = "/home/shreyaspadhy_gmail_com/MNIST/non_gprior"
    # config.save_dir = "/home/shreyaspadhy_gmail_com/MNIST/not_gprior"
    ###################### Decide on what optimisations to use ######################
    config.perform_w_lin_optimization = False
    config.perform_STO_optimization = True

    ##################### EM Step Configs #####################
    config.prior_prec = 1.0
    config.num_em_steps = 10

    ###################### Linear Mode Evaluation Configs ####################
    config.linear = ml_collections.ConfigDict()
    # Training Configs
    config.linear.process_batch_size = 2000
    config.linear.eval_process_batch_size = 2000  # 10000/125

    config.linear.n_epochs = 15
    config.linear.perform_eval = True
    config.linear.eval_interval = 5
    config.linear.save_interval = 10

    # Optimizer Configs
    config.linear.optim_name = "adam"
    config.linear.optim = ml_collections.ConfigDict()
    config.linear.optim.lr = 0.1
    config.linear.optim.nesterov = False
    config.linear.optim.momentum = 0.9

    config.linear.lr_schedule_name = "exponential_decay"
    config.linear.lr_schedule = ml_collections.ConfigDict()
    config.linear.lr_schedule.decay_rate = 0.995
    config.linear.lr_schedule.transition_steps = 1

    ######################## Sample-then-Optimise Configs #####################
    config.sampling = ml_collections.ConfigDict()
    config.sampling.prediction_method = "dyadic"
    config.sampling.compute_exact_w_samples = True
    config.sampling.init_samples_at_prior = True
    config.sampling.recompute_sto_samples = True
    config.sampling.recompute_ggn_matrix = False

    config.sampling.use_g_prior = True
    config.sampling.use_zero_target_samples = True

    config.sampling.reinit_params = "diag_hessian"

    config.sampling.prior_samples_path = (
        "/home/shreyaspadhy_gmail_com/flax_models/MNIST/0/prior_samples.pkl"
    )
    config.sampling.target_samples_path = (
        "/home/shreyaspadhy_gmail_com/flax_models/MNIST/0/target_samples.h5"
    )
    config.sampling.exact_samples_path = "/scratch3/gosset/sp2058/repos/bayesian-lottery-tickets/saves/classification/MNIST/LeNetSmall/exact_samples.npy"
    config.sampling.ggn_matrix_path = (
        "/home/shreyaspadhy_gmail_com/flax_models/MNIST/0/H.npy"
    )
    # config.sampling.exact_samples_path = "/scratch3/gosset/sp2058/repos/bayesian-lottery-tickets/saves/classification/MNIST/LeNetSmall/w_samples.npy"

    config.sampling.w_0s_path = "/scratch3/gosset/sp2058/debugging/jax/unflat_w_0s.pkl"
    config.sampling.num_samples = 8

    config.sampling.H_L_jitter = 1e-6

    # Training Configs
    config.sampling.process_batch_size = 2000
    config.sampling.eval_process_batch_size = 2000  # 10000/125
    config.sampling.n_epochs = 30
    config.sampling.perform_eval = True
    config.sampling.eval_interval = 5
    config.sampling.save_interval = 10

    # Optimizer Configs
    config.sampling.optim_name = "sgd"
    config.sampling.optim = ml_collections.ConfigDict()
    config.sampling.optim.lr = 250.0
    config.sampling.optim.nesterov = True
    config.sampling.optim.momentum = 0.9

    config.sampling.update_scheme = "polyak"  # polyak
    config.sampling.update_step_size = 0.01
    config.sampling.lr_schedule_name = None  # "exponential_decay"
    config.sampling.lr_schedule = ml_collections.ConfigDict()
    config.sampling.lr_schedule.decay_rate = 0.997
    config.sampling.lr_schedule.transition_steps = 1

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.load_model = False
    config.wandb.project = "sampled-laplace"
    config.wandb.entity = "cbl-mlg"
    config.wandb.model_artifact_name = "lenetsmall_mnist"
    config.wandb.artifact_name = "mnist_linear"
    config.wandb.log_params = False
    config.wandb.params_log_interval = -1
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/linearised-NNs"

    return config
