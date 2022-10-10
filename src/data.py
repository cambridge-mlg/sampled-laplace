import h5py
import numpy as np
import tensorflow as tf
import torch


class TargetSamplesDataset(torch.utils.data.Dataset):
    """Augment an image dataset to also return a random sampled y_samples.

    This dataset assumes target_samples have been previously sampled from the
    target distribution y_samples ~ N(0, H_L^{-1}) and saved in an h5py file.
    The dataset interleaves y_samples with the image_dataset, so that it can
    return tuples of (x, y, y_samples).
    """

    def __init__(self, image_dataset, h5_filename, num_samples):
        self.dataset = image_dataset  # len = N_train
        self.filename = h5_filename
        self.num_samples = num_samples
        with h5py.File(self.filename, "r") as f:
            # N x K x O
            self.sampled_targets = np.array(f["target_samples"][:])
            self.sampled_target_labels = np.array(f["targets"][:])
            # If O == 1, append a dimension to make it 3D
            if len(self.sampled_targets.shape) == 2:
                self.sampled_targets = self.sampled_targets[:, :, None]

            # Subselect num_samples samples from all saved.
            if self.num_samples > self.sampled_targets.shape[-1]:
                raise ValueError(
                    "The number of samples requested is greater than the number"
                    "of samples saved in the h5 file."
                )

            self.sampled_targets = self.sampled_targets[:, :, : self.num_samples]

        print(f"sample_targets shape: {self.sampled_targets.shape}")

    def __getitem__(self, i):
        if self.sampled_target_labels[i] != self.dataset[i][1]:
            raise ValueError(
                "The saved sampled_targets have a label mismatch with the "
                "image_dataset. This usually means that the sampled_targets "
                "were sampled from a different shuffle/split of the "
                "image_dataset."
            )
        return (
            self.dataset[i][0],
            self.dataset[i][1],
            np.array(self.sampled_targets[i, ...]),
        )

    def __len__(self):
        return len(self.dataset)


def tf_target_samples_dataset(
    image_dataset, target_samples_path: str, num_samples: int
):
    """Create a dataset from a saved target samples file."""
    with h5py.File(target_samples_path, "r") as f:
        # N x K x O
        sampled_targets = np.array(f["target_samples"][:])
        sampled_target_labels = np.array(f["targets"][:])
        # If O == 1, append a dimension to make it 3D
        if len(sampled_targets.shape) == 2:
            sampled_targets = sampled_targets[:, :, None]

        # Subselect num_samples samples from all saved.
        if num_samples > sampled_targets.shape[-1]:
            raise ValueError(
                "The number of samples requested is greater than the number"
                "of samples saved in the h5 file."
            )

        sampled_targets = sampled_targets[:, :, :num_samples]

        samples_dataset = tf.data.Dataset.from_tensor_slices(
            (sampled_targets, sampled_target_labels)
        )

        dataset = tf.data.Dataset.zip((image_dataset, samples_dataset))

        def preprocessing_fn(tuple_1, tuple_2):
            x, y = tuple_1["image"], tuple_1["label"]
            y_samples, y_check = tuple_2
            y_check = tf.cast(y_check, tf.int64)

            return {
                "image": x,
                "label": y,
                "y_samples": y_samples,
                "label_check": y_check,
            }

        dataset = dataset.map(
            preprocessing_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return dataset
