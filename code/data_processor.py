import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from albumentations import (
    Compose, RandomBrightnessContrast, MotionBlur, MedianBlur, HueSaturationValue,
    Rotate, Blur, ChannelShuffle, OneOf, IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion,
    GridDistortion, IAAPiecewiseAffine, CLAHE, IAASharpen, IAAEmboss
)
from configs.config import ModelConfig

class DataProcessor:
    def __init__(self, df):
        image_paths = np.array([os.path.join(ModelConfig.IMG_DIR, image) for image in df["images"].values])
        labels = np.array(df["labels"].values)
        self.classes = list(set([ch for label in labels for ch in label]))
        self.idx_to_char = {idx + 1: ch for idx, ch in enumerate(self.classes)}
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(image_paths, labels)
        self.transforms = Compose([
            Rotate(limit=10),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3)

        ], p=0.5)

    def aug_fn(self, image):
        data = {"image": image}
        aug_data = self.transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = tf.image.resize(aug_img, size=[ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])
        return aug_img

    def get_encoded_label(self, label):
        label = label.decode("utf-8")
        return np.array([self.char_to_idx[ch] for ch in label])

    def get_data(self, image_path, label):
        print(image_path)
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=3)
        # normalize
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE])

        # Apply augmentation
        image = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)

        # img dim: (height, width, depth) -> (width, height, depth)
        image = tf.transpose(image, perm=[1, 0, 2])

        label = tf.numpy_function(func=self.get_encoded_label, inp=[label], Tout=tf.int64)
        print("done")

        return {"images": image, "labels": label}


    def get_batches(self):
        train_data = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        val_data = tf.data.Dataset.from_tensor_slices((self.val_x, self.val_y))

        train_data = (
            train_data.map(
                self.get_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(ModelConfig.BATCH_SIZE)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        val_data = (
            val_data.map(
                self.get_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
                .batch(ModelConfig.BATCH_SIZE)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return train_data, val_data

if __name__ == "__main__":
    df = pd.read_csv("dataset/dataset.csv")
    print(df.head(5))
    dp = DataProcessor(df)
    print(dp.classes)
    train_data, val_data = dp.get_batches()

    # check if data is augumented properly or not
    _, ax = plt.subplots(4, 4, figsize=(10, 10))
    for batch in train_data.take(1):
        images = batch["images"]
        labels = batch["labels"]
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")

            label = labels[i].numpy().astype("uint8")
            print(list(label))
            label = "".join([dp.idx_to_char[ch] for ch in label])
            ax[i // 4, i % 4].imshow(img[:, :, 0].T)
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
    plt.show()