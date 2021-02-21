import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

from code.data_processor import DataProcessor
from code.model import get_model
from configs.config import ModelConfig

def decode_pred(preds):
    preds = tf.math.softmax(preds, axis=2)
    preds = tf.math.argmax(preds, axis=2)
    print(preds)

def train(model, train_data, val_data):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=ModelConfig.EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(ModelConfig.MODEL_DIR, "weights-{epoch:02d}-{val_loss:.2f}.hdf5"),
                save_weights_only=True,
                mode='max', save_best_only=False
            ),
            tf.keras.callbacks.TensorBoard(log_dir=ModelConfig.LOG_DIR)
        ]
    )

    return history

def visualize(val_data, idx_to_char, trained_model):

    _, ax = plt.subplots(4, 4, figsize=(10, 10))
    for batch in val_data.take(1):
        images = batch["images"]
        orig_labels = batch["labels"]
        pred_labels = trained_model.predict(images)
        decode_pred(pred_labels)
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")

            label = orig_labels[i].numpy().astype("uint8")
            print(list(label))
            label = "".join([idx_to_char[ch] for ch in label])
            ax[i // 4, i % 4].imshow(img[:, :, 0].T)
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
        plt.show()

if __name__ == "__main__":
    is_train = True
    df = pd.read_csv("dataset/dataset.csv")
    print(df.head(5))
    dp = DataProcessor(df)
    print(dp.classes)
    train_data, val_data = dp.get_batches()

    model = get_model(len(dp.classes) + 1)

    if is_train:
        history = train(model, train_data, val_data)
    else:
        model_name = "weights-33-17.61.hdf5"
        model.load_weights(os.path.join(ModelConfig.MODEL_DIR, model_name))

    trained_model = Model(
        model.get_layer(name="images").input, model.get_layer(name="dense_2").output
    )
    print(trained_model.summary())

    visualize(val_data, dp.idx_to_char, trained_model)






