import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import Levenshtein

from code.data_processor import DataProcessor
from code.model import get_model, get_tl_model
from configs.config import ModelConfig

def sequence_to_vehicle_no(sequence):
    s = ""
    for ch in sequence:
        if len(s) != 0:
            if ch != s[-1]:
                s = s + ch
        else:
            s = s + ch
    # print("STR: ", s)
    s = s.replace("-", "")
    return s



def decode_pred(preds, idx_to_char):
    predictions = []
    preds = tf.math.softmax(preds, axis=2)
    preds = tf.math.argmax(preds, axis=2)
    for pred in preds:
        pred = pred.numpy().astype("uint8")
        pred_str = "".join([idx_to_char[val] for val in pred])
        vehicle_no = sequence_to_vehicle_no(pred_str)
        # print(pred_str, vehicle_no)
        predictions.append(vehicle_no)
    return predictions

def pred_data(trained_model, val_data, data_len, idx_to_char):
    predictions = []
    for batch in val_data.take(data_len//ModelConfig.VAL_BATCH_SIZE):
        images = batch["inputs"]
        orig_labels = batch["labels"]
        pred_labels = trained_model.predict(images)
        pred = decode_pred(pred_labels, idx_to_char)


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

def visualize(val_data, idx_to_char, trained_model, data_len):

    _, ax = plt.subplots(4, 3, figsize=(10, 10))
    cnt = 0
    for batch in val_data.take(data_len//ModelConfig.VAL_BATCH_SIZE):
        images = batch["inputs"]
        orig_labels = batch["labels"]
        pred_labels = trained_model.predict(images)
        predictions = decode_pred(pred_labels, idx_to_char)
        for i in range(ModelConfig.VAL_BATCH_SIZE):
            img = (images[i] * 255).numpy().astype("uint8")

            orig_label = orig_labels[i].numpy().astype("uint8")
            orig_label = "".join([idx_to_char[ch] for ch in orig_label])
            ax[i // 3, i % 3].imshow(img[:, :, 0].T)
            ax[i // 3, i % 3].set_title(f"{orig_label}=> {predictions[i]}, {Levenshtein.distance(orig_label, predictions[i])}")
            ax[i // 3, i % 3].axis("off")
        cnt += 1
        plt.savefig(f"saved_img/pred_{cnt}.png")

def get_accuracy(val_data, idx_to_char, trained_model, data_len):
    correct_wl = 0
    correct_cl = 0
    for batch in val_data.take(data_len//ModelConfig.VAL_BATCH_SIZE):
        images = batch["inputs"]
        orig_labels = batch["labels"]
        pred_labels = trained_model.predict(images)
        predictions = decode_pred(pred_labels, idx_to_char)

        for i in range(ModelConfig.VAL_BATCH_SIZE):
            img = (images[i] * 255).numpy().astype("uint8")

            orig_label = orig_labels[i].numpy().astype("uint8")
            orig_label = "".join([idx_to_char[ch] for ch in orig_label])
            if orig_label == predictions[i]:
                correct_wl += 1
            correct_cl += 7 - Levenshtein.distance(orig_label, predictions[i])
    print(f"The word accuracy is {correct_wl*100/data_len}%")
    print(f"The char accuracy is {correct_cl * 100 / (data_len*7)}%")

if __name__ == "__main__":
    is_train = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            is_train = True
            print(is_train)

    df = pd.read_csv("dataset/dataset.csv")
    print(df.head(5))
    dp = DataProcessor(df)
    print(dp.classes)
    train_data, val_data = dp.get_batches()

    model = get_model(len(dp.classes) + 1)
    if ModelConfig.TRAINING_TYPE == "pretrained":
        model = get_tl_model(len(dp.classes) + 1)

    if is_train:
        history = train(model, train_data, val_data)
    else:
        model_name = "weights-72-2.03.hdf5"
        model.load_weights(os.path.join(ModelConfig.MODEL_DIR, model_name))

    trained_model = Model(
        model.get_layer(name="inputs").input, model.get_layer(name="dense_2").output
    )
    print(trained_model.summary())

    dp.idx_to_char[31] = "-"

    visualize(val_data, dp.idx_to_char, trained_model, len(dp.val_x))
    # print(len(dp.val_x))
    # pred_data(trained_model, val_data, len(dp.val_x), dp.idx_to_char)

    get_accuracy(val_data, dp.idx_to_char, trained_model, len(dp.val_x))








