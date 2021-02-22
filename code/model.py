import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers

from configs.config import ModelConfig


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss().
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def get_model(num_classes, type=None):
    inp = layers.Input(
        shape=(ModelConfig.IMG_WIDTH, ModelConfig.IMG_HEIGHT, 3), name="inputs", dtype="float32"
    )
    labels = layers.Input(name="labels", shape=(None,), dtype="float32")

    conv_1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1")(inp)
    pool_1 = layers.MaxPool2D((2, 2), name="pool_1")(conv_1)

    conv_2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2")(pool_1)
    pool_2 = layers.MaxPool2D((2, 2), name="pool_2")(conv_2)

    reshape = tf.keras.layers.Reshape(target_shape=(ModelConfig.IMG_WIDTH // 4, ModelConfig.IMG_HEIGHT // 4 * 64), name="reshape")(pool_2)
    dense_1 = tf.keras.layers.Dense(64, activation="relu", name="Dense_1")(reshape)

    bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(dense_1)
    bilstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(bilstm_1)

    dense_2 = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_2")(bilstm_2)

    output = CTCLayer(name="ctc_loss")(labels, dense_2)

    final_model = Model(
        inputs=[inp, labels], outputs=output, name="final_model"
    )


    final_model.compile(optimizer=tf.keras.optimizers.Adam())

    return final_model

def get_tl_model(num_classes):
    labels = tf.keras.layers.Input(name="labels", shape=(None,), dtype="float32")
    base_model = VGG19(weights="imagenet", include_top=False,
                       input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))

    base_model.get_layer("input_1")._name = "inputs"

    # EXP 1- VGG16: block4_pool
    # model = base_model.get_layer('block4_pool').output
    # model = tf.keras.layers.Reshape(target_shape=(14, 14 * 512), name="reshape")(model)

    #Exp 2- VGG16: block4_conv4
    model = base_model.get_layer('block4_conv4').output
    model = tf.keras.layers.Reshape(target_shape=(28, 28 * 512), name="reshape")(model)

    model = tf.keras.layers.Dense(64, activation="relu", name="Dense_1")(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(model)

    model = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_2")(model)

    output = CTCLayer(name="ctc_loss")(labels, model)

    final_model = Model(
        inputs=[base_model.input, labels], outputs=output, name="final_model"
    )

    for layer in base_model.layers:
        layer.trainable = False

    final_model.compile(optimizer=tf.keras.optimizers.Adam())
    print(final_model.summary())

    return final_model
