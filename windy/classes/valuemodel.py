#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides the ValueModel class, for defining a neural network model of the value function."""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers


def reload_model(model_dir, inputshape):
    """Load a model.

    Args:
        model_dir (str):
            path to the model
        inputshape (ndarray):
            shape of the neural network input given by :attr:`otto.classes.sourcetracking.SourceTracking.NN_input_shape`
    """
    model_name = os.path.basename(model_dir)
    weights_path = os.path.abspath(os.path.join(model_dir, model_name))
    config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
    with open(config_path, 'rb') as filehandle:
        config = pickle.load(filehandle)
    if "discount" not in config:
        config["discount"] = 1.0
    if "shaping" not in config:
        config["shaping"] = "0"
    if "conv_layers" not in config:
        config["conv_layers"] = 0
        config["conv_coord"] = None
        config["conv_filters"] = 0
        config["conv_sizes"] = 0
        config["pool_sizes"] = 0
    config = {k.lower(): v for k, v in config.items()}  # for retrocompatibility
    model = ValueModel(**config)
    model.build_graph(input_shape_nobatch=inputshape)
    model.load_weights(weights_path)
    return model


class ValueModel(Model):
    """Neural network model used to predict the value of the belief state
    (i.e. the expected remaining time to find the source).

    Args:
        ndim (int):
            number of space dimensions (1D, 2D, ...) for the search problem
        conv_layers (int):
            number of conv layers
        conv_coord (np.array or None):
            meshgrid of coordinates to add as extra channels, or None for not adding coordinates
        conv_filters (tuple(int)):
            number of filters for each conv layer
        conv_sizes (tuple(int)):
            size of the conv kernel for each conv layer
        pool_sizes (tuple(int)):
            size of the max pool for each conv layer
        fc_layers (int):
            number of hidden layers
        fc_units (int or tuple(int)):
            units per layer
        regularization_factor (float, optional):
            factor for regularization losses (default=0.0)
        loss_function (str, optional):
            either 'mean_absolute_error', 'mean_absolute_percentage_error' or 'mean_squared_error' (default)
        discount (float, optional):
            discount factor (gamma)
        shaping (str, optional):
            reward shaping function, as a string

    Attributes:
        config (dict):
            saves the args in a dictionary, can be used to recreate the model

    """

    def __init__(self,
                 ndim,
                 conv_layers,
                 conv_coord,
                 conv_filters,
                 conv_sizes,
                 pool_sizes,
                 fc_layers,
                 fc_units,
                 regularization_factor=0.0,
                 loss_function='mean_squared_error',
                 discount=1.0,
                 shaping="0",
                 ):
        """Constructor.


        """
        super(ValueModel, self).__init__()
        self.config = {"ndim": ndim,
                       "conv_layers": conv_layers,
                       "conv_coord": conv_coord,
                       "conv_filters": conv_filters,
                       "conv_sizes": conv_sizes,
                       "pool_sizes": pool_sizes,
                       "fc_layers": fc_layers,
                       "fc_units": fc_units,
                       "regularization_factor": regularization_factor,
                       "loss_function": loss_function,
                       "discount": discount,
                       "shaping": shaping,
                       }

        self.ndim = ndim

        if loss_function == 'mean_absolute_error':
            self.loss_function = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_function == 'mean_absolute_percentage_error':
            self.loss_function = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_function == 'mean_squared_error':
            self.loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        else:
            raise Exception("This loss function has not been made available")

        regularizer = regularizers.l2(regularization_factor)

        self.discount = discount
        self.shaping = shaping

        # Convolutional layers
        self.conv_block = None
        if conv_layers > 0:
            if conv_coord is not None:
                self.conv_coord = np.array(conv_coord, dtype=np.float32)
            else:
                self.conv_coord = None
            conv_convpad = 'valid'
            conv_poolpad = 'valid'
            if len(conv_sizes) != conv_layers or len(conv_filters) != conv_layers or len(
                    pool_sizes) != conv_layers:
                raise Exception("Must provide 1 convsize/filter/poolsize per CV layer")
            if conv_convpad not in ('valid', 'same'):
                raise Exception("This padding is not allowed for conv")
            if conv_poolpad not in ('valid', 'same'):
                raise Exception("This padding is not allowed for pool")

            self.conv_block = []
            for i in range(conv_layers):
                conv_ = layers.Conv2D(
                    filters=conv_filters[i],
                    kernel_size=conv_sizes[i],
                    padding=conv_convpad,
                    activation='relu',
                    activity_regularizer=regularizer,
                )
                self.conv_block.append(conv_)
                if pool_sizes[i] > 1:
                    pool_ = layers.MaxPooling2D(
                        pool_size=pool_sizes[i],
                        padding=conv_poolpad,
                    )
                    self.conv_block.append(pool_)

        # flattening
        self.flatten = layers.Flatten()

        # fully connected layers
        self.fc_block = None
        if fc_layers > 0:
            if isinstance(fc_units, int):
                fc_units = tuple([fc_units] * fc_layers)
            if len(fc_units) != fc_layers:
                raise Exception("Must provide nb of units for each dense layer or provide a single int")
            self.fc_block = []
            for i in range(fc_layers):
                dense_ = layers.Dense(
                    units=fc_units[i],
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activity_regularizer=regularizer,
                )
                self.fc_block.append(dense_)

        # last linear layer
        self.densefinal = layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1),
            kernel_constraint=tf.keras.constraints.non_neg(),
            bias_constraint=tf.keras.constraints.non_neg(),
        )

    def call(self, x, training=False, sym_avg=False):
        """Call the value model

        Args:
            x (ndarray or tf.tensor with shape (batch_size, input_shape)):
                array containing a batch of inputs
            training (bool, optional):
                whether this call is done during training (as opposed to evaluation) (default=False)
            sym_avg (bool, optional):
                whether to take the average value of symmetric duplicates (default=False)

        Returns:
            x (tf.tensor with shape (batch_size, 1))
                array containing a batch of values
        """
        shape = x.shape  # (batch_size, input_shape)
        ensemble_sym_avg = False
        if sym_avg and (shape[0] is not None):
            ensemble_sym_avg = True

        # create symmetric duplicates
        if ensemble_sym_avg:
            if self.ndim == 2:
                Nsym = 2
                x = x[tf.newaxis, ...]
                _ = tf.reverse(x, axis=[3])  # symmetry: y -> -y
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            else:
                raise Exception("symmetric duplicates for ndim != 2 is not implemented")

        # convolutions
        if self.conv_block is not None:
            x = x[..., tf.newaxis]  # adding the channel dim
            batchdim_size = x.shape[0]
            if self.conv_coord is not None:  # adding coordinates as channels
                for coord in self.conv_coord:
                    batch_coord = coord[tf.newaxis, ...]
                    if batchdim_size is not None:
                        batch_coord = tf.repeat(batch_coord, batchdim_size, axis=0)
                    batch_coord = batch_coord[..., tf.newaxis]
                    x = tf.concat([x, batch_coord], axis=-1)
            for i in range(len(self.conv_block)):
                x = self.conv_block[i](x, training=training)

        # flatten input
        x = self.flatten(x)

        # fully connected
        if self.fc_block is not None:
            for i in range(len(self.fc_block)):
                x = self.fc_block[i](x, training=training)

        x = self.densefinal(x)

        # reduce the symmetric outputs
        if ensemble_sym_avg:
            x = tf.reshape(x, shape=(Nsym, shape[0], 1))
            x = tf.reduce_mean(x, axis=0)

        return x  # (batch_size, 1)

    def build_graph(self, input_shape_nobatch):
        """Builds the model. Use this function instead of model.build() so that a call to
        model.summary() gives shape information.

        Args:
            input_shape_nobatch (tuple(int)):
                shape of the neural network input given by :attr:`otto.classes.sourcetracking.SourceTracking.NN_input_shape`
        """
        input_shape_nobatch = tuple(input_shape_nobatch)
        input_shape_withbatch = tuple([1] + list(input_shape_nobatch))
        self.build(input_shape_withbatch)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)

    # note: the tf.function decorator prevent using tensor.numpy() for performance reasons, use only tf operations
    @tf.function
    def train_step(self, x, y, augment=False):
        """A training step.

        Args:
            x (tf.tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.tensor with shape=(batch_size, 1)): batch of target values

        Returns:
            loss (tf.tensor with shape=()): total loss
        """

        # Add symmetric duplicates
        if augment:
            shape = x.shape
            if self.ndim == 2:
                Nsym = 2
                x = x[tf.newaxis, ...]
                _ = tf.reverse(x, axis=[3])  # symmetry: y -> -y
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            else:
                raise Exception("augmentation with symmetric duplicates is not implemented for ndim != 2")

            # repeat target
            y = y[tf.newaxis, ...]
            y = tf.repeat(y, Nsym, axis=0)
            y = tf.reshape(y, shape=tuple([Nsym * shape[0]] + [1]))

        # Compute predictions
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            loss_err = self.loss_function(y, y_pred)  # compute loss
            loss_reg = tf.math.reduce_sum(self.losses)  # adding the regularization losses
            loss = tf.math.add(loss_err, loss_reg)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute total loss
        loss = tf.math.reduce_mean(loss)

        return loss

    @tf.function
    def test_step(self, x, y):
        """ A test step.

        Args:
            x (tf.tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.tensor with shape=(batch_size, 1)): batch of target values

        Returns:
            loss (tf.tensor with shape=()): total loss
        """

        # Compute predictions
        y_pred = self(x, training=False)

        # Compute the loss
        loss_err = self.loss_function(y, y_pred)  # compute loss
        loss_reg = tf.math.reduce_sum(self.losses)  # adding the regularization losses
        loss = tf.math.add(loss_err, loss_reg)

        # Compute total loss
        loss = tf.math.reduce_mean(loss)

        return loss

    def save_model(self, model_dir):
        """Save the model to model_dir."""
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_name = os.path.basename(model_dir)
        weights_path = os.path.abspath(os.path.join(model_dir, model_name))
        self.save_weights(weights_path, save_format='h5')
        config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
        with open(config_path, 'wb') as filehandle:
            pickle.dump(self.config, filehandle)

        