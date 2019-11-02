from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import tensorflow as tf
import numpy as np


def gelu(x):
    return 0.5 * x * (1 + tf.nn.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def layerNorm(inputs, dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):
        scale = tf.get_variable("scale", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.ones_initializer())

        shift = tf.get_variable("shift", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

    mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)

    epsilon = 1e-9

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)), (inputs - mean)) + shift

    return LN


def sparse_init(shape, l=1, sparsity=0.5, d=2):
    l = 1/l
    if d >= 2:
        a = -np.sqrt(6*l/(shape[-1]+shape[-2]))
        b = +np.sqrt(6*l/(shape[-1]+shape[-2]))
    else:
        a = -np.sqrt(3*l/(shape[-1]))
        b = +np.sqrt(3*l/(shape[-1]))
    dense_values = np.random.uniform(a, b, shape)
    sparse_mask = np.random.binomial(1, 1.0-sparsity, shape)

    values = np.asarray(dense_values*sparse_mask, np.float32)

    return values


def k_winner(logits, duty, dim, k_rate, train):
    return tf.cond(train,
                   lambda: k_winner_train(logits, duty, dim, k_rate),
                   lambda: k_winner_eval(logits, duty, dim, k_rate))


def k_winner_eval(logits, duty, dim, k=0.3):
    k = tf.reshape(k, [1])
    k = tf.cast(tf.round(k*dim), tf.int32)

    threshold, _ = tf.math.top_k(logits,
                                 k=k[0],
                                 sorted=True)
    threshold = threshold[:, -1]
    threshold = tf.reshape(threshold, [-1, 1])

    logic = logits-threshold
    binary_mask = tf.where(tf.less_equal(logic, 0.0),
                           x=tf.zeros_like(logits, tf.float32),
                           y=tf.ones_like(logits, tf.float32))

    out = logits*binary_mask
    return out, duty


def k_winner_train(logits, duty, dim, k=0.3, B=1.5):
    k = tf.reshape(k, [1])
    k = tf.cast(tf.round(k*dim), tf.int32)

    duty = tf.reshape(duty, [1, dim])

    alpha = tf.cast(k/dim, tf.float32)

    b = tf.exp(B*(alpha-duty))

    threshold, _ = tf.math.top_k(b*logits,
                                 k=k[0],
                                 sorted=True)
    threshold = threshold[:, -1]
    threshold = tf.reshape(threshold, [-1, 1])

    logic = logits-threshold
    binary_mask = tf.where(tf.less_equal(logic, 0.0),
                           x=tf.zeros_like(logits, tf.float32),
                           y=tf.ones_like(logits, tf.float32))

    out = logits*binary_mask
    duty = (1-alpha)*duty + alpha*binary_mask

    duty = tf.reduce_mean(duty, axis=0)

    return out, duty


# copied from: https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/sparsemax/python/ops/sparsemax.py#L30-L94


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sparsemax op."""

__all__ = ["sparsemax"]


def sparsemax(logits, name=None):
    """Computes sparsemax activations [1].
    For each batch `i` and class `j` we have
      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
    [1]: https://arxiv.org/abs/1602.02068
    Args:
      logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
        `float64`.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, "sparsemax", [logits]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        obs = array_ops.shape(logits)[0]
        dims = array_ops.shape(logits)[1]

        # In the paper, they call the logits z.
        # The mean(logits) can be substracted from logits to make the algorithm
        # more numerically stable. the instability in this algorithm comes mostly
        # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
        # to zero. However, in practise the numerical instability issues are very
        # minor and substacting the mean causes extra issues with inf and nan
        # input.
        z = logits

        # sort z
        z_sorted, _ = nn.top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = math_ops.cumsum(z_sorted, axis=1)
        k = math_ops.range(
            1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

        # calculate tau(z)
        # If there are inf values or all values are -inf, the k_z will be zero,
        # this is mathematically invalid and will also cause the gather_nd to fail.
        # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
        # fixed later (see p_safe) by returning p = nan. This results in the same
        # behavior as softmax.
        k_z_safe = math_ops.maximum(k_z, 1)
        indices = array_ops.stack([math_ops.range(0, obs), k_z_safe - 1], axis=1)
        tau_sum = array_ops.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

        # calculate p
        p = math_ops.maximum(
            math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])
        # If k_z = 0 or if z = nan, then the input is invalid
        p_safe = array_ops.where(
            math_ops.logical_or(
                math_ops.equal(k_z, 0), math_ops.is_nan(z_cumsum[:, -1])),
            array_ops.fill([obs, dims], math_ops.cast(float("nan"), logits.dtype)),
            p)

        return p_safe
