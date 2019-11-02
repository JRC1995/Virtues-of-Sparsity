import tensorflow as tf
import numpy as np
import math
import sys
sys.path.append("../../")
from lib.multiheaded_attention import multiheaded_attention
from lib.functions import gelu
from lib.functions import layerNorm


class Transformer:

    def __init__(self, hyperparameters, query, value, dim,
                 current_depth, layers,
                 true_q_len, true_v_len,
                 train, name, self_attention=True):

        self.hyperparameters = hyperparameters

        self.D = dim
        self.current_depth = current_depth
        self.T = layers
        self.true_q_len = true_q_len
        self.true_v_len = true_v_len

        self.N = tf.shape(query)[0]
        self.qS = tf.shape(query)[1]
        self.vS = tf.shape(value)[1]
        self.global_q_len = tf.tile(tf.constant([1], tf.int32), [self.N])

        self.train = train
        self.name = name
        self.self_attention = self_attention

        self.in_dropout = hyperparameters['in_dropout']
        self.dropout = hyperparameters['Transformer_dropout']
        self.fc_dim = hyperparameters['fc_dim']
        self.adaptive_span = hyperparameters['adaptive_span']

        Q_mask = tf.sequence_mask(self.true_q_len, maxlen=self.qS, dtype=tf.float32)
        self.Q_mask = tf.reshape(Q_mask, [self.N, self.qS, 1])

        V_mask = tf.sequence_mask(self.true_v_len, maxlen=self.vS, dtype=tf.float32)
        self.V_mask = tf.reshape(V_mask, [self.N, self.vS, 1])

        self.Q = query*self.Q_mask
        self.V = value*self.V_mask

        self.Z = tf.constant(0.0, tf.float32)

        self.Transformer()

    # %%

    def single_unit(self, Q, V, t):

        l = self.current_depth+t

        with tf.variable_scope(self.name+"_"+str(t), reuse=tf.AUTO_REUSE, dtype=tf.float32):

            init = tf.initializers.variance_scaling(scale=1/l,
                                                    mode='fan_avg', distribution='uniform')

            W1 = tf.get_variable("W1", [self.D, self.fc_dim], dtype=tf.float32,
                                 initializer=init)
            B1 = tf.get_variable("Bias1", [self.fc_dim], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            W2 = tf.get_variable("W2", [self.fc_dim, self.D], dtype=tf.float32,
                                 initializer=init)
            B2 = tf.get_variable("Bias2", [self.D], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

        if t == 0:
            Q = layerNorm(Q, self.D, self.name+"/layer_norm1_Q"+str(t))

            if not self.self_attention:
                V = layerNorm(V, self.D, self.name+"/layer_norm1_V"+str(t))
            else:
                V = Q

        attention = multiheaded_attention(hyperparameters=self.hyperparameters,
                                          query=Q, value=V,
                                          dim=self.D,
                                          true_q_len=self.true_q_len,
                                          true_v_len=self.true_v_len,
                                          current_depth=self.current_depth,
                                          time=t,
                                          layers=self.T,
                                          train=self.train,
                                          name=self.name,
                                          adaptive_span=self.adaptive_span)

        sublayer1 = attention.output
        self.Z += attention.Z

        sublayer1 = tf.layers.dropout(sublayer1, rate=self.dropout, training=self.train)

        sublayer1 = layerNorm(sublayer1+Q, self.D, self.name+"/layer_norm2_"+str(t))

        sublayer2 = tf.reshape(sublayer1, [self.N*self.qS, self.D])
        sublayer2 = gelu(tf.matmul(sublayer2, W1)+B1)
        sublayer2 = tf.matmul(sublayer2, W2)+B2

        sublayer2 = tf.reshape(sublayer2, [self.N, self.qS, self.D])
        sublayer2 = tf.layers.dropout(sublayer2, rate=self.dropout, training=self.train)

        sublayer2 = layerNorm(sublayer2+sublayer1, self.D, self.name+"/layer_norm3_"+str(t))

        return sublayer2

    # %%
    def Transformer(self):

        Q = self.Q
        V = self.V

        Q = tf.layers.dropout(Q, rate=self.in_dropout, training=self.train)
        V = tf.layers.dropout(V, rate=self.in_dropout, training=self.train)

        layers = [Q]

        for i in range(self.T):
            Q = self.single_unit(Q, V, i)
            V = Q
            self.true_v_len = self.true_q_len
            layers.append(Q)

        self.layers = layers
        self.Z = self.Z/tf.constant(max([self.T, 1]), tf.float32)
