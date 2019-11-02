import tensorflow as tf
import numpy as np
import math
import sys
sys.path.append("../../")
from lib.multiheaded_attention_sparse import multiheaded_attention
from lib.functions import gelu
from lib.functions import layerNorm
from lib.functions import sparse_init
from lib.functions import k_winner


class Transformer:

    def __init__(self, hyperparameters, query, value, dim,
                 duty_cycles, duty_index, new_duty_cycles, current_depth, layers,
                 true_q_len, true_v_len,
                 train, name, self_attention=True):

        self.hyperparameters = hyperparameters

        self.D = dim
        self.current_depth = current_depth
        self.T = layers
        self.true_q_len = true_q_len
        self.true_v_len = true_v_len

        self.duty_cycles = duty_cycles
        self.duty_index = duty_index
        self.new_duty_cycles = new_duty_cycles

        self.N = tf.shape(query)[0]
        self.qS = tf.shape(query)[1]
        self.vS = tf.shape(value)[1]
        self.global_q_len = tf.tile(tf.constant([1], tf.int32), [self.N])

        self.train = train
        self.name = name
        self.self_attention = self_attention

        self.in_k = hyperparameters['in_k']
        self.k = hyperparameters['Transformer_k']
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

            W1 = tf.get_variable("W1", dtype=tf.float32,
                                 initializer=tf.constant(sparse_init((self.D, self.fc_dim), l)))
            B1 = tf.get_variable("Bias1", [self.fc_dim], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            W2 = tf.get_variable("W2", dtype=tf.float32,
                                 initializer=tf.constant(sparse_init((self.fc_dim, self.D), l)))
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

        k_rate = tf.cond(self.train,
                         lambda: self.k,
                         lambda: tf.minimum(self.in_k+0.5*self.in_k, 1.0))

        sublayer1 = tf.reshape(sublayer1, [-1, self.D])

        duty = self.get_duty(self.D)
        sublayer1, duty = k_winner(sublayer1, duty, self.D, k_rate, self.train)
        self.update_duty(duty, self.D)

        sublayer1 = tf.reshape(sublayer1, [self.N, self.qS, self.D])

        sublayer1 = layerNorm(sublayer1+Q, self.D, self.name+"/layer_norm2_"+str(t))

        sublayer2 = tf.reshape(sublayer1, [self.N*self.qS, self.D])
        sublayer2 = tf.matmul(sublayer2, W1)+B1

        duty = self.get_duty(self.fc_dim)
        sublayer2, duty = k_winner(sublayer2, duty, self.fc_dim, k_rate, self.train)
        self.update_duty(duty, self.fc_dim)

        sublayer2 = tf.matmul(sublayer2, W2)+B2

        duty = self.get_duty(self.D)
        sublayer2, duty = k_winner(sublayer2, duty, self.D, k_rate, self.train)
        self.update_duty(duty, self.D)

        sublayer2 = tf.reshape(sublayer2, [self.N, self.qS, self.D])

        sublayer2 = layerNorm(sublayer2+sublayer1, self.D, self.name+"/layer_norm3_"+str(t))

        return sublayer2

    # %%

    def update_duty(self, duty, dim):
        self.new_duty_cycles.append(duty)
        self.duty_index += dim

    # %%

    def get_duty(self, dim):
        return self.duty_cycles[self.duty_index:self.duty_index+dim]

    # %%

    def Transformer(self):

        Q = self.Q
        V = self.V

        Q = tf.reshape(Q, [-1, self.D])
        V = tf.reshape(V, [-1, self.D])

        k_rate = tf.cond(self.train,
                         lambda: self.in_k,
                         lambda: tf.minimum(self.in_k+0.5*self.in_k, 1.0))

        duty = self.get_duty(self.D)
        Q, duty = k_winner(Q, duty, self.D, k_rate, self.train)
        self.update_duty(duty, self.D)

        if self.self_attention:
            V = Q
        else:
            duty = self.get_duty(self.D)
            V, duty = k_winner(V, duty, self.D, k_rate, self.train)
            self.update_duty(duty, self.D)

        Q = tf.reshape(Q, [self.N, self.qS, self.D])
        V = tf.reshape(V, [self.N, self.vS, self.D])

        layers = [Q]

        for i in range(self.T):
            Q = self.single_unit(Q, V, i)
            V = Q
            self.true_v_len = self.true_q_len
            layers.append(Q)

        self.layers = layers
        self.Z = self.Z/tf.constant(max([self.T, 1]), tf.float32)
