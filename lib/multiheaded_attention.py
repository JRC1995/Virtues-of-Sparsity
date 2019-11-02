import tensorflow as tf
import math
import numpy as np
import sys
sys.path.append("../../")
from lib.functions import sparsemax


class multiheaded_attention:

    def __init__(self, hyperparameters, query, value, dim,
                 true_q_len, true_v_len,
                 current_depth, time, layers,
                 train, name, global_present=False, adaptive_span=True, out_dim=None):

        self.neg_inf = -(2**32)

        self.Q = query
        self.K = value
        self.V = value

        self.D = dim
        self.true_q_len = true_q_len
        self.true_v_len = true_v_len

        self.N = tf.shape(query)[0]
        self.qS = tf.shape(query)[1]
        self.vS = tf.shape(value)[1]

        self.current_depth = current_depth
        self.t = time
        self.T = layers

        self.train = train
        self.name = name
        self.heads = hyperparameters['heads']
        self.max_len = hyperparameters['max_len']
        self.dropout = hyperparameters['attention_dropout']
        self.sparsegen = hyperparameters['sparsegen']
        self.sparsegen_lambda = hyperparameters['sparsegen_lambda']
        self.R = tf.constant(hyperparameters['R'], tf.float32)  # hyperparameters['R']

        self.adaptive_span = adaptive_span
        self.global_present = global_present

        if out_dim is None:
            self.out_dim = dim
        else:
            self.out_dim = out_dim

        self.d = int(self.D/self.heads)

        Q_mask = tf.sequence_mask(self.true_q_len, maxlen=self.qS, dtype=tf.float32)
        self.Q_mask = tf.reshape(Q_mask, [self.N, self.qS, 1])

        V_mask = tf.sequence_mask(self.true_v_len, maxlen=self.vS, dtype=tf.float32)
        self.V_mask = tf.reshape(V_mask, [self.N, self.vS, 1])

        self.PE = self.spatial_encoding()
        self.binary_mask, self.softmax_mask = self.create_mask()

        self.multiheaded_attention()

    # %%
    def spatial_encoding(self):
        S = self.max_len
        D = self.D

        pe = np.zeros((2*S+1, D,), np.float32)

        for pos in range(-S, S+1):
            for i in range(0, D):
                if i % 2 == 0:
                    pe[pos+S, i] = math.sin(pos/(10000**(i/D)))
                else:
                    pe[pos+S, i] = math.cos(pos/(10000**((i-1)/D)))

        return tf.constant(pe.reshape((2*S+1, D)), tf.float32)

    # %%
    def create_mask(self):

        all_zeros = tf.zeros([self.N, self.qS, self.vS], tf.float32)
        all_neg_inf = tf.ones([self.N, self.qS, self.vS], tf.float32)*self.neg_inf

        binary_mask = tf.reshape(self.V_mask, [self.N, 1, self.vS])
        binary_mask = tf.tile(binary_mask, [1, self.qS, 1])
        binary_mask = binary_mask*self.Q_mask

        softmax_mask = tf.where(tf.equal(binary_mask, tf.constant(0, tf.float32)),
                                x=all_neg_inf,
                                y=all_zeros)

        softmax_mask = tf.reshape(softmax_mask, [1, self.N, self.qS, self.vS])
        softmax_mask = tf.tile(softmax_mask, [self.heads, 1, 1, 1])
        softmax_mask = tf.reshape(softmax_mask, [self.heads*self.N, self.qS, self.vS])

        binary_mask = tf.reshape(binary_mask, [1, self.N, self.qS, self.vS])
        binary_mask = tf.tile(binary_mask, [self.heads, 1, 1, 1])
        binary_mask = tf.reshape(binary_mask, [self.heads*self.N, self.qS, self.vS])

        return binary_mask, softmax_mask

    # %%

    # ADAPTED FROM: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    def generate_relative_embd(self, embeddings):

        S = tf.maximum(self.vS, self.qS)

        range_vec = tf.reshape(tf.range(S), [1, S])
        range_mat = tf.tile(range_vec, [S, 1])

        relative_pos_mat = range_mat - tf.transpose(range_mat)

        if not self.global_present:
            relative_pos_mat = relative_pos_mat[0:self.qS, 0:self.vS]
        else:
            relative_pos_mat = relative_pos_mat[0:self.qS, 0:self.vS-1]
            relative_pos_zeros = tf.zeros([self.qS, 1], tf.int32)
            relative_pos_mat = tf.concat([relative_pos_zeros, relative_pos_mat], axis=-1)

        relative_pos_mat_shifted = relative_pos_mat + self.max_len
        # will represent -self.max_len by 0,-self.max_len+1 by 1, and so on

        RE = tf.nn.embedding_lookup(embeddings, relative_pos_mat_shifted)

        return RE

    # %%

    def mask_pos(self, z):

        S = tf.maximum(self.vS, self.qS)

        range_vec = tf.reshape(tf.range(S), [1, S])
        range_mat = tf.tile(range_vec, [S, 1])

        relative_pos_mat = range_mat - tf.transpose(range_mat)

        relative_pos_mat = relative_pos_mat[0:self.qS, 0:self.vS]

        relative_pos_mat = tf.reshape(relative_pos_mat, [1, self.qS, self.vS])

        relative_pos_mat = tf.abs(tf.cast(relative_pos_mat, tf.float32))

        return tf.minimum(tf.maximum((self.R+z-relative_pos_mat)/self.R, 0.0), 1.0)

    # %%

    def masked_softmax(self, logits, z=None):

        epsilon = 1e-9

        if self.adaptive_span:
            pos_mask = self.mask_pos(z)

            exp_logits = tf.exp(logits)
            masked_exp_logits = pos_mask*tf.exp(logits)
            norm = tf.reduce_sum(masked_exp_logits, axis=-1, keepdims=True)+epsilon
            out = masked_exp_logits/norm
        else:
            out = tf.nn.softmax(logits, axis=-1)

        return out

    # %%

    def masked_sparsegen(self, logits, z=None):

        if self.adaptive_span:

            pos_mask = self.mask_pos(z)

            logits = pos_mask*logits
            logits = tf.reshape(logits, [self.heads*self.N*self.qS, self.vS])
            logits = logits/(1.0-self.sparsegen_lambda)

        out = sparsemax(logits)

        out = tf.reshape(out, [self.heads*self.N, self.qS, self.vS])

        return out

    # %%

    def multiheaded_attention(self):

        t = self.t

        l = self.current_depth+t

        init = tf.initializers.variance_scaling(scale=1/l,
                                                mode='fan_avg',
                                                distribution='uniform')

        with tf.variable_scope(self.name+"_"+str(t), reuse=tf.AUTO_REUSE, dtype=tf.float32):

            Wq = tf.get_variable("Wq", [self.heads, self.D,  self.d],
                                 dtype=tf.float32, initializer=init)

            Wk = tf.get_variable("Wk", [self.heads, self.D, self.d],
                                 dtype=tf.float32, initializer=init)

            Wv = tf.get_variable("Wv", [self.heads, self.D, self.d],
                                 dtype=tf.float32, initializer=init)

            if self.adaptive_span:

                Vz = tf.get_variable("Vz", [self.heads, self.d, 1],
                                     dtype=tf.float32, initializer=init)

                Bz = tf.get_variable("Bz", [self.heads, 1, 1],
                                     dtype=tf.float32, initializer=tf.zeros_initializer())

            Wq = tf.transpose(Wq, [1, 0, 2])
            Wq = tf.reshape(Wq, [self.D, self.heads*self.d])

            Wk = tf.transpose(Wk, [1, 0, 2])
            Wk = tf.reshape(Wk, [self.D, self.heads*self.d])

            Wv = tf.transpose(Wv, [1, 0, 2])
            Wv = tf.reshape(Wv, [self.D, self.heads*self.d])

            Wo = tf.get_variable("Wo", [self.heads*self.d, self.out_dim],
                                 dtype=tf.float32, initializer=init)

        # Position bias and weights are shared accross layers

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, dtype=tf.float32):

            u = tf.get_variable("u_bias", [self.heads, 1, 1, self.d],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            v = tf.get_variable("v_bias", [self.heads, 1, 1, self.d],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            Wrk = tf.get_variable("Wrk", [self.heads, self.D, self.d],
                                  dtype=tf.float32, initializer=init)

            Wrk = tf.transpose(Wrk, [1, 0, 2])
            Wrk = tf.reshape(Wrk, [self.D, self.heads*self.d])

        Q = tf.reshape(self.Q*self.Q_mask, [self.N*self.qS, self.D])
        K = tf.reshape(self.K*self.V_mask, [self.N*self.vS, self.D])
        V = tf.reshape(self.V*self.V_mask, [self.N*self.vS, self.D])

        Q = tf.matmul(Q, Wq)
        K = tf.matmul(K, Wk)
        V = tf.matmul(V, Wv)

        Q = tf.reshape(Q, [self.N, self.qS, self.heads*self.d])
        K = tf.reshape(K, [self.N, self.vS, self.heads*self.d])
        V = tf.reshape(V, [self.N, self.vS, self.heads*self.d])

        Q = tf.concat(tf.split(Q, self.heads, axis=-1), axis=0)  # (h*N, T_q, d_model/h)
        K = tf.concat(tf.split(K, self.heads, axis=-1), axis=0)  # (h*N, T_k, d_model/h)
        V = tf.concat(tf.split(V, self.heads, axis=-1), axis=0)  # (h*N, T_k, d_model/h)

        Q = tf.reshape(Q, [self.heads, self.N, self.qS, self.d])

        if self.adaptive_span:

            # Dynamically compute maximum span Z

            S = tf.cast(tf.maximum(self.qS, self.vS), tf.float32)
            Q_ = tf.reshape(Q, [self.heads, self.N*self.qS, self.d])
            Z = S*tf.nn.sigmoid(tf.matmul(Q_, Vz)+Bz)
            Z = tf.reshape(Z, [self.heads, self.N, self.qS, 1])
            Z = tf.reshape(Z, [self.heads*self.N, self.qS, 1])

        # ATTENTION

        Qc = tf.reshape(Q+u, [self.heads*self.N, self.qS, self.d])

        content_scores = tf.matmul(Qc, tf.transpose(K, [0, 2, 1]))

        PEk = tf.matmul(self.PE, Wrk)
        REk = self.generate_relative_embd(PEk)

        REk = tf.reshape(REk, [self.qS, self.vS, self.heads, self.d])
        REk = tf.transpose(REk, [2, 0, 1, 3])

        Qr = Q+v
        Qr = tf.transpose(Qr, [0, 2, 1, 3])
        position_scores = tf.matmul(Qr, tf.transpose(REk, [0, 1, 3, 2]))
        position_scores = tf.transpose(position_scores, [0, 2, 1, 3])
        position_scores = tf.reshape(position_scores, [self.heads*self.N, self.qS, self.vS])

        scalar_d = tf.sqrt(tf.constant(self.d, tf.float32))

        compatibility = (content_scores + position_scores)/scalar_d

        if not self.adaptive_span:
            Z = None

        if self.sparsegen:
            compatibility = compatibility*self.binary_mask
            compatibility = self.masked_sparsegen(compatibility, Z)
        else:
            compatibility = compatibility*self.binary_mask + self.softmax_mask
            compatibility = self.masked_softmax(compatibility, Z)

        # Attention Dropout

        compatibility = tf.layers.dropout(compatibility, rate=self.dropout, training=self.train)

        attended_content = tf.matmul(compatibility, V)

        attended_heads = attended_content
        attended_heads = tf.concat(tf.split(attended_heads, self.heads, axis=0), axis=2)
        attended_heads = tf.reshape(attended_heads, [self.N*self.qS, self.heads*self.d])

        head_composition = tf.matmul(attended_heads, Wo)

        head_composition = tf.reshape(head_composition, [self.N, self.qS, self.out_dim])

        self.output = head_composition

        if self.adaptive_span:
            self.Z = tf.reduce_mean(Z)
        else:
            self.Z = 0.0
