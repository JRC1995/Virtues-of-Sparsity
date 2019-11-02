import tensorflow as tf
import numpy as np
import math
from Config.config import config
import sys
sys.path.append("../../")
from lib.Transformer import Transformer
from lib.functions import gelu


class Encoder:

    def __init__(self, config_key, text, labels, dim,
                 true_seq_len, train, learning_rate,
                 word_embeddings, tags_size, class_weights):

        hyperparameters = config[config_key]

        self.hyperparameters = hyperparameters

        self.config_key = config_key

        self.text = text
        self.labels = labels

        self.N = tf.shape(self.text)[0]
        self.S = tf.shape(self.text)[1]
        self.D = dim

        self.true_seq_len = true_seq_len

        self.train = train
        self.learning_rate = learning_rate

        self.word_embeddings = word_embeddings

        self.tags_size = tags_size

        self.in_dropout = hyperparameters['in_dropout']
        self.MLP_dropout = hyperparameters['MLP_dropout']
        self.Z_lambda = hyperparameters['Z_lambda']
        self.l2 = hyperparameters['l2']
        self.T = hyperparameters['layers']
        self.max_grad_norm = hyperparameters['max_grad_norm']
        self.dim = hyperparameters['dim']

        self.class_weights = class_weights

        self.initiate_graph()
        self.compute_cost()
        self.optimizer()
        self.predict()

    # %%

    def initiate_graph(self):

        init = tf.zeros_initializer()

        with tf.variable_scope("embedding"):

            embeddings = tf.get_variable("embedding", trainable=False,
                                         dtype=tf.float32,
                                         initializer=tf.constant(self.word_embeddings.tolist()))

            Wc = tf.get_variable("Wc", shape=[self.D, self.dim],
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.glorot_uniform_initializer())

            Bc = tf.get_variable("Bc", shape=[self.dim],
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.zeros_initializer())

            embd_text = tf.nn.embedding_lookup(embeddings, self.text)

            embd_text = tf.reshape(embd_text, [self.N*self.S, self.D])

            embd_text = tf.matmul(embd_text, Wc) + Bc

            embd_text = tf.reshape(embd_text, [self.N, self.S, self.dim])

        input = embd_text

        attention = Transformer(hyperparameters=self.hyperparameters,
                                query=input,
                                value=input,
                                dim=self.dim,
                                current_depth=1,
                                layers=self.T,
                                true_q_len=self.true_seq_len,
                                true_v_len=self.true_seq_len,
                                train=self.train,
                                name="encode_source")

        layers = attention.layers
        self.Z = attention.Z
        top_layer = layers[-1]

        with tf.variable_scope("softmax"):

            self.W_score1 = tf.get_variable("W_score1", shape=[self.dim, 300],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=tf.glorot_uniform_initializer())

            self.B_score1 = tf.get_variable("Bias_score1", dtype=tf.float32,
                                            shape=[300],
                                            trainable=True, initializer=tf.zeros_initializer())

            self.W_score2 = tf.get_variable("W_score2", shape=[300, self.tags_size],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=tf.glorot_uniform_initializer())

            self.B_score2 = tf.get_variable("Bias_score2", dtype=tf.float32,
                                            shape=[self.tags_size],
                                            trainable=True, initializer=tf.zeros_initializer())

            top_layer = tf.reshape(top_layer, [-1, self.dim])

            score1 = gelu(tf.matmul(top_layer, self.W_score1)+self.B_score1)
            score1 = tf.layers.dropout(score1, rate=self.MLP_dropout, training=self.train)
            score = tf.matmul(score1, self.W_score2) + self.B_score2

            self.score = tf.reshape(score, [self.N, self.S, self.tags_size])

    # %%

    def compute_cost(self):

        filtered_trainables = [var for var in tf.trainable_variables() if
                               not("Bias" in var.name or "bias" in var.name
                                   or "noreg" in var.name)]

        regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in filtered_trainables])

        with tf.variable_scope("loss"):

            weights = tf.gather(self.class_weights, self.labels)

            # labels = tf.cast(self.labels, tf.float32)*tf.constant(1-self.label_smoothing-(self.label_smoothing /
            # self.tags_size), tf.float32)+(self.label_smoothing/self.tags_size)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self.score,
            )

            loss = loss*weights

            loss_mask = tf.sequence_mask(self.true_seq_len,
                                         maxlen=self.S,
                                         dtype=tf.float32)
            loss_mask = tf.reshape(loss_mask, [self.N, self.S])

            masked_loss = tf.multiply(loss, loss_mask)

            self.loss = tf.reduce_mean(masked_loss) + self.l2*regularization + self.Z_lambda*self.Z
            self.display_loss = tf.reduce_mean(masked_loss)

    # %%

    def optimizer(self):

        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        filtered_vars = all_vars

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-09,
                                           use_locking=False,
                                           name='Adam')

        gvs = optimizer.compute_gradients(self.loss, var_list=filtered_vars)

        self.train_ops = optimizer.apply_gradients([(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1])
                                                    for i, gv in enumerate(gvs)])

    # %%

    def predict(self):

        self.predictions = tf.argmax(self.score,
                                     axis=-1,
                                     output_type=tf.int32)

        # Comparing predicted sequence with labels

        comparison = tf.cast(tf.equal(self.predictions, self.labels),
                             tf.float32)

        # Masking to ignore the effect of pads while calculating accuracy
        pad_mask = tf.sequence_mask(self.true_seq_len,
                                    maxlen=self.S,
                                    dtype=tf.bool)

        masked_comparison = tf.boolean_mask(comparison, pad_mask)

        # Accuracy
        self.accuracy = tf.reduce_mean(masked_comparison)
