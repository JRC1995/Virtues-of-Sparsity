import sys
sys.path.append("../")
import pickle
import json
import numpy as np
import string
import random
import tensorflow as tf
from model.Transformer_NER import Encoder
from DataLoader.bucket_and_batch import bucket_and_batch
sys.path.append("../../")
from utils.eval import eval


bnb = bucket_and_batch()
eval = eval()

test_batch_size = int(64)

with open('../Processed_Data/CoNLL_NER_2003.pkl', 'rb') as fp:
    data = pickle.load(fp)

tags2idx = data['tags2idx']
test_text = data['test_text']
test_tags = data['test_tags']
vocab2idx = data['vocab2idx']
embd = data['embd']

idx2vocab = {v: k for k, v in vocab2idx.items()}

tf_text = tf.placeholder(tf.int32, [None, None])
tf_labels = tf.placeholder(tf.int32, [None, None])
tf_true_seq_lens = tf.placeholder(tf.int32, [None])
tf_train = tf.placeholder(tf.bool)
tf_learning_rate = tf.placeholder(tf.float32)

labels2idx = tags2idx
idx2labels = {v: k for k, v in labels2idx.items()}

class_weights = [1.0 for tag in tags2idx]
class_weights[labels2idx['O']] = 1.0


model = Encoder(config_key='CoNLL2003_Hyperparameters',
                text=tf_text,
                labels=tf_labels,
                dim=300,
                tags_size=len(labels2idx),
                true_seq_len=tf_true_seq_lens,
                train=tf_train,
                learning_rate=tf_learning_rate,
                word_embeddings=embd,
                class_weights=class_weights)


with tf.Session() as sess:  # Start Tensorflow Session

    test_batches_source, test_batches_labels, test_batches_true_seq_lens = bnb.bucket_and_batch(
        test_text, test_tags, vocab2idx, labels2idx,  test_batch_size)

    print("Testing batches loaded")

    display_step = 100
    # patience = 5

    filtered_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=filtered_var_list)

    print('Loading pre-trained weights for the model...')

    saver.restore(sess, '../Model_Backup/Transformer/Transformer.ckpt')

    print('\nRESTORATION COMPLETE\n')

    total_test_cost = 0
    total_tp = 0
    total_pred_len = 0
    total_gold_len = 0

    for i in range(0, len(test_batches_source)):

        if i % 100 == 0:
            print("Testing batch {}".format(i+1))

        cost, prediction,\
            acc = sess.run([model.display_loss,
                            model.predictions,
                            model.accuracy],
                           feed_dict={tf_text: test_batches_source[i],
                                      tf_labels: test_batches_labels[i],
                                      tf_true_seq_lens: test_batches_true_seq_lens[i],
                                      tf_train: False})

        tp, pred_len, gold_len = eval.stats(test_batches_labels[i],
                                            prediction,
                                            test_batches_true_seq_lens[i],
                                            idx2labels)

        total_tp += tp
        total_pred_len += pred_len
        total_gold_len += gold_len
        total_test_cost += cost

    test_len = len(test_batches_source)

    test_prec, test_rec, test_F1 = eval.F1(total_tp, total_pred_len, total_gold_len)

    avg_test_cost = total_test_cost/test_len

    print("\n\nTEST RESULTS\n\n")

    print("Test Loss= " +
          "{:.5f}".format(avg_test_cost) + ", Test F1= " +
          "{:.5f}".format(test_F1)+", Test Prec= " +
          "{:.5f}".format(test_prec)+", Test Rec= " +
          "{:.5f}".format(test_rec))
