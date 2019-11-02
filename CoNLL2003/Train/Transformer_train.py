import sys
sys.path.append("../")
import pickle
from Config.config import config
import tensorflow as tf
from model.Transformer_NER import Encoder
from DataLoader.bucket_and_batch import bucket_and_batch
sys.path.append("../../")
from utils.eval import eval
import numpy as np
import math
import string
import random


bnb = bucket_and_batch()
eval = eval()

hyperparameters = config['CoNLL2003_Hyperparameters']

val_batch_tokens = 64
train_batch_tokens = 64

max_iterations = 1000

dim = hyperparameters['dim']

with open('../Processed_Data/CoNLL_NER_2003.pkl', 'rb') as fp:
    data = pickle.load(fp)

tags2idx = data['tags2idx']
train_text = data['train_text']
train_tags = data['train_tags']
val_text = data['val_text']
val_tags = data['val_tags']
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


parameter_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("\n\nNumber of Parameters: {}\n\n".format(parameter_count))


epochs = 100

with tf.Session() as sess:  # Start Tensorflow Session

    val_batches_source, val_batches_labels, val_batches_true_seq_lens = bnb.bucket_and_batch(
        val_text, val_tags, vocab2idx, labels2idx,  val_batch_tokens)

    print("Validation batches loaded")

    train_batches_source, train_batches_labels, train_batches_true_seq_lens = bnb.bucket_and_batch(
        train_text, train_tags, vocab2idx, labels2idx,  train_batch_tokens)

    print("Train batches loaded")

    display_step = 100
    patience = hyperparameters['patience']

    load = 'n'  # input("\nLoad checkpoint? y/n: ")
    print("")

    filtered_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=filtered_var_list)

    if load.lower() == 'y':

        print('Loading pre-trained weights for the model...')

        saver.restore(sess, '../Model_Backup/Transformer/Transformer.ckpt')
        sess.run(tf.global_variables())
        sess.run(tf.tables_initializer())

        with open('../Model_Backup/Transformer/Transformer.pkl', 'rb') as fp:
            train_data = pickle.load(fp)

        epochs_covered = train_data['epochs_covered']
        best_loss = train_data['best_loss']
        best_F1 = train_data['best_F1']
        impatience = train_data['impatience']
        steps = train_data['steps']
        train_losses = train_data['train_losses']
        val_losses = train_data['val_losses']
        train_F1s = train_data['train_F1s']
        val_F1s = train_data['val_F1s']

        print('\nRESTORATION COMPLETE\n')

    else:

        epochs_covered = 0
        best_loss = math.inf
        best_F1 = -math.inf
        impatience = 0
        steps = 1
        train_losses = []
        train_F1s = []
        val_losses = []
        val_F1s = []

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.tables_initializer())

    for epoch in range(epochs_covered, epochs):

        batches_indices = [i for i in range(0, len(train_batches_source))]
        random.shuffle(batches_indices)

        total_train_loss = 0
        total_tp = 0
        total_pred_len = 0
        total_gold_len = 0

        for i in range(0, min(1000, len(train_batches_source))):

            #warmup_steps = 4000
            #lrate =  (dim**(-.5))*min(steps**(-0.5), steps*(warmup_steps**(-1.5)))

            lrate = hyperparameters['lrate']

            j = int(batches_indices[i])

            cost, prediction,\
                _ = sess.run([model.display_loss,
                              model.predictions,
                              model.train_ops],
                             feed_dict={tf_text: train_batches_source[j],
                                        tf_labels: train_batches_labels[j],
                                        tf_true_seq_lens: train_batches_true_seq_lens[j],
                                        tf_train: True,
                                        tf_learning_rate: lrate})

            steps += 1

            tp, pred_len, gold_len = eval.stats(train_batches_labels[j],
                                                prediction,
                                                train_batches_true_seq_lens[j],
                                                idx2labels)

            total_tp += tp
            total_pred_len += pred_len
            total_gold_len += gold_len
            total_train_loss += cost

            _, _, F1 = eval.F1(tp, pred_len, gold_len)

            if i % display_step == 0:

                print("Iter "+str(i)+", Cost = " +
                      "{:.3f}".format(cost)+", F1 = " +
                      "{:.3f}".format(F1))

        _, _, train_F1 = eval.F1(total_tp, total_pred_len, total_gold_len)

        train_F1s.append(train_F1)
        train_len = len(train_batches_source)
        train_losses.append(total_train_loss/train_len)

        print("\n\n")

        total_val_cost = 0
        total_tp = 0
        total_pred_len = 0
        total_gold_len = 0

        for i in range(0, len(val_batches_source)):

            if (i+1) % 100 == 0:
                print("Validating Batch {}".format(i+1))

            cost, prediction,\
                acc = sess.run([model.display_loss,
                                model.predictions,
                                model.accuracy],
                               feed_dict={tf_text: val_batches_source[i],
                                          tf_labels: val_batches_labels[i],
                                          tf_true_seq_lens: val_batches_true_seq_lens[i],
                                          tf_train: False})

            tp, pred_len, gold_len = eval.stats(val_batches_labels[i],
                                                prediction,
                                                val_batches_true_seq_lens[i],
                                                idx2labels)

            total_tp += tp
            total_pred_len += pred_len
            total_gold_len += gold_len
            total_val_cost += cost

        _, _, val_F1 = eval.F1(total_tp, total_pred_len, total_gold_len)

        val_len = len(val_batches_source)

        avg_val_cost = total_val_cost/val_len

        val_F1s.append(val_F1)
        val_losses.append(avg_val_cost)

        print("\n\nVALIDATION\n\n")

        print("Epoch " + str(epoch) + ", Validation Loss= " +
              "{:.3f}".format(avg_val_cost) + ", Validation F1= " +
              "{:.3f}".format(val_F1))

        flag = 0
        impatience += 1

        if avg_val_cost < best_loss:

            impatience = 0

            best_loss = avg_val_cost

        if val_F1 >= best_F1:

            impatience = 0

            best_F1 = val_F1

            flag = 1

        if flag == 1:

            saver.save(sess, '../Model_Backup/Transformer/Transformer.ckpt')

            PICKLE_dict = {'epochs_covered': epoch+epochs_covered+1,
                           'best_loss': best_loss,
                           'best_F1': best_F1,
                           'impatience': impatience,
                           'steps': steps,
                           'train_losses': train_losses,
                           'val_losses': val_losses,
                           'train_F1s': train_F1s,
                           'val_F1s': val_F1s}

            with open('../Model_Backup/Transformer/Transformer.pkl', 'wb') as fp:
                pickle.dump(PICKLE_dict, fp)

            print("Checkpoint created!")

        print("\n")

        if impatience > patience:
            break
