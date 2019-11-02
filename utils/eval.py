from __future__ import division
import numpy as np


class eval:

    def stats(self, batch_labels, batch_predictions, true_seq_lens, tags):

        tp = 0
        pred_len = 0
        gold_len = 0

        for i in range(0, len(batch_labels)):
            j = 0
            while j < true_seq_lens[i]:

                init_j = j

                if tags[batch_labels[i][j]][0] == 'B':
                    gold_len += 1
                    chunk_label = []
                    chunk_prediction = []
                    chunk_label.append(batch_labels[i][j])
                    chunk_prediction.append(batch_predictions[i][j])

                    j += 1

                    while j < true_seq_lens[i]:
                        if tags[batch_labels[i][j]][0] in ['I']:
                            chunk_label.append(batch_labels[i][j])
                            chunk_prediction.append(batch_predictions[i][j])
                            j += 1
                        else:
                            break

                    label_entity = np.asarray(chunk_label, dtype=np.int32)
                    prediction_entity = np.asarray(chunk_prediction, dtype=np.int32)

                    if np.all(np.equal(label_entity, prediction_entity)):
                        tp += 1

                elif tags[batch_labels[i][j]][0] == 'O':
                    j += 1

            j = 0

            while j < true_seq_lens[i]:

                if tags[batch_predictions[i][j]][0] in ['B']:
                    pred_len += 1
                j += 1

        return tp, pred_len, gold_len

    def F1(self, tp, pred_len, gold_len):

        prec = tp/pred_len if pred_len > 0 else 0
        rec = tp/gold_len if gold_len > 0 else 0
        F1 = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0

        return prec, rec, F1
