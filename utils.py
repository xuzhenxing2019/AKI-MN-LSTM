""" Code for loading data. """
import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re
import pickle as pkl
# import hickle as hkl
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import StratifiedKFold

def load_data(data_type):
    """Load data."""
    f_label = open('/Users/xuzhenxing/PycharmProjects/AKI-MN-LSTM/label.pkl', 'rb')

    label = pkl.load(f_label)
    f_label.close()
    all_label = list(label.values())
    labels = np.array(all_label)
    # print labels,len(labels)
    print 'labels are loaded.'

    # load clinical sequence
    f_cli_seq = open('/Users/xuzhenxing/PycharmProjects/AKI-MN-LSTM/clinical_sequnce.pkl', 'rb')
    cli_seq = pkl.load(f_cli_seq)
    f_cli_seq.close()
    all_cli_seq = list(cli_seq.values())
    clinical_sequences = np.array(all_cli_seq)

    print 'clinical_sequences are loaded.'

    # load clinical notes
    f_cli_note = open('/Users/xuzhenxing/PycharmProjects/AKI-MN-LSTM/clinical_notes.pkl', 'rb')
    cli_note = pkl.load(f_cli_note)
    f_cli_note.close()
    all_cli_note = list(cli_note.values())
    clinical_notes = np.array(all_cli_note)

    print 'notes are loaded.'

    # load demographic and commorbiditiea
    f_demo_comm = open('/Users/xuzhenxing/PycharmProjects/AKI-MN-LSTM/dict_demo_comm.pkl', 'rb')
    #
    demo_comm = pkl.load(f_demo_comm)
    f_demo_comm.close()
    all_demo_comm = list(demo_comm.values())
    demo_comm_ss = np.array(all_demo_comm)

    print demo_comm_ss.shape

    print 'demo_comm are loaded.'


    # clinical_sequences = clinical_sequences.reshape(len(clinical_sequences),-1)
    # clinical_notes = clinical_notes.reshape(len(clinical_notes),-1)

    # train, validate, test split
    skf = StratifiedKFold(n_splits=5)
    x_seq_set = list()
    x_notes_set = list()
    x_demo_comm_set = list()
    y_set = list()
    for train_index, test_index in skf.split(clinical_sequences,labels):
        # train_s, test_s = clinical_sequences[train_index], clinical_sequences[test_index]
        # train_n, test_n = clinical_notes[train_index], clinical_notes[test_index]
        # train_y, test_y = labels[train_index], labels[test_index]
        #
        # val_s = test_s
        # val_n = test_n
        # val_y = test_y
        #
        # x_seq_set.append((train_s,  val_s,  test_s))
        # x_notes_set.append((train_n, val_n, test_n))
        # y_set.append((train_y, val_y, test_y))

        # train_s_1, test_s = clinical_sequences[train_index], clinical_sequences[test_index]
        # train_n_2, test_n = clinical_notes[train_index], clinical_notes[test_index]
        # train_y_3, test_y = labels[train_index], labels[test_index]
        #
        # train_s = train_s_1[:20000]
        # train_n = train_n_2[:20000]
        # train_y = train_y_3[:20000]
        #
        # val_s = train_s_1[20001:]
        # val_n = train_n_2[20001:]
        # val_y = train_y_3[20001:]
        #
        # x_seq_set.append((train_s,  val_s,  test_s))
        # x_notes_set.append((train_n, val_n, test_n))
        # y_set.append((train_y, val_y, test_y))

        train_s_1, test_s = clinical_sequences[train_index], clinical_sequences[test_index]
        train_n_2, test_n = clinical_notes[train_index], clinical_notes[test_index]

        train_y_3, test_y = labels[train_index], labels[test_index]
        train_d_4, test_d = demo_comm_ss[train_index], demo_comm_ss[train_index]

        skf_2 = StratifiedKFold(n_splits=3)
        # skf_2 = StratifiedKFold(n_splits=5)

        for train_index_2, test_index_2 in skf_2.split(train_s_1, train_y_3):
            i=0
            if i ==0:
                train_s, val_s = train_s_1[train_index_2],train_s_1[test_index_2]
                train_n, val_n = train_n_2[train_index_2],train_n_2[test_index_2]
                train_y, val_y = train_y_3[train_index_2],train_y_3[test_index_2]

                train_d, val_d = train_d_4[train_index_2],train_d_4[test_index_2]

                x_seq_set.append((train_s, val_s, test_s))
                x_notes_set.append((train_n, val_n, test_n))
                x_demo_comm_set.append((train_d,val_d,test_d))
                y_set.append((train_y, val_y, test_y))

                i = i+1

        # print len(train_s), len(test_s)
        #
        # print train_s.shape,test_s.shape

        print 'loading data is over.'
    return x_seq_set, x_notes_set, x_demo_comm_set, y_set
    # return x_seq_set, x_notes_set, y_set

