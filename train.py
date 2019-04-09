# %load_ext autoreload
# %autoreload 2
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.insert(0, '..')
import models, utils
# from utils import model_perf

import tensorflow as tf
import numpy as np
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import scipy.sparse as sp
import pickle as pkl

# from tensorflow.examples.tutorials.mnist import input_data

# %matplotlib inline
flags = tf.app.flags
FLAGS = flags.FLAGS

results_auc = dict()
results_accuracy = dict()
aucs = list()
accuracies = list()
results = list()

class model_perf(object):

    def __init__(self, i_fold):
        self.i_fold = i_fold
        # self.pairs_label = pairs_label
        self.names, self.params = set(), {}
        self.fit_auc, self.fit_accuracy, self.fit_losses, self.fit_time = {}, {}, {}, {}
        self.train_auc, self.test_accuracy, self.train_loss = {}, {}, {}
        self.test_auc, self.train_accuracy, self.test_loss = {}, {}, {}
        self.s_represent = dict()
        self.s_count = dict()

    def test(self, model, name, params, train_data, train_recs, train_demo, train_labels, val_data, val_recs, val_demo, val_labels, test_data, test_recs, test_demo, test_labels):
        self.params[name] = params
        print ('start train.....')
        self.fit_auc[name], self.fit_accuracy[name], self.fit_losses[name], self.fit_time[name] = \
                model.fit(train_data, train_recs, train_demo, train_labels, val_data, val_recs, val_demo, val_labels)
        del val_data, val_demo, val_labels

        print('End training.....')

        # string, self.test_auc[name], self.test_accuracy[name], self.test_loss[name], _, test_represent, test_prob = \
        #         model.evaluate(test_data, test_recs, test_labels)

        string, self.test_auc[name], self.test_accuracy[name], self.test_loss[name], predictions = \
                model.evaluate(test_data, test_recs, test_demo, test_labels)

        print('test  {}'.format(string))
        self.names.add(name)

        print('End testing.....')

    def save(self, data_type):
        results = list()
        for name in sorted(self.names):
            results.append([name, self.test_accuracy[name], self.train_accuracy[name],
            self.test_f1[name], self.train_f1[name], self.test_loss[name],
            self.train_loss[name], self.fit_time[name]*1000])

        if os.path.exists(data_type + '_results.csv'):
            old = pd.read_csv(data_type + '_results.csv', header=None)
            new = pd.DataFrame(data=results)
            r = pd.concat([old, new], ignore_index=True)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])
        else:
            r = pd.DataFrame(data=results)
            r.to_csv(data_type + '_results.csv', index=False, header=['method', 'test_acc',
            'train_acc', 'test_f1', 'train_f1', 'test_loss', 'train_loss', 'fit_time'])


    def fin_result(self, data_type, i_fold=None):
        for name in sorted(self.names):
            if name not in results_auc:
                results_auc[name] = 0
            if name not in results_accuracy:
                results_accuracy[name] = 0
            results_auc[name] += self.test_auc[name]
            results_accuracy[name] += self.test_accuracy[name]
            aucs.append(self.test_auc[name])
            accuracies.append(self.test_accuracy[name])
            results.append([i_fold, self.test_auc[name], self.test_accuracy[name]])
        if i_fold == 4:
            for name in sorted(self.names):
                results_auc[name] /= 5
                results_accuracy[name] /= 5
                # print('{:5.2f}  {}'.format(
                #     results_auc[name], name))
                # print('{:5.2f}  {}'.format(
                #     results_accuracy[name], name))
            std_auc = np.std(np.array(aucs))
            std_accuracy = np.std(np.array(accuracies[:-1]))
            results.append([name, results_auc[name], std_auc, results_accuracy[name], std_accuracy])
            r = pd.DataFrame(data=results)
            r.to_csv(data_type + '_fin_results', index=False, header=['method', 'test_auc', 'std_auc', 'test_accuracy', 'std_accuracy'])


    def show(self, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)         # controls default text sizes
            plt.rc('axes', titlesize=fontsize)    # fontsize of the axes title
            plt.rc('axes', labelsize=fontsize)    # fontsize of the x any y labels
            plt.rc('xtick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)   # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)   # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('  auc      loss        time [ms]  name')
        print('test  train   test  train   test     train')
        for name in sorted(self.names):
            print('{:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                    self.test_auc[name], self.train_auc[name],
                    self.test_loss[name], self.train_loss[name], self.fit_time[name]*1000, name))

def get_notes_data(notes, labels, max_sentence_num, max_sentence_length):
    pairs = notes
    train_pairs, val_pairs, test_pairs = pairs
    train_labels, val_labels, test_labels = labels

    m = max_sentence_num
    f = max_sentence_length

    # notes data
    train_x = np.zeros([train_pairs.shape[0], m, f])
    val_x = np.zeros([val_pairs.shape[0], m, f])
    test_x = np.zeros([test_pairs.shape[0], m, f])

    # store notes
    train_x[:,:,:] = train_pairs
    val_x[:,:,:] = val_pairs
    test_x[:,:,:] = test_pairs

    train_y = train_labels
    val_y = val_labels
    test_y = test_labels

    print (train_y.shape)
    print (val_y.shape)
    print (test_y.shape)

    # print(train_x[2, :, :])

    return train_x, train_y, val_x, val_y, test_x, test_y

def get_records_data(records, mem_size, code_size):

    pairs=records
    train_pairs, val_pairs, test_pairs = pairs

    train_pairs = train_pairs.transpose(0, 2, 1)
    val_pairs = val_pairs.transpose(0, 2, 1)
    test_pairs = test_pairs.transpose(0, 2, 1)

     # clinical records
    train_r = np.zeros([train_pairs.shape[0], mem_size, code_size])
    val_r = np.zeros([val_pairs.shape[0], mem_size, code_size])
    test_r = np.zeros([test_pairs.shape[0], mem_size, code_size])

    # store clinical records
    train_r[:,:,:] = train_pairs
    val_r[:,:,:] = val_pairs
    test_r[:,:,:] = test_pairs

    print (train_r.shape)
    print (val_r.shape)
    print (test_r.shape)

    # print (train_r[2,:,:])

    return train_r, val_r, test_r

def get_demo_comm_data(demo_comm, demo_comm_num, demo_comm_num_dim):
    pairs = demo_comm
    train_pairs, val_pairs, test_pairs = pairs

    # f = demo_comm_num
    # m = demo_comm_num_dim

    m = demo_comm_num
    f = demo_comm_num_dim

    # # demo_comm data
    # train_d = np.zeros([train_pairs.shape[0], m, f])
    # val_d = np.zeros([val_pairs.shape[0], m, f])
    # test_d = np.zeros([test_pairs.shape[0], m, f])

    # demo_comm data
    train_d = np.zeros([train_pairs.shape[0], m])
    val_d = np.zeros([val_pairs.shape[0], m])
    test_d = np.zeros([test_pairs.shape[0], m])

    # store demo_comm
    train_d[:,:] = train_pairs
    val_d[:,:] = val_pairs
    test_d[:,:] = test_pairs

    # print (train_d.shape)
    # print (val_d.shape)
    # print (test_d.shape)

    # print(train_d[2, :, :])

    return train_d, val_d, test_d



def train(modality, method, data_type, distance, k, fdim, nhops, mem_size, code_size, n_words, edim, n_epoch, batch_size, time_step, clinical_words, max_sentence_num, max_sentence_length, demo_comm_num, demo_comm_num_dim, demo_comm,  notes, records, labels, i_fold):
    str_params = '_' + modality + '_' + distance + '_k' + str(k) + '_fdim' + str(fdim) + '_nhops' + str(nhops) + '_memsize' + str(mem_size) + '_codesize' + str(code_size) + '_nwords' + str(n_words) + '_edim' + str(edim)

    print ('Set parameters...')
    mp = model_perf(i_fold)
    # Architecture.
    common = {}
    common['dir_name']       = 'ppmi/'
    common['num_epochs']     = n_epoch
    common['batch_size']     = batch_size
    common['eval_frequency'] = 5 * common['num_epochs']
    common['patience']       = 5
    common['regularization'] = 1e-2
    common['dropout']        = 0.25
    common['learning_rate']  = 5e-3
    common['decay_rate']     = 0.95
    common['momentum']       = 0.9
    common['init_std']       = 5e-2

    train_x, train_y, val_x, val_y, test_x, test_y = get_notes_data(notes, labels, max_sentence_num, max_sentence_length) # data, labels

    train_r, val_r, test_r = get_records_data(records,mem_size, code_size)

    train_d, val_d, test_d = get_demo_comm_data(demo_comm, demo_comm_num, demo_comm_num_dim)


    C = max(train_y)+1
    common['decay_steps']    = train_x.shape[0] / common['batch_size']

    if method == 'MemGCN':
        # str_params += ''
        name = 'cgconv_softmax'
        params = common.copy()
        params['method'] = method

        params['time_step'] = time_step
        params['clinical_words'] = clinical_words
        params['max_sentence_num'] = max_sentence_num
        params['max_sentence_length'] = max_sentence_length

        params['demo_comm_num'] = demo_comm_num
        params['demo_comm_num_dim'] = demo_comm_num_dim


        params['p']              = [1] # pooling size
        params['M']              = [C]
        params['K']              = k    # support number
        params['nhops']          = nhops # hop number
        params['fdim']           = fdim # filters dimension
        params['edim']           = edim # embeddings dimension
        params['mem_size']       = mem_size # the length of sequential records
        params['code_size']      = code_size # the size of one record
        params['n_words']        = n_words # feature dimension
        params['distance']       = distance
        params['fin'] = train_x.shape[2]
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['brelu'] = 'b2relu'
        params['pool'] = 'apool1'

        mp.test(models.siamese_cgcnn_mem(**params), name, params, train_x, train_r, train_d, train_y, val_x, val_r, val_d, val_y, test_x, test_r, test_d, test_y)

    # mp.save(data_type)
    method_type = method + '_'
    mp.fin_result(method_type + data_type + str_params, i_fold)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('modality', type=str)
    # parser.add_argument('method', type=str)
    # parser.add_argument('data_type', type=str)
    # parser.add_argument('distance', type=str)
    # parser.add_argument('K', type=int)
    # parser.add_argument('fdim', type=int)
    # parser.add_argument('nhops', type=int)
    # parser.add_argument('mem_size', type=int)
    # parser.add_argument('code_size', type=int)
    # parser.add_argument('n_words', type=int)
    # parser.add_argument('edim', type=int)
    # parser.add_argument('n_epoch', type=int)
    # parser.add_argument('batch_size', type=int)
    #
    # parser.add_argument('time_step', type=int)
    # parser.add_argument('clinical_words', type=int)
    #
    # parser.add_argument('max_sentence_num', type=int)
    # parser.add_argument('max_sentence_length', type=int)
    # args = parser.parse_args()
    # print ('-----------------START-------------------')
    # print (args.method)


    # See function train for all possible parameter and there definition.
    records, notes, demo_comm, labels = utils.load_data(data_type='dti_fact')    # notes (queries) ,  records (EHR-chat-lab-events) , labels (0/1)
    # records, notes, labels = utils.load_data(data_type='dti_fact')    # notes (queries) ,  records (EHR-chat-lab-events) , labels (0/1)

    print ("5-fold cross validation ...")
    for l in range(5): # 5-fold cross validation
        print ("********* The %d fold ... *********" %(l+1))
        train(modality= 'AKI',  #args.modality,
              method= 'MemGCN',  # args.method,
              data_type= 'str',  #rgs.data_type,
              distance=  'in',  #args.distance,
              k=  30,  #args.K,
              fdim= 32,  #args.fdim,
              nhops= 3,  #args.nhops,
              mem_size= 12,  #args.mem_size,  12
              code_size= 38,  #args.code_size, the number of features for clinical chart and lab events;  38
              n_words= 190,  #args.nwords,  #  bins , each feature has 5 bins.;; 38*5 =190
              # edim=  32,  #args.edim,
              edim=128,  # args.edim,

              n_epoch=  10,  #args.n_epoch,
              batch_size= 32,  #args.batch_size,

              time_step=  12,  #args.time_step,
              clinical_words= 9754,  #args.clinical_words, # the number of words for clinical notes


              max_sentence_num =  5,  #args.max_sentence_num,
              max_sentence_length = 200,  #args.max_sentence_length,

              demo_comm_num =  12,  # the number of demo_comm
              demo_comm_num_dim = 1,  # dim is 1 , for demo_comm information
              demo_comm=demo_comm[l],

              notes=notes[l],
              records=records[l],

              labels=labels[l],

              i_fold=l)
    print ('-----------------DONE-------------------')
