import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil
from tensorflow.contrib import layers

import time


# Common methods for all models
class base_model(object):

    def __init__(self):
        self.regularizers = []

    def predict(self, data, recs, demo, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            batch_recs = np.zeros((self.batch_size, recs.shape[1], recs.shape[2]))
            batch_demo = np.zeros((self.batch_size, demo.shape[1]))

            tmp_data = data[begin:end, :, :]
            tmp_recs = recs[begin:end, :, :]
            tmp_demo = demo[begin:end, :]


            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
                tmp_recs = tmp_recs.toarray()
            batch_data[:end-begin] = tmp_data
            batch_recs[:end-begin] = tmp_recs
            batch_demo[:end - begin] = tmp_demo

            feed_dict = {self.ph_data: batch_data, self.ph_recs: batch_recs, self.ph_demo: batch_demo, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]
            # represent[begin:end, :] = batch_rep[:end-begin, :]
            # prob[begin:end, :] = batch_prob[:end-begin, :]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            # return predictions
            return predictions, loss


    def evaluate(self, data, recs, demo, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        recs: size N x Q
            N: number of signals (samples)
            Q: number of timestamps (sequence length)
        labels: size N
            N: number of signals (samples)
        """
        # t_process, t_wall = time.process_time(), time.time()
        t_process, t_wall = time.clock(), time.time()
        predictions, loss = self.predict(data, recs, demo, labels, sess)

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
        auc = 100 * sklearn.metrics.auc(fpr, tpr)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        # string = 'auc: {:.2f}, accuracy: {:.2f} ({:d} / {:d}), loss: {:.2e}'.format(auc, accuracy, ncorrects, len(labels), loss)
        string = 'acc: {:.2f}, loss: {:.2e}'.format( accuracy, loss)


        if sess is None:
            # string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall)

        # return string, auc, loss, predictions
        return string, auc, accuracy, loss, predictions


    def fit(self, train_data, train_recs, train_demo, train_labels, val_data, val_recs, val_demo, val_labels):
        # t_process, t_wall = time.process_time(), time.time()
        t_process, t_wall = time.clock(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        count = 0
        bad_counter = 0
        accuracies = []
        aucs = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        estop = False  # early stop

        print '#######'

        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            count += len(idx)
            batch_data, batch_recs, batch_demo, batch_labels = train_data[idx, :, :], train_recs[idx, :, :], train_demo[idx, :], train_labels[idx]

            # print 'batch is  ok !'
            # print 'the  data:'
            # print batch_data
            # print 'the labels of data:'
            # print batch_labels
            # print batch_data.shape, batch_recs.shape, batch_labels.shape

            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                batch_recs = batch_recs.toarray()
                batch_demo = batch_demo.toarray()
            feed_dict = {self.ph_data: batch_data, self.ph_recs: batch_recs, self.ph_demo: batch_demo, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            # learning_rate, loss_average = sess.run([self.op_train, self.op_loss], feed_dict)


            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                print ('Seen samples: %d' % count)
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, auc, accuracy, loss, predictions = self.evaluate(val_data, val_recs, val_demo, val_labels, sess)
                aucs.append(auc)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                # print(predictions.tolist()[:50])
                # print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))
                # print('  time: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall))


                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validataion/auc', simple_value=auc)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

                if len(aucs) > (self.patience+5) and auc > np.array(aucs).max():
                    bad_counter = 0

                if len(aucs) > (self.patience+5) and auc <= np.array(aucs)[:-self.patience].max():
                    bad_counter += 1
                    if bad_counter > self.patience:
                        # print('Early Stop!')
                        estop = True
                        break
            if estop:
                break
        # print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        # print('validation auc: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))

        # print('validation auc: '.format(max(accuracies)))
        # print('validation auc: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))

        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps
        return aucs, accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph with memory network.
    def build_gcn_graph_mem(self, M_0, M_1):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, M_1), 'data')   #  # clinical notes
                self.ph_recs = tf.placeholder(tf.int32, (self.batch_size, self.mem_size, self.code_size), 'recs') # clinical records
                self.ph_demo = tf.placeholder(tf.float32, (self.batch_size, self.demo_comm_num), 'demo') # demo

                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            print 'build_model_mem is starting.'
            # op_logits = self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            # self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            # # self.op_loss, self.op_loss_average, self.op_var_loss, self.op_mean_loss, self.op_same_var, self.op_diff_var = self.loss(op_logits, self.ph_labels, self.regularization)

        # adding start
            # op_logits = self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            # self.final_out_0 = self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            # # self.final_out = tf.layers.dense(self.final_out_0, 2)
            #
            # self.final_out= self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            #
            # # self.op_loss = self.inference(self.ph_data, self.ph_recs, self.ph_dropout)
            #
            # # self.op_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.ph_labels, logits=self.final_out)  # compute cost
            #
            # self.op_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.ph_labels, logits=self.final_out)  # compute cost
        # adding end

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_recs, self.ph_demo, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            # self.op_loss, self.op_loss_average, self.op_var_loss, self.op_mean_loss, self.op_same_var, self.op_diff_var = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()


    def inference(self, data, recs, demo, dropout):
        # """

        # TODO: optimizations for sparse data
        # logits, represent, prob = self._inference(data, recs, dropout)
        logits = self._inference(data, recs, demo, dropout)


        # return logits, represent, prob

        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""

        # print '&&&&&&', logits

        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average


    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class siamese_cgcnn_mem(base_model):
    """
    eps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, time_step, clinical_words, max_sentence_num, max_sentence_length,demo_comm_num, demo_comm_num_dim, fdim, K, p, M, fin, n_words, mem_size, code_size, edim, nhops, distance, method='GCN', filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0.6, batch_size=100, eval_frequency=200, patience=10, init_std=0.05,
                dir_name=''):
        # super().__init__()

        # Verify the consistency
        self.regularizers = []
        # assert fdim == edim

        # M_0 = L[0].shape[0]
        M_0 = max_sentence_num
        M_1 = max_sentence_length

        self.M_0 = M_0
        self.M_1 = M_1

        self.M_2 = 32

        self.demo_comm_num = demo_comm_num
        self.demo_comm_num_dim = demo_comm_num_dim


        # j = 0
        # self.L = []
        # for pp in p:
        #     self.L.append(L[j])
        #     j += int(np.log2(pp)) if pp > 1 else 0
        # L = self.L

        # Store attributes and bind operations.
        self.distance = distance
        # self.L, self.fdim, self.K, self.p, self.M, self.fin = L, fdim, K, p, M, fin #  hyper-parameters
        self.fdim, self.K, self.p, self.M, self.fin = fdim, K, p, M, fin #  hyper-parameters

        self.n_words, self.mem_size, self.code_size, self.edim = n_words, mem_size, code_size, edim # memory hyper-parameters

        self.n_nodes = 1

        self.num_epochs, self.learning_rate, self.patience = num_epochs, learning_rate, patience
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.method = method
        # self.filter = getattr(self, filter)
        # self.brelu = getattr(self, brelu)
        # self.pool = getattr(self, pool)
        self.init_std = init_std
        self.nhops = nhops

        # for clinical notes
        self.time_step = time_step
        self.clinical_words = clinical_words

        # Build the computational graph with memory network.
        # print '%%%%%%%%%%%%'
        self.build_gcn_graph_mem(M_0,M_1)

    def build_var(self):  # memory
        self.A = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        self.image = tf.Variable(tf.random_normal([self.batch_size, self.M_0, self.M_1], stddev=self.init_std))
        # self.image_2 = tf.Variable(tf.random_normal([self.batch_size, self.M_0, self.M_2], stddev=self.init_std))


    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        # print 'x.shape', x.shape

        N, Min = x.get_shape()

        x = tf.squeeze(x)

        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference_Hie_lstm(self, x_0): # x_0 [batch,time step,clinical_words], [batch_size, M_0, M_1]

        self.image = x_0
        # self.image_2 = c_inference_lstm(x_0)
        with tf.variable_scope('lstm_1'):

            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=200)
            outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                rnn_cell,  # cell you have chosen
                self.image,  # input
                initial_state=None,  # the initial hidden state
                dtype=tf.float32,  # must given if set initial_state = None
                time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
            )
        with tf.variable_scope('lstm_2'):
            rnn_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
            outputs_2, (h_c_2, h_n_2) = tf.nn.dynamic_rnn(
                rnn_cell_2,  # cell you have chosen
                outputs,  # input
                initial_state=None,  # the initial hidden state
                dtype=tf.float32,  # must given if set initial_state = None
                time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
            )

        x_00 = h_n_2
        # x_00 = h_n
        # print '********,outputs.shape', outputs.shape
        # print '********,outputs_2.shape', outputs_2.shape
        # print x_00.shape

        return x_00



    # def _inference_Hie_lstm(self, x_0): # x_0 [batch,time step, clinical_words], [batch_size, M_0, M_1]
    #     # lstm
    #     self.image = x_0
    #
    #     han = model_Hie.HAN(vocab_size=190, # vocab_size", 190,
    #                     num_classes=2,
    #                     embedding_size=128,  #embedding_size 200
    #                     hidden_size=64) # hidden_size
    #
    #     rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=32)
    #     outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    #         rnn_cell,  # cell you have chosen
    #         self.image,  # input
    #         initial_state=None,  # the initial hidden state
    #         dtype=tf.float32,  # must given if set initial_state = None
    #         time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
    #     )
    #     x_00 = h_n  # x_00 = outputs[:,-1,:]
    #
    #     # print '********,x_00.shape'
    #     # print x_00.shape
    #
    #     x_00 = han.out
    #
    #     return x_00

    def build_memory(self, recs_0):
        with tf.variable_scope("memory"):
            Ain_0 = tf.nn.embedding_lookup(self.A, recs_0) # recs_0 size is (batch_size, mem_size, code_size)
            Ain_0 = tf.reduce_sum(Ain_0, 2)

            Bin_0 = tf.nn.embedding_lookup(self.B, recs_0)
            Bin_0 = tf.reduce_sum(Bin_0, 2)

            Ain = Ain_0
            Bin = Bin_0

        return Ain, Bin

    def _inference_memory(self, y_0, Ain_0, Bin_0):

        # y_0 shape [batch_size, num_units]

        # print '%%%%%%%%%%%%%%------'
        # print self.n_nodes,self.edim
        # print y_0
        # print 'y_0.shape', y_0.shape
        #
        # print 'Ain_0.shape', Ain_0.shape
        # print 'Ain_0.shape', Bin_0.shape

        w_str = 0.6
        w_unstr = 0.4   # w_str + w_unstr = = 1.0


        # compute weights for attention
        hid3dim_0 = tf.reshape(y_0, [-1, self.n_nodes, self.edim])  # self.edim = num_units
        Aout_0 = tf.matmul(hid3dim_0, Ain_0, adjoint_b=True)

        Aout3dim_0 = tf.reshape(Aout_0, [-1, self.n_nodes, self.mem_size])
        P_0 = tf.nn.softmax(Aout3dim_0) # batch_size x n_nodes x mem_size

        # output memory
        probs3dim_0 = tf.reshape(P_0, [-1, self.n_nodes, self.mem_size])
        Bout_0 = tf.matmul(probs3dim_0, Bin_0) # Bout_0 size is (batch_size, n_nodes, edim)
        Bout3dim_0 = tf.reshape(Bout_0, [-1, self.n_nodes, self.edim])

        # compute the output
        # batch, n_nodes = y_0.get_shape() # (batch, n_nodes, edim)
        y_0 = tf.reshape(y_0, [-1, self.edim])
        Cout_0 = tf.matmul(y_0, self.C)
        Cout_0 = tf.reshape(Cout_0, [-1, self.n_nodes, self.edim])

        Dout_0 = tf.add(w_str*Cout_0, w_unstr*Bout3dim_0)

        Dout = tf.reshape(Dout_0, [-1, self.edim])

        return Dout

    def _inference(self, x, recs, demo, dropout): # ru kou
       # xu add
        self.build_var()
        u = x   # x image, queries -> clinical notes;
        recs = recs
        # print 'recs.shape', recs.shape
        # print recs
        Ain, Bin, = self.build_memory(recs)

        for ihop in range(self.nhops):
            if ihop ==0:
                y = self._inference_Hie_lstm(u)  # y shape [batch_size, num_units]
                # y = self._inference_Hie_lstm

                # print 'hop = 000'
            else:
                y = u

            u = self._inference_memory(y, Ain, Bin)

        u_ = u # shape [batch_size, num_units]

        # print 'u.shape',u, u.shape
        #
        # print 'demo.shape1',demo, demo.shape

        demo = tf.reshape(demo, [-1, self.demo_comm_num])

        # print 'demo.shape2', demo, demo.shape

        u_ = tf.concat([u_, demo],1)

        # print 'u_.shape',u_.shape



        #
        # for i, M in enumerate(self.M[:-1]):
        #     with tf.variable_scope('fc{}'.format(i+1)):
        #         u_ = self.fc(u_, M)
        #         u_ = tf.nn.dropout(u_, dropout)

        # with tf.variable_scope('fc{}'.format(1)):
        #     u_ = self.fc(u_, self.M[:-1])
        #     u_ = tf.nn.dropout(u_, dropout)

        # Logits linear layer, i.e. softmax without normalization.

        # with tf.variable_scope('logits'):
        #     prob = self.fc(u_, self.M, relu=False)

# start adding
        with tf.variable_scope('logits'):
            # out_2 = layers.fully_connected(inputs=u_, num_outputs=2, activation_fn=None)
            out_2 = self.fc(u_, 2)

            out_2 = tf.nn.dropout(out_2, dropout)

        # prob = tf.argmax(out_2, axis=1, name='predict')

        prob = out_2

# end adding

        # return prob
        # print 'u_.shape and prob are :', u_.shape, prob
        # return prob, u_
        return prob



