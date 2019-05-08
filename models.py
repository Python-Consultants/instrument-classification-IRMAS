from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
from data_utils import read_file
import os



class FC(object):

    def __init__(self, config):
        self.lr = config.lr
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.dropout = config.dropout

        self.input = tf.placeholder(tf.float32, shape=[None, None, None])
        self.label = tf.placeholder(tf.int32, shape=[None, None])
        self.dropout_rate = tf.placeholder(tf.float32)

        inputs = tf.reshape(self.input, [-1, config.feature_dim * config.frame_num])
        inputs = tf.nn.dropout(inputs, self.dropout_rate)

        w = tf.get_variable("cnn_weight", shape=[config.feature_dim * config.frame_num, 11])
        b = tf.get_variable("cnn_bias", shape=[11], initializer=tf.constant_initializer(0.1))
        self.prediction = tf.matmul(inputs, w) + b

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                              logits=self.prediction))

        global_step = tf.get_variable("cnn_global_step", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer)
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.RMSPropOptimizer(self.lr, 0.9)
        optimizer = tf.train.AdamOptimizer(self.lr, 0.9)
        self.optimize = optimizer.minimize(self.loss_op, global_step=global_step)

        self.pred_label = tf.argmax(self.label, 1)

        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.dir_model = 'fc_model/model'

    def save_model(self):
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_model(self):
        print ("Reloading the latest trained model...")
        if not os.path.exists(self.dir_model):
            print ("No trained model!")
        self.saver.restore(self.sess, self.dir_model)

    def train(self):

        data_train, target_train = read_file('train')
        data_valid, target_valid = read_file('valid')
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        temp_high = 0.0
        temp_high_epoch = 0
        for epoch in range(self.epoch_num):
            losses = []
            accs = []
            data_size = len(data_train)
            iteration = int(data_size / self.batch_size) + 1
            for i in range(iteration):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, data_size)
                inputs = data_train[start:end]
                targets = target_train[start:end]
                loss, acc, _ = self.sess.run([self.loss_op, self.accuracy, self.optimize],
                                             feed_dict={self.input: inputs,
                                                        self.label: targets,
                                                        self.dropout_rate: self.dropout})
                losses.append(loss)
                accs.append(acc)

            train_losses.append(np.mean(losses))
            train_accs.append(np.mean(accs))
            valid_loss, valid_acc = self.sess.run([self.loss_op, self.accuracy],
                                                  feed_dict={self.input: data_valid,
                                                             self.label: target_valid,
                                                             self.dropout_rate: 1.0})
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_accs[-1] > temp_high:
                temp_high = valid_accs[-1]
                temp_high_epoch = epoch
                print("new high! Epoch {}: acc {} loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))
                self.save_model()
            elif epoch == temp_high_epoch + 20:
                print ("early stop since no increase for 20 epoch!")
                break
            elif (epoch + 1) % 1 == 0:
                print("Epoch {}: acc {} loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))

        return train_losses, valid_losses

    def predict(self):

        data_test, target_test = read_file('test')
        test_loss, test_acc, test_pred = self.sess.run([self.loss_op, self.accuracy, self.pred_label],
                                                       feed_dict={self.input: data_test,
                                                                  self.label: target_test,
                                                                  self.dropout_rate: 1.0})
        print (test_pred.shape)

        print ("testing...")
        print ("acc {}  loss {}".format(test_acc, test_loss))


class CNN(object):

    def __init__(self, config):
        self.lr = config.lr
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.dropout = config.dropout

        self.input = tf.placeholder(tf.float32, shape=[None, None, None])
        self.label = tf.placeholder(tf.int32, shape=[None, None])
        self.dropout_rate = tf.placeholder(tf.float32)

        inputs = tf.expand_dims(self.input, -1)

        k = tf.get_variable("cnn_k", [2, 2, 1, 10])
        output = tf.nn.conv2d(inputs, k, [1, 1, 1, 1], "SAME")
        output = tf.nn.max_pool(output, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

        inputs = tf.reshape(output, [-1, 20 * 65 * 10])
        inputs = tf.nn.dropout(inputs, self.dropout_rate)

        w = tf.get_variable("cnn_weight", shape=[20 * 65 * 10, 11])
        b = tf.get_variable("cnn_bias", shape=[11], initializer=tf.constant_initializer(0.1))
        self.prediction = tf.matmul(inputs, w) + b

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                              logits=self.prediction))

        global_step = tf.get_variable("cnn_global_step", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer)
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.RMSPropOptimizer(self.lr, 0.9)
        optimizer = tf.train.AdamOptimizer(self.lr, 0.9)
        self.optimize = optimizer.minimize(self.loss_op, global_step=global_step)

        self.pred_label = tf.argmax(self.label, 1)

        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.dir_model = 'cnn_model/model'

    def save_model(self):
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_model(self):
        print ("Reloading the latest trained model...")
        if not os.path.exists(self.dir_model):
            print ("No trained model!")
        self.saver.restore(self.sess, self.dir_model)

    def train(self):

        data_train, target_train = read_file('train')
        data_valid, target_valid = read_file('valid')
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        temp_high = 0.0
        temp_high_epoch = 0
        for epoch in range(self.epoch_num):
            losses = []
            accs = []
            data_size = len(data_train)
            iteration = int(data_size / self.batch_size) + 1
            for i in range(iteration):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, data_size)
                inputs = data_train[start:end]
                targets = target_train[start:end]
                loss, acc, _ = self.sess.run([self.loss_op, self.accuracy, self.optimize],
                                             feed_dict={self.input: inputs,
                                                        self.label: targets,
                                                        self.dropout_rate: self.dropout})
                losses.append(loss)
                accs.append(acc)

            train_losses.append(np.mean(losses))
            train_accs.append(np.mean(accs))
            valid_loss, valid_acc = self.sess.run([self.loss_op, self.accuracy],
                                                  feed_dict={self.input: data_valid,
                                                             self.label: target_valid,
                                                             self.dropout_rate: 1.0})
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_accs[-1] > temp_high:
                temp_high = valid_accs[-1]
                temp_high_epoch = epoch
                print("new high! Epoch {}: acc {} loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))
                self.save_model()
            elif epoch == temp_high_epoch + 20:
                print ("early stop since no increase for 20 epoch!")
                break
            elif (epoch + 1) % 1 == 0:
                print("Epoch {}: acc {} loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))

        return train_losses, valid_losses

    def predict(self):

        data_test, target_test = read_file('test')
        test_loss, test_acc, test_pred = self.sess.run([self.loss_op, self.accuracy, self.pred_label],
                                                       feed_dict={self.input: data_test,
                                                                  self.label: target_test,
                                                                  self.dropout_rate: 1.0})
        print (test_pred.shape)

        print ("testing...")
        print ("acc {}  loss {}".format(test_acc, test_loss))


class RNN(object):

    def __init__(self, config):
        self.lr = config.lr
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.dropout = config.dropout

        self.input = tf.placeholder(tf.float32, shape=[None, None, config.frame_num])
        self.label = tf.placeholder(tf.int32, shape=[None, None])
        self.dropout_rate = tf.placeholder(tf.float32)

        cell = tf.nn.rnn_cell.BasicRNNCell(config.num_units)

        output, state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)

        output = output[:, -1, :]
        output = tf.nn.dropout(output, self.dropout_rate)
        inputs = tf.reshape(output, [-1, config.num_units])

        w = tf.get_variable("cnn_weight", shape=[config.num_units, 11])
        b = tf.get_variable("cnn_bias", shape=[11], initializer=tf.constant_initializer(0.1))
        self.prediction = tf.matmul(inputs, w) + b

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                              logits=self.prediction))

        global_step = tf.get_variable("rcnn_global_step", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer)
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.RMSPropOptimizer(lr, 0.9)
        optimizer = tf.train.AdamOptimizer(self.lr, 0.9)
        self.optimize = optimizer.minimize(self.loss_op, global_step=global_step)

        self.pred_label = tf.argmax(self.label, 1)

        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.dir_model = 'rnn_model/model'

    def save_model(self):
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_model(self):
        print ("Reloading the latest trained model...")
        if not os.path.exists(self.dir_model):
            print ("No trained model!")
        self.saver.restore(self.sess, self.dir_model)

    def train(self):

        data_train, target_train = read_file('train')
        data_valid, target_valid = read_file('valid')
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        temp_high = 0.0
        temp_high_epoch = 0
        for epoch in range(self.epoch_num):
            losses = []
            accs = []
            data_size = len(data_train)
            iteration = int(data_size / self.batch_size) + 1
            for i in range(iteration):
                # idx = np.random.choice(data_size, batch_size, replace=False)
                # inputs = data_train[idx]
                # targets = vola_train[idx]
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, data_size)
                inputs = data_train[start:end]
                targets = target_train[start:end]
                loss, acc, _ = self.sess.run([self.loss_op, self.accuracy, self.optimize],
                                             feed_dict={self.input: inputs,
                                                        self.label: targets,
                                                        self.dropout_rate: self.dropout})
                losses.append(loss)
                accs.append(acc)

            train_losses.append(np.mean(losses))
            train_accs.append(np.mean(accs))
            valid_loss, valid_acc = self.sess.run([self.loss_op, self.accuracy],
                                                  feed_dict={self.input: data_valid,
                                                             self.label: target_valid,
                                                             self.dropout_rate: 1.0})
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_accs[-1] > temp_high:
                temp_high = valid_accs[-1]
                temp_high_epoch = epoch
                print("new high! Epoch {}: acc {} train loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))
                self.save_model()
            elif epoch == temp_high_epoch + 20:
                print ("early stop since no increase for 20 epoch!")
                break
            elif (epoch + 1) % 1 == 0:
                print("Epoch {}: acc {} train loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))

        return train_losses, valid_losses


class RCNN(object):

    def __init__(self, config):
        self.lr = config.lr
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.dropout = config.dropout

        self.input = tf.placeholder(tf.float32, shape=[None, None, config.frame_num])
        self.label = tf.placeholder(tf.int32, shape=[None, None])
        self.dropout_rate = tf.placeholder(tf.float32)

        cell = tf.nn.rnn_cell.BasicRNNCell(config.num_units)

        output, state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)

        output = tf.nn.dropout(output, self.dropout_rate)
        output_shape = tf.shape(output)
        output = tf.expand_dims(output, -1)

        k = tf.get_variable("cnn_k", [2, 2, 1, 3])
        output = tf.nn.conv2d(output, k, [1, 1, 1, 1], "SAME")
        inputs = tf.reshape(output, [-1, 40 * config.num_units * 3])

        w = tf.get_variable("cnn_weight", shape=[40 * config.num_units * 3, 11])
        b = tf.get_variable("cnn_bias", shape=[11], initializer=tf.constant_initializer(0.1))
        self.prediction = tf.matmul(inputs, w) + b

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                              logits=self.prediction))

        global_step = tf.get_variable("rcnn_global_step", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer)
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.RMSPropOptimizer(lr, 0.9)
        optimizer = tf.train.AdamOptimizer(self.lr, 0.9)
        self.optimize = optimizer.minimize(self.loss_op, global_step=global_step)

        self.pred_label = tf.argmax(self.label, 1)

        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.dir_model = 'rcnn_model/model'

    def save_model(self):
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_model(self):
        print ("Reloading the latest trained model...")
        if not os.path.exists(self.dir_model):
            print ("No trained model!")
        self.saver.restore(self.sess, self.dir_model)

    def train(self):

        data_train, target_train = read_file('train')
        data_valid, target_valid = read_file('valid')
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        temp_high = 0.0
        temp_high_epoch = 0
        for epoch in range(self.epoch_num):
            losses = []
            accs = []
            data_size = len(data_train)
            iteration = int(data_size / self.batch_size) + 1
            for i in range(iteration):
                # idx = np.random.choice(data_size, batch_size, replace=False)
                # inputs = data_train[idx]
                # targets = vola_train[idx]
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, data_size)
                inputs = data_train[start:end]
                targets = target_train[start:end]
                loss, acc, _ = self.sess.run([self.loss_op, self.accuracy, self.optimize],
                                             feed_dict={self.input: inputs,
                                                        self.label: targets,
                                                        self.dropout_rate: self.dropout})
                losses.append(loss)
                accs.append(acc)

            train_losses.append(np.mean(losses))
            train_accs.append(np.mean(accs))
            valid_loss, valid_acc = self.sess.run([self.loss_op, self.accuracy],
                                                  feed_dict={self.input: data_valid,
                                                             self.label: target_valid,
                                                             self.dropout_rate: 1.0})
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_accs[-1] > temp_high:
                temp_high = valid_accs[-1]
                temp_high_epoch = epoch
                print("new high! Epoch {}: acc {} train loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))
                self.save_model()
            elif epoch == temp_high_epoch + 20:
                print ("early stop since no increase for 20 epoch!")
                break
            elif (epoch + 1) % 1 == 0:
                print("Epoch {}: acc {} train loss {}".format((epoch + 1), valid_accs[-1], valid_losses[-1]))

        return train_losses, valid_losses
