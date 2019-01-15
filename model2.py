import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
from utils import get_batch_index


class IAN(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.embedding_matrix = config.embedding_matrix
        self.sess = sess

    def build_model(self):
        with tf.name_scope('inputs'):
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len], name='aspects')
            self.lcontexts = tf.placeholder(tf.int32, [None, self.max_context_len], name='lcontexts')
            self.rcontexts = tf.placeholder(tf.int32, [None, self.max_context_len], name='rcontexts')
            self.labels = tf.placeholder(tf.int32, [None, self.n_class], name='labels')
            self.aspect_lens = tf.placeholder(tf.int32, None, name='aspect_lens')
            self.lcontext_lens = tf.placeholder(tf.int32, None, name='lcontext_lens')
            self.rcontext_lens = tf.placeholder(tf.int32, None, name='rcontext_lens')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

            lcontext_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.lcontexts)
            lcontext_inputs = tf.cast(lcontext_inputs, tf.float32)
            lcontext_inputs = tf.nn.dropout(lcontext_inputs, keep_prob=self.dropout_keep_prob)

            rcontext_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.rcontexts)
            rcontext_inputs = tf.cast(rcontext_inputs, tf.float32)
            rcontext_inputs = tf.nn.dropout(rcontext_inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('weights'):
            weights = {
                'laspect_score': tf.get_variable(
                    name='W_la',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'raspect_score': tf.get_variable(
                    name='W_ra',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'lcontext_score': tf.get_variable(
                    name='W_lc',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'rcontext_score': tf.get_variable(
                    name='W_rc',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 4, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'laspect_score': tf.get_variable(
                    name='B_la',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'raspect_score': tf.get_variable(
                    name='B_ra',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'lcontext_score': tf.get_variable(
                    name='B_lc',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'rcontext_score': tf.get_variable(
                    name='B_rc',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('dynamic_rnn'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.n_hidden),
                inputs=aspect_inputs,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            batch_size = tf.shape(aspect_outputs)[0]
            aspect_avg = tf.reduce_mean(aspect_outputs, 1)

            lcontext_outputs, lcontext_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.n_hidden),
                inputs=lcontext_inputs,
                sequence_length=self.lcontext_lens,
                dtype=tf.float32,
                scope='lcontext_lstm'
            )
            lcontext_avg = tf.reduce_mean(lcontext_outputs, 1)

            rcontext_outputs, rcontext_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.n_hidden),
                inputs=rcontext_inputs,
                sequence_length=self.rcontext_lens,
                dtype=tf.float32,
                scope='rcontext_lstm'
            )
            rcontext_avg = tf.reduce_mean(rcontext_outputs, 1)

            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs)
            lcontext_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            lcontext_avg_iter = lcontext_avg_iter.unstack(lcontext_avg)
            rcontext_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            rcontext_avg_iter = rcontext_avg_iter.unstack(rcontext_avg)
            aspect_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            aspect_lens_iter = aspect_lens_iter.unstack(self.aspect_lens)
            laspect_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            laspect_att = tf.TensorArray(size=batch_size, dtype=tf.float32)
            raspect_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            raspect_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, laspect_rep, laspect_att, raspect_rep, raspect_att):
                a = aspect_outputs_iter.read(i)
                b1 = lcontext_avg_iter.read(i)
                b2 = rcontext_avg_iter.read(i)
                l = math_ops.to_int32(aspect_lens_iter.read(i))

                laspect_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['laspect_score']), tf.reshape(b1, [-1, 1])) + biases[
                        'laspect_score']), [1, -1])
                laspect_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(laspect_score, [0, 0], [1, l])), tf.zeros([1, self.max_aspect_len - l])], 1)
                laspect_att = laspect_att.write(i, laspect_att_temp)
                laspect_rep = laspect_rep.write(i, tf.matmul(laspect_att_temp, a))

                raspect_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['raspect_score']), tf.reshape(b2, [-1, 1])) + biases[
                        'raspect_score']), [1, -1])
                raspect_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(raspect_score, [0, 0], [1, l])), tf.zeros([1, self.max_aspect_len - l])], 1)
                raspect_att = raspect_att.write(i, raspect_att_temp)
                raspect_rep = raspect_rep.write(i, tf.matmul(raspect_att_temp, a))
                return (i + 1, laspect_rep, laspect_att, raspect_rep, raspect_att)

            def condition(i, laspect_rep, laspect_att, raspect_rep, raspect_att):
                return i < batch_size

            _, laspect_rep_final, laspect_att_final, raspect_rep_final, raspect_att_final = tf.while_loop(
                cond=condition, body=body, loop_vars=(0, laspect_rep, laspect_att, raspect_rep, raspect_att))
            self.laspect_atts = tf.reshape(laspect_att_final.stack(), [-1, self.max_aspect_len])
            self.laspect_reps = tf.reshape(laspect_rep_final.stack(), [-1, self.n_hidden])
            self.raspect_atts = tf.reshape(raspect_att_final.stack(), [-1, self.max_aspect_len])
            self.raspect_reps = tf.reshape(raspect_rep_final.stack(), [-1, self.n_hidden])

            lcontext_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            lcontext_outputs_iter = lcontext_outputs_iter.unstack(lcontext_outputs)
            rcontext_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            rcontext_outputs_iter = rcontext_outputs_iter.unstack(rcontext_outputs)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            lcontext_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            lcontext_lens_iter = lcontext_lens_iter.unstack(self.lcontext_lens)
            rcontext_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            rcontext_lens_iter = rcontext_lens_iter.unstack(self.rcontext_lens)
            lcontext_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            lcontext_att = tf.TensorArray(size=batch_size, dtype=tf.float32)
            rcontext_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            rcontext_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, lcontext_rep, lcontext_att, rcontext_rep, rcontext_att):
                a1 = lcontext_outputs_iter.read(i)
                a2 = rcontext_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                l1 = math_ops.to_int32(lcontext_lens_iter.read(i))
                l2 = math_ops.to_int32(rcontext_lens_iter.read(i))

                lcontext_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a1, weights['lcontext_score']), tf.reshape(b, [-1, 1])) + biases[
                        'lcontext_score']), [1, -1])
                lcontext_att_temp = tf.concat([tf.nn.softmax(tf.slice(lcontext_score, [0, 0], [1, l1])),
                                               tf.zeros([1, self.max_context_len - l1])], 1)
                lcontext_att = lcontext_att.write(i, lcontext_att_temp)
                lcontext_rep = lcontext_rep.write(i, tf.matmul(lcontext_att_temp, a1))

                rcontext_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a2, weights['rcontext_score']), tf.reshape(b, [-1, 1])) + biases[
                        'rcontext_score']), [1, -1])
                rcontext_att_temp = tf.concat([tf.nn.softmax(tf.slice(rcontext_score, [0, 0], [1, l2])),
                                               tf.zeros([1, self.max_context_len - l2])], 1)
                rcontext_att = rcontext_att.write(i, rcontext_att_temp)
                rcontext_rep = rcontext_rep.write(i, tf.matmul(rcontext_att_temp, a2))

                return (i + 1, lcontext_rep, lcontext_att, rcontext_rep, rcontext_att)

            def condition(i, lcontext_rep, lcontext_att, rcontext_rep, rcontext_att):
                return i < batch_size

            _, lcontext_rep_final, lcontext_att_final, rcontext_rep_final, rcontext_att_final = tf.while_loop(
                cond=condition, body=body, loop_vars=(0, lcontext_rep, lcontext_att, rcontext_rep, rcontext_att))
            self.lcontext_atts = tf.reshape(lcontext_att_final.stack(), [-1, self.max_context_len])
            self.lcontext_reps = tf.reshape(lcontext_rep_final.stack(), [-1, self.n_hidden])
            self.rcontext_atts = tf.reshape(rcontext_att_final.stack(), [-1, self.max_context_len])
            self.rcontext_reps = tf.reshape(rcontext_rep_final.stack(), [-1, self.n_hidden])

            self.reps = tf.concat([self.laspect_reps, self.raspect_reps, self.lcontext_reps, self.rcontext_reps], 1)
            self.predict = tf.matmul(self.reps, weights['softmax']) + biases['softmax']
            self.predict = tf.identity(self.predict, name='predict_id')

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predict, labels=self.labels), name='cost')
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                               global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32), name='accuracy')

        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        timestamp = str(int(time.time()))
        _dir = 'logs2/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(
            self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def train(self, data):
        aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens = data
        cost, cnt = 0., 0

        for sample, num in self.get_batch_data(aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens,
                                               rcontext_lens, self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op],
                                                   feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens = data
        cost, acc, cnt = 0., 0, 0

        for sample, num in self.get_batch_data(aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens,
                                               rcontext_lens, len(aspects), False, 1.0):
            loss, accuracy, step, summary = self.sess.run(
                [self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))

        aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens = train_data
        with open('analysis2/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens,
                                                   rcontext_lens, len(aspects), False, 1.0):
                laspect_atts, raspect_atts, lcontext_atts, rcontext_atts, correct_pred = self.sess.run(
                    [self.laspect_atts, self.raspect_atts, self.lcontext_atts, self.rcontext_atts, self.correct_pred],
                    feed_dict=sample)

                for a, b, c, d, e in zip(laspect_atts, raspect_atts, lcontext_atts, rcontext_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    c = str(c).replace('\n', '')
                    d = str(d).replace('\n', '')
                    f.write('%s\n%s\n%s\n%s\n%s\n' % (a, b, c, d, e))
        print('Finishing analyzing training data')

        aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens = test_data
        with open('analysis2/test_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens,
                                                   rcontext_lens, len(aspects), False, 1.0):
                laspect_atts, raspect_atts, lcontext_atts, rcontext_atts, correct_pred = self.sess.run(
                    [self.laspect_atts, self.raspect_atts, self.lcontext_atts, self.rcontext_atts, self.correct_pred],
                    feed_dict=sample)
                for a, b, c, d, e in zip(laspect_atts, raspect_atts, lcontext_atts, rcontext_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    c = str(c).replace('\n', '')
                    d = str(d).replace('\n', '')
                    f.write('%s\n%s\n%s\n%s\n%s\n' % (a, b, c, d, e))
        print('Finishing analyzing testing data')

    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())

        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        inputs = {
            "aspects": self.aspects,
            "lcontexts": self.lcontexts,
            "rcontexts": self.rcontexts,
            "labels": self.labels,
            "aspect_lens": self.aspect_lens,
            "lcontext_lens": self.lcontext_lens,
            "rcontext_lens": self.rcontext_lens,
            "dropout_keep_prob": self.dropout_keep_prob,
        }
        outputs = {
            "cost": self.cost,
            "accuracy": self.accuracy,
            "predict": self.predict,
        }

        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models2/model_iter', global_step=step)
                tf.saved_model.simple_save(self.sess, 'models2/lr_iter_' + str(step), inputs, outputs)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; test-loss=%.6f; test-acc=%.6f;' % (
            str(i), train_loss, train_acc, test_loss, test_acc))
        saver.save(self.sess, 'models2/model_final')
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))

        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models2/'))
        self.analysis(train_data, test_data)

    def get_batch_data(self, aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens,
                       batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(aspects), batch_size, is_shuffle):
            feed_dict = {
                self.aspects: aspects[index],
                self.lcontexts: lcontexts[index],
                self.rcontexts: rcontexts[index],
                self.labels: labels[index],
                self.aspect_lens: aspect_lens[index],
                self.lcontext_lens: lcontext_lens[index],
                self.rcontext_lens: rcontext_lens[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)