import tensorflow as tf
from utils import get_data_info, read_data_lr, load_word_embeddings, get_batch_index
from model2 import IAN
from tensorflow.python.saved_model import tag_constants
import numpy as np
import os
import time
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#=====================first run=======================
word2id_file = open('word2id2.obj','wb')
train_data_file = open('train_data2.obj','wb')
test_data_file = open('test_data2.obj','wb')
embedding_file = open('embedding_file2.obj','wb')


#=====================second run=======================
#word2id_file = open('word2id2.obj','rb')
#train_data_file = open('train_data2.obj','rb')
#test_data_file = open('test_data2.obj','rb')
#embedding_file = open('embedding_file2.obj','rb')


# word2id_file = open('word2id2l.obj','rb')
# train_data_file = open('train_data2l.obj','rb')
# test_data_file = open('test_data2l.obj','rb')
# embedding_file = open('embedding_file2l.obj','rb')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 10, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_file_name', 'data/glove.840B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('dataset', 'data/laptop/', 'the directory of dataset')

tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')


def main(_):
    start_time = time.time()

    #=====================first run=======================
    print('Loading data info ...')
    word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(FLAGS.dataset, FLAGS.pre_processed)
    print(FLAGS.max_aspect_len, FLAGS.max_context_len)
    pickle.dump(word2id,word2id_file)
    print('Loading training data and testing data ...')
    train_data = read_data_lr(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'train', FLAGS.pre_processed)
    test_data = read_data_lr(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'test', FLAGS.pre_processed)
    pickle.dump(train_data,train_data_file)
    pickle.dump(test_data,test_data_file)
    print('Loading pre-trained word vectors ...')
    FLAGS.embedding_matrix = load_word_embeddings(FLAGS.embedding_file_name, FLAGS.embedding_dim, word2id)
    pickle.dump(FLAGS.embedding_matrix,embedding_file)

    #=====================second run=======================
    # train_data=pickle.load(train_data_file)
    # test_data=pickle.load(test_data_file)
    # FLAGS.max_aspect_len=10
    # FLAGS.max_context_len=82
    # print('loading embeddings')
    # FLAGS.embedding_matrix = pickle.load(embedding_file)

    with tf.Session() as sess:
        #=====================first run=======================
        model = IAN(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)

    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))
    word2id_file.close()
    train_data_file.close()
    test_data_file.close()
    embedding_file.close()

def predict_sentiment(sess, test_data):
    tf.saved_model.loader.load(sess,[tag_constants.SERVING],'models2/iter_best')

    graph = tf.get_default_graph()
    for v in tf.get_default_graph().as_graph_def().node:
        print(v.name)
    selfaspects=graph.get_tensor_by_name('inputs/aspects:0')
    selflcontexts=graph.get_tensor_by_name('inputs/lcontexts:0')
    selfrcontexts=graph.get_tensor_by_name('inputs/rcontexts:0')
    selflabels=graph.get_tensor_by_name('inputs/labels:0')
    selfaspect_lens=graph.get_tensor_by_name('inputs/aspect_lens:0')
    selflcontext_lens=graph.get_tensor_by_name('inputs/lcontext_lens:0')
    selfrcontext_lens=graph.get_tensor_by_name('inputs/rcontext_lens:0')
    selfdropout_keep_prob=graph.get_tensor_by_name('inputs/dropout_keep_prob:0')

    selfcost=graph.get_tensor_by_name('loss/cost:0')
    selfaccuracy=graph.get_tensor_by_name('predict/accuracy:0')
    selfpredict=graph.get_tensor_by_name('dynamic_rnn/predict_id:0')

    cost, acc, cnt = 0., 0, 0
    aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens = test_data
    with open('predict2/test_demo.txt', 'w') as f:
        for sample, num in get_batch_data(selfaspects, selflcontexts, selfrcontexts, selflabels, selfaspect_lens, selflcontext_lens, selfrcontext_lens, selfdropout_keep_prob, aspects, lcontexts, rcontexts,labels, aspect_lens, lcontext_lens, rcontext_lens, len(aspects), False, 1.0):
            loss, accuracy, predict = sess.run([selfcost, selfaccuracy, selfpredict], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num
            for pred in predict:
                index=np.argmax(pred)
                sent=''
                if index == 0:
                    sent='negative'
                elif index==1:
                    sent='neutral'
                else:
                    sent='positive'
                f.write('%s\n' % (sent,))

        f.write('%f\n%f\n' % (acc/cnt,cost/cnt))
    print('Finishing analyzing testing data')

def get_batch_data(selfaspects, selflcontexts, selfrcontexts, selflabels, selfaspect_lens, selflcontext_lens, selfrcontext_lens, selfdropout_keep_prob, aspects, lcontexts, rcontexts, labels, aspect_lens, lcontext_lens, rcontext_lens, batch_size, is_shuffle, keep_prob):
    for index in get_batch_index(len(aspects), batch_size, is_shuffle):
        feed_dict = {
            selfaspects: aspects[index],
            selflcontexts: lcontexts[index],
            selfrcontexts: rcontexts[index],
            selflabels: labels[index],
            selfaspect_lens: aspect_lens[index],
            selflcontext_lens: lcontext_lens[index],
            selfrcontext_lens: rcontext_lens[index],
            selfdropout_keep_prob: keep_prob,
        }
        yield feed_dict, len(index)

if __name__ == '__main__':
    tf.app.run()