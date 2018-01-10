"""
Implementation of Dynamic Memory Networks (DyMenMM)

Reference
1. https://arxiv.org/pdf/1506.07285.pdf

cuteboydot@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import pickle
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from utils import data_loader

np.core.arrayprint._line_width = 1000
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

file_data_train = "./data/qa10_indefinite-knowledge_train.txt"
file_data_test = "./data/qa10_indefinite-knowledge_test.txt"
#file_data_train = "./data/qa11_basic-coreference_train.txt"
#file_data_test = "./data/qa11_basic-coreference_test.txt"

file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_list_train = "./data/data_list_train.bin"
file_data_idx_list_train = "./data/data_idx_list_train.bin"
file_data_list_test = "./data/data_list_test.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
dir_summary = "./model/summary/"

pre_trained = 0
my_device = "/cpu:0"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(now)

if pre_trained == 0:
    print("Load data file & make vocabulary...")

    data_list_train, data_list_test, data_idx_list_train, data_idx_list_test, rdic, dic, max_len = \
        data_loader(file_data_train, file_data_test)
    max_story_len = max_len[0]
    max_words_len = max_len[1]

    # save dictionary
    with open(file_data_list_train, 'wb') as handle:
        pickle.dump(data_list_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list_train, 'wb') as handle:
        pickle.dump(data_idx_list_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_list_test, 'wb') as handle:
        pickle.dump(data_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list_test, 'wb') as handle:
        pickle.dump(data_idx_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic, 'wb') as handle:
        pickle.dump(rdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_max_len, 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Load vocabulary from model file...")

    # load dictionary
    with open(file_data_list_train, 'rb') as handle:
        data_list_train = pickle.load(handle)
    with open(file_data_idx_list_train, 'rb') as handle:
        data_idx_list_train = pickle.load(handle)
    with open(file_data_list_test, 'rb') as handle:
        data_list_test = pickle.load(handle)
    with open(file_data_idx_list_test, 'rb') as handle:
        data_idx_list_test = pickle.load(handle)
    with open(file_dic, 'rb') as handle:
        dic = pickle.load(handle)
    with open(file_rdic, 'rb') as handle:
        rdic = pickle.load(handle)
    with open(file_max_len, 'rb') as handle:
        max_len = pickle.load(handle)
    max_story_len = max_len[0]
    max_words_len = max_len[1]

print("len(data_list_train) = %d" % len(data_idx_list_train))
print("len(data_list_test) = %d" % len(data_idx_list_test))
print("len(dic) = %d" % len(dic))
print("max_story_len = %d" % max_story_len)
print("max_words_len = %d" % max_words_len)
print()

print("data_list_train[0] example")
print(data_list_train[0])
print("data_idx_list_train[0] example")
print(data_idx_list_train[0])
print("rdic example")
print(rdic[:20])
print()


def generate_batch(size):
    np.random.seed(int(time()))
    assert size <= len(data_idx_list_train)

    data_story = np.zeros((size, max_story_len, max_words_len), dtype=np.int)
    data_story_mask = np.zeros((size, max_story_len), dtype=np.int)
    data_story_len = np.zeros(size, dtype=np.int)
    data_question = np.zeros((size, max_words_len), dtype=np.int)
    data_question_len = np.zeros(size, dtype=np.int)
    data_answer = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_train)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        sto = data_idx_list_train[idx][0]
        que = data_idx_list_train[idx][1]
        ans = data_idx_list_train[idx][2]

        for b in range(len(sto)):
            m = sto[b] + [dic["<nil>"]] * (max_words_len - len(sto[b]))
            data_story[a][b] = m
            data_story_mask[a][b] = 1

        data_story_len[a] = len(sto)
        data_question[a] = que + [dic["<nil>"]] * (max_words_len - len(que))
        data_question_len[a] = len(que)
        data_answer[a] = ans

    return data_story, data_story_mask, data_story_len, data_question, data_question_len, data_answer


def generate_test_batch(size):
    np.random.seed(int(time()))
    assert size <= len(data_idx_list_test)

    data_story = np.zeros((size, max_story_len, max_words_len), dtype=np.int)
    data_story_mask = np.zeros((size, max_story_len), dtype=np.int)
    data_story_len = np.zeros(size, dtype=np.int)
    data_question = np.zeros((size, max_words_len), dtype=np.int)
    data_question_len = np.zeros(size, dtype=np.int)
    data_answer = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_test)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        sto = data_idx_list_test[idx][0]
        que = data_idx_list_test[idx][1]
        ans = data_idx_list_test[idx][2]

        for b in range(len(sto)):
            m = sto[b] + [dic["<nil>"]] * (max_words_len - len(sto[b]))
            data_story[a][b] = m
            data_story_mask[a][b] = 1

        data_story_len[a] = len(sto)
        data_question[a] = que + [dic["<nil>"]] * (max_words_len - len(que))
        data_question_len[a] = len(que)
        data_answer[a] = ans

    return data_story, data_story_mask, data_story_len, data_question, data_question_len, data_answer


MAX_STORY_LEN = max_story_len
MAX_WORDS_LEN = max_words_len
SIZE_VOC = len(rdic)
SIZE_MEMORY_STEP = 4
SIZE_EMBED_DIM = 20
SIZE_STATE_DIM = 30
SIZE_ATT_DIM = 30
SIZE_HIDDEN_DIM = 20


def position_encoding():
    pos_enc = np.ones((SIZE_EMBED_DIM, MAX_WORDS_LEN), dtype=np.float32)

    for a in range(1, SIZE_EMBED_DIM):
        for b in range(1, MAX_WORDS_LEN):
            pos_enc[a][b] = (1 - b / MAX_WORDS_LEN) - (a / SIZE_EMBED_DIM) * (1 - 2 * b / MAX_WORDS_LEN)
    pos_enc = np.transpose(pos_enc)
    return pos_enc


with tf.Graph().as_default():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Build Graph...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        with tf.device(my_device):
            story = tf.placeholder(tf.int32, [None, MAX_STORY_LEN, MAX_WORDS_LEN], name="story")
            story_mask = tf.placeholder(tf.int32, [None, MAX_STORY_LEN], name="story_mask")
            story_len = tf.placeholder(tf.int32, [None, ], name="stroy_len")
            question = tf.placeholder(tf.int32, [None, MAX_WORDS_LEN], name="question")
            question_len = tf.placeholder(tf.int32, [None, ], name="question_len")
            answer = tf.placeholder(tf.int32, [None, ], name="answer")
            one_hot_answer = tf.one_hot(answer, SIZE_VOC, name="one_hot_answer")
            keep_prob = tf.placeholder(tf.float32, [None, ], name="keep_prob")

            global_step = tf.Variable(0, name="global_step", trainable=False)
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            # Position Encoding
            pe = position_encoding()

            with tf.name_scope("embedding_layer"):
                embed_nil = tf.zeros([1, SIZE_EMBED_DIM], name="nil_emb")

                embeddings_tmp = tf.Variable(tf.random_normal([SIZE_VOC - 1, SIZE_EMBED_DIM], mean=0, stddev=0.1))
                embeddings_tmp2 = tf.concat([embed_nil, embeddings_tmp], axis=0)
                embeddings = tf.Variable(embeddings_tmp2, name="embeddings")

                embed_input = tf.nn.embedding_lookup(embeddings, story, name="embed_input")
                embed_quest = tf.nn.embedding_lookup(embeddings, question, name="embed_quest")
                embed_answer = tf.nn.embedding_lookup(embeddings, answer, name="embed_answer")

            with tf.variable_scope("input_module"):
                rnn_input = tf.reduce_sum(embed_input * pe, axis=2)                         # [batch, story, embed]
                rnn_outputs_input, _ = bi_rnn(GRUCell(SIZE_STATE_DIM), GRUCell(SIZE_STATE_DIM),
                                              inputs=rnn_input, sequence_length=story_len, dtype=tf.float32)

                c = tf.reduce_sum(tf.stack(rnn_outputs_input), axis=0, name="c")            # [batch, story, state]

            with tf.variable_scope("question_module"):
                _, rnn_state_quest = bi_rnn(GRUCell(SIZE_STATE_DIM), GRUCell(SIZE_STATE_DIM),
                                            inputs=embed_quest, sequence_length=question_len, dtype=tf.float32)
                q = tf.reduce_sum(tf.stack(rnn_state_quest), axis=0, name="q")              # [batch, state]

            with tf.variable_scope("episodic_memory_module") as scope:
                #concat_feature_size = SIZE_STATE_DIM * 9
                concat_feature_size = SIZE_STATE_DIM * 7

                W_b = tf.get_variable("W_b", [SIZE_STATE_DIM, 1])
                Att_W1 = tf.get_variable("Att_W1", [concat_feature_size, SIZE_ATT_DIM])
                Att_b1 = tf.get_variable("Att_b1", [SIZE_ATT_DIM])
                Att_W2 = tf.get_variable("Att_W2", [SIZE_ATT_DIM, 1])
                Att_b2 = tf.get_variable("Att_b2", [1])
                gru_episode = GRUCell(SIZE_STATE_DIM)
                gru_memory = GRUCell(SIZE_STATE_DIM)

                m = q   # memory
                attention_list = []
                for k in range(SIZE_MEMORY_STEP):
                    c_0 = tf.reshape(c[:, 0, :], [-1, SIZE_STATE_DIM])
                    h = tf.zeros_like(c_0)                                                      # [batch, state]
                    final_h = tf.zeros_like(h)

                    m_prev = m
                    attention_context_list = []
                    for t in range(max_story_len):
                        c_t = tf.reshape(c[:, t, :], [-1, SIZE_STATE_DIM], name="c_t")          # [batch, state]

                        '''''''''
                        feature = [c_t,
                                   m_prev,
                                   q,
                                   c_t * q,
                                   c_t * m_prev,
                                   tf.abs(c_t - q),
                                   tf.abs(c_t - m_prev),
                                   tf.matmul(c_t, W_b) * q,
                                   tf.matmul(c_t, W_b) * m_prev]
                        '''''''''
                        feature = [c_t,
                                   m_prev,
                                   q,
                                   c_t * q,
                                   c_t * m_prev,
                                   (c_t - q) * (c_t - q),
                                   (c_t - m_prev) * (c_t - m_prev)]
                        z = tf.concat(feature, 1, name="z")                                     # [batch, state * 9]

                        feature_dim = z.get_shape()[1]
                        assert feature_dim == concat_feature_size

                        g = tf.tanh(tf.matmul(z, Att_W1) + Att_b1)                              # [batch, att_dim]
                        g = tf.sigmoid(tf.matmul(g, Att_W2) + Att_b2, name="g")                 # [batch, 1]

                        attention_context_list.append(g)

                        h_prev = h

                        episode_out, episode_state = gru_episode(c_t, h_prev)

                        h = g * episode_state + (1 - g) * h                                     # [batch, state]

                        condition_where = (t >= story_len)
                        h = tf.where(condition_where, tf.zeros_like(h), h)
                        final_h = tf.where(condition_where, final_h, h)

                        scope.reuse_variables()

                    e = final_h

                    # memory update..
                    _, m = gru_memory(e, m_prev)

                    # save attention prob
                    attention_context = tf.convert_to_tensor(attention_context_list)            # [story, batch, 1]
                    attention_context = tf.reshape(attention_context, [max_story_len, -1])      # [story, batch]
                    attention_context = tf.transpose(attention_context, [1, 0])                 # [batch, story]
                    mask_nil = tf.cast(story_mask, dtype=tf.bool)                               # [batch, story]
                    attention_context = tf.where(mask_nil, attention_context, -9999 * tf.ones_like(attention_context))
                    attention_context = tf.nn.softmax(attention_context)                        # [batch, story]
                    attention_list.append(attention_context)

                # summary attention prob
                attention_map = tf.convert_to_tensor(attention_list)                            # [hop, batch, story]
                attention_map = tf.transpose(attention_map, [1, 2, 0], name="attention_map")    # [batch, story, hop]

            with tf.variable_scope("answer_module"):
                W_a = tf.get_variable("W_a", [SIZE_STATE_DIM, SIZE_VOC])
                b_a = tf.get_variable("b_a", [SIZE_VOC])

                hypothesis = tf.matmul(m, W_a) + b_a

            with tf.name_scope("train_optimizer"):
                variables = tf.trainable_variables()

                loss = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=tf.cast(one_hot_answer, tf.float32))
                loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * 0.001
                loss = tf.reduce_sum(loss, name="loss") + loss_l2

                train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

            with tf.name_scope("predict_and_accuracy"):
                predict = tf.cast(tf.argmax(hypothesis, 1), tf.int32)
                correct = tf.equal(answer, predict)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            out_dir = os.path.abspath(os.path.join("./model", timestamp))
            print("LOGDIR = %s" % out_dir)
            print()

        # Summaries
        loss_summary = tf.summary.scalar("loss", loss)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_dir = os.path.join(out_dir, "summary", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model-step")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        if pre_trained == 2:
            print("Restore model file...")
            file_model = "./model/2018-01-04 20:44/checkpoints/"
            saver.restore(sess, tf.train.latest_checkpoint(file_model))

        BATCHS = 75
        BATCHS_TEST = len(data_idx_list_test)
        EPOCHS = 150
        STEPS = int(len(data_idx_list_train) / BATCHS)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        print(now)
        print("Train start!!")

        loop_step = 0
        for epoch in range(EPOCHS):
            for step in range(STEPS):
                data_s, data_s_mask, data_s_len, data_q, data_q_len, data_a = generate_batch(BATCHS)

                feed_dict = {
                    story: data_s,
                    story_mask: data_s_mask,
                    story_len: data_s_len,
                    question: data_q,
                    question_len: data_q_len,
                    answer: data_a
                }

                _, batch_loss, batch_acc, train_sum, g_step = \
                    sess.run([train_op, loss, accuracy, train_summary_op, global_step], feed_dict)

                if loop_step % 50 == 0:
                    train_summary_writer.add_summary(train_sum, g_step)

                if loop_step % 100 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] glob_step[%05d] - batch_loss:%.4f, batch_acc:%.4f (%s) " %
                          (epoch, g_step, batch_loss, batch_acc, now))

                # test
                if loop_step % 100 == 0:
                    data_s, data_s_mask, data_s_len, data_q, data_q_len, data_a = generate_test_batch(BATCHS_TEST)

                    feed_dict = {
                        story: data_s,
                        story_mask: data_s_mask,
                        story_len: data_s_len,
                        question: data_q,
                        question_len: data_q_len,
                        answer: data_a
                    }

                    pred, test_loss, test_acc, test_sum, g_step = \
                        sess.run([predict, loss, accuracy, test_summary_op, global_step], feed_dict)
                    test_summary_writer.add_summary(test_sum, g_step)

                if loop_step % 200 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] glob_step[%05d] - test_loss:%.4f, test_acc:%.4f (%s)" %
                          (epoch, g_step, test_loss, test_acc, now))

                if loop_step % 5000 == 0:
                    print("INTERMEDIATES TEST")
                    cnt = 0
                    for a in range(100):
                        print("[%03d]: Y=%02d, Y^=%02d  " % (a, data_a[a], pred[a]), end="")
                        if data_a[a] == pred[a]:
                            cnt += 1
                        if (a + 1) % 5 == 0:
                            print()
                    print("INTERMEDIATES TEST ACC : %02d " % cnt)

                loop_step += 1
            saver.save(sess, checkpoint_prefix, global_step=g_step)

        print("Train finished..")
        print()

        print("TEST EXAMPLE")
        data_s, data_s_mask, data_s_len, data_q, data_q_len, data_a = generate_test_batch(BATCHS_TEST)

        feed_dict = {
            story: data_s,
            story_mask: data_s_mask,
            story_len: data_s_len,
            question: data_q,
            question_len: data_q_len,
            answer: data_a
        }

        pred, attention = sess.run([predict, attention_map], feed_dict)
        print(attention.shape)

        for a in range(min(50, BATCHS_TEST)):
            print("Test [%03d]" % (a + 1))
            for b in range(max_story_len):
                if data_s_mask[a][b] == 0 :
                    break

                print("Story [#%02d] " % b, end="")
                sen = [rdic[w] for w in data_s[a][b]]
                hop_prob = attention[a][b]

                str_sen = "{0:65}".format(str(sen))
                str_att = "{0:60}".format(str(hop_prob))
                print(" %s, %s" % (str_sen, str_att))

            que = [rdic[q] for q in data_q[a]]
            print("Question : %s" % str(que))

            ans = rdic[data_a[a]]
            pre = rdic[pred[a]]
            print("Answer : %s, \t Predict : %s" % (ans, pre))