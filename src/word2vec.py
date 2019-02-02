import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
from scipy import spatial
from sklearn.manifold import TSNE
import re
import nltk

nltk.download('gutenberg')
from nltk.corpus import gutenberg

nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import OrderedDict


# ---------------------------------------------------------- #
# Process vocabulary
# Using gutenberg data
# Minimal pre-processing done
# ---------------------------------------------------------- #

def get_vocab():
    # Save the data - no need to preprocess every time!
    w2v_sample_vocab_file = 'w2v_sample_vocab.pkl'
    w2v_sample_word_index_file = 'w2v_sample_word_index.pkl'
    w2v_sample_corpus_file = 'w2v_sample_corpus.pkl'

    stop_words = set(stopwords.words('english'))
    vocab = None
    corpus_data = None
    word_index = None

    if os.path.exists(w2v_sample_vocab_file):
        with open(w2v_sample_vocab_file, 'rb') as f:
            vocab = pickle.load(f)

    if os.path.exists(w2v_sample_word_index_file):
        with open(w2v_sample_word_index_file, 'rb') as f:
            word_index = pickle.load(f)

    if os.path.exists(w2v_sample_corpus_file):
        with open(w2v_sample_corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)

    if word_index is not None and vocab is not None and corpus_data is not None:
        print(' Number of words ', len(vocab))
        print(' Number of sentences', len(corpus_data))
        return word_index, vocab, corpus_data

    # Data not yet present #
    # --- Use gutenberg data as sample --- #

    words = nltk.corpus.gutenberg.words('austen-emma.txt')

    sentences = gutenberg.sents('austen-emma.txt')
    words = set(words)
    words = [str(w).lower() for w in words]

    sentences = [[s.lower() for s in sent] for sent in sentences]
    words = list(sorted(words))
    valid_words = []
    for w in words:
        if w in stop_words:
            continue
        res = re.match("^[a-z]{2,}$", w)
        if res is not None:
            valid_words.append(res.group(0))
    vocab = list(sorted(set(valid_words)))
    word_index = OrderedDict({i[0]: i[1] for i in enumerate(vocab, 0)})
    inv_word_index = {v: k for k, v in word_index.items()}
    corpus_data = []
    for sent in sentences:
        _sent = []
        for s in sent:
            try:
                _sent.append(inv_word_index[s])
            except:
                # simply ignore if word not in vocab
                pass
        corpus_data.append(_sent)
    print(corpus_data[0])

    if not os.path.exists(w2v_sample_vocab_file):
        file = open(w2v_sample_vocab_file, 'wb')
        pickle.dump(vocab, file, protocol=4)
    else:
        with open(w2v_sample_vocab_file, 'rb') as f:
            vocab = pickle.load(f)

    if not os.path.exists(w2v_sample_word_index_file):
        file = open(w2v_sample_word_index_file, 'wb')
        pickle.dump(word_index, file, protocol=4)
    else:
        with open(w2v_sample_word_index_file, 'rb') as f:
            word_index = pickle.load(f)

    if not os.path.exists(w2v_sample_corpus_file):
        file = open(w2v_sample_corpus_file, 'wb')
        pickle.dump(corpus_data, file, protocol=4)
    else:
        with open(w2v_sample_corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)

    return word_index, vocab, corpus_data


# ---------------------------------------------------------- #
# Create positive and negative samples
# ---------------------------------------------------------- #
def create_data(data, vocab, ctxt_size, neg_samples, use_saved=True):
    vocab_size = len(vocab)
    x_pos = None  # set of surrounding(context nodes)
    y_pos = None  # target node
    x_neg = None  # Random nodes

    x_pos_file = 'w2v_sample_data_posSamples.pkl'
    y_pos_file = 'w2v_sample_data_word.pkl'
    x_neg_file = 'w2v_sample_data_negSamples.pkl'
    # If saved - use them!
    if (use_saved):
        if os.path.exists(x_pos_file):
            with open(x_pos_file, 'rb') as f:
                x_pos = pickle.load(f)
        if os.path.exists(y_pos_file):
            with open(y_pos_file, 'rb') as f:
                y_pos = pickle.load(f)
        if os.path.exists(x_neg_file):
            with open(x_neg_file, 'rb') as f:
                x_neg = pickle.load(f)

    if x_pos is None or y_pos is None or x_neg is None:
        x_pos = []  # set of surrounding(context nodes)
        y_pos = []  # target node
        x_neg = []  # Random nodes

        ctxt_l = ctxt_r = int(ctxt_size / 2)
        all_sentences = []
        # parse each sentence

        for sentence in data:
            if len(sentence) < ctxt_size + 1:
                continue

            for _idx in range(ctxt_l, len(sentence) - ctxt_r):
                cur = sentence[_idx]
                y_pos.append([cur])

                tmp = sentence[_idx - ctxt_l: _idx]
                tmp.extend(sentence[_idx + 1: _idx + ctxt_r + 1])
                x_pos.append(tmp)

                mult_p_idx = np.ones(vocab_size)
                exclude = list(tmp)
                exclude.append(cur)

                np.put(mult_p_idx, exclude, 0)
                mult_p_idx = mult_p_idx / np.sum(mult_p_idx)
                neg = [np.nonzero(
                    np.random.multinomial(
                        1, mult_p_idx)
                )[0][0] for _ in range(neg_samples)]
                x_neg.append(neg)

        x_pos = np.array(x_pos)
        x_neg = np.array(x_neg)
        y_pos = np.array(y_pos)

        # Save the data
        file = open(x_pos_file, 'wb')
        pickle.dump(x_pos, file, protocol=4)

        file = open(y_pos_file, 'wb')
        pickle.dump(y_pos, file, protocol=4)

        file = open(x_neg_file, 'wb')
        pickle.dump(x_neg, file, protocol=4)

    print(' x_pos :', x_pos.shape)
    print(' y_pos :', y_pos.shape)
    print(' x_neg :', x_neg.shape)
    return x_pos, y_pos, x_neg


# # -------------------- Model class--------------------#

class word2vec:

    def __init__(self):
        return

    def set_model_params(
            self,
            vocab_size,
            op_dim,
            context_size,
            neg_samples,
            batch_size=32,
            num_epochs=5
    ):

        self.context_size = context_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.show_loss = True
        self.inp_dim = vocab_size
        self.op_dim = op_dim
        self.neg_samples = neg_samples
        return

    # data : word, context, negative samples
    def set_train_data(self, data_w, data_c, data_n):
        self.y_pos_data = data_w
        self.x_pos_data = data_c
        self.x_neg_data = data_n
        return

    def get_train_data(self):
        return self.y_pos_data, self.x_pos_data, self.x_neg_data

    def build_model(self):
        with tf.variable_scope('model'):
            self.x_pos_inp = tf.placeholder(tf.int32, [None, self.context_size], name='x_pos')
            self.y_pos_inp = tf.placeholder(tf.int32, [None, 1], name='y_pos')
            self.x_neg_inp = tf.placeholder(tf.int32, [None, self.neg_samples], name='x_neg')

            # do one hot decoding
            self.x_pos = tf.one_hot(
                indices=self.x_pos_inp,
                depth=self.inp_dim
            )

            self.y_pos = tf.one_hot(
                indices=self.y_pos_inp,
                depth=self.inp_dim
            )

            self.x_neg = tf.one_hot(
                indices=self.x_neg_inp,
                depth=self.inp_dim
            )
            # declare weights #
            print('Shape : x_pos  x_neg y_pos ', self.x_pos.shape, self.x_neg.shape, self.y_pos.shape)

            initial = tf.truncated_normal([self.inp_dim, self.op_dim], stddev=0.1)
            self.W = tf.Variable(initial)
            initial = tf.truncated_normal([1, self.op_dim], stddev=0.1)
            self.B = tf.Variable(initial)

            # self.emb1 = tf.nn.xw_plus_b(self.y_pos, self.W, self.B)

            self.emb1 = tf.einsum('ijk,kl->ijl', self.y_pos, self.W)
            self.emb1 = tf.add(self.emb1, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_pos, self.W)
            self.emb2 = tf.add(tmp, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_neg, self.W)
            self.emb3 = tf.add(tmp, self.B)

            print('Shape : emb1  emb2 emb3 ', self.emb1.shape, self.emb2.shape, self.emb3.shape)

            # ------- Loss function --------- #
            # Expand tensor to do dot product
            tmp1 = tf.stack([self.emb1] * self.x_pos.shape[1], axis=1)
            tmp1 = tf.squeeze(tmp1, axis=2)
            t1 = tf.nn.l2_normalize(self.emb2, -1)
            t2 = tf.nn.l2_normalize(tmp1, -1)
            cs1 = tf.reduce_sum(tf.multiply(t1, t2))

            tmp2 = tf.stack([self.emb1] * self.neg_samples, axis=1)
            tmp2 = tf.squeeze(tmp2, axis=2)
            # do dot product
            t1 = tf.nn.l2_normalize(self.emb3, -1)
            t2 = tf.nn.l2_normalize(tmp2, -1)
            cs2 = tf.multiply(t1, t2)
            cs2 = tf.reduce_sum(cs2, axis=-1)
            # do exp
            cs2 = tf.math.exp(cs2)
            cs2 = -tf.log(tf.reduce_sum(cs2))
            loss = -(cs1 - cs2)

            self.loss = loss
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
            self.train = self.optimizer.minimize(self.loss)

    def train_model(self):
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        y_pos, x_pos, x_neg = self.get_train_data()
        bs = self.batch_size
        num_batches = x_pos.shape[0] // bs

        losses = []

        for epoch in range(self.num_epochs):
            _loss = []
            for i in range(num_batches):
                data_x_pos = x_pos[i * bs: (i + 1) * bs]
                data_x_neg = x_neg[i * bs: (i + 1) * bs]
                data_y_pos = y_pos[i * bs: (i + 1) * bs]

                loss, _ = self.sess.run(
                    [self.loss, self.train],
                    feed_dict={
                        self.x_pos_inp: data_x_pos,
                        self.x_neg_inp: data_x_neg,
                        self.y_pos_inp: data_y_pos,
                    })
                _loss.append(loss)
            _loss = np.mean(_loss)
            if epoch % 2 == 0:
                print('Loss', _loss)
            losses.append(_loss)

        if self.show_loss == True:
            plt.plot(range(len(losses)), losses, 'r-')
            plt.show()
        return

    def get_emb_dict(self, word_index):
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        x = list(word_index.keys())
        x = np.reshape(x, [-1, 1])
        emb = self.sess.run(
            self.emb1,
            feed_dict={
                self.y_pos_inp: x
            }
        )
        res = OrderedDict()
        for wi, w in word_index.items():
            res[w] = emb[wi]
        return res


# ------------------------------------ #
# Run the model
# ------------------------------------ #
# Set the parameters of model
# ------------------------------------- #
model_data_file = 'w2v_1.pkl'
word_index, vocab, corpus_data = get_vocab()
vocab_size = len(vocab)
context_size = 2
neg_samples = 50
op_dim = 32
num_epochs = 15
x_pos, y_pos, x_neg = create_data(corpus_data, vocab, context_size, neg_samples)
emb_dict = None

# ---- If model not saved -> Build ----- #
if not os.path.exists(model_data_file):
    w2v_model = word2vec()
    w2v_model.set_model_params(
        vocab_size=vocab_size,
        op_dim=op_dim,
        context_size=context_size,
        neg_samples=neg_samples,
        num_epochs=num_epochs
    )

    w2v_model.set_train_data(y_pos, x_pos, x_neg)
    w2v_model.build_model()
    w2v_model.train_model()
    emb_dict = w2v_model.get_emb_dict(word_index)
    file = open(model_data_file, 'wb')
    pickle.dump(
        emb_dict,
        file,
        protocol=4
    )
else:
    with open(model_data_file, 'rb') as f:
        emb_dict = pickle.load(f)

emb_mat = np.array(list(emb_dict.values()))
emb_matrix = np.reshape(emb_mat, [emb_mat.shape[0], emb_mat.shape[-1]])


# ----------------------------------------------- #
# Find k closest words
# ----------------------------------------------- #

def closest(word, emb_dict, k=2):
    n_emb = emb_dict[word]
    dist_dict = {}
    for i, e in emb_dict.items():
        if i == word:
            continue
        cos_sim = 1 - spatial.distance.cosine(n_emb, e)
        dist_dict[i] = cos_sim

    # sort by distance
    tmp = sorted(dist_dict.items(), key=lambda x: x[1])

    # return top k
    return [i[0] for i in tmp[:k]]


print(closest('happy', emb_dict, 15))


# ------------------------------------------------ #
# Visualization
# ------------------------------------------------ #

def visualize(emb_matrix, emb_dict):
    tsne_model = TSNE(
        perplexity=40,
        n_components=2,
        init='pca',
        n_iter=2500,
        random_state=23
    )
    new_values = tsne_model.fit_transform(emb_matrix)

    x = []
    y = []
    labels = emb_dict.keys()
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x[:100])):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    # model = TSNE(n_components=2, random_state=0)
    # tsne_embedding = model.fit_transform(emb_matrix)
    # n_vectors = tsne_embedding[list(range(emb_matrix.shape[0]))]
    # print(n_vectors)
    # plt.subplots(figsize=(10, 10))
    # for idx in range(emb_matrix.shape[0]):
    #     plt.scatter(
    #         n_vectors[idx, 0],
    #         n_vectors[idx, 1],
    #         color='steelblue'
    #     )
    #
    # plt.show()


visualize(emb_matrix, emb_dict)
