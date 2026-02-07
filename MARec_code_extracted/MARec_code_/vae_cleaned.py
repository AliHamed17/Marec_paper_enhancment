import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Embedding, Flatten, Dot, MultiHeadAttention, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2, l1, l2
from tensorflow.keras.callbacks import Callback
import numpy as np
from scipy import sparse
import time
import bottleneck as bn
import pickle

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()

    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / X_true_binary.sum(axis=1)
    return recall

class MultiVAE(object):
    def __init__(self, p_dims,X, q_dims=None,  lam=0.01, lr=1e-3, lambda2=50, perc_value=10, random_seed=None, n_additional_layer=0):
        self.n_additional_layer = n_additional_layer
        self.p_dims = p_dims
        self.lambda2 = lambda2
        self.perc_value = perc_value
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = [self.q_dims[0]]# + self.p_dims[1:]
        prev_dim = self.q_dims[0]
        for i in range(self.n_additional_layer):
            self.dims.append(prev_dim//2)
            prev_dim = prev_dim//2
        self.dims.append(self.q_dims[1])
        for i in range(self.n_additional_layer):
            self.dims.append(prev_dim)
            prev_dim = prev_dim * 2
        self.dims.append(self.q_dims[0])
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed
        self.all_X = X

        self.construct_placeholders()

    def construct_placeholders(self):        
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        # xtilde or Siamese pred_val
        self.input_xtilde = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)
        
        vector = np.sum(self.all_X, axis=0)
        percentile = max(np.percentile(vector, self.perc_value),1)
        k = self.lambda2/percentile
        vector_tr = np.zeros(len(vector))
        for counter,item in enumerate(vector):
            if item <= percentile:
                vector_tr[counter] = k*(percentile-item)

        D_R = tf.linalg.diag(vector_tr)
        D_R = tf.cast(D_R, tf.float32)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph + tf.matmul(log_softmax_var * self.input_xtilde, D_R),
            axis=1))

        regularization_loss = tf.losses.get_regularization_loss()
        loss = neg_ll + 2 * regularization_loss
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                regularizer=tf.keras.regularizers.L2(self.lam),
                initializer=tf.compat.v1.keras.initializers.glorot_normal(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])


def run_vae(train_data, test_data, Xtilde, batch_size, batch_size_vad, N_vad, n_epochs, chkpt_dir, p_dim=200, lr=1e-4, k=25, lambda2=1, random_seed=98765, n_additional_layer=0):
    X = train_data.toarray()
    data_shape = X.shape
    N = data_shape[0]
    idxlist = range(N)

    # training batch size
    batches_per_epoch = int(np.ceil(float(N) / batch_size))

    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2

    n_items = data_shape[1]
    idxlist = np.array(idxlist)

    p_dims = [p_dim, n_items]

    tf.disable_eager_execution()
    tf.reset_default_graph()
    dae = MultiVAE(p_dims, X, lr=lr, lam=0.01 / batch_size, random_seed=random_seed, lambda2=lambda2, n_additional_layer=n_additional_layer)

    saver, logits_var, loss_var, train_op_var, merged_var = dae.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    idxlist_vad = range(N_vad)
    idxlist_vad = np.array(idxlist_vad)

    # todo: change code order: B-> Xtilde->train, initialise DR, add it into loss

    ndcgs_vad = []
    vae_train_data = train_data.tocsr()
    cold_test_data = test_data.tocsr()
    nonzeros = cold_test_data.nonzero()

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf
        
        for epoch in range(n_epochs):
            np.random.shuffle(idxlist)
            
            # train for one epoch
            epoch_time = 0
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = vae_train_data[idxlist[st_idx:end_idx]]
                X_tilde_train = Xtilde[idxlist[st_idx:end_idx]]
                
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')           
                
                feed_dict = {dae.input_ph: X,
                            dae.input_xtilde: X_tilde_train,
                            dae.keep_prob_ph: 0.5}       
                t1 = time.time()
                sess.run(train_op_var, feed_dict=feed_dict)
                t2 = time.time()
                epoch_time += (t2-t1)
            print("Epoch: {}, time {}".format(epoch, epoch_time))
                        
            # compute validation NDCG
            ndcg_dist = []
            n10_list, r10_list, n20_list, r20_list, n50_list, r50_list, n100_list, r100_list = [], [],[], [],[], [],[], []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vae_train_data[idxlist_vad[st_idx:end_idx]]
                X_tilde_train = Xtilde[idxlist_vad[st_idx:end_idx]]
                
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')
            
                pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X, dae.input_xtilde: X_tilde_train} )

                pred_val[X.nonzero()] = -np.inf

                # use either NDCG or Recall
                ndcg_dist.append(Recall_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=k))
                r10_list.append(Recall_at_k_batch(pred_val,cold_test_data[idxlist_vad[st_idx:end_idx]], k=10))
                n10_list.append(NDCG_binary_at_k_batch(pred_val,cold_test_data[idxlist_vad[st_idx:end_idx]], k=10))
                r20_list.append(Recall_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=25))
                n20_list.append(NDCG_binary_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=25))
                r50_list.append(Recall_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=50))
                n50_list.append(NDCG_binary_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=50))
                r100_list.append(Recall_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=100))
                n100_list.append(NDCG_binary_at_k_batch(pred_val, cold_test_data[idxlist_vad[st_idx:end_idx]], k=100))
        
            ndcg_dist = np.concatenate(ndcg_dist)
            r10_list = np.concatenate(r10_list)
            n10_list = np.concatenate(n10_list)
            r20_list = np.concatenate(r20_list)
            n20_list = np.concatenate(n20_list)
            r50_list = np.concatenate(r50_list)
            n50_list = np.concatenate(n50_list)
            r100_list = np.concatenate(r100_list)
            n100_list = np.concatenate(n100_list)
            ndcg_ = np.nanmean(ndcg_dist)
            ndcgs_vad.append(ndcg_)
            print('Metrics@{}: {}'.format(k, ndcg_))
            print("Test NDCG@10=%.5f (%.5f)" % (np.nanmean(n10_list), np.nanstd(n10_list) / np.sqrt(len(n10_list))))
            print("Test NDCG@25=%.5f (%.5f)" % (np.nanmean(n20_list), np.nanstd(n20_list) / np.sqrt(len(n20_list))))
            print("Test NDCG@50=%.5f (%.5f)" % (np.nanmean(n50_list), np.nanstd(n50_list) / np.sqrt(len(n50_list))))
            print("Test NDCG@100=%.5f (%.5f)" % (np.nanmean(n100_list), np.nanstd(n100_list) / np.sqrt(len(n100_list))))
            print("Test Recall@10=%.5f (%.5f)" % (np.nanmean(r10_list), np.nanstd(r10_list) / np.sqrt(len(r10_list))))
            print("Test Recall@25=%.5f (%.5f)" % (np.nanmean(r20_list), np.nanstd(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (np.nanmean(r50_list), np.nanstd(r50_list) / np.sqrt(len(r50_list))))
            print("Test Recall@100=%.5f (%.5f)" % (np.nanmean(r100_list), np.nanstd(r100_list) / np.sqrt(len(r100_list))))

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_

def test_vae(train_data, test_data, Xtilde, batch_size, chkpt_dir, p_dim=200, lr=1e-4, k=25, lambda2=1, test_num=100000, random_seed=98765):
    tf.disable_eager_execution()
    tf.reset_default_graph()
    p_dims = [p_dim, train_data.shape[1]]
    X = train_data.toarray()
    dae = MultiDAE(p_dims, X, lr=lr, lam=0.01 / batch_size, random_seed=random_seed, lambda2=lambda2)
    saver, logits_var, loss_var, train_op_var, merged_var = dae.build_graph()
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))
        X = X[:test_num]
        Xtilde = Xtilde[:test_num]
        test_data = test_data.tocsr()[:test_num,:]
        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')
        pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X, dae.input_xtilde: Xtilde})
        pred_val[X.nonzero()] = -np.inf
    
        n10_list, r10_list, n20_list, r20_list, n50_list, r50_list, n100_list, r100_list = [], [],[], [],[], [],[], []
        r10_list.append(Recall_at_k_batch(pred_val,test_data, k=10))
        n10_list.append(NDCG_binary_at_k_batch(pred_val,test_data, k=10))
        r20_list.append(Recall_at_k_batch(pred_val, test_data, k=25))
        n20_list.append(NDCG_binary_at_k_batch(pred_val, test_data, k=25))
        r50_list.append(Recall_at_k_batch(pred_val, test_data, k=50))
        n50_list.append(NDCG_binary_at_k_batch(pred_val, test_data, k=50))
        r100_list.append(Recall_at_k_batch(pred_val, test_data, k=100))
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data, k=100))
        print("Test NDCG@10=%.5f (%.5f)" % (np.nanmean(n10_list), np.nanstd(n10_list) / np.sqrt(len(n10_list))))
        print("Test NDCG@25=%.5f (%.5f)" % (np.nanmean(n20_list), np.nanstd(n20_list) / np.sqrt(len(n20_list))))
        print("Test NDCG@50=%.5f (%.5f)" % (np.nanmean(n50_list), np.nanstd(n50_list) / np.sqrt(len(n50_list))))
        print("Test NDCG@100=%.5f (%.5f)" % (np.nanmean(n100_list), np.nanstd(n100_list) / np.sqrt(len(n100_list))))
        print("Test Recall@10=%.5f (%.5f)" % (np.nanmean(r10_list), np.nanstd(r10_list) / np.sqrt(len(r10_list))))
        print("Test Recall@25=%.5f (%.5f)" % (np.nanmean(r20_list), np.nanstd(r20_list) / np.sqrt(len(r20_list))))
        print("Test Recall@50=%.5f (%.5f)" % (np.nanmean(r50_list), np.nanstd(r50_list) / np.sqrt(len(r50_list))))
        print("Test Recall@100=%.5f (%.5f)" % (np.nanmean(r100_list), np.nanstd(r100_list) / np.sqrt(len(r100_list))))
