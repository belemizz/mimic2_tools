from collections import OrderedDict

import numpy as np
from  sklearn import cross_validation, linear_model

import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import generate_sample
from alg_classification import ClassificationResult, calc_classification_result, sumup_classification_result

import sys
sys.path.append('../../deep_tutorial/sample_codes/')

## utility
def ar_float(data):
    return np.asarray(data, dtype=th.config.floatX)

def th_imatrix(name):
    return T.matrix(name, dtype= 'int64')

def th_ivector(name):
    return T.vector(name, dtype= 'int64')

def th_fmatrix(name):
    return T.matrix(name, dtype= 'float64')

def print_info(word):
    print '[INFO] ' + word

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)

def prepare_data(seqs, labels, maxlen = None):
        # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(th.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


## Optimizer
def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [th.shared(p.get_value() * ar_float(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [th.shared(p.get_value() * ar_float(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [th.shared(p.get_value() * ar_float(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = th.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = th.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


class Lstm():
    def __init__( self,
                  n_epochs = 100,
                  patience=2
                  ):

        # loop
        self.n_epochs = n_epochs
        self.patience = patience
#        self.validFreq=370  # Compute the validation error after this number of update.
        self.validFreq=100  # Compute the validation error after this number of update.
        self.saveFreq=1110  # Save the parameters after every saveFreq updates
        self.dispFreq=10  # Display to stdout the training progress every N updates
        self.batch_size=16  # The batch size during training.
        
        # optimization
        self.optimizer=adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        self.lrate=0.01  # Learning rate for sgd (not used for adadelta and rmsprop)
        
        # embedding params
        self.dim_feature = 128
        self.maxlen=100  # Sequence longer then this get ignored
        self.n_words=10000  # Vocabulary size
        self.ydim = 2

        # lstm params

        # regularization
        self.decay_c=1.  # Weight decay for the classifier applied to the U weights.

        # classification params
        self.use_dropout=True

        # random seed
        self.trng = RandomStreams(123)
        np.random.seed(123)

        self.__init_param()

    def __init_param(self):
        # parameters
        self.tparams = self.__init_params()

    def __init_params(self):
        def ortho_w(ndim):
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            return u.astype(th.config.floatX)
        
        params = OrderedDict()

        # weight for embedding
        randn = np.random.rand(self.n_words, self.dim_feature)
        params['Wemb'] = (0.01 * randn).astype(th.config.floatX)

        # lstm
        params['lstm_W'] = np.concatenate([ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature)], axis=1)
        params['lstm_U'] = np.concatenate([ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature),
                                           ortho_w(self.dim_feature)], axis=1)
        params['lstm_b'] = np.zeros((4 * self.dim_feature,)).astype(th.config.floatX)
        
        # classifier
        params['U'] = 0.01 * np.random.randn(self.dim_feature, self.ydim).astype(th.config.floatX)
        params['b'] = np.zeros((self.ydim,)).astype(th.config.floatX)

        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = th.shared(params[kk], name=kk)
        return tparams

    def dropout_layer(self, state_before, use_noise, trng):
        proj = T.switch(use_noise,
                        (state_before * trng.binomial(state_before.shape,
                                                      p=0.5, n=1,
                                                      dtype=state_before.dtype)),
                        state_before * 0.5)
        return proj

    def pred_error(self, f_pred, prepare_data, data, iterator, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        valid_err = 0
        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      np.array(data[1])[valid_index],
                                      maxlen=None)
            preds = f_pred(x, mask)
            targets = np.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - ar_float(valid_err) / len(data[0])
        return valid_err

    def lstm_layer(self, x, mask=None):
        n_steps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        assert mask is not None
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = x_ + T.dot(h_, self.tparams['lstm_U']) 

            i = T.nnet.sigmoid(_slice(preact, 0, self.dim_feature))
            f = T.nnet.sigmoid(_slice(preact, 1, self.dim_feature))
            o = T.nnet.sigmoid(_slice(preact, 2, self.dim_feature))
            c = T.tanh(_slice(preact, 3, self.dim_feature))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
            # if mask = 1 then c, else c_prev

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            # if mask = 1 then h, else h_prev

            return h, c

        wx_b = T.dot(x, self.tparams['lstm_W'] ) + self.tparams['lstm_b']

        rval, updates = th.scan(_step,
                                sequences=[mask, wx_b],
                                outputs_info=[T.alloc(ar_float(0.),
                                                      n_samples,
                                                      self.dim_feature),
                                              T.alloc(ar_float(0.),
                                                      n_samples,
                                                      self.dim_feature)
                                              ],
                                name='lstm_layers',
                                n_steps=n_steps)
        return rval[0] #return only sequence of hidden


    def get_lstm_model_with_emb(self, x, mask, y):
        ## embedding
        emb = self.tparams['Wemb'][x.flatten()].reshape([x.shape[0],
                                                         x.shape[1],
                                                         self.dim_feature])

        ## lstm get sequence of hidden variable
        proj = self.lstm_layer(emb, mask=mask)

        ## calcurate_mean
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

        ## dropout
        if self.use_dropout:
            use_noise = th.shared( ar_float(0.) )
            proj = self.dropout_layer(proj, use_noise, self.trng)
        
        ## classification
        pred_prob = T.nnet.softmax(T.dot(proj, self.tparams['U']) + self.tparams['b'])

        pred = pred_prob.argmax(axis = 1)
        
#        f_pred_prob = th.function([x, mask], pred_prob, name='f_pred_prob')
#        f_pred = th.function([x, mask], pred_prob.argmax(axis=1), name='f_pred')
        off = 1e-8
        if pred_prob.dtype == 'float16':
            off = 1e-6
        cost = -T.log(pred_prob[T.arange(x.shape[1]), y] + off).mean()

        return cost, pred

    def split_train_and_valid(self, r_valid, train_x, train_y):
        if r_valid < 0 or r_valid > 1:
            raise ValueError('should be 0 <= r_valid <= 1')
        
        n_valid = int(len(train_x) * r_valid)
        n_train = len(train_x) - n_valid
        valid_x = train_x[n_train:len(train_x)]
        valid_y = train_y[n_train:len(train_x)]

        train_x = train_x[0:n_train]
        train_y = train_y[0:n_train]

        return [train_x, train_y], [valid_x, valid_y]
        
    def fit(self, train_x, train_y):

        ## Parameters
        valid_batch_size=64  # The batch size used for validation/test set.

        print_info('building model')

        # variables
        x = th_imatrix('x')
        y = th_ivector('y')
        mask = th_fmatrix('mask')
        l_rate = T.scalar(name='learning rate')

        # models
        cost, pred = self.get_lstm_model_with_emb(x, mask, y)

        if decay_c > 0.:
            decay_c = th.shared(ar_float(decay_c), name='decay_c')
            weight_decay = 0.
            weight_decay += (self.tparams['U'] ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        f_pred = th.function([x, mask], pred, name='f_pred')
        grads = T.grad(cost, wrt = self.tparams.values())

        # optimization
        f_grad_shared, f_update = adadelta(l_rate, self.tparams, grads, x, mask, y, cost)

#        kf_valid = lstm.get_minibatches_idx(len(valid[0]), valid_batch_size)
#        kf_test = lstm.get_minibatches_idx(len(test[0]), valid_batch_size)

        print_info('loop begins')

        [train_x, train_y], [valid_x, valid_y] = self.split_train_and_valid(0.05, train_x, train_y)
        
        n_updates = 0  # the number of update done
        bad_count = 0

        l_errors = []
        b_estop = False
        best_p = None

        kf_valid = get_minibatches_idx(len(valid_x), self.batch_size, True)
        
        for i_epoch in xrange(self.n_epochs):
            kf = get_minibatches_idx(len(train_x), self.batch_size, True)
            n_samples = 0
            l_costs = []

            for _, train_index in kf:
                n_updates += 1
                y = [train_y[i] for i in train_index]
                x = [train_x[i] for i in train_index]

                x, mask, y = prepare_data(x,y)
                cost = f_grad_shared(x, mask, y)
                f_update(self.lrate)

                l_costs.append(cost)

                if np.mod(n_updates, self.dispFreq) == 0:
                    print 'Epoch:', i_epoch, 'Update:', n_updates, 'Cost:', cost

                if np.mod(n_updates, self.validFreq) == 0:
                    train_err = self.pred_error(f_pred, prepare_data, [train_x, train_y], kf)
                    valid_err = self.pred_error(f_pred, prepare_data, [valid_x, valid_y], kf_valid)

                    print 'Train:' , train_err, 'Valid:', valid_err

                    l_errors.append([train_err, valid_err])

                    ## TODO: if best score save the parameter

                    if( len(l_errors) > self.patience and
                        valid_err >= np.array(l_errors)[:-self.patience,0].min()):
                        bad_count +1
                        if bad_count > self.patience:
                            print_info('Early stopping')
                            b_estop = True
                            break
                        
                    print "Estop Counter: %d/%d"%(bad_count, self.patience)

            print "Averave cost in epoch %d: %f"%(i_epoch, np.mean(l_costs))
            if b_estop:
                break

        print l_errors
        import ipdb
        ipdb.set_trace()

    def predict(self, test_x):
        print '___prediction___'
        predict_y = np.zeros( len(test_x))
        
        print 'not implemented'
        for i in range(len(test_x)):
            predict_y[i] = np.random.randint(2)
        return predict_y


class LR_ts():
    def __init__(self, n_dim = 10):
        self.clf = linear_model.LogisticRegression(random_state =0)
        self.n_dim = n_dim

    def fit(self, train_x, train_y):
        s_train_x = [ts[0: self.n_dim] for ts in train_x]
        self.clf.fit(s_train_x, train_y)

    def predict(self, test_x):
        s_test_x = [ts[0:self.n_dim] for ts in test_x]
        predict_y = self.clf.predict(s_test_x)
        return predict_y

class Cointoss():
    def __init__(self, seed = 0):
        np.random.seed(seed)

    def fit(self, train_x, train_y):
        pass

    def predict(self, test_x):
        predict_y = np.zeros( len(test_x))
        
        print 'not implemented'
        for i in range(len(test_x)):
            predict_y[i] = np.random.randint(2)
        return predict_y
        
algorithm_list = ['lstm', 'lr', 'coin']
def get_algorithm(algorithm):
    if algorithm is 'lstm':
        clf = Lstm()
    elif algorithm is 'lr':
        clf = LR_ts()
    elif algorithm is 'coin':
        clf = Cointoss()
    else:
        raise ValueError("algorithm has to be either %s"%algorithm_list)
    return clf

def fit_and_test(train_x, train_y, test_x, test_y, algorithm = 'lr'):
    clf = get_algorithm(algorithm)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)

    return calc_classification_result(predict_y, test_y)
    
if __name__ == '__main__':
    [x, y] = generate_sample.time_series(1)

    n_train = 1998 + 105
    n_test = 500

    train_x = x[0:n_train]
    train_y = y[0:n_train]
    test_x = x[n_train:n_test+n_train]
    test_y = y[n_train:n_test+n_train]

    algorithm = 'lstm'
    result = fit_and_test(train_x, train_y, test_x, test_y, algorithm)


    
    ## # plot
    ## kf = cross_validation.KFold(len(x), n_folds = 4, shuffle = True, random_state = 0)
    ## 
    

    ## result_list = []
    ## for train, test in kf:
    ##     train_x = x[train]
    ##     train_y = y[train]

    ##     test_x = x[test]
    ##     test_y = y[test]

    ##     result = fit_and_test(train_x, train_y, test_x, test_y, algorithm)
    ##     print result
    ##     result_list.append(result)

    ## print 'sumup'
    ## print sumup_classification_result(result_list)


    ## ## initialization of params
## def init_params(options):
##     params = OrderedDict()
##     # embedding
##     randn = np.random.rand(options['n_words'],
##                            options['dim_proj'])
##     params['Wemb'] = (0.01 * randn).astype(th.config.floatX)

##     # lstm
##     params = param_init_lstm(options, params)
    
##     # classifier
##     params['U'] = 0.01 * np.random.randn(options['dim_proj'],
##                                          options['ydim']).astype(th.config.floatX)
##     params['b'] = np.zeros((options['ydim'],)).astype(th.config.floatX)

##     return params

## def param_init_lstm(options, params, prefix='lstm'):

##     def ortho_w(ndim):
##         W = np.random.randn(ndim, ndim)
##         u, s, v = np.linalg.svd(W)
##         return u.astype(th.config.floatX)

##     W = np.concatenate([ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj'])], axis=1)
##     params[_p(prefix, 'W')] = W
##     U = np.concatenate([ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj']),
##                         ortho_w(options['dim_proj'])], axis=1)
##     params[_p(prefix, 'U')] = U
##     b = np.zeros((4 * options['dim_proj'],))
##     params[_p(prefix, 'b')] = b.astype(th.config.floatX)

##     return params


## def init_tparams(params):
##     tparams = OrderedDict()
##     for kk, pp in params.iteritems():
##         tparams[kk] = th.shared(params[kk], name=kk)
##     return tparams

## ## model
## def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    
##     nsteps = state_below.shape[0]
##     if state_below.ndim == 3:
##         n_samples = state_below.shape[1]
##     else:
##         n_samples = 1

##     assert mask is not None
    
##     def _slice(_x, n, dim):
##         if _x.ndim == 3:
##             return _x[:, :, n * dim:(n + 1) * dim]
##         return _x[:, n * dim:(n + 1) * dim]

##     def _step(m_, x_, h_, c_):
##         preact = T.dot(h_, tparams[_p(prefix, 'U')])
##         preact += x_

##         i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
##         f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
##         o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
##         c = T.tanh(_slice(preact, 3, options['dim_proj']))

##         c = f * c_ + i * c
##         c = m_[:, None] * c + (1. - m_)[:, None] * c_
##         # if mask = 1 then c, else c_prev

##         h = o * T.tanh(c)
##         h = m_[:, None] * h + (1. - m_)[:, None] * h_
##         # if mask = 1 then h, else h_prev

##         return h, c

##     state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
##                    tparams[_p(prefix, 'b')])

##     dim_proj = options['dim_proj']
##     rval, updates = th.scan(_step,
##                             sequences=[mask, state_below],
##                             outputs_info=[T.alloc(ar_float(0.),
##                                                   n_samples,
##                                                   dim_proj),
##                                           T.alloc(ar_float(0.),
##                                                   n_samples,
##                                                   dim_proj)
##                                           ],
##                             name=_p(prefix, '_layers'),
##                             n_steps=nsteps)
##     return rval[0] #return only sequence of hidden


