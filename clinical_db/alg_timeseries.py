from collections import OrderedDict, namedtuple

import numpy as np
from  sklearn import cross_validation, linear_model

import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import get_sample
from get_sample import select_tseries
from alg_classification import ClassificationResult, calc_classification_result, sumup_classification_result

import sys
sys.path.append('../../deep_tutorial/sample_codes/')

from mutil import p_info

## utility
def ar_float(data):
    return np.asarray(data, dtype=th.config.floatX)

def th_imatrix(name):
    return T.matrix(name, dtype= 'int64')

def th_ivector(name):
    return T.vector(name, dtype= 'int64')

def th_fmatrix(name):
    return T.matrix(name, dtype= 'float64')

def th_ftensor3(name):
    return T.tensor3(name, dtype= 'float64')

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
## Optimizer

def myadadelta(lr, tparams, grads, x, mask, y, cost):
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
                              name='%d_grad' % i)
                    for i, p in enumerate(tparams)]
    running_up2 = [th.shared(p.get_value() * ar_float(0.),
                             name='%d_rup2' % i)
                   for i, p in enumerate(tparams)]
    running_grads2 = [th.shared(p.get_value() * ar_float(0.),
                                name='%d_rgrad2' % i)
                      for i, p in enumerate(tparams)]

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
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = th.function([lr], [], updates=ru2up + param_up,
                            on_unused_input='ignore',
                           name='adadelta_f_update')

    return f_grad_shared, f_update

def ortho_w(ndim):
    """ generate random orthogonal weight"""
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(th.config.floatX)

class Lstm():
    def __init__( self,
                  n_epochs = 100,
                  patience = 20
                  ):

        # loop
        self.n_epochs = n_epochs
        self.patience = patience
        self.validFreq = 100
        self.dispFreq = 10
        self.batch_size = 10

        # random seed
        self.trng = RandomStreams(123)
        np.random.seed(123)
        
    def __init_params(self, dim_feature, dim_class):
        """ initialize the parameters to be optimized """
        self.p_Wl = th.shared( np.hstack([ortho_w(dim_feature),
                                          ortho_w(dim_feature),
                                          ortho_w(dim_feature),
                                          ortho_w(dim_feature)] ),
                               name = 'Wl', borrow = True)
        self.p_Ul = th.shared( np.hstack([ortho_w(dim_feature),
                                          ortho_w(dim_feature),
                                          ortho_w(dim_feature),
                                          ortho_w(dim_feature)]),
                               name = 'Wl', borrow = True)
        self.p_bl = th.shared( np.zeros(4 * dim_feature).astype(th.config.floatX),
                               name = 'bl', borrow = True)
        self.p_Wc = th.shared( 0.01 * np.random.randn(dim_feature, dim_class).astype(th.config.floatX),
                               name = 'Wc', borrow = True)
        self.p_bc = th.shared( np.zeros(dim_class).astype(th.config.floatX),
                               name = 'bc', borrow = True)

    def __l_params(self):
        return [self.p_Wl, self.p_Ul, self.p_bl, self.p_Wc, self.p_bc]

    def __keep_params(self):
        params = self.__l_params()
        self.best_params = [param.copy() for param in params]

    def __retrieve_best_params(self):
        [self.p_Wl, self.p_Ul, self.p_bl, self.p_Wc, self.p_bc] = self.best_params

    def __lstm_layer(self, t_x, t_m):

        n_steps = t_x.shape[0]
        n_samples = t_x.shape[1]

        def vslice( mat, idx, dim):
            return mat[:, idx * dim: (idx+1) * dim]

        def _step(x_, m_, h_, c_):
            preact = T.dot(x_, self.p_Wl) + T.dot(h_, self.p_Ul) + self.p_bl

            i = T.nnet.sigmoid( vslice(preact, 0, self.dim_feature))
            f = T.nnet.sigmoid( vslice(preact, 1, self.dim_feature))
            o = T.nnet.sigmoid( vslice(preact, 2, self.dim_feature))
            c = T.tanh(vslice(preact, 3, self.dim_feature))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_ # if mask = 1 then c, else c_prev

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_ # if mask = 1 then h, else h_prev
            
            return h, c

        rval, updates = th.scan(_step,
                                sequences = [t_x, t_m],
                                outputs_info = [T.alloc(ar_float(0.),
                                                        n_samples,
                                                        self.dim_feature),
                                                T.alloc(ar_float(0.),
                                                        n_samples,
                                                        self.dim_feature)
                                                ],
                                name = 'lstm_layers',
                                n_steps = n_steps
                                )
        return rval[0] # return only sequence of hidden
                                                        
    def __get_lstm_model(self, x, m, y):
        hidden = self.__lstm_layer(x, m)
        mean = (hidden * m[:, :, None]).sum(axis=0) / m.sum(axis=0)[:, None]

        pred_prob = T.nnet.softmax(T.dot( mean, self.p_Wc) + self.p_bc)
        pred = pred_prob.argmax(axis = 1)

        off = 1e-8
        cost = -T.log(pred_prob[T.arange(x.shape[1]), y] + off).mean()

        return cost, pred

    def __get_test(self, t_x):
        return t_x.shape[0]
    
    def __split_train_and_valid(self, r_valid, train_set):
        if r_valid < 0 or r_valid > 1:
            raise ValueError('should be 0 <= r_valid <= 1')
        
        n_valid = int(len(train_set[0]) * r_valid)
        n_train = len(train_set[0]) - n_valid

        valid_set = select_tseries(train_set, range(n_train, n_train+n_valid))
        train_set = select_tseries(train_set, range(0, n_train))
        
        return train_set, valid_set

    def __prediction_error(self, sample_set):
        predicted_y = self.f_pred(sample_set[0], sample_set[1])
        error_rate = 1. - calc_classification_result(predicted_y, sample_set[2]).acc
        return error_rate
        
    def __validation(self, train_set, valid_set, l_errors):
        train_err = self.__prediction_error(train_set)
        valid_err = self.__prediction_error(valid_set)
        print 'Train:' , train_err, 'Valid:', valid_err

        if (len(l_errors) == 0 or valid_err < np.array(l_errors)[:, 1].min()):
            p_info('Save best parameters')
            self.__keep_params()
        l_errors.append([train_err, valid_err])

    def __judge_early_stopping(self, l_errors):
        valid_errors = [error[1] for error in l_errors]
        min_index= valid_errors.index(min(valid_errors))
        p_info("Patience: %d/%d"%(len(valid_errors)-min_index, self.patience))
        if len(valid_errors) > min_index + self.patience:
            return True
        else:
            return False

    def fit(self, train_set):
        """ train the model by training set """

        self.dim_feature = len(train_set[0][0][0])
        self.dim_class = 2
        self.__init_params(self.dim_feature, self.dim_class)

        p_info('Building the model')
        t_x = th_ftensor3('t_x')
        t_m = th_fmatrix('t_m')
        t_y = th_ivector('t_y')

        m_test = self.__get_test(t_x)
        f_test = th.function([t_x], m_test, name = 'f_test')
        
        m_cost, m_pred = self.__get_lstm_model(t_x, t_m, t_y) 
        m_grads = T.grad(m_cost, wrt = self.__l_params())

        self.f_pred = th.function([t_x, t_m], m_pred, name = 'f_pred')
        
        t_lr = T.scalar(name='t_lr')
        f_cost, f_update = myadadelta(t_lr, self.__l_params(), m_grads, t_x, t_m, t_y, m_cost)

        p_info('Loop begins')
        
        n_updates = 0  # the number of update done
        bad_count = 0

        l_errors = []
        b_estop = False
        
        train_set, valid_set = self.__split_train_and_valid(0.1, train_set)

        self.__validation(train_set, valid_set, l_errors)
        for i_epoch in xrange(self.n_epochs):
            kf = get_minibatches_idx(train_set[0].shape[1], self.batch_size, True)
            n_samples = 0
            l_costs = []

            for _, train_index in kf:
                n_updates += 1
                [x, m, y] = select_tseries(train_set, train_index)

                cost = f_cost(x, m, y)
                l_costs.append(cost)
                f_update(0.01)

                if np.mod(n_updates, self.dispFreq) == 0:
                    print 'Epoch:', i_epoch, 'Update:', n_updates, 'Cost:', cost

                if np.mod(n_updates, self.validFreq) == 0:
                    self.__validation(train_set, valid_set, l_errors)
                    b_estop = self.__judge_early_stopping(l_errors)
                    if b_estop:
                        p_info('Early stopping')
                        break
                    
            print "Averave m_cost in epoch %d: %f"%(i_epoch, np.mean(l_costs))
            if b_estop: break

        self.__retrieve_best_params()
        self.__validation(train_set, valid_set, l_errors)

        print l_errors
        print self.best_params

    def predict(self, test_sample):
        if self.f_pred == None:
            raise ValueError("Fitting must be done before prediction")
        self.__retrieve_best_params()
        return self.f_pred(test_sample[0], test_sample[1])
    
class LR_ts():
    def __init__(self, n_dim = 10):
        self.clf = linear_model.LogisticRegression(random_state =0)
        self.n_dim = n_dim

    def fit(self, train_set):
        train_x = train_set[0]
        train_y = train_set[2]

        n_sample = train_x.shape[1]
        s_train_x = np.array([train_x[:self.n_dim, idx, :].flatten() for idx in range(n_sample)])

        self.clf.fit(s_train_x, train_y)

    def predict(self, test_sample):
        test_x = test_sample[0]
        n_sample = test_x.shape[1]
        s_test_x = np.array([test_x[:self.n_dim, idx, :].flatten() for idx in range(n_sample)])
        predict_y = self.clf.predict(s_test_x)
        return predict_y

class Cointoss():
    def __init__(self, seed = 0):
        np.random.seed(seed)

    def fit(self, train_set):
        pass

    def predict(self, test_sample):
        n_sample = test_sample[0].shape[1]
        predict_y = np.zeros( n_sample )
        
        for i in range(n_sample):
            predict_y[i] = np.random.randint(2)
        return predict_y

algorithm_list = ['lstm', 'lr', 'coin']
def get_algorithm(algorithm):
    if algorithm is 'lstm':
        clf = Lstm(n_epochs = 20)
    elif algorithm is 'lr':
        clf = LR_ts(n_dim = 10)
    elif algorithm is 'coin':
        clf = Cointoss()
    else:
        raise ValueError("algorithm has to be either %s"%algorithm_list)
    return clf

def cv(sample_set, n_fold, algorithm = 'lr'):

    n_sample = sample_set[0].shape[1]
    
    kf = cross_validation.KFold(sample_set[0].shape[1], n_folds = n_fold, shuffle = True, random_state = 0)
    i_cv = 0
    results = []
    for train_idx, test_idx in kf:
        i_cv = i_cv + 1
        p_info("___cv%d___"%i_cv)
        train_set = select_tseries( sample_set, train_idx)
        test_set = select_tseries( sample_set, test_idx)
        results.append(fit_and_test(train_set, test_set, algorithm))

    return sumup_classification_result(results)

def fit_and_test(train_set, test_set, algorithm = 'lr'):
    clf = get_algorithm(algorithm)
    clf.fit(train_set)
    predict_y = clf.predict(test_set[0:2])
    return calc_classification_result(predict_y, test_set[2])

if __name__ == '__main__':
    sample_set = get_sample.time_series(0,2)
    result = cv(sample_set, 4, 'lstm')
    
    ## n_train = int( sample_set[0].shape[1] * 0.75)
    ## n_test = sample_set[0].shape[1] - n_train

    ## train_set = select_tseries( sample_set, range(0,n_train) )
    ## test_set = select_tseries( sample_set, range(n_train, n_train+n_test))

    ## algorithm = 'coin'
    ## result = fit_and_test(train_set, test_set, algorithm)

    print result
