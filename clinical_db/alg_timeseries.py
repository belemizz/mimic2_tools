import numpy as np
import theano
import theano.tensor as T
from  sklearn import cross_validation, linear_model

import generate_sample
from alg_classification import ClassificationResult, calc_classification_result, sumup_classification_result


from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.path.append('../../deep_tutorial/sample_codes/')
import lstm

class Lstm():
    def __init__(self, n_epochs = 1):
        self.n_epochs = 1

    def fit(self, train_x, train_y):
        print '___fit training sample___'

        ## Parameters
        dim_proj=128  # word embeding dimension and LSTM number of hidden units.
        patience=10  # Number of epoch to wait before early stop if no progress
        max_epochs=5000  # The maximum number of epoch to run
        dispFreq=10  # Display to stdout the training progress every N updates
        decay_c=0.  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000  # Vocabulary size
        optimizer=lstm.adadelta  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm'  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz'  # The best model will be saved there
        validFreq=370  # Compute the validation error after this number of update.
        saveFreq=1110  # Save the parameters after every saveFreq updates
        maxlen=100  # Sequence longer then this get ignored
        batch_size=16  # The batch size during training.
        valid_batch_size=64  # The batch size used for validation/test set.
        dataset='imdb'

        # Parameter for extra option
        noise_std=0.
        use_dropout=True  # if False slightly faster, but worst test error
                           # This frequently need a bigger model.
        reload_model=None  # Path to a saved model we want to start from.
        test_size=-1  # If >0, we keep only this number of test example.
        

        model_options = locals().copy()
        ydim = np.max(train_x[1]) + 1
        model_options['ydim'] = ydim

        params = lstm.init_params(model_options)
        tparams = lstm.init_tparams(params)

        (use_noise, x, mask,
         y, f_pred_prob, f_pred, cost) = lstm.build_model(tparams, model_options)

        f_cost = theano.function([x,mask, y], cost, name = 'f_cost')
        
        grads = T.grad(cost, wrt=tparams.values())
        f_grad = theano.function([x, mask, y], grads, name='f_grad')

        lr = T.scalar(name='lr')
        f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                            x, mask, y, cost)
        
        print 'Optimization'

#        kf_valid = lstm.get_minibatches_idx(len(valid[0]), valid_batch_size)
#        kf_test = lstm.get_minibatches_idx(len(test[0]), valid_batch_size)

        history_errs = []
        best_p = None
        bad_count = 0

        uidx = 0  # the number of update done
        estop = False  # early stop

        for eidx in xrange(self.n_epochs):
            n_samples = 0

            kf = lstm.get_minibatches_idx(len(train_x), batch_size, shuffle = True)

            for _, train_index in kf:
                print 'hello'
                
            
            


        
        
                    
            

        print 'not implemented'

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
    [x, y] = generate_sample.generate_time_series()

    # plot
    kf = cross_validation.KFold(len(x), n_folds = 4, shuffle = True, random_state = 0)
    algorithm = 'lstm'

    result_list = []
    for train, test in kf:
        train_x = x[train]
        train_y = y[train]

        test_x = x[test]
        test_y = y[test]

        result = fit_and_test(train_x, train_y, test_x, test_y, algorithm)
        print result
        result_list.append(result)

    print 'sumup'
    print sumup_classification_result(result_list)
