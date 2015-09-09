import numpy as np
import random


def sample(source_num=0, n_dim=0, n_flag=2):
    """Get n-dim vector samples.

    :return: [x,y]
    x: 2-d array [sample, feature]
    y: 1-d array of labels
    """
    if source_num is 0:
        l_amount = [100] * n_flag
        bias = range(n_flag)
        [x, y] = normal_dist(n_dim, l_amount, bias, seed=1)

    elif source_num is 1:
        from sklearn import datasets
        iris = datasets.load_iris()
        x, y = chop_data(iris.data, iris.target, n_dim, n_flag)

    elif source_num is 2:
        from logistic_sgd import load_data
        datasets = load_data('mnist.pkl.gz')
        [shared_x, shared_y] = datasets[0]
        if n_dim > 0 and n_flag > 0:
            x, y = chop_data(shared_x.get_value(), shared_y.eval(),
                             n_dim, n_flag)
        else:
            x, y = shared_x.get_value(), shared_y.eval()

    else:
        raise ValueError
    return x, y


def chop_data(all_data, all_target, data_dim, n_flag):
    """Reduce the number of category of the flags to n_flag."""
    all_flag = np.unique(all_target)
    flags = all_flag[0: min(all_flag.shape[0], n_flag)]

    x_list = []
    y_list = []
    for flag in flags:
        x_list.append(all_data[all_target == flag])
        y_list.append(all_target[all_target == flag])

    x = np.vstack(x_list)
    y = np.hstack(y_list)

    if data_dim > 0:
        x = x[:, 0:data_dim]

    return x, y


def normal_dist(n_dim=2, l_amount=[100, 100], bias=[-2, 2], seed=1):
    """Generate 2 element samples of normal distribution."""
    data = []

    random.seed(seed)
    np.random.seed(seed)

    for i, amount in enumerate(l_amount):
        for j in xrange(amount):
            vec = np.random.randn(1, n_dim) + bias[i]
            flag = i
            data.append([vec, flag])

    random.shuffle(data)

    x = np.array([item[0][0] for item in data])
    y = np.array([item[1] for item in data])

    return [x, y]


def uniform_dist(n_dim=2, n_sample=100, minimum=0.0, maximum=1.0, seed=1):
    """Generate samples by uniform distribution."""
    np.random.seed(seed)
    return np.random.uniform(minimum, maximum, (n_sample, n_dim))
