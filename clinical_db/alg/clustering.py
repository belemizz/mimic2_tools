"""Algorithm for basic classtering."""
import numpy as np
from mutil import p_info
from bunch import Bunch

L_algorithm = ['alg1']
Default_param = Bunch(name='lr')


def example():
    comat, l_freq = generate_comat()
    group_comat(comat, l_freq, n_sample=100)


def group_comat(comat, hist, n_sample, th_pmi=0.0, th_js=0.0):
    n_label = comat.shape[0]
    mat_pmi = np.zeros((n_label, n_label))
    mat_js = np.zeros((n_label, n_label))

    p_info('Compute Distance Matrix')
    for i in range(n_label):
        for j in range(n_label):
            if j < i:
                co_pmi = pmi(comat[i, j], hist[i], hist[j], n_sample)
                mat_pmi[i][j] = mat_pmi[j][i] = co_pmi

                co_js = js_div(distribution(comat[i]), distribution(comat[j]))
                mat_js[i][j] = mat_js[j][i] = co_js

    p_info('Compute Group')
    l_group = np.arange(n_label)
    js_counter = 0
    pmi_counter = 0
    for i in range(1, n_label):
        for j in range(i):
            if mat_js[i][j] < th_js:
                l_group[i] = l_group[j]
                js_counter += 1
            elif mat_pmi[i][j] > th_pmi:
                l_group[i] = l_group[j]
                pmi_counter += 1
    print (js_counter, pmi_counter)
    return l_group


def pmi(freq_ij, freq_i, freq_j, n_all):
    return np.log2(float(n_all * freq_ij) / (freq_i * freq_j))


def js_div(p_1, p_2):
    def kld(p1, p2):
        p2[np.where(p2 == 0.)] = p1[np.where(p2 == 0.)]
        kld = p1 * np.log2(p1 / p2)
        kld[np.where(np.isnan(kld))] = 0.
        return sum(kld)

    m = 0.5 * (p_1 + p_2)
    jsd = 0.5 * (kld(p_1, m) + kld(p_2, m))
    return jsd


def euc_div(p_1, p_2):
    return np.linalg.norm(p_1 - p_2)


def distribution(histogram):
    return histogram.astype('float') / sum(histogram)


def generate_comat():
    n_size = 10
    min_freq = 10
    max_freq = 30
    comat = np.zeros((n_size, n_size)).astype('int')

    l_freq = [np.random.randint(min_freq, max_freq) for i in range(n_size)]

    for i in range(n_size):
        for j in range(n_size):
            if j < i:
                max_val = min(l_freq[i], l_freq[j])
                value = np.random.randint(max_val)
                comat[i][j] = value
                comat[j][i] = value
            elif j == i:
                comat[i][i] = 0
    return comat, l_freq
