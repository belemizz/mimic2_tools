''' 
Experiments are recorded in this script
Experiment date, update date, and purpose should be recorded
'''
import matplotlib.pyplot as plt
import mutil.graph
import alg.classification


# Experiment Date: 07/16/2015
# Update Date: 07/17/2015
def classify_vital_and_lab_timeseries():
    import mutil.graph
    graphs = mutil.graph.Graph()

    import evaluate_feature
    import alg.classification
    l_dbd = [0,2]
    l_nsteps = range(1, 6)
    for dbd in l_dbd:
        result = []
        for n_steps in l_nsteps:
            efo = evaluate_feature.evaluate_fetaure(
                max_id=200000,
                n_lab = 20,
                days_before_discharge = dbd,
                rp_learn_flag = False,
                n_cv_folds = 10,
                tseries_freq = 1.0,
                tseries_steps = n_steps,
                )
            result.append(efo.tseries_eval())

        long_label = ['Lab', 'Vital']
        short_label = ['Lab', 'Vit']
        for i in range(2):
            rec = [item[i].rec for item in result]
            prec = [item[i].prec for item in result]
            f = [item[i].f for item in result]
            title = "%s_tseriess_dbd_%d"%(short_label[i], dbd)
            filename = "%s_tseries_dbd_%d"%(short_label[i], dbd)
            graphs.bar_pl([rec, prec, f], l_nsteps, ['recall', 'precision', 'F-measure'],
                          xlim = [0,1], title = title, filename = filename)    

    
# Experiment Date: 07/01/2015
# Update Deate
def classify_vital_and_lab():
    import mutil.graph
    graphs = mutil.graph.Graph()

    import evaluate_feature
    import alg.classification

    dbd_list = [0,2]
    for dbd in dbd_list:
        result = []
        for alg in alg.classification.algorithm_list:
            efo = evaluate_feature.evaluate_fetaure(
                max_id = 200000,
                n_lab = 20,
                days_before_discharge = dbd,
                rp_learn_flag = False,
                n_cv_folds = 4,
                class_alg = alg
                )
            result.append(efo.point_eval())

        for i in [10, 20]:
            recall = [item['lab_class'][i-1].rec for item in result]
            precision = [item['lab_class'][i-1].prec for item in result]
            f_measure = [item['lab_class'][i-1].f for item in result]
            alg = [item['param']['class_alg'] for item in result]
            title = "Lab/ n_metrics = %d/ dbd=%d"%(i,dbd)
            filename = "Lab_n_metrics_%d_dbd_%d"%(i,dbd)
            graphs.bar_pl([recall, precision, f_measure], alg, ['recall', 'precision', 'F-measure'], xlim = [0,1], title = title, filename = filename)

        i = 4
        recall = [item['vital_class'][i-1].rec for item in result]
        precision = [item['vital_class'][i-1].prec for item in result]
        f_measure = [item['vital_class'][i-1].f for item in result]
        alg = [item['param']['class_alg'] for item in result]
        title = "Vital/ n_metrics = %d/ dbd=%d"%(i,dbd)
        filename = "Vital_n_metrics_%d_dbd_%d"%(i,dbd)
        graphs.bar_pl([recall, precision, f_measure], alg, ['recall', 'precision', 'F-measure'], xlim = [0,1], title = title, filename = filename)


# Experiment Date:06/24/2015
# Update Date:
def compare_lab_tests_and_vitals():
    ''' Compare the ability of prediction with lab tests and vital test for HF patients'''

    import evaluate_feature
    efo = evaluate_feature.evaluate_fetaure( max_id = 200000,
                                             target_codes = ['428.0'],
                                             n_lab = 20,
                                             days_before_discharge = 0,
                                             rp_learn_flag = False,
                                             n_cv_folds = 1,
                                             pca_components = 4,
                                             ica_components = 4,
                                             dae_hidden = 40,
                                             dae_corruption = 0.3)
                                             
    efo.compare_dbd([0., 0.25, 0.5, 1., 2., 3.])
    
    efo.rp_learn_flag = True
    efo.n_cv_folds = 4
    efo.point_eval()

    efo.n_lab = 10
    efo.dae_hidden = 20
    efo.point_eval()

if __name__ == '__main__':
#    compare_lab_tests_and_vitals()
    classify_vital_and_lab_timeseries()
    plt.waitforbuttonpress()
    
    
