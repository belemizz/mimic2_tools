'''
Experiments are recorded in this script
Experiment date, update date, and purpose should be recorded
'''
import alg.classification

from mutil import Graph, Stopwatch
graph = Graph()
sw = Stopwatch()


def class_weight_comparison():
    from get_sample.timeseries import TimeSeries
    import alg.timeseries
    ts = TimeSeries()
    sample = ts.normal(length=10, n_dim=2,
                       n_negative=900, n_positive=100, bias=[9, 10])
    train_set, test_set = sample.split_train_test()

    param = alg.timeseries.Default_param

    step = 0.005
    minimum = 0.1
    maximum = 0.16

    n_step = int((maximum - minimum) / step) + 1
    for idx in range(n_step):
        weight_0 = minimum + idx * step
        weight_1 = 1.0 - weight_0
        param.class_weight = {0: weight_0,
                              1: weight_1}
        result = alg.timeseries.fit_and_test(train_set, test_set, param=param)
        print (weight_0, result.f, result.recall, result.prec)

    print ('---')
    param.class_weight = "auto"
    result = alg.timeseries.fit_and_test(train_set, test_set, param=param)
    print ('auto', result.f, result.recall, result.prec)

    param.class_weight = None
    result = alg.timeseries.fit_and_test(train_set, test_set, param=param)
    print ('None', result.f, result.recall, result.prec)


def cycle_comparison(predictor, l_cycle, filename):
    result = predictor.compare_cycle(l_cycle)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    label = [int(1. / cycle) for cycle in l_cycle]
    comparison_label = 'Points per Day'
    graph.bar_classification(l_lab_result, label,
                             comparison_label=comparison_label,
                             title='Cycle Comparison: Lab',
                             filename=filename + '_CycleLab')
    graph.bar_classification(l_vit_result, label,
                             comparison_label=comparison_label,
                             title='Cycle Comparison: Vital',
                             filename=filename + '_CycleVit')
    print [r.lab.n_posi for r in result]
    print [r.lab.n_nega for r in result]


def duration_comparison(predictor, l_duration, filename):
    result = predictor.compare_duration(l_duration)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    label = [int(duration) for duration in l_duration]
    comparison_label = 'Duration in Days'

    graph.bar_classification(l_lab_result, label,
                             comparison_label=comparison_label,
                             title='Duration Comparison: Lab',
                             filename=filename + '_DurLab')
    graph.bar_classification(l_vit_result, label,
                             comparison_label=comparison_label,
                             title='Duration Comparison: Vital',
                             filename=filename + '_DurVit')
    print [r.lab.n_posi for r in result]
    print [r.lab.n_nega for r in result]


def death_prediction():
    from predict_death import PredictDeath

    class_param = alg.classification.Default_param
    tseries_param = alg.timeseries.Default_param
    pd = PredictDeath(max_id=0,
                      target_codes='chf',
                      n_lab=20,
                      disch_origin=False,
                      l_poi=0.,
                      tseries_duration=2.,
                      tseries_cycle=0.1,
                      class_param=class_param,
                      tseries_param=tseries_param,
                      n_cv_fold=10)

    l_cycle = [1., 0.5, 0.25, 0.2, 0.1]
    cycle_comparison(pd, l_cycle, 'death')

    l_duration = [1., 2., 3., 4., 5.]
    duration_comparison(pd, l_duration, 'death')
    graph.waitforbuttonpress()


def readmission_prediction():
    from predict_readmission import PredictReadmission

    # Parameter
    class_param = alg.classification.Default_param
    tseries_param = alg.timeseries.Default_param
    pr = PredictReadmission(max_id=0,
                            target_codes='chf',
                            matched_only=False,
                            n_lab=20,
                            disch_origin=True,
                            l_poi=0.,
                            tseries_flag=True,
                            tseries_duration=2.,
                            tseries_cycle=0.1,
                            visualize_data=False,
                            class_param=class_param,
                            tseries_param=tseries_param,
                            n_cv_fold=10)
    # comparison in cycle parameter
    l_cycle = [1., 0.5, 0.25, 0.2, 0.1]
    cycle_comparison(pr, l_cycle, 'readm')

    l_duration = [1., 2., 3., 4., 5.]
    duration_comparison(pr, l_duration, 'readm')
    graph.waitforbuttonpress()


def classify_vital_and_lab_timeseries():
    '''Predict death with vital and lab timeseries data
    :Experiment Date: 07/16/2015
    :Updated Date: 07/17/2015
    '''

    import evaluate_feature
    l_dbd = [0., 2.]
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
            graphs.comparison_bar([rec, prec, f], l_nsteps, ['recall', 'precision', 'F-measure'],
                          lim = [0,1], title = title, filename = filename)    

# Experiment Date: 07/01/2015
# Update Deate
def classify_vital_and_lab():
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
            graphs.comparison_bar([recall, precision, f_measure], alg, ['recall', 'precision', 'F-measure'], lim = [0,1], title = title, filename = filename)

        i = 4
        recall = [item['vital_class'][i-1].rec for item in result]
        precision = [item['vital_class'][i-1].prec for item in result]
        f_measure = [item['vital_class'][i-1].f for item in result]
        alg = [item['param']['class_alg'] for item in result]
        title = "Vital/ n_metrics = %d/ dbd=%d"%(i,dbd)
        filename = "Vital_n_metrics_%d_dbd_%d"%(i,dbd)
        graphs.comparison_bar([recall, precision, f_measure], alg, ['recall', 'precision', 'F-measure'], lim = [0,1], title = title, filename = filename)


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
    sw.reset()
#    readmission_prediction()
    death_prediction()
    sw.stop()
    sw.print_cpu_elapsed()
