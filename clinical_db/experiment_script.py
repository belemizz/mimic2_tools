'''
Experiments are recorded in this script
Experiment date, update date, and purpose should be recorded
'''
from patient_classification import Default_db_param, Default_data_param, Default_alg_param

from mutil import Graph, Stopwatch
graph = Graph()
sw = Stopwatch()
from bunch import Bunch
from mutil import Csv


def list_matched_id():
    from predict_readmission import PredictReadmission
    db_param = Default_db_param
    data_param = Default_data_param
    alg_param = Default_alg_param

    db_param.matched_only = True

    pd = PredictReadmission(db_param, data_param, alg_param)
    csv = Csv('../data/matched_id.csv')
    csv.write_single_list(pd.id_list)


def class_weight_comparison():
    import alg.timeseries
    from get_sample.timeseries import TimeSeries
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
    result = predictor.compare_cycle(l_cycle, True)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    label = ['Point'] + [int(1. / cycle) for cycle in l_cycle]
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


def span_comparison(predictor, l_span, filename):
    result = predictor.compare_span(l_span, True)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    label = ['Point'] + l_span
    comparison_label = 'Point / Timeseries'

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


def coef_span_comparison(predictor, l_span, filename):
    result = predictor.compare_coef(l_span, True, True)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    label = ['Point'] + ['Tseries'] + l_span
    comparison_label = 'coef_span'

    graph.bar_classification(l_lab_result, label,
                             comparison_label=comparison_label,
                             title='Coef Span Comparison:Lab',
                             filename=filename + '_CoefLab')
    graph.bar_classification(l_vit_result, label,
                             comparison_label=comparison_label,
                             title='Coef Span Comparison:Vit',
                             filename=filename + '_CoefVit')

    print [r.lab.n_posi for r in result]
    print [r.lab.n_nega for r in result]


def alg_comparison(predictor, l_alg_param, l_label, filename):
    result = predictor.compare_class_alg(l_alg_param)
    l_lab_result = [r.lab for r in result]
    l_vit_result = [r.vit for r in result]

    comparison_label = 'Algorithm'
    graph.bar_classification(l_lab_result, l_label,
                             comparison_label=comparison_label,
                             title='Alg Comparison:Lab',
                             filename=filename + '_AlgLab')
    graph.bar_classification(l_vit_result, l_label,
                             comparison_label=comparison_label,
                             title='Alg Comparison:Vit',
                             filename=filename + '_AlgVit')
    print [r.lab.n_posi for r in result]
    print [r.lab.n_nega for r in result]
    print [r.lab.f for r in result]


def death_prediction():
    from predict_death import PredictDeath

    db_param = Default_db_param
    data_param = Default_data_param
    alg_param = Default_alg_param

    # base parameter
    data_param.disch_origin = False
    data_param.tseries_flag = False
    data_param.span = [0., 1.]
    data_param.tseries_cycle = 0.25

    pd = PredictDeath(db_param, data_param, alg_param)

    def feature_importance_graph(importance, filename):
        ent_reduction = [item[0] for item in importance]
        labels = [item[3] for item in importance]
        graph.bar_feature_importance(ent_reduction, labels, filename)
    importance = pd.ent_reduction_point()
    feature_importance_graph(importance, 'pd')

    algorithm_flag = True
    if algorithm_flag:
        l_alg_label = ['lr', 'dt', 'svm']
        l_param = []
        for name in l_alg_label:
            param = Bunch(alg_param.class_param.copy())
            param.name = name
            l_param.append(param)
        alg_comparison(pd, l_param, l_alg_label, 'pd_alg1')

        l_alg_label = ['lr', 'pca_lr', 'dae_lr']
        l_param = []
        for name in l_alg_label:
            param = Bunch(alg_param.class_param.copy())
            param.name = name
            l_param.append(param)
        alg_comparison(pd, l_param, l_alg_label, 'pd_alg2')

    span_flag = False
    if span_flag:
        l_span = [data_param.span]
        span_comparison(pd, l_span, 'pd')

    cycle_flag = False
    if cycle_flag:
        l_cycle = [0.5, 0.25, 0.2, 0.1]
        cycle_comparison(pd, l_cycle, 'pd')

    coef_flag = False
    if coef_flag:
        l_span = [data_param.span]
        coef_span_comparison(pd, l_span, 'pd')

    graph.waitforbuttonpress()


def readmission_prediction():
    from predict_readmission import PredictReadmission

    # Parameter
    db_param = Default_db_param
    data_param = Default_data_param
    alg_param = Default_alg_param

    data_param.disch_origin = True
    data_param.tseries_flag = False
    data_param.span = [-2., 0.]
    data_param.tseries_cycle = 0.25

    pd = PredictReadmission(db_param, data_param, alg_param)

    def feature_importance_graph(importance, filename):
        ent_reduction = [item[0] for item in importance]
        labels = [item[3] for item in importance]
        graph.bar_feature_importance(ent_reduction, labels, filename)
    importance = pd.ent_reduction_point()
    feature_importance_graph(importance, 'pd')

    algorithm_flag = False
    if algorithm_flag:
        l_alg_label = ['lr', 'dt', 'svm']
        l_param = []
        for name in l_alg_label:
            param = Bunch(alg_param.class_param.copy())
            param.name = name
            l_param.append(param)
        alg_comparison(pd, l_param, l_alg_label, 'pd_alg1')

        l_alg_label = ['lr', 'pca_lr', 'dae_lr']
        l_param = []
        for name in l_alg_label:
            param = Bunch(alg_param.class_param.copy())
            param.name = name
            l_param.append(param)
        alg_comparison(pd, l_param, l_alg_label, 'pd_alg2')

    span_flag = False
    if span_flag:
        l_span = [data_param.span]
        span_comparison(pd, l_span, 'pd')

    cycle_flag = False
    if cycle_flag:
        l_cycle = [0.5, 0.25, 0.2, 0.1]
        cycle_comparison(pd, l_cycle, 'pd')

    coef_flag = False
    if coef_flag:
        l_span = [data_param.span]
        coef_span_comparison(pd, l_span, 'pd')

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
    list_matched_id()
#    readmission_prediction()
#    death_prediction()
    sw.stop()
    sw.print_cpu_elapsed()
