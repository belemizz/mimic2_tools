''' 
Experiments are recorded in this script
Experiment date, update date, and purpose should be recorded
'''

import matplotlib.pyplot as plt

# Experiment Date:06/24/2015
# Update Date:
def compare_lab_tests_and_vitals():
    ''' Compare the ability of prediction with lab tests and vital test for HF patients'''

    import evaluate_feature
    efo = evaluate_feature.evaluate_fetaure( max_id = 2000,
                                             target_codes = ['428.0'],
                                             n_lab = 20,
                                             days_before_discharge = 0,
                                             rp_learn_flag = False,
                                             n_cv_folds = 1
                                             )
    print efo.compare_dbd([0,1])
   

    ## result = []
    ## for day in [0, 1, 2, 3]:
    ##     result.append( evaluate_feature.main(
    ##         max_id = 200000,
    ##         target_codes = ['428.0'],
    ##         n_lab = 20,
    ##         days_before_discharge = day,
    ##         rp_learn_flag = False,
    ##         n_cv_folds = 1)
    ##         )
    

if __name__ == '__main__':
    compare_lab_tests_and_vitals()
    plt.waitforbuttonpress()
    
