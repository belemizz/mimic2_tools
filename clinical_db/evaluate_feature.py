"""
Evaluate the importance of the feature
"""

import control_mimic2db
import numpy
import collections
import datetime


def main( max_id = 200000):

    mimic2db = control_mimic2db.control_mimic2db()

    target_codes = ['428.0']
    id_list =  mimic2db.subject_with_icd9_codes(target_codes)
    subject_ids = [item for item in id_list if item < max_id]

    print "Number of candidates %d"%len(subject_ids)

    lab_ids = []
    for subject_id in subject_ids:
        patient = mimic2db.get_subject(subject_id)
        if patient:
            final_adm = patient.get_final_admission()
            if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:
                lab_ids = lab_ids+[item.itemid for item in final_adm.labs]

    counter =  collections.Counter(lab_ids)
    n_feature = 20
    most_commons = counter.most_common(n_feature)
    most_common_tests = [item[0] for item in most_commons]
    print most_commons
    print most_common_tests
    
    ## data aquisition

    ids = []
    values = []
    flags = []
    dbd = 1
    
    for subject_id in subject_ids:
        patient = mimic2db.get_subject(subject_id)
        if patient:
            final_adm = patient.get_final_admission()
            if len(final_adm.icd9)>0 and final_adm.icd9[0][3] == target_codes[0]:
                
                time_of_interest = final_adm.disch_dt + datetime.timedelta(1-dbd)
                lab_result =  final_adm.get_newest_lab_at_time(time_of_interest)

                value = [float('NaN')] * n_feature
                for item in lab_result:
                    if item[0] in most_common_tests:
                        index = most_common_tests.index(item[0])
                        value[index] = float(item[4])

                if True in numpy.isnan(value):
                    print value
                else:
                    values.append(value)
                    flags.append(patient.hospital_expire_flg)
                    ids.append(patient.subject_id)

    print ids
    print values
    print flags
                    
    X = numpy.array(values)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, flags)

    from sklearn.externals.six import StringIO
    with open("feature.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file = f)

    importance = clf.feature_importances_
    for index in range(len(most_commons)):
        print "%d:  %f"%(most_common_tests[index],importance[index])


if __name__ == '__main__':
    main()
