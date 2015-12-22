# Before use
1. Prepare mimic2 clinical database
2. Install libraries
3. Run either of the following scripts

- The scripts are tested on only on Ubuntu 14.04 and python 2.7.6.

# Scripts
- analyze_icd9.py: Count frequency of icd 9 code
- list_id_form_code.py Generate a list of subject id that have the selected icd 9 code 
- patient_statistics.py: Show patient statistics
- evaluate_feature.py: Evaluate importance of the features
- show_medical_record.py: Display graphs of a selected patient's medical data
- predict_death.py: Predict death of the patients by physiological and lab test data
- predict_readmission.py: Predict readmission of the patients by the physiological and lab test data
- experiment_script.py: Examples of experiments

# Packages
- alg: algorithm implementation
- get_sample: package for getting data
- mutil: Utility classes and functions
- test: Test scripts

# The following scripts are experimental
- patient_classification.py
- classify_patients.py
- learn_visualize_feature.py

# Preparing mimic2 clinical database
1. Make physionet account and get approved to access to the database. Detailed instruction is available on the following link. https://physionet.org/mimic2/mimic2_clinical_flatfiles.shtml
2. You should be able to access the folloing url when you get approval. https://physionet.org/works/MIMICIIClinicalDatabase/files/
3. Download the MIMIC2 importer from there and setup database in your own computer. PostgreSQL is recommended. For full install, huge diskspace (~100GB) is needed.
