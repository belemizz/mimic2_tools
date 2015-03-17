function icustay_datail = LoadIcustayDetail()
  f = fopen('../data/icu_admission_details.csv');
  filedata = textscan(f, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter',',');
  fclose(f);

  icustay_datail.icustay_id = cellfun(@str2double, filedata{1});
  icustay_datail.subject_id = cellfun(@str2double, filedata{2});
  icustay_datail.gender = filedata{3};
  icustay_datail.dob = datetime(strrep(filedata{4},'\N',''));
  icustay_datail.dod = datetime(strrep(filedata{5},'\N',''));
  icustay_datail.expire_flg = filedata{6};
  icustay_datail.subject_icustay_total_num = cellfun(@str2double, filedata{7});
  icustay_datail.subject_icustay_seq = cellfun(@str2double, filedata{8});
  icustay_datail.hadm_id = cellfun(@str2double, filedata{9});
  icustay_datail.hospital_total_num = cellfun(@str2double, filedata{10});
  icustay_datail.hospital_seq = cellfun(@str2double, filedata{11});
  icustay_datail.hospital_first_flg = filedata{12};
  icustay_datail.hospital_last_flg = filedata{13};
  icustay_datail.hospital_admit_dt = datetime(strrep(filedata{14},'\N',''));
  icustay_datail.hospital_disch_dt = datetime(strrep(filedata{15},'\N',''));
  icustay_datail.hospital_los = cellfun(@str2double, filedata{16});
  icustay_datail.hospital_expire_flg = filedata{17};
  icustay_datail.icustay_total_num = cellfun(@str2double, filedata{18});
  icustay_datail.icustay_seq = cellfun(@str2double, filedata{19});
  icustay_datail.icustay_first_flg = filedata{20};
  icustay_datail.icustay_last_flg = filedata{21};
  icustay_datail.icustay_intime = datetime(strrep(filedata{22},'\N',''));
  icustay_datail.icustay_outtime = datetime(strrep(filedata{23},'\N',''));
  icustay_datail.icustay_admit_age = cellfun(@str2double, filedata{24});
  icustay_datail.icustay_age_group = filedata{25};
  icustay_datail.icustay_los = cellfun(@str2double, filedata{26});
  icustay_datail.icustay_expire_flg = filedata{27};
  icustay_datail.icustay_first_careunit = filedata{28};
  icustay_datail.icustay_last_careunit = filedata{29};
  icustay_datail.icustay_first_service = filedata{30};
  icustay_datail.icustay_last_service = filedata{31};
  icustay_datail.height = cellfun(@str2double, filedata{32});
  icustay_datail.weight_first = cellfun(@str2double, filedata{33});
  icustay_datail.weight_min = cellfun(@str2double, filedata{34});
  icustay_datail.weight_max = cellfun(@str2double, filedata{35});
  icustay_datail.sapsi_first = cellfun(@str2double, filedata{36});
  icustay_datail.sapsi_min = cellfun(@str2double, filedata{37});
  icustay_datail.sapsi_max = cellfun(@str2double, filedata{38});
  icustay_datail.sofa_first = cellfun(@str2double, filedata{39});
  icustay_datail.sofa_min = cellfun(@str2double, filedata{40});
  icustay_datail.sofa_max = cellfun(@str2double, filedata{41});
  icustay_datail.matched_waveforms_num = cellfun(@str2double, filedata{42});

end

