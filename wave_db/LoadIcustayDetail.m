function icustay_datail = LoadIcustayDetail()
  data = RawIcustayDetail();

  icustay_datail.icustay_id = cellfun(@str2double, data{1});
  icustay_datail.subject_id = cellfun(@str2double, data{2});
  icustay_datail.gender = data{3};
  icustay_datail.dob = datetime(strrep(data{4},'\N',''));
  icustay_datail.dod = datetime(strrep(data{5},'\N',''));
  icustay_datail.expire_flg = data{6};
  icustay_datail.subject_icustay_total_num = cellfun(@str2double, data{7});
  icustay_datail.subject_icustay_seq = cellfun(@str2double, data{8});
  icustay_datail.hadm_id = cellfun(@str2double, data{9});
  icustay_datail.hospital_total_num = cellfun(@str2double, data{10});
  icustay_datail.hospital_seq = cellfun(@str2double, data{11});
  icustay_datail.hospital_first_flg = data{12};
  icustay_datail.hospital_last_flg = data{13};
  icustay_datail.hospital_admit_dt = datetime(strrep(data{14},'\N',''));
  icustay_datail.hospital_disch_dt = datetime(strrep(data{15},'\N',''));
  icustay_datail.hospital_los = cellfun(@str2double, data{16});
  icustay_datail.hospital_expire_flg = data{17};
  icustay_datail.icustay_total_num = cellfun(@str2double, data{18});
  icustay_datail.icustay_seq = cellfun(@str2double, data{19});
  icustay_datail.icustay_first_flg = data{20};
  icustay_datail.icustay_last_flg = data{21};
  icustay_datail.icustay_intime = datetime(strrep(data{22},'\N',''));
  icustay_datail.icustay_outtime = datetime(strrep(data{23},'\N',''));
  icustay_datail.icustay_admit_age = cellfun(@str2double, data{24});
  icustay_datail.icustay_age_group = data{25};
  icustay_datail.icustay_los = cellfun(@str2double, data{26});
  icustay_datail.icustay_expire_flg = data{27};
  icustay_datail.icustay_first_careunit = data{28};
  icustay_datail.icustay_last_careunit = data{29};
  icustay_datail.icustay_first_service = data{30};
  icustay_datail.icustay_last_service = data{31};
  icustay_datail.height = cellfun(@str2double, data{32});
  icustay_datail.weight_first = cellfun(@str2double, data{33});
  icustay_datail.weight_min = cellfun(@str2double, data{34});
  icustay_datail.weight_max = cellfun(@str2double, data{35});
  icustay_datail.sapsi_first = cellfun(@str2double, data{36});
  icustay_datail.sapsi_min = cellfun(@str2double, data{37});
  icustay_datail.sapsi_max = cellfun(@str2double, data{38});
  icustay_datail.sofa_first = cellfun(@str2double, data{39});
  icustay_datail.sofa_min = cellfun(@str2double, data{40});
  icustay_datail.sofa_max = cellfun(@str2double, data{41});
  icustay_datail.matched_waveforms_num = cellfun(@str2double, data{42});

end

