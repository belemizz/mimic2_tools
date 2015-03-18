function [ pidx_list ] = PidOfIcustayExpireFlagEq(flag)
% list pidx of the patient who has icustay expire flag equals given value
% flag should be true or false

  data = RawIcustayDetail();
  subject_id = cellfun(@str2double, data{2});
  icustay_expire_flg = data{27};
  
  y_match_flag = ismember(icustay_expire_flg, 'Y');
  y_pidx_list = unique(subject_id(y_match_flag));
  
  if flag
    pidx_list = y_pidx_list;
  else %flag = false
    pidx_list = setdiff(subject_id, y_pidx_list);
  end

end

