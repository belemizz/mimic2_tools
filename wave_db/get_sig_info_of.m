function sig_info = get_sig_info_of(sig_url, metric_list)
% return signal information of sig_url for given metric_list

info = wfdbdesc(sig_url);

field = fieldnames(info);
n_field = length(field);
init = cell(1, n_field);

blank_struct = cell2struct(init, field,2);

if ~isempty(info)
  for midx = 1:length(metric_list)
    metric_in_info = strrep({info.Description}, ' ','');
    metric_index = find(ismember(metric_in_info, metric_list{midx}));
    if metric_index > 0
      sig_info(midx) = info(metric_index);
    else
      sig_info(midx) = blank_struct;
    end
  end
end

if ~exist('sig_info','var')
  sig_info = struct([]);
end

end
