function nidx_list = get_nidx_list_for(pid, numerics_all)
% return indexs of numerics list for a specific id

nidx_list = [];
for nidx = 1:length(numerics_all)
  if id_of(numerics_all{nidx}) == pid
    nidx_list = [nidx_list, nidx]; %#ok<AGROW>
  end
end

end

function pat_id = id_of(numerics)
pat_id = str2num(numerics(2:6));
end
