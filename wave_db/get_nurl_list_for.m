function nurl_list = get_nurl_list_for(pid, numerics_all)
% return url of numerics list for a given specific patient id, pid

base = 'mimic2wdb/matched';
sig_url = @(nidx) sprintf('%s/%s',base, numerics_all{nidx});

nurl_list = {};
for nidx = 1:length(numerics_all)
  if id_of(numerics_all{nidx}) == pid
    nurl = sig_url(nidx);
    nurl_list = [nurl_list, nurl]; %#ok<AGROW>
  end
end

end

function pat_id = id_of(numerics)

pat_id = str2double(numerics(2:6));

end
