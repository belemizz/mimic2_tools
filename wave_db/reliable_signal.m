function [tm_reliable, sig_reliable, tm_excluded] = reliable_signal(tm,sig, var_width)
% extract reliable data points and return:
% tm_reliable
%   timestamps of the reliable points
% sig_reliable
%   values of the reliable points
% tm_excluded
%   timestapms of the excluded datapoints
%

if nargin < 3
  var_width = 7;
end

zero = 0.000001;

if var_width > 0
  sig2_ave = smooth(sig.^2,var_width,'moving');
  sig_ave = smooth(sig,var_width,'moving');
  var = sig2_ave - sig_ave.^2;
  
  if isempty(sig2_ave)
    tm_reliable = [];
    sig_reliable = [];
    tm_excluded = tm;
  else
    tm_reliable = tm( ~isnan(sig) & sig > 0 & var > zero);
    sig_reliable = sig( ~isnan(sig) & sig > 0 & var> zero);
    tm_excluded = tm( isnan(sig) | sig<=0 | var<=zero);
  end
else
  tm_reliable = tm(~isnan(sig) & sig > 0);
  sig_reliable = sig(~isnan(sig) & sig > 0);
  tm_excluded = tm( isnan(sig) & sig<=0);
end

end
