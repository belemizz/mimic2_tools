function [tm_reliable, sig_reliable, tm_excluded] = reliable_signal(tm,sig, var_width)

if nargin < 3
  var_width = 5;
end

zero = 0.000001;

if var_width > 0
  var_width = 5;
  sig2_ave = smooth(sig.^2,var_width,'moving');
  sig_ave = smooth(sig,var_width,'moving');
  var = sig2_ave - sig_ave.^2;
  
  tm_reliable = tm(sig > 0 & var > zero);
  sig_reliable = sig(sig > 0 & var> zero);
  tm_excluded = tm(sig<=0 | var<=zero);
else
  tm_reliable = tm(sig > 0);
  sig_reliable = sig(sig > 0);
  tm_excluded = tm(sig<=0);
end

end
