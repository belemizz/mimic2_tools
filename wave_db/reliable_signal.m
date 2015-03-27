function [tm_reliable, sig_reliable, tm_excluded] = reliable_signal(tm,sig)

var_width = 5;
sig2_ave = smooth(sig.^2,var_width,'moving');
sig_ave = smooth(sig,var_width,'moving');

var = sig2_ave - sig_ave.^2;

zero = 0.000001;
tm_reliable = tm(sig > 0 & var > zero);
sig_reliable = sig(sig > 0 & var> zero);
tm_excluded = tm(sig<=0 | var<=zero);

end
