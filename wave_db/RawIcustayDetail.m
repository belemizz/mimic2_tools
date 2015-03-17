function [ data ] = RawIcustayDetail()
  f = fopen('../data/icu_admission_details.csv');
  data = textscan(f, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter',',');
  fclose(f);
end

