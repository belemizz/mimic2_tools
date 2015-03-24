function numerics_all = load_numerics_all()
% load the names of the numerics files from the file
f = fopen('../data/numerics_list.dat');
temp = textscan(f,'%s');
numerics_all = temp{1};
fclose(f);

end



