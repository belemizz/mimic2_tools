import glob
import os
ret = glob.glob("./HR_SpO2*.png")
cols = 4
rows = 4

group = [ret[i:i+cols*rows] for i in range(0,len(ret), cols*rows)]

for idx in range(0,len(group)):
    files = " ".join(group[idx])
    command = "montage -tile %dx%d -resize 100%% -geometry +0+0 %s out%d.png"%(cols,rows,files,idx)
    os.system(command)
