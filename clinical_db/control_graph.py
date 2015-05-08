"""
A class for controling graph
"""

import numpy as np
import matplotlib.pyplot as plt

import pdb

class control_graph:
#    def __init__(self):

#    def __del__(self):


    def draw_med_icu(self, icustay, title, filename="", show_flag = True):
        result = icustay.medications
        base = icustay.intime

        item = result[0]
        fig = plt.figure(figsize = plt.figaspect(0.5))
        ax = fig.add_axes([.05,.1,.6,.8])

        for item in result:

            time_diff = self.time_diff_in_hour(item[3], base)

            if max(item[5]) > 100:
                value = np.array(item[5])/1000
                tag = "%s [1000 %s]"%(item[1],item[2])
            elif max(item[5]) > 10:
                value = np.array(item[5])/100
                tag = "%s [100 %s]"%(item[1],item[2])
            elif max(item[5]) > 1:
                value = np.array(item[5])/10
                tag = "%s [10 %s]"%(item[1],item[2])
            else:
                value = np.array(item[5])
                tag = "%s [%s]"%(item[1],item[2])

            ax.plot(time_diff,value, 'o', label = tag)
                
        ax.set_title(title)
        ax.set_xlabel("Hours since ICU Admission")
        ax.set_ylabel("Amount of Dose")
        ax.legend(bbox_to_anchor = (1.02, 1),  loc = 'upper left', borderaxespad = 0)

        if len(filename) > 0:
            fig.savefig(filename)

        if show_flag:
            fig.show()

    def time_diff_in_hour(self, time_seq, base_time):
        return [(item - base_time).total_seconds()/3600 for item in time_seq]


