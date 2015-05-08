"""
A class for controling graph
"""

import numpy as np
import matplotlib.pyplot as plt

import math

class control_graph:
#    def __init__(self):

#    def __del__(self):


    def draw_med_icu(self, icustay, title, filename="", show_flag = True):
        result = icustay.medications
        base = icustay.intime
        fig, ax = self.figure_with_side_legend()

        for item in result:
            time_diff = self.time_diff_in_hour(item[3], base)
            value = np.array(item[5])
            plot_value, order = self.normalize(value)
            tag = "%s [%d %s]"%(item[1],order,item[2])
            ax.plot(time_diff,plot_value, 'o', label = tag)
                
        ax.set_title(title)
        ax.set_xlabel("Hours since ICU Admission")
        ax.set_ylabel("Amount of Dose")
        ax.legend(bbox_to_anchor = (1.02, 1),  loc = 'upper left', borderaxespad = 0)
        self.show_and_save(fig, filename, show_flag)

    def draw_lab_adm(self, admission, title, filename="", show_flag = True):
        base_time = admission.admit_dt
        data = admission.labs
        fig, ax = self.figure_with_side_legend()

        for item in data:
            time_diff = self.time_diff_in_hour(item[3], base_time)
            try:
                value = np.array([float(num) for num in item[4]])
                plot_val, order = self.normalize(value)
                tag = "%s [%d %s]"%(item[1], order, item[2])
                ax.plot(time_diff, plot_val, label = tag)                    
            except ValueError:
                print "Can't plot %s"%item[1]
                #print item[4]

        ax.set_title(title)
        ax.set_xlabel("Hours since Admission")
        ax.legend(bbox_to_anchor = (1.02, 1),  loc = 'upper left', borderaxespad = 0)
        self.show_and_save(fig, filename, show_flag)

    def draw_lab_adm_itemid(self, admission, itemids, title, filename = "", show_flag = True):
        base_time = admission.admit_dt
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_title(title)
        ax1.set_xlabel("Hours since Admission")
        
        colors = ['b', 'r']
        axis = [ax1, ax2]

        for idx, id in enumerate(itemids):
            data = admission.get_lab_itemid(id)
            time_diff = self.time_diff_in_hour(data[3], base_time)
            values = data[4]
            axis[idx].plot(time_diff, values, "%ss--"%colors[idx])
            axis[idx].set_ylabel("%s [%s]"%(data[1], data[2]), color = colors[idx])

        self.show_and_save(fig, filename, show_flag)

    def show_and_save(self, fig, filename, show_flag):
        if len(filename) > 0:
            fig.savefig(filename)
        if show_flag:
            fig.show()

    def normalize(self, value):
        max_val = max(abs(value))
        order = 10 ** int(math.log10(float(max_val)))
        n_value = value / order
        return n_value, order

    def figure_with_side_legend(self):
        fig = plt.figure(figsize = plt.figaspect(0.5))
        ax = fig.add_axes([.05,.1,.6,.8])
        return fig, ax

    def time_diff_in_hour(self, time_seq, base_time):
        return [(item - base_time).total_seconds()/3600 for item in time_seq]












