"""
A class for controling graph
"""
import numpy as np
import matplotlib.pyplot as plt
import math

class control_graph:
#    def __init__(self):

#    def __del__(self):

    def draw_med_icu(self, icustay, base_time,  title, filename="", show_flag = True):
        data = icustay.medications
        fig, ax = self.figure_with_side_legend()

        for item in data:
            time_diff = self.time_diff_in_hour(item[3], base_time)
            value = np.array(item[5])
            plot_value, order = self.normalize(value)
            tag = "%s [%0.1f %s]"%(item[1], order, item[2])
            ax.plot(time_diff,plot_value, 'o', label = tag)
                
        ax.set_title(title)
        ax.set_xlabel("Hours since Admission")

        self.show_legend(ax)
        self.show_icustay_span_in_icu(ax, icustay, base_time)
        self.show_and_save(fig, filename, show_flag)
        
    def draw_chart_icu(self, icustay, base_time, title, filename="", show_flag = True):
        data = icustay.charts
        fig, ax = self.figure_with_side_legend()

        for item in data:
            time_diff = self.time_diff_in_hour(item[3], base_time)

            try:
                value = np.array([float(num) for num in item[5]])
                plot_val, order = self.normalize(value)
                tag = "%s [%0.1f %s]"%(item[1], order, item[2])
                ax.plot(time_diff, plot_val,  label = tag )
            except ValueError:
                print "Can't plot %s"%item[1]

        ax.set_title(title)
        ax.set_xlabel("Hours since Admission")

        self.show_legend(ax)
        self.show_icustay_span_in_icu(ax, icustay, base_time)
        self.show_and_save(fig, filename, show_flag)

    def draw_io_icu(self, icustay, base_time, title, filename="", show_flag = True):
        data = icustay.ios
        fig, ax = self.figure_with_side_legend()
        
        for item in data:
            time_diff = self.time_diff_in_hour(item[3], base_time)

            try:
                value = np.array([float(num) for num in item[5]])
                plot_val, order = self.normalize(value)
                tag = "%s [%0.1f %s]"%(item[1], order, item[2])
                ax.plot(time_diff, plot_val,'o',  label = tag )
            except ValueError:
                print "Can't plot %s"%item[1]

        ax.set_title(title)
        ax.set_xlabel("Hours since Admission")

        self.show_legend(ax)
        self.show_icustay_span_in_icu(ax, icustay, base_time)
        self.show_and_save(fig, filename, show_flag)

    def draw_lab_adm(self, admission, title, filename="", show_flag = True):
        base_time = admission.admit_dt
        data = admission.labs
        fig, ax = self.figure_with_side_legend()
        counter = 0

        for item in data:
            time_diff = self.time_diff_in_hour(item[3], base_time)
            try:
                value = np.array([float(num) for num in item[4]])
                plot_val, order = self.normalize(value)
                tag = "%s [%0.1f %s]"%(item[1],order,item[2])
                ax.plot(time_diff, plot_val, label = tag)
                counter = counter + 1
                if counter >= 30:
                    break
            except ValueError:
                print "Can't plot %s"%item[1]

        ax.set_title(title)
        ax.set_xlabel("Hours since Admission")

        self.show_legend(ax)
        self.show_icustay_span(ax, admission)
        self.show_and_save(fig, filename, show_flag)

    def draw_lab_adm_itemid(self, admission, itemids, title, filename = "", show_flag = True):
        base_time = admission.admit_dt

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        colors = ['b', 'r']
        axis = [ax1, ax2]

        for idx, id in enumerate(itemids):
            data = admission.get_lab_itemid(id)
            time_diff = self.time_diff_in_hour(data[3], base_time)
            values = data[4]
            axis[idx].plot(time_diff, values, "%ss--"%colors[idx])
            axis[idx].set_ylabel("%s [%s]"%(data[1], data[2]), color = colors[idx])

        ax1.set_title(title)
        ax1.set_xlabel("Hours since Admission")
        
        self.show_icustay_span(ax1, admission)
        self.show_and_save(fig, filename, show_flag)

    def draw_lab_distribution(self, expire_values, recover_values, title, filename = "", show_flag = True):
        fig, ax = plt.subplots()
        for value in expire_values:
            ax.plot(value[0], value[1], "ro")
        for value in recover_values:
            ax.plot(value[0], value[1], "bo")

        ax.set_xlabel("Creatinine [mg/dL]")
        ax.set_ylabel("Urea Nitrogen[mg/dL]")

        self.show_and_save(fig, filename, show_flag)
        
    def show_icustay_span_in_icu(self, ax, icustay, base_time):
        icu_io = self.time_diff_in_hour([icustay.intime, icustay.outtime],base_time)
        ax.axvspan(icu_io[0], icu_io[1], alpha = 0.2, color = 'red')

    def show_icustay_span(self, ax, admission):
        base_time = admission.admit_dt
        icu_ios = [self.time_diff_in_hour([icustay.intime, icustay.outtime],base_time) for icustay in admission.icustays]
        for span in icu_ios:
            ax.axvspan(span[0], span[1], alpha = 0.2, color = 'red')


    def show_and_save(self, fig, filename, show_flag):
        if len(filename) > 0:
            fig.savefig(filename)
        if show_flag:
            fig.show()

    def normalize(self, value):
        max_val = max(abs(value))
        order = 10.0 ** int(math.log10(float(max_val)))
        n_value = value / order
        return n_value, order

    def figure_with_side_legend(self):
        fig = plt.figure(figsize = plt.figaspect(0.5))
        ax = fig.add_axes([.05,.1,.5,.8])
        return fig, ax

    def show_legend(self, ax):
        ax.legend(bbox_to_anchor = (1.02, 1),  loc = 'upper left', borderaxespad = 0, prop = {'size':8})

    def time_diff_in_hour(self, time_seq, base_time):
        return [(item - base_time).total_seconds()/3600 for item in time_seq]