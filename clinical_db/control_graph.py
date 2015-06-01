"""
A class for controling graph
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from more_itertools import chunked

class control_graph:
    def __init__(self):
        self.limit_timeseries = 25

#    def __del__(self):

    def draw_lab_adm(self, admission, title, filename="", show_flag = True):
        base_time = admission.admit_dt
        data = admission.labs
        plot_list = self.__get_plot_list(base_time, data)
        icu_ios = [self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time) for icustay in admission.icustays]
        self.__draw_series_with_legend(plot_list, icu_ios, title, filename, show_flag)

    def draw_med_icu(self, icustay, base_time,  title, filename="", show_flag = True):
        data = icustay.medications
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime],base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag, 'o')
        
    def draw_chart_icu(self, icustay, base_time, title, filename="", show_flag = True):
        data = icustay.charts
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime],base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag)

    def draw_io_icu(self, icustay, base_time, title, filename="", show_flag = True):
        data = icustay.ios
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime],base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag,'o')

    def draw_lab_adm_itemid(self, admission, itemids, title, filename = "", show_flag = True):
        base_time = admission.admit_dt

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        colors = ['b', 'r']
        axis = [ax1, ax2]

        for idx, id in enumerate(itemids):
            data = admission.get_lab_itemid(id)
            time_diff = self.__time_diff_in_hour(data.timestamps, base_time)
            values = data.values
            axis[idx].plot(time_diff, values, "%ss--"%colors[idx])
            axis[idx].set_ylabel("%s [%s]"%(data.description, data.unit), color = colors[idx])

        ax1.set_title(title)
        ax1.set_xlabel("Hours since Admission")
        
        base_time = admission.admit_dt
        icu_ios = [self.__time_diff_in_hour([icustay.intime, icustay.outtime],base_time) for icustay in admission.icustays]
        for span in icu_ios:
            ax1.axvspan(span[0], span[1], alpha = 0.2, color = 'red')
        self.__show_and_save(fig, filename, show_flag)

    def draw_lab_distribution(self, expire_values, recover_values, title, filename = "", show_flag = True):
        fig, ax = plt.subplots()
        for value in expire_values:
            ax.plot(value[0], value[1], "ro")
        for value in recover_values:
            ax.plot(value[0], value[1], "bo")

        ax.set_xlabel("Creatinine [mg/dL]")
        ax.set_ylabel("Urea Nitrogen[mg/dL]")

        self.__show_and_save(fig, filename, show_flag)

    def plot_classification(self, positive, negative, line, title, filename = "", show_flag = True, x_label = "", y_label = ""):
        fig, ax = plt.subplots()
        ax.plot(positive[:,0], positive[:,1], 'ro')
        ax.plot(negative[:,0], negative[:,1], 'bo')
        ax.plot([line[0], line[1]], [line[2], line[3]])

        margin_rate = 0.05
        x_max = max(max(positive[:,0]), max(negative[:,0]))
        x_min = min(min(positive[:,0]), min(negative[:,0]))
        x_margin = (x_max - x_min) * margin_rate
        ax.set_xlim([x_min-x_margin, x_max+x_margin] )

        y_max = max(max(positive[:,1]), max(negative[:,1]))
        y_min = min(min(positive[:,1]), min(negative[:,1]))
        y_margin = (y_max - y_min) * margin_rate
        ax.set_ylim([y_min - y_margin, y_max + y_margin] )

        if len(x_label) > 0:
            ax.set_xlabel(x_label)
        if len(y_label) > 0:
            ax.set_ylabel(y_label)

#        fig.show()
        self.__show_and_save(fig, filename, show_flag)

    def bar_feature_importance(self, entropy_reduction, labels, filename = "", show_flag = True):
        fig, ax = plt.subplots()
        Y = range( len(entropy_reduction))
        Y.reverse()

        ax.barh(Y, entropy_reduction, height = 0.4)
        plt.yticks(Y, labels)
        ax.set_xlabel("Entropy Reduction")
        plt.tick_params(axis = 'both', which = 'major', labelsize = 8)
        plt.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        plt.tight_layout()

        self.__show_and_save(fig, filename, show_flag)
        
    def normalize(self, value):
        max_val = max(abs(value))
        order = 10.0 ** int(math.log10(float(max_val)))
        n_value = value / order
        return n_value, order

    def __show_and_save(self, fig, filename, show_flag):
        if len(filename) > 0:
            fig.savefig(filename)
        if show_flag:
            fig.show()

    def __figure_with_legend(self):
        fig = plt.figure(figsize = plt.figaspect(0.5))
        ax = fig.add_axes([.05,.1,.5,.8])
        return fig, ax

    def __show_legend(self, ax):
        ax.legend(bbox_to_anchor = (1.02, 1),  loc = 'upper left', borderaxespad = 0, prop = {'size':8})

    def __time_diff_in_hour(self, time_seq, base_time):
        return [(item - base_time).total_seconds()/3600 for item in time_seq]

    def __get_plot_list(self, base_time, time_series):
        plot_list = []
        for item in time_series:
            try:
                time_diff = self.__time_diff_in_hour(item.timestamps, base_time)
                value = np.array([float(num) for num in item.values])
                plot_val, order = self.normalize(value)
                tag = "%s [%0.1f %s]"%(item.description,order,item.unit)
                plot_list.append([time_diff, plot_val, tag])
            except ValueError:
                print "Can't plot %s"%item.description
        return plot_list

    def __draw_series_with_legend(self,plot_list, icu_ios, title, filename, show_flag, style = '-'):
        plot_all = list(chunked(plot_list,self.limit_timeseries))
        for plot_list in plot_all:
            fig, ax = self.__figure_with_legend()
            for item in plot_list:
                ax.plot(item[0], item[1], style, label = item[2])

            for span in icu_ios:
                ax.axvspan(span[0], span[1], alpha = 0.2, color = 'red')

            ax.set_title(title)
            ax.set_xlabel("Hours since Admission") 

            self.__show_legend(ax)
            self.__show_and_save(fig, filename, show_flag)


