"""A class for controling graph."""
import numpy as np
import matplotlib.pyplot as plt
from math import log10, ceil, sqrt
from more_itertools import chunked
import cPickle


class Graph:

    """Control all the graphs and visualizations."""

    def __init__(self):
        """Initializer of Graph class."""
        self.limit_timeseries = 25
        self.dir_to_save = "../data/"

    def visualize_image(self, data,
                        h_len=28, n_cols=0, filename="", show_flag=True):
        """Visualizer of image data."""
        if data.ndim == 1:
            v_len = data.shape[0] / h_len
            if n_cols == 0:
                n_cols = 1
                n_rows = 1
        elif data.ndim == 2:
            v_len = data.shape[1] / h_len
            if n_cols == 0:
                n_cols = int(ceil(sqrt(data.shape[0])))
            n_rows = int(ceil(float(data.shape[0]) / n_cols))
        else:
            raise ValueError

        plt.gray()
        fig, axes = plt.subplots(n_rows, n_cols)

        X, Y = np.meshgrid(range(h_len), range(v_len))
        for i_v in range(n_rows):
            for i_h in range(n_cols):
                index = i_h + i_v * n_cols
                if index < data.shape[0]:

                    if n_rows > 1:
                        ax = axes[i_v, i_h]
                        Z = data[index].reshape(v_len, h_len)
                    elif n_cols > 1:
                        ax = axes[i_h]
                        Z = data[index].reshape(v_len, h_len)
                    else:
                        ax = axes
                        Z = data.reshape(v_len, h_len)

                    Z = Z[::-1, :]
                    ax.set_xlim(0, h_len - 1)
                    ax.set_ylim(0, v_len - 1)
                    ax.pcolor(X, Y, Z)
                    ax.tick_params(labelbottom='off')
                    ax.tick_params(labelleft='off')

        self.__show_and_save(fig, filename, show_flag)

    def draw_lab_adm(self, admission, title, filename="", show_flag=True):
        """Draw lab tests data of admissions."""
        base_time = admission.admit_dt
        data = admission.labs
        plot_list = self.__get_plot_list(base_time, data)
        icu_ios = [self.__time_diff_in_hour(
            [icustay.intime, icustay.outtime], base_time)
            for icustay in admission.icustays]
        self.__draw_series_with_legend(plot_list, icu_ios,
                                       title, filename, show_flag)

    def draw_med_icu(self, icustay, base_time, title, filename="", show_flag=True):
        data = icustay.medications
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag, 'o')

    def draw_chart_icu(self, icustay, base_time, title, filename="", show_flag=True):
        data = icustay.charts
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag)

    def draw_selected_chart_icu(self, icustay, itemid_list, base_time, title,
                                filename="", show_flag=True):
        selected_ids = itemid_list
        data = [item for item in icustay.charts if item.itemid in selected_ids]
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag)

    def draw_io_icu(self, icustay, base_time, title, filename="", show_flag=True):
        data = icustay.ios
        plot_list = self.__get_plot_list(base_time, data)
        icu_io = self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time)
        self.__draw_series_with_legend(plot_list, [icu_io], title, filename, show_flag, 'o')

    def draw_lab_adm_itemid(self, admission, itemids, title, filename="", show_flag=True):
        base_time = admission.admit_dt

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        colors = ['b', 'r']
        axis = [ax1, ax2]

        for idx, id in enumerate(itemids):
            data = admission.get_lab_itemid(id)
            time_diff = self.__time_diff_in_hour(data.timestamps, base_time)
            values = data.values
            axis[idx].plot(time_diff, values, "%ss--" % colors[idx])
            axis[idx].set_ylabel("%s [%s]" % (data.description, data.unit), color=colors[idx])

        ax1.set_title(title)
        ax1.set_xlabel("Hours since Admission")

        base_time = admission.admit_dt
        icu_ios = [self.__time_diff_in_hour([icustay.intime, icustay.outtime], base_time)
                   for icustay in admission.icustays]
        for span in icu_ios:
            ax1.axvspan(span[0], span[1], alpha=0.2, color='red')
        self.__show_and_save(fig, filename, show_flag)

    def draw_lab_distribution(self, expire_values, recover_values, title,
                              filename="", show_flag=True):
        fig, ax = plt.subplots()
        for value in expire_values:
            ax.plot(value[0], value[1], "ro")
        for value in recover_values:
            ax.plot(value[0], value[1], "bo")

        ax.set_xlabel("Creatinine [mg/dL]")
        ax.set_ylabel("Urea Nitrogen[mg/dL]")

        self.__show_and_save(fig, filename, show_flag)

    def plot_classification(self, positive, negative, line, title,
                            filename="", show_flag=True, x_label="", y_label=""):
        fig, ax = plt.subplots()
        ax.plot(positive[:, 0], positive[:, 1], 'ro')
        ax.plot(negative[:, 0], negative[:, 1], 'bo')
        ax.plot([line[0], line[1]], [line[2], line[3]])

        margin_rate = 0.05
        x_max = max(max(positive[:, 0]), max(negative[:, 0]))
        x_min = min(min(positive[:, 0]), min(negative[:, 0]))
        x_margin = (x_max - x_min) * margin_rate
        ax.set_xlim([x_min - x_margin, x_max + x_margin])

        y_max = max(max(positive[:, 1]), max(negative[:, 1]))
        y_min = min(min(positive[:, 1]), min(negative[:, 1]))
        y_margin = (y_max - y_min) * margin_rate
        ax.set_ylim([y_min - y_margin, y_max + y_margin])

        if len(x_label) > 0:
            ax.set_xlabel(x_label)
        if len(y_label) > 0:
            ax.set_ylabel(y_label)

        self.__show_and_save(fig, filename, show_flag)

    def plot_classification_with_contour(self, x, y, xx, yy, z, x_label, y_label,
                                         filename="", show_flag=True):
        fig, ax = plt.subplots()

        ax.contourf(xx, yy, z, cmap=plt.cm.rainbow, alpha=0.2)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.rainbow)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        self.__show_and_save(fig, filename, show_flag)

    def bar_feature_importance(self, entropy_reduction, labels, filename="", show_flag=True):
        fig, ax = plt.subplots()
        Y = range(len(entropy_reduction))
        Y.reverse()

        ax.barh(Y, entropy_reduction, height=0.4)
        plt.yticks(Y, labels)
        ax.set_xlabel("Entropy Reduction")
        plt.tick_params(axis='both', which='both', labelsize=8)
        plt.tight_layout()

        self.__show_and_save(fig, filename, show_flag)

    def series_classification(self, l_classification_result, timestamp, x_label,
                              title="", filename="", show_flag=True):
        l_rec = [item.rec for item in l_classification_result]
        l_prec = [item.prec for item in l_classification_result]
        l_f = [item.f for item in l_classification_result]
        legend = ['recall', 'precision', 'f_measure']
        self.line_series([l_rec, l_prec, l_f], timestamp, legend, ylim=[0, 1],
                         x_label=x_label, title=title)

    # general graphs
    def comparison_bar(self, data, labels, legend="", metric_label="", lim=[], horizontal=False,
                       title="", filename="", show_flag=True):
        original_data = locals().copy()
        fig, ax = plt.subplots()
        Y = range(len(labels))
        Y.reverse()

        if horizontal:
            [set_lim1, set_lim2] = [ax.set_ylim, ax.set_xlim]
            set_label1 = ax.set_xlabel
            set_ticks = plt.yticks
        else:
            [set_lim1, set_lim2] = [ax.set_xlim, ax.set_ylim]
            set_label1 = ax.set_ylabel
            set_ticks = plt.xticks

        if isinstance(data[0], (int, float)):
            bar_height = 0.5
            if horizontal:
                ax.barh(Y, data, height=bar_height)
            else:
                ax.bar(Y, data, width=bar_height)
            set_ticks([item + 0.25 for item in Y], labels)

        else:
            bar_height = 1. / (len(data) + 1)
            cmap = plt.cm.rainbow
            cmap_v = cmap.N / (len(data) - 1)
            for idx, d in enumerate(data):
                if horizontal:
                    ax.barh([y + bar_height * (len(data) - idx - 1) for y in Y], d,
                            height=bar_height, color=cmap(idx * cmap_v))
                else:
                    ax.bar([y + bar_height * (len(data) - idx - 1) for y in Y], d,
                           width=bar_height, color=cmap(idx * cmap_v))
                set_ticks([item + bar_height / 2 * len(data) for item in Y], labels)

        set_lim1([-bar_height, len(labels)])
        if lim:
            set_lim2(lim)
        if metric_label:
            set_label1(metric_label)
        if legend:
            plt.legend(legend)
        if title:
            ax.set_title(title)
        self.__show_and_save(fig, filename, show_flag, original_data)

    def line_series(self, data, y_points, legend, x_label="", y_label="", ylim=[], markersize=10,
                    title="", filename="", show_flag=True):
        original_data = locals().copy()
        fig, ax = plt.subplots()
        for item in data:
            if markersize is 0:
                ax.plot(y_points, item, '-')
            else:
                ax.plot(y_points, item, 'o--', markersize=markersize)
        ax.legend(legend)
        ax.set_xlim(self.__calc_lim(y_points, 0.05))

        if ylim:
            ax.set_ylim(ylim)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        self.__show_and_save(fig, filename, show_flag, original_data)

    def waitforbuttunpress(self):
        plt.waitforbuttonpress()

    def __calc_lim(self, values, margin_ratio):
        margin = (max(values) - min(values)) * margin_ratio
        return [min(values) - margin, max(values) + margin]

    def normalize(self, value):
        max_val = max(abs(value))
        order = 10.0 ** int(log10(float(max_val)))
        n_value = value / order
        return n_value, order

    def __show_and_save(self, fig, filename, show_flag, data=None):
        if len(filename) > 0:
            path = self.dir_to_save + filename
            fig.savefig(path)

            if data is not None:
                p_path = path + '.pkl'
                f = open(p_path, 'w')
                cPickle.dump(data, f)
                f.close()

        if show_flag:
            fig.show()

    def __figure_with_legend(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_axes([.05, .1, .5, .8])
        return fig, ax

    def __show_legend(self, ax):
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 8})

    def __time_diff_in_hour(self, time_seq, base_time):
        return [(item - base_time).total_seconds() / 3600 for item in time_seq]

    def __get_plot_list(self, base_time, time_series):
        plot_list = []
        for item in time_series:
            try:
                time_diff = self.__time_diff_in_hour(item.timestamps, base_time)
                value = np.array([float(num) for num in item.values])
                plot_val, order = self.normalize(value)
                tag = "%s [%0.1f %s]" % (item.description, order, item.unit)
                plot_list.append([time_diff, plot_val, tag])
            except ValueError:
                print "Can't plot %s" % item.description
        return plot_list

    def __draw_series_with_legend(self, plot_list, icu_ios, title, filename, show_flag, style='-'):
        plot_all = list(chunked(plot_list, self.limit_timeseries))
        for plot_list in plot_all:
            fig, ax = self.__figure_with_legend()
            for item in plot_list:
                ax.plot(item[0], item[1], style, label=item[2])

            for span in icu_ios:
                ax.axvspan(span[0], span[1], alpha=0.2, color='red')

            ax.set_title(title)
            ax.set_xlabel("Hours since Admission")

            self.__show_legend(ax)
            self.__show_and_save(fig, filename, show_flag)
