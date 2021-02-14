import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('ggplot')

''' Shot Accuracy Plot
ticks = [5,6,7,8,9,10,11,12,13,14,15]#[1,2,3,4,5]
data_lists = [
    [92.35,92.52,93.2,93.71,93.85,94.15,94.22,94.37,94.68,94.73,94.82],
    [89.15,89.74,90.41,90.88,91.31,91.47,91.84,92.03,92.2,92.3,92.48],
    [86.13,86.98,87.8,88.15,88.71,89.22,89.43,89.6,89.87,90.05,90.16],
    [80.04,81.38,82.39,83.09,83.61,84.21,84.6,85.16,85.35,85.79,85.99]
]#[[0.4,1.2,2.3,4,5.5]]
label_lists = [
    'VirusShare_00177 5-way',
    'VirusShare_00177 10-way',
    'APIMDS 5-way',
    'APIMDS 10-way'
]#['test1']
color_lists = ['red', 'red', 'royalblue', 'royalblue'] #['red']
marker_lists = ['o', '^', 'o', "^"]#['.']
'''


acc_data_lists = [
    [91.04,91.71,92.11,92.35,91.8,91.55,90.71,91.05,90.22,90.12, 91.13, 90.32, 90.48, 90.84, 90.42, 91.14, 90.49, 90.49, 90.87, 90.77],
    [87.44, 88.64, 88.7, 89.15, 88.07, 87.88, 87.77, 87.64, 87.46, 87.02, 86.93, 87.05, 86.87, 87.43, 87.56, 87.72, 87.38, 86.98, 87.31, 87.28]
]
time_data_lists = [
    [14.2, 19.6, 25.1, 29.4, 36.9, 42.4, 48.8, 53.6, 58.6, 64.5, 70.1, 75.1, 80.5, 83.2, 90.5, 93.4, 100.6, 106.1, 111.5, 115.6],
    [22.4, 32.0, 41.1, 50.2, 61.5, 71.4, 79.9, 89.8, 98.8, 108.5, 116.3, 122.4, 131.8, 142.6, 154.5, 164.3, 170.7, 187.9, 195.2, 201.9]
]
acc_label_lists = [
    "VirusShare_00177 5-shot 5-way accuracy",
    "VirusShare_00177 5-shot 10-way accuracy",
    # "APIMDS 5-shot 5-way",
    # "APIMDS 5-shot 10-way"
]
time_label_list = [
    "VirusShare_00177 5-shot 5-way test time per episode",
    "VirusShare_00177 5-shot 10-way test time per episode"
]

color_lists = ['orange', 'green']
marker_lists = ['s', 's']

bar_width = 10
ticks = np.arange(50, 1050, 50)
num_list = len(time_data_lists)
bar_ticks = [
    np.arange(50, 1050, 50) - (num_list/2 - i - 0.5) * bar_width
    for i in range(num_list)
]

marker_size = 6
title = ''
x_title = 'Sequence Length'
acc_y_title = 'Accuracy(%)'
time_y_title = 'ms / Episode'

fig_size = (15,6)
dpi = 300

fig = plt.figure(figsize=fig_size, dpi=dpi)
plt.xticks(ticks)
plt.title(title)
# plt.xlabel(x_title)
# plt.ylabel(y_title)
plt.grid(True, axis='y')

acc_axis = fig.add_subplot(111)
time_axis = acc_axis.twinx()

acc_axis.set_xlabel('Maximum Sequence Length')
acc_axis.set_ylabel(acc_y_title)
time_axis.set_ylabel(time_y_title)

acc_axis.set_ylim(75, 95)
time_axis.set_ylim(0, 350)

for acc_data, time_data, bar_tick, acc_label, time_label, color, marker in zip(acc_data_lists, time_data_lists, bar_ticks, acc_label_lists, time_label_list, color_lists, marker_lists):
    acc_axis.plot(ticks, acc_data, color=color, marker=marker, label=acc_label, markersize=marker_size)
    time_axis.bar(bar_tick, time_data, color=color, width=10, label=time_label, zorder=2)

acc_axis.legend(loc='upper left')
time_axis.legend(loc='upper right')
# plt.legend()
plt.show()
# plt.savefig('C:/Users/Asichurter/Desktop/截图/virushare.jpg', format='JPEG', dpi=300)

