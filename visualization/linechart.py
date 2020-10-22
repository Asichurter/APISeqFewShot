import numpy as np
import matplotlib.pyplot as plt

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
color_list = ['red', 'red', 'royalblue', 'royalblue'] #['red']
marker_list = ['o', '^', 'o', "^"]#['.']

marker_size = 6
title = ''
x_title = 'Shot'
y_title = 'Accuracy(%)'

fig_size = (10,8)
dpi = 300

plt.figure(dpi=dpi)
plt.xticks(ticks)
plt.title(title)
plt.xlabel(x_title)
plt.ylabel(y_title)
plt.grid(True)

for data,label,color,marker in zip(data_lists,label_lists,color_list,marker_list):
    plt.plot(ticks, data, color=color, marker=marker, label=label, markersize=marker_size)

plt.legend()
plt.show()

