import numpy as np
import matplotlib.pyplot as plt

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

ticks = [50,100,150,200,250,300,350,400,450,500]
data_lists = [
    [91.04,91.71,92.11,92.35,91.8,91.55,90.71,91.05,90.22,90.12],
    [87.44,88.64,88.7,89.15,88.07,87.88,87.77,87.64,87.46,87.02],
    [77.7,82.37,84.97,85.57,85.92,86.16,86.32,83.78,84.3,84.27],
    [69.09,75.63,79,80.04,79.61,80.04,79.42,77.09,78.87,76.9]
]
label_lists = [
    "VirusShare_00177 5-shot 5-way",
    "VirusShare_00177 5-shot 10-way",
    "APIMDS 5-shot 5-way",
    "APIMDS 5-shot 10-way"
]
color_lists = ['orange', 'orange', 'lightgreen', 'lightgreen']
marker_lists = ['S', 'D', 'S', 'D']

marker_size = 6
title = ''
x_title = 'Sequence Length'
y_title = 'Accuracy(%)'

fig_size = (10,8)
dpi = 300

plt.figure(dpi=dpi)
plt.xticks(ticks)
plt.title(title)
plt.xlabel(x_title)
plt.ylabel(y_title)
plt.grid(True)

for data,label,color,marker in zip(data_lists,label_lists,color_lists,marker_lists):
    plt.plot(ticks, data, color=color, marker=marker, label=label, markersize=marker_size)

plt.legend()
plt.show()

