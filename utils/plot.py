import visdom
import numpy as np

class VisdomPlot:

    def __init__(self, env_title, types, titles, xlabels, ylabels, legends):
        self.Handle = visdom.Visdom(env=env_title)
        self.Types = {title: type_ for title,type_ in zip(titles, types)}
        self.XLabels = {title: xlabel for title, xlabel in zip(titles, xlabels)}
        self.YLabels = {title: ylabel for title, ylabel in zip(titles, ylabels)}
        self.Legends = {title: legend for title, legend in zip(titles, legends)}

    def update(self, title, x_val, y_val, update={'flag':False, 'val':None}):
        if not update['flag']:
            update_flag = None if x_val==0 else 'append'
        else:
            update_flag = update['val']

        y_val = np.array(y_val)
        y_size = y_val.shape[1]

        x_val = np.ones((1, y_size)) * x_val

        plot_func = self.getType(self.Types[title])

        plot_func(X=x_val, Y=y_val, win=title,
                  opts=dict(
                      legend=self.Legends[title],
                      title=title,
                      xlabel=self.XLabels[title],
                      ylabel=self.YLabels[title],
                  ),
                  update= update_flag)

    def getType(self, t):
        if t == 'line':
            return self.Handle.line
        else:
            raise NotImplementedError('暂未实现的类型:%d'%t)