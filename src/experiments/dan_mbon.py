from src.utilities import imaging as im
from src.utilities import plotting as pl
import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
import seaborn as sns
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
import scipy.signal as sg
from scipy import stats

importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)
importlib.reload(im)

folder = ['20200110_58E02jRGECO1a_VT026001jGCaMP7f_Fly3_trial6',
            '20200311_58E02jRGECO1a_VT026001jGCaMP7f_Fly1_trail3',
            '20200311_58E02jRGECO1a_VT026001jGCaMP7f_Fly1_trial2',
            '20200719_58E02jRGECO1a_VT026001jGCaMP7f_Fly1_trial2']
analyze = [True,
            False,
            True,
            True]
offset = [5,0,0,0]

class dan_mbon:
    def __init__(self):
        self.datafol = "/Volumes/LACIE/Andy/dan_mbon"
        self.figurefol = "/Volumes/LACIE/Andy/dan_mbon/figures"
        self.df = pd.DataFrame({
        'folder': folder,
        'analyze': analyze,
        'offset': offset
        })

class dan_mbon_ex():
    """
    class for manipulating data and plotting examples from individual mbon examples
    """
    def __init__(self, dict, datafol):
        for key in list(dict.keys()):
            setattr(self,key, dict[key])

        self.datafol = os.path.join(datafol, self.folder)
        self.figure_folder = os.path.join(self.datafol, 'figures')
        if not os.path.exists(self.figure_folder):
            os.makedirs(self.figure_folder)
        self.fly = im.fly(self.folder, self.datafol, **dict)
        post_processing_file = os.path.join(self.datafol, 'processed','postprocessing.h5')
        df = pd.read_hdf(post_processing_file, 'pv2')
        df_dan = df.filter(regex='DAN').mean(axis=1)
        df_mbon = df.filter(regex='MBON').mean(axis=1)
        df = pd.DataFrame({'dan': df_dan, 'mbon': df_mbon})
        self.pv2 = df
        #self.ft = pd.read_hdf(post_processing_file, 'ft')
        self.ft2 = pd.read_hdf(post_processing_file, 'ft2')
        #self.ix = pd.read_hdf(post_processing_file, 'ix')

    def split_data(self):
        """
        split data into inside and out
        """

        # get compartments, exclude ones for which no ROI was drawn
        data = self.ft2
        data['dan'] = self.pv2['dan']
        data['mbon'] = self.pv2['mbon']
        # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
        #data = fn.exclude_lost_tracking(data, thresh=10)
        data = fn.consolidate_in_out(data)
        data = fn.calculate_speeds(data)
        d, di, do = fn.inside_outside(data)
        dict_temp = {"data": data,
                    "d": d,
                    "di": di,
                    "do": do}


        pickle_name = os.path.join(self.fly.processedfol, 'bouts.p')
        fn.save_obj(dict_temp, pickle_name)
        return dict_temp

    def load_split_data(self, overwrite=True):
        """
        load the inside/outside split data
        if the data hasn't been saved, split and save the data
        """
        pickle_name = os.path.join(self.fly.processedfol, 'bouts.p')
        if (not overwrite) and os.path.exists(pickle_name):
            data = fn.load_obj(pickle_name)
        else:
            data = self.split_data()
        return data

    def plot_trajectory(self):
        fig, axs = plt.subplots(1,1)
        axs = pl.plot_trajectory(self.ft2, axs)
        axs = pl.plot_trajectory_odor(self.ft2, axs)
        fig.savefig(os.path.join(self.figure_folder, 'trajectory.pdf'))

    def plot_activity(self):
        data = self.load_split_data()
        di = data['di']
        do =data['do']
        df = data['data']
        t = df.seconds.to_numpy()
        t = t - t[0]
        dan = df.dan.to_numpy()
        mbon = df.mbon.to_numpy()

        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(t, dan)
        axs[1].plot(t, mbon)

        height = np.max(dan)-np.min(dan)
        y0=np.min(dan)

        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            t0 = t[ix[0]]
            t1 = t[ix[-1]]
            width = t1-t0
            rect = Rectangle((t0, y0), width, height, fill='k', alpha=0.3)
            axs[0].add_patch(rect)
        fig.savefig(os.path.join(self.figure_folder, 'activity.pdf'))

    def plot_triggered_average(self):
        """
        plot entry and exit triggered averages for an individual trial
        """

        fig, axs = plt.subplots(1,2)
        data = self.load_split_data()
        di = data['di']
        do =data['do']

        if len(di)==0:
            return
        df = data['data']
        traces = ['dan', 'mbon']
        colors = ['red', 'green']
        d = {}
        for i, trace in enumerate(traces):
            trace_dff = df[trace]
            in_out = []
            out_in = []
            ix_in = []
            ix_out = []

            buf_pts = 20 # 2 seconds

            #find entry points
            for key in list(di.keys()):
                ix_in.append(di[key].index[0])
                f = di[key][trace].to_numpy()
                t = di[key].seconds.to_numpy()
                t = t-t[0]
                #axs[0].plot(t, f, 'r')

            for key in list(do.keys())[1:]:
                ix_out.append(do[key].index[0])

            ix_in = np.array(ix_in)+self.offset
            ix_out = np.array(ix_out)+self.offset

            # plot out to in
            for ix in ix_in:
                ixs = np.arange(ix-buf_pts, ix+buf_pts)
                t = (np.arange(len(ixs))-buf_pts)/10
                axs[0].plot(t, trace_dff[ixs], color=colors[i], alpha = 0.1)
                out_in.append(trace_dff[ixs])
            out_in = np.array(out_in)
            axs[0].plot(t, np.mean(out_in, axis=0), color=colors[i])

            # plot in to out
            for ix in ix_out:
                ixs = np.arange(ix-buf_pts, ix+buf_pts)
                t = (np.arange(len(ixs))-buf_pts)/10
                axs[1].plot(t, trace_dff[ixs], color=colors[i], alpha = 0.1)
                in_out.append(trace_dff[ixs])
            in_out = np.array(in_out)
            axs[1].plot(t, np.mean(in_out, axis=0), color=colors[i])

            d[trace] = out_in

        fig.savefig(os.path.join(self.figure_folder, 'triggered_in_out.pdf'))
        return d


%matplotlib
dm = dan_mbon()
datafol = dm.datafol
figure_folder = dm.figurefol
df = dm.df
fig, axs = plt.subplots(1,2)
colors = ['red', 'green']
all = {'dan':[],'mbon':[]}
for i, row in df.iterrows():
    d = dict(row)
    ex = dan_mbon_ex(d, datafol)
    # ex.plot_trajectory()
    trig = ex.plot_triggered_average()
    if d['analyze']:
        for i, celltype in enumerate(['dan', 'mbon']):
            temp = trig[celltype]
            color = colors[i]
            for entry in temp:
                t = np.linspace(-2,2,len(entry))
                axs[i].plot(t, entry, color = color, alpha = 0.1)
            an_avg = np.mean(temp, axis=0)
            t = np.linspace(-2,2, len(an_avg))
            axs[i].plot(t, an_avg, color)
            all[celltype].append(temp)
for i, celltype in enumerate(['dan', 'mbon']):
    color = colors[i]
    all[celltype] = np.concatenate(all[celltype])
    m = np.mean(all[celltype], axis=0)
    error = stats.sem(all[celltype], axis=0)
    t = np.linspace(-2,2, len(m))
    axs[i].plot(t, m, color = 'k', linewidth=2)
    axs[i].fill_between(t, m+error, m-error, color=color)
fig.savefig(os.path.join(figure_folder, 'triggered_average.pdf'))
