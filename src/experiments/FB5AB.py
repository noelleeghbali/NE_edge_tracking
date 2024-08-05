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

folder = ['20220103_21D07_sytjGCaMP7f_Fly1_001',
            '20220103_21D07_sytjGCaMP7f_Fly1_002',
            '20220107_21D07_sytjGCaMP7f_Fly1_001',
            '20220107_21D07_sytjGCaMP7f_Fly1_002',
            '20220110_21D07_sytjGCaMP7f_Fly1_001',
            '20220110_21D07_sytjGCaMP7f_Fly1_002',
            '20220111_21D07_sytjGCaMP7f_Fly1_001',
            '20220111_21D07_sytjGCaMP7f_Fly1_003',
            '20220331_PAMChr_21D07GCaMP6s_Fly1_001',
            '20220331_PAMChr_21D07GCaMP6s_Fly1_002',
            '20220331_PAMChr_21D07GCaMP6s_Fly1_003']
analyze = [True,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False
            ]
class FB5AB:
    def __init__(self):
        self.datafol = "/Volumes/LACIE/Andy/21D07"
        self.figurefol = "/Volumes/LACIE/Andy/21D07/figures"
        self.df = pd.DataFrame({
        'folder': folder,
        'analyze': analyze
        })

class fb5ab_ex():
    """
    class for manipulating data and plotting examples from individual FB5AB examples
    """
    def __init__(self, dict, datafol):
        for key in list(dict.keys()):
            setattr(self,key, dict[key])

        self.datafol = os.path.join(datafol, self.folder)
        self.figure_folder = os.path.join(self.datafol, 'figures')
        if not os.path.exists(self.figure_folder):
            os.makedirs(self.figure_folder)
        self.fly = im.fly(self.folder, self.datafol, **dict)
        pv2, _, ft2, _ = self.fly.load_postprocessing()
        self.ft2 = ft2
        self.pv2 = pv2

    def split_data(self):
        """
        split data into inside and out
        """

        # get compartments, exclude ones for which no ROI was drawn
        data = self.ft2
        data['fb5ab_dff'] = self.pv2['fb5ab_dff']
        # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
        data = fn.exclude_lost_tracking(data, thresh=10)
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

    def load_split_data(self, overwrite=False):
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
        fig.suptitle(self.folder)
        fig.savefig(os.path.join(self.figure_folder, 'trajectory.pdf'))

    def plot_activity(self):
        data = self.load_split_data()
        di = data['di']
        do =data['do']
        df = data['data']
        t = df.seconds.to_numpy()
        t = t - t[0]
        fb5ab = df.fb5ab_dff.to_numpy()

        fig, axs = plt.subplots(1,1)
        axs.plot(t, fb5ab)

        height = np.max(fb5ab)-np.min(fb5ab)
        y0=np.min(fb5ab)
        for key in list(di.keys()):
            df_temp = di[key]
            ix = df_temp.index.to_numpy()
            t0 = t[ix[0]]
            t1 = t[ix[-1]]
            width = t1-t0
            rect = Rectangle((t0, y0), width, height, fill='k', alpha=0.3)
            axs.add_patch(rect)
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
        fb5ab_dff = df['fb5ab_dff']
        in_out = []
        out_in = []
        ix_in = []
        ix_out = []

        buf_pts = 20 # 2 seconds

        #find entry points
        for key in list(di.keys()):
            ix_in.append(di[key].index[0])
            f = di[key].fb5ab_dff.to_numpy()
            t = di[key].seconds.to_numpy()
            t = t-t[0]

        for key in list(do.keys())[1:]:
            ix_out.append(do[key].index[0])

        # plot out to in
        for ix in ix_in:
            ixs = np.arange(ix-buf_pts, ix+buf_pts)
            t = (np.arange(len(ixs))-buf_pts)/10
            axs[0].plot(t, fb5ab_dff[ixs], 'k', alpha = 0.1)
            out_in.append(fb5ab_dff[ixs])
        out_in = np.array(out_in)
        axs[0].plot(t, np.mean(out_in, axis=0), 'k')

        # plot in to out
        for ix in ix_out:
            ixs = np.arange(ix-buf_pts, ix+buf_pts)
            t = (np.arange(len(ixs))-buf_pts)/10
            axs[1].plot(t, fb5ab_dff[ixs], 'k', alpha = 0.1)
            in_out.append(fb5ab_dff[ixs])
        in_out = np.array(in_out)
        axs[1].plot(t, np.mean(in_out, axis=0), 'k')

        fig.savefig(os.path.join(self.figure_folder, 'triggered_in_out.pdf'))
        return out_in



# %%
"""
plot a specific example heatmap.  Bit of a hack job here, just for an example heatmap for Janelia conference
"""
%matplotlib
fb5ab = FB5AB()
datafol = fb5ab.datafol
figurefol = fb5ab.figurefol
df = fb5ab.df
for i, row in df.iterrows():
    if row.folder == '20220331_PAMChr_21D07GCaMP6s_Fly1_002':
        d = dict(row)
        ex = fb5ab_ex(d, datafol)
        ex.fly.crop=False
        datafol = os.path.join(datafol, row.folder)
        zslices = 1
        fb = im.FSB2(row.folder, datafol, z_slices=[1])
        fb.load_processed()
        wedges = fb.pv2.filter(regex='fb')
        wedges.fillna(method='ffill', inplace=True)
        wedges = wedges.iloc[0:1600,:]
        wedges = wedges.apply(fn.lnorm).to_numpy()
        wedges = np.rot90(wedges, k=1, axes=(0,1))

# cropped dataframe
df = fb.ft2
df = df.iloc[0:1600, :]

# plot heatmap with odor
fig, axs = plt.subplots(1,1)
sns.heatmap(wedges, ax=axs, cmap='Blues', cbar=True, rasterized=True)
d, di, do = fn.inside_outside(df)
axs_r = axs.twinx()
axs_r.set_ylim(-np.pi, np.pi)
for key in list(di.keys()):
    df_temp = di[key]
    ix = df_temp.index.to_numpy()
    width = ix[-1]-ix[0]
    height = 2*np.pi
    rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
    axs_r.add_patch(rect)
fig.savefig(os.path.join(fb5ab.figurefol, 'example_heatmap.pdf'), transparent=True)

# plot trajectory
fig, axs = plt.subplots(1,1)
axs = pl.plot_trajectory(df, axs)
axs = pl.plot_trajectory_odor(df, axs)
axs = pl.plot_vertical_edges(df, axs, width=10)
axs.axis('equal')
fig.savefig(os.path.join(fb5ab.figurefol, 'example_trajectory.pdf'), transparent=True)
print(total_time)








# %%
"""
plot the triggered averages for FB5AB
"""
%matplotlib

fb5ab = FB5AB()
datafol = fb5ab.datafol
figurefol = fb5ab.figurefol
df = fb5ab.df
fig, axs = plt.subplots(1,1)
all = []
for i, row in df.iterrows():
    d = dict(row)
    if d['analyze']:
        ex = fb5ab_ex(d, datafol)
        #ex.plot_trajectory()
        temp = ex.plot_triggered_average()
        for entry in temp:
            t = np.linspace(-2,2,len(entry))
            axs.plot(t, entry, color = 'k', alpha = 0.1)
        an_avg = np.mean(temp, axis=0)
        t = np.linspace(-2,2, len(an_avg))
        axs.plot(t, an_avg, 'k')
        all.append(temp)
all = np.concatenate(all)
m = np.mean(all, axis=0)
error = stats.sem(all, axis=0)
t = np.linspace(-2,2, len(m))
axs.plot(t, m, color = 'k', linewidth=2)
axs.fill_between(t, m+error, m-error, color='k')
fig.savefig(os.path.join(figurefol, 'triggered_average.pdf'))
