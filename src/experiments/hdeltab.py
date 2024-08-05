from src.utilities import imaging as im
from src.utilities import plotting as pl
import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import seaborn as sns
from skimage import io
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from scipy import stats
matplotlib.rcParams['pdf.fonttype'] = 42

importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)
importlib.reload(im)

class hdeltab:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '1RhzNSO-wDWqFJ3RLa7zPCrLgd3Lj94UY_VKHTi9F_z0'
        self.datafol = "/Volumes/LACIE/Andy/hdb"
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')

        # df.analyze.loc[df.analyze == 'TRUE'] = True
        # df.analyze.loc[df.analyze == 'FALSE'] = False
        # df.DP_analysis.loc[df.DP_analysis == 'TRUE'] = True
        # df.DP_analysis.loc[df.DP_analysis == 'FALSE'] = False
        # df.crop.loc[df.crop == 'FALSE'] = False
        df = df.replace('TRUE',True, regex=True)
        df = df.replace('FALSE',False, regex=True)
        self.sheet = df


class hdb_example:
    """
    class for plotting and analyzing individual examples.  Could eventually fold
    this into hdeltc class
    """
    def __init__(self, dict, datafol):
        for key in list(dict.keys()):
            setattr(self,key, dict[key])
        # convert these columns to arrays
        keys = ['FB_slices', 'PB_slices', 'EB_slices']
        for key in keys:
            string = dict[key]
            if string == 'None':
                setattr(self, key, None)
            else:
                setattr(self, key, [int(s) for s in string.split(',')])

        keys = ['FB_masks']
        for key in keys:
            string = dict[key]
            if string == 'None':
                setattr(self, key, None)
            else:
                setattr(self, key, [mask for mask in string.split(',')])

        # specify where the figures should be stored
        cwd = os.getcwd()
        save_folder = os.path.join(cwd, 'figures/hdb', self.folder)

        save_folder = os.path.join('/Volumes/LACIE/Andy/hdeltab/figures', self.folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.figure_folder = save_folder

        # specify where the data is stored
        self.datafol = os.path.join(datafol, self.folder)

        if self.FB_masks is not None:
            self.fb = im.FSB2(self.folder, self.datafol, self.FB_slices)
        if self.PB_slices is not None:
            self.pb = im.PB2(self.folder, self.datafol, self.PB_slices)
        if self.EB_slices is not None:
            self.eb = im.EB(self.folder, self.datafol, self.EB_slices)

        self.fly = im.fly(self.folder, self.datafol, **dict)

    def plot_trajectory(self):
        self.fb.load_processed()
        fig, axs = plt.subplots(1,2)
        axs[0].axis('equal')
        ft2 = self.fb.ft2
        axs[0] = pl.plot_vertical_edges(ft2, axs[0], width=10)
        axs[0] = pl.plot_trajectory(ft2, axs[0])
        axs[0] = pl.plot_trajectory_odor(ft2, axs[0])
        axs[1] = pl.plot_vertical_edges(ft2, axs[1], width=10)
        axs[1] = pl.plot_trajectory(ft2, axs[1])
        axs[1] = pl.plot_trajectory_odor(ft2, axs[1])
        fig.suptitle(self.folder)
        fig.savefig(os.path.join(self.figure_folder, 'trajectory.pdf'))
        return fig, axs

    def plot_FB_heatmap(self):
        """
        plot the fan-shaped body heatmaps
        overlay the extracted phase
        not upsampled so cannot be compared directly with behavior
        """
        if self.FB_masks is None:
            print('NO FB specified')
        else:
            num_masks = len(self.FB_masks)
            fig, axs = plt.subplots(num_masks, 1, sharex=True, squeeze=False)
            for i, mask in enumerate(self.FB_masks):
                wedges = self.fb.get_wedge_rois(mask)
                phase = self.fb.get_centroidphase(wedges)
                axs_r = axs[i, 0].twinx()
                axs_r.plot(phase, 'r.', alpha=0.5)
                axs_r.set_ylim(-np.pi, np.pi)
                wedges = np.rot90(wedges, k=1, axes=(0,1))
                sns.heatmap(wedges, ax=axs[i, 0], cmap='Blues',vmin=0,vmax=1, cbar=True)

    def plot_FB_heatmap_heading(self):
        """
        plot the fan-shaped body heatmaps
        overlay the extracted phase
        not upsampled so cannot be compared directly with behavior
        """
        self.fb.load_processed()
        heading = self.fb.ft2.ft_heading
        # for hdeltab it's hard to assign a phase when the fly is not moving so find stops
        df = self.fb.ft2
        _, df = fn.find_stops(df)
        stops = df.stop.to_numpy()
        if self.FB_masks is None:
            print('NO FB specified')
        else:
            num_masks = len(self.FB_masks)
            fig, axs = plt.subplots(num_masks+1, 1, figsize = (15,2*(num_masks+1)), sharex=True)
            for i, mask in enumerate(self.FB_masks):
                print(mask)

                # plot the wedges and the phase
                wedges = self.fb.get_layer_wedges(mask)
                phase = self.fb.get_centroidphase(wedges)
                diff = stops*(phase-heading)
                diff = diff[~np.isnan(diff)]
                offset = fn.circmean(diff)
                offset_phase = fn.wrap(phase-offset)
                #axs[num_masks].plot(offset_phase, '.')
                axs[num_masks].plot(stops*offset_phase, '.')
                axs_r = axs[i].twinx()
                #axs_r.plot(phase, '.')#, color='red', alpha=0.5)
                axs_r.set_ylim(-np.pi, np.pi)
                wedges = np.rot90(wedges, k=1, axes=(0,1))
                sns.heatmap(wedges, ax=axs[i], cmap='Blues', cbar=False, rasterized=True)

                # plot the odor
                ft2 = self.fb.ft2
                ft2 = fn.consolidate_in_out(ft2)
                d, di, do = fn.inside_outside(ft2)
                for key in list(di.keys()):
                    df_temp = di[key]
                    ix = df_temp.index.to_numpy()
                    width = ix[-1]-ix[0]
                    height = 2*np.pi
                    rect = Rectangle((ix[0], -np.pi), width, height, fill=False)
                    rect2 = Rectangle((ix[0], -np.pi), width, height, fill=False)
                    axs_r.add_patch(rect)
                    axs[num_masks].add_patch(rect2)

            axs[i+1].plot(heading)
            # make example
            if ex.folder=='20220504_Fly1_005':
                axs[i+1].set_xlim(2300,5160)
            fig.suptitle(self.folder)
            fig.savefig(os.path.join(self.figure_folder, 'hDB_heatmap.pdf'))
        return fig, axs

    def epg_hdb_phase_diff(self):
        for i, mask in enumerate(self.FB_masks):
            # load data
            self.fb.load_processed()
            # only look at phase over stops
            df = self.fb.ft2
            _, df = fn.find_stops(df)
            stops = df.stop.to_numpy()
            # calculate offset phase
            heading = self.fb.ft2.ft_heading
            wedges = self.fb.get_layer_wedges(mask)
            phase = self.fb.get_centroidphase(wedges)
            diff = stops*(phase-heading)
            diff = diff[~np.isnan(diff)]
            offset = fn.circmean(diff)
            offset_phase = fn.wrap(phase-offset)
            # now calculate diff between HDB and phase
            diff = fn.wrap(offset_phase-heading)
            diff = stops*diff
            diff = diff[~np.isnan(diff)]
            # plot
            fig, axs  = plt.subplots(1,1)
            #sns.histplot(diff, stat='probability', kde=True,bins=36)
            axs = sns.kdeplot(diff, ax=axs, cut=0)
            return axs


%matplotlib

def heading_hdb_phase_diff():
    hdb = hdeltab()
    df = hdb.sheet
    fig,axs = plt.subplots(1,1, figsize=(2,2))
    all_kdes = []
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            print(d['folder'])
            datafol = '/Volumes/LACIE/Andy/hdeltab/'
            ex = hdb_example(d,datafol)
            ax = ex.epg_hdb_phase_diff()
            kde_line = ax.get_lines()[0].get_data()
            x = kde_line[0]
            all_kdes.append(kde_line[1])
            axs.plot(kde_line[0], kde_line[1], 'k', alpha=0.2)
    all_kdes = np.array(all_kdes)
    axs.plot(x, np.mean(all_kdes, axis=0), color='k')
    axs.set_xticks([-np.pi, 0, np.pi])
    axs.set_xticklabels([-180, 0, 180])
    axs.set_xlabel('hDB phase-heading (degrees)')
    #fig.savefig(os.path.join('/Volumes/LACIE/Andy/hdeltab/figures', 'phase_diff.pdf'))


def in_out_odor():
    hdb = hdeltab()
    df = hdb.sheet
    moving, still = [],[]
    for i in np.arange(len(df)):
        d = df.iloc[i].to_dict()
        if d['analyze'] == True:
            print(d['folder'])
            datafol = '/Volumes/LACIE/Andy/hdeltab/'
            ex = hdb_example(d,datafol)
            wedges = ex.fb.get_layer_wedges('fbmask')

            bumps_still, bumps_moving = ex.fb.bumps_moving_still(wedges)
            moving.append(np.mean(bumps_moving, axis=0))
            still.append(np.mean(bumps_still, axis=0))
            wedges = np.rot90(wedges, axes=(1,0))
            fig, axs = plt.subplots(1,1)
            sns.heatmap(wedges,ax=axs)
            fig.suptitle(d['folder'])
    fig,axs = plt.subplots(1,1, figsize=(2,2))
    for i in np.arange(len(moving)):
        axs.plot(moving[i], 'k', alpha=0.2)
        axs.plot(still[i], color = pl.dmagenta, alpha=0.2)
    axs.plot(np.mean(moving, axis=0), 'k', alpha=1)
    axs.plot(np.mean(still, axis=0), color = pl.dmagenta, alpha=1)
    fig.savefig(os.path.join('/Volumes/LACIE/Andy/hdeltab/figures', 'bumps_moving_still.pdf'))
#
# wedges.shape
# wedges[:,:15].shape
# %%
