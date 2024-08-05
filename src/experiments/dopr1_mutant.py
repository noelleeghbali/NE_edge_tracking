import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from src.utilities import imaging as im
from scipy import interpolate, stats
from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)
os.getcwd()

class dopr1_mutant:
    def __init__(self, directory='M1'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'

        self.sheet_id = '1yPqZ_jmJIzkRJ2aLkrepZEsdJAFaKazB4kYoYP-H0VA'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
        self.logfol = os.path.join(self.cwd, 'data/dopr1_mutant/logs')
        self.picklefol = os.path.join(self.cwd, 'data/dopr1_mutant/pickles')
        self.picklesname = os.path.join(self.picklefol, 'trajectories.p')
        self.figurefol = os.path.join(self.cwd, 'figures/dopr1_mutant')

    def split_trajectories(self):
        all_data = {}

        for i, log in enumerate(self.sheet.log):
            data = fn.read_log(os.path.join(self.logfol, log))
            mfc = self.sheet.mfc.iloc[i]
            # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
            data = fn.exclude_lost_tracking(data, thresh=10)
            if mfc == 'old': # all experiments performed with old MFCs
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            data = fn.consolidate_in_out(data)
            data = fn.calculate_speeds(data)
            d, di, do = fn.inside_outside(data)
            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do}
            all_data[log] = dict_temp
        fn.save_obj(all_data, self.picklesname)

    def load_trajectories(self):
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def plot_individual_trajectories(self):
        all_data = self.load_trajectories()
        for i, log in enumerate(self.sheet.log):
            print(log)
            df = all_data[log]['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_vertical_edges(df, axs)
            pl.plot_trajectory(df, axs)
            pl.plot_trajectory_odor(df, axs)
            axs.axis('equal')
            fig.suptitle(log)

    def plot_example_trajectory(self):
        all_data = self.load_trajectories()
        examples = ['09042020-134647_Fly4_DopR1mut_003.log', '09042020-112635_Fly1_CantonS_001.log']
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        j=0
        for i, log in enumerate(self.sheet.log):
            if log in examples:
                df = all_data[log]['data']
                pl.plot_vertical_edges(df, axs[j])
                pl.plot_trajectory(df, axs[j])
                pl.plot_trajectory_odor(df, axs[j])
                j+=1
        fig.savefig(os.path.join(self.figurefol, 'example_trajectories.pdf'))

    def heatmap(self,cmap='gist_gray', overwrite = True, plot_individual=False, res = 5):
        @jit(nopython=True)
        def populate_heatmap(xi, yi, x_bounds, y_bounds, fly_density):
            x_previous,y_previous = 1000,1000
            for i in np.arange(len(xi)):
                if min(x_bounds)<=xi[i]<=max(x_bounds) and min(y_bounds)<=yi[i]<=max(y_bounds):
                    x_index = np.argmin((xi[i]-x_bounds)**2)
                    y_index = np.argmin((yi[i]-y_bounds)**2)
                    if x_index != x_previous or y_index != y_previous:
                        fly_density[x_index, y_index] += 1
                        x_previous = x_index
                        y_previous = y_index
            return fly_density

        filename = os.path.join(self.picklefol, 'heatmaps.p')
        d = {}


        if not os.path.exists(filename) or overwrite:
            # set up arrays for heatmap
            x_bounds = np.arange(-200, 200, res)
            y_bounds = np.arange(0, 1000, res)
            x_previous = [1000, 1000]
            y_previous = [1000, 1000]
            x_previous_rotate = [1000, 1000]
            y_previous_rotate = [1000, 1000]
            x_bounds_rotate = np.arange(-300, 300, res)
            y_bounds_rotate = np.arange(0, 1000, res)
            d['x_bounds'] = x_bounds
            d['y_bounds'] = y_bounds
            d['x_bounds_rotate'] = x_bounds_rotate
            d['y_bounds_rotate'] = y_bounds_rotate
            fly_density = np.zeros((len(x_bounds), len(y_bounds)))

            all_data = self.load_trajectories()

            for i, log in enumerate(self.sheet.log):
                print(log)
                df = all_data[log]['data']
                if self.sheet.iloc[i].genotype=='wt':
                    print('enter')
                    continue
                # crop trajectories so that they start when the odor turns on
                if self.sheet.iloc[i].mfc=='new':
                    idx_start = df.index[df.mfc2_stpt>0.01].tolist()[0]
                    df = df.iloc[idx_start:]
                xi = df.ft_posx.to_numpy()
                yi = df.ft_posy.to_numpy()
                xi=xi-xi[0]
                yi=yi-yi[0]

                # # save and example trajectory to plot on top of the heatmap
                # if log in examples:
                #     x_ex = xi
                #     y_ex = yi
                # if plot_individual == True:
                #     fig, axs = plt.subplots(1,1)
                #     axs.plot(xi,yi)
                #     axs.title.set_text(log)
                fly_density = populate_heatmap(xi, yi, x_bounds, y_bounds, fly_density)

            fly_density = np.rot90(fly_density, k=1, axes=(0,1))
            fly_density = fly_density/np.sum(fly_density)
            d['fly_density'] = fly_density
            fn.save_obj(d, filename)
        elif os.path.exists(filename):
            print('enter')
            d = fn.load_obj(filename)
            print(d.keys())
            fly_density = d['fly_density']
            x_bounds = d['x_bounds']
            y_bounds = d['y_bounds']
        fig, axs = plt.subplots(1,1)
        vmin = np.percentile(fly_density[fly_density>0],0)
        vmax = np.percentile(fly_density[fly_density>0],90)
        im = axs.imshow(fly_density, cmap=cmap, vmin=vmin,vmax = vmax, rasterized=True, extent=(min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds)))
        axs.plot([-25,-25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)
        axs.plot([25,25], [np.min(y_bounds), np.max(y_bounds)], 'w', alpha=0.2)

        # axs.plot(x_ex, y_ex, 'red', linewidth=1)#, alpha = 0.8, linewidth=0.5)
        fig.colorbar(im)
        fig.savefig(os.path.join(self.figurefol, 'dopr1_mutant_heatmap.pdf'))

        fig, axs = plt.subplots(1,1)
        # vmin = np.percentile(fly_density_rotate[fly_density_rotate>0],10)
        # vmax = np.percentile(fly_density_rotate[fly_density_rotate>0],90)
        #axs.imshow(fly_density_rotate, cmap=cmap, vmin=1,vmax = 6, extent=(min(x_bounds_rotate), max(x_bounds_rotate), min(y_bounds_rotate), max(y_bounds_rotate)))


        # boundaries
        boundary=25
        fly_density_projection = fly_density[0:-40, :]/np.sum(fly_density[0:-40, :])
        x_mean = np.sum(fly_density_projection, axis = 0)
        fig, axs = plt.subplots(1,1)
        axs.plot([-boundary, -boundary], [min(x_mean), max(x_mean)], 'k', alpha=0.5)
        axs.plot([boundary, boundary], [min(x_mean), max(x_mean)],'k', alpha=0.5)
        axs.plot(x_bounds, x_mean)
        axs.set_ylim(0,0.06)
        fig.savefig(os.path.join(self.figurefol, 'dopr1_mutant_density.pdf'))
        return fly_density

    #
    # def distance_tracked_upwind(self):
    #     all_data = self.load_trajectories()
    #
    #     df = self.sheet
    #     upwind_dist=[]
    #     for fly in df.fly.unique():
    #         df_fly = df[df.fly==fly]
    #         for i, log in enumerate(df_fly.log):
    #             # find where odor turns on
    #             data = all_data[log]['data']
    #             ix0 = data.index[data.instrip==True].tolist()[0]
    #             data = data.iloc[ix0:]
    #             data.ft_posy = data.ft_posy-data.ft_posy.iloc[0]
    #             data_instrip = data.mask(data.instrip==False)
    #             y = data_instrip.ft_posy.to_numpy()
    #             experiment = df_fly.iloc[i].experiment
    #             if experiment=='constant':
    #                 y_constant = np.nanmax(y)/1000
    #             elif experiment=='reverse':
    #                 y_reverse = np.nanmax(y)/1000
    #             else:
    #                 print('NO EXPERIMENT FOUND')
    #                 return
    #         upwind_dist.append([y_constant, y_reverse])
    #     upwind_dist = np.array(upwind_dist)
    #     fig, axs = plt.subplots(1,1, figsize=(2,6))
    #     axs = pl.paired_plot(axs, upwind_dist[:,0],upwind_dist[:,1])
    #     axs.set_xticks([0,1])
    #     axs.set_xticklabels(['constant', 'reverse'])
    #     axs.set_ylabel('distance tracked up plume (m)')
    #     axs.set_ylim(0,1.2)
    #     fig.savefig(os.path.join(self.figurefol, 'paired_plot.pdf'))

%matplotlib
rg = dopr1_mutant()
rg.heatmap()
# rg.plot_example_trajectory()
