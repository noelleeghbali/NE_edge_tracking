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

class spontaneous_et:
    def __init__(self, directory='M1'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'

        # specify data sheet
        self.sheet_id = '1v2z5npPBzF1et2OQoA2CQuQp8ScBT5HX3uv8KEaAXlk'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
    
        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/spontaneous_et/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'et_manuscript_spontaneous_et.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/spontaneous_et')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')

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

    def distance_tracked_upwind_plots(self):
        """
        sort trajectories into 4 different categories for upwind tracking
        """

        # plot parameters
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style("white")
        sns.set_style("ticks")

        # load the data
        all_data = self.load_trajectories()
        df = self.sheet

        fig, axs = plt.subplots(1,4, figsize=(3,2))
        
        titles = ['0-250', '250-500', '500-750', '750-1000']
        
        # make plume cartoons
        for i in np.arange(4):
            axs[i] = pl.plot_plume_corridor(axs[i])
            axs[i].set_ylim(-100,1000)
            axs[i].set_xlim(-100,100)
            axs[i].axis('off')
            axs[i].set_title(titles[i], rotation=45)
        
        # make scale bar
        axs[1].plot([-25, 25], [-20,-20])
        axs[1].text(-50, -100, '50 mm')
        
        upwind_dist=[]
        
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            for i, log in enumerate(df_fly.log):
                # find where odor turns on
                data = all_data[log]['data']
                ix0 = data.index[data.instrip==True].tolist()[0]
                data = data.iloc[ix0:]
                
                data.ft_posy = data.ft_posy-data.ft_posy.iloc[0]
                data.ft_posx = data.ft_posx-data.ft_posx.iloc[0]
                data_instrip = data.mask(data.instrip==False)
                y = data_instrip.ft_posy.to_numpy()
                experiment = df_fly.iloc[i].experiment
                
                y_upwind = np.nanmax(y)
                if y_upwind>1000:
                    y_upwind=999
                if 0<y_upwind<=250:
                    ax=axs[0]
                elif 250<y_upwind<=500:
                    ax=axs[1]
                elif 500<y_upwind<=750:
                    ax=axs[2]
                elif 750<y_upwind<=1000:
                    ax=axs[3]
                ax = pl.plot_trajectory(data,ax, linewidth=0.5)
                ax = pl.plot_trajectory_odor(data, ax, linewidth=0.5)
                

            upwind_dist.append(y_upwind)
        upwind_dist = np.array(upwind_dist)
        
        # save the trajectory plot
        fig.tight_layout()
        plt.subplots_adjust(wspace=-0.7)
        fig.savefig(os.path.join(self.figurefol, 'spontaneous_et_trajectories.pdf'))

        # save the upwind distance tracked
        fig, axs = plt.subplots(1,1, figsize = (2,2))
        sns.histplot(upwind_dist,ax=axs, bins = np.linspace(0,1000,5), color='grey')
        sns.despine()
        axs.set_xticks([0,250,500,750,1000])
        axs.set_xlabel('distance tracked up plume (mm)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'spontaneous_et_distance_tracked_upwind.pdf'))

        # make distance tracked upwind plot
        fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5))
        sorted = -np.sort(-upwind_dist/1000)
        fly_ix = np.arange(len(sorted))+1
        axs.plot(fly_ix, sorted, '.',color='k')
        axs.set_xticks(fly_ix)
        axs.set_xticklabels('')
        axs.set_yticks([0,0.5,1])
        axs.set_xlabel('fly')
        axs.set_ylabel('dist. tracked (m)')
        for axis in ['top','bottom','left','right']:
            axs.spines[axis].set_linewidth(0.75)
        axs.tick_params(width=0.75, length=4)
        # axs.tick_params(axis='y',width=0.75, length=4, pad=-1, labelsize=6)
        # axs.tick_params(axis='x', width=0.0, length=0)
        sns.despine(ax=axs)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'spontaneous_et_individual.pdf'))

        return upwind_dist
