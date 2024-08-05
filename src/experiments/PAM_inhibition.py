import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs
from src.utilities import plotting as pl
from src.utilities import imaging as im
from scipy import interpolate, stats
#from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(funcs)

class pam_inhibition():
    """
    1) specify experiment
    2) split up log files based on inside/outside
    3) pickle results
    """
    def __init__(self, experiment = 'T', download_logs = False):
        d = dr.drive_hookup()
        # your working directory
        self.cwd = cwd = os.getcwd()
        # LACIE working directory
        self.cwd = '/Volumes/LACIE/edge-tracking'
        self.experiment = experiment
        self.sheet_id = '1luntmJJj6tIsQEsFBoMbYDW8xBgd8wiUOVGgb6nsyzs'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
        self.logfol = os.path.join(self.cwd, 'data/pam_gtacr1/logs')
        self.picklefol = os.path.join(self.cwd, 'data/pam_gtacr1/pickles')
        self.picklesname = os.path.join(self.picklefol, 'pam_gtacr1.p')
        self.figurefol = os.path.join(self.cwd, 'figures/pam_gtacr1')

    def split_trajectories(self):
        all_data = {}
        for i, log in enumerate(self.sheet.log):
            data = funcs.read_log(os.path.join(self.logfol, log))
            mfc = self.sheet.mfc.iloc[i]
            # data['instrip'] = np.where(np.abs(data.ft_posx)<25, True, False)
            data = funcs.exclude_lost_tracking(data, thresh=10)
            if mfc == 'old': # all experiments performed with old MFCs
                data['instrip'] = np.where(np.abs(data.mfc3_stpt)>0, True, False)
            data = funcs.consolidate_in_out(data)
            data = funcs.calculate_speeds(data)
            d, di, do = funcs.inside_outside(data)
            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do}
            all_data[log] = dict_temp
        funcs.save_obj(all_data, self.picklesname)

    def load_trajectories(self):
        all_data = funcs.load_obj(self.picklesname)
        return all_data

    def plot_example_trajectory(self):
        no_lights = '10222020-123725_Fly2_PAMgtACR1_no_lights_001.log'
        lights = '10222020-124911_Fly2_PAMgtACR1_inside_lights_002.log'
        examples = [no_lights, lights]
        all_data = self.load_trajectories()
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        for i, log in enumerate(examples):
            df = all_data[log]['data']
            di = all_data[log]['di']
            axs[i] = pl.plot_vertical_edges(df, axs[i])
            axs[i] = pl.plot_trajectory(df, axs[i])
            axs[i] = pl.plot_trajectory_odor(df, axs[i])
            print(len(df))
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'example_trajectories.pdf'))

    def plot_distance_up_plume(self):
        """
        plot the distance tracked up the plume and the average upwind velocity
        """
        df = self.sheet
        all_data = self.load_trajectories()
        fig, axs = plt.subplots(1,2)
        dist_upwind = [[],[]]
        speed_upwind = [[],[]]
        conditions = ['on', 'off']
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            for i,condition in enumerate(conditions):
                log = df_fly[df_fly.experiment == condition].log.to_list()[0]
                print(log)
                ft2 = all_data[log]['data']
                maxy = ft2.mask(ft2.instrip==False).ft_posy.max()
                dist_upwind[i].append(maxy)
                del_t = np.mean(np.diff(ft2.seconds))
                speed_y = np.gradient(ft2.ft_posy)/del_t
                speed_y = np.mean(speed_y)
                speed_upwind[i].append(speed_y)
        dist_upwind = np.array(dist_upwind)
        speed_upwind = np.array(speed_upwind)
        for col in np.arange(dist_upwind.shape[1]):
            jitter = np.random.normal(loc=0, scale=0.05, size=2)
            axs[0].plot([0,1]+jitter, dist_upwind[:,col], 'k', alpha=0.2)
            axs[1].plot([0,1]+jitter, speed_upwind[:,col], 'k', alpha=0.2)
            #axs.plot([0,1,2]+jitter, dist_upwind[:,col], color='k',marker='.', alpha=0.3, markersize=10)
        axs[0].plot([0,1], np.mean(dist_upwind, axis=1), color='k')
        axs[0].plot([0,1], np.mean(dist_upwind, axis=1), color='k', marker='.', markersize=10)
        axs[0].set_xticks([0,1])
        axs[0].set_xticklabels(['on', 'off'])
        axs[0].set_ylabel('distance tracked up plume (mm)')
        axs[1].plot([0,1], np.mean(speed_upwind, axis=1), color='k')
        axs[1].plot([0,1], np.mean(speed_upwind, axis=1), color='k', marker='.', markersize=10)
        axs[1].set_xticks([0,1])
        axs[1].set_xticklabels(['on', 'off'])
        axs[1].set_ylabel('average upwind speed (mm/s)')
        fig.savefig(os.path.join(self.figurefol, 'distance_speed_upwind.pdf'))
        fig.tight_layout()

%matplotlib
pam_inhibition().plot_distance_up_plume()
