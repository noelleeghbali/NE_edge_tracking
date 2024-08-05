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

class fed_pam_chrimson():
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
        self.sheet_id = '1mFPXUYS8vRbGXEB5hvK0uro3r3Dztvsq8KVelRPfOoI'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
        self.logfol = os.path.join(self.cwd, 'data/fed_pam_chrimson/logs')
        self.picklefol = os.path.join(self.cwd, 'data/fed_pam_chrimson/pickles')
        self.picklesname = os.path.join(self.picklefol, 'fed_pam_chrimson.p')
        self.figurefol = os.path.join(self.cwd, 'figures/fed_pam_chrimson')

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
        no_lights = '09232020-174714_Fly1_PAMChr_fed_no_lights.log'
        single_light = '09232020-180045_Fly1_PAMChr_fed_single.log'
        repeated_light = '09232020-181222_Fly1_PAMChr_fed_lights.log'
        examples = [no_lights, single_light, repeated_light]
        all_data = self.load_trajectories()
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
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
        dist_upwind = [[],[],[]]
        speed_upwind = [[],[],[]]
        conditions = ['no_lights', 'single', 'multiple']
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
            jitter = np.random.normal(loc=0, scale=0.05, size=3)
            axs[0].plot([0,1,2]+jitter, dist_upwind[:,col], 'k', alpha=0.2)
            axs[1].plot([0,1,2]+jitter, speed_upwind[:,col], 'k', alpha=0.2)
            #axs.plot([0,1,2]+jitter, dist_upwind[:,col], color='k',marker='.', alpha=0.3, markersize=10)
        axs[0].plot([0,1,2], np.mean(dist_upwind, axis=1), color='k')
        axs[0].plot([0,1,2], np.mean(dist_upwind, axis=1), color='k', marker='.', markersize=10)
        axs[0].set_xticks([0,1,2])
        axs[0].set_xticklabels(['none', 'single', 'multiple'])
        axs[0].set_ylabel('distance tracked up plume (mm)')
        axs[1].plot([0,1,2], np.mean(speed_upwind, axis=1), color='k')
        axs[1].plot([0,1,2], np.mean(speed_upwind, axis=1), color='k', marker='.', markersize=10)
        axs[1].set_xticks([0,1,2])
        axs[1].set_xticklabels(['none', 'single', 'multiple'])
        axs[1].set_ylabel('average upwind speed (mm/s)')
        fig.savefig(os.path.join(self.figurefol, 'distance_speed_upwind.pdf'))

    def plot_average_traj(self):
        """
        plot the average inside and outside trajectories
        try x and y sem for the 2d trajectories
        """
        from scipy.stats import sem
        pts=10000
        df = self.sheet
        all_data = self.load_trajectories()
        conditions = ['no_lights', 'single', 'multiple']
        fig,axs = plt.subplots(1,3, sharex=True, sharey=True)
        averages_out = [[[],[]], [[],[]], [[],[]]]
        averages_in = [[[],[]], [[],[]], [[],[]]]
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            for i,condition in enumerate(conditions):
                avg_x_out, avg_y_out = averages_out[i]
                avg_x_in, avg_y_in = averages_in[i]
                avg_x_out_temp, avg_y_out_temp = [],[]
                avg_x_in_temp, avg_y_in_temp = [],[]
                log = df_fly[df_fly.experiment == condition].log.to_list()[0]
                do = all_data[log]['do']
                di = all_data[log]['di']
                for key in list(do.keys())[1:]:
                    temp = do[key]
                    if len(temp)>10:
                        temp = funcs.find_cutoff(temp)
                        x = temp.ft_posx.to_numpy()
                        y = temp.ft_posy.to_numpy()
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        if np.abs(x[-1]-x[0])<1:
                            if np.mean(x)>0: # align insides to the right and outsides to the left
                                x = -x
                            t = np.arange(len(x))
                            t_common = np.linspace(t[0], t[-1], pts)
                            fx = interpolate.interp1d(t, x)
                            fy = interpolate.interp1d(t, y)
                            #axs.plot(fx(t_common), fy(t_common))
                            avg_x_out_temp.append(fx(t_common))
                            avg_y_out_temp.append(fy(t_common))
                avg_x_out.append(avg_x_out_temp)
                avg_y_out.append(avg_y_out_temp)
                averages_out[i] = avg_x_out, avg_y_out
                for key in list(di.keys())[1:]:
                    temp = di[key]
                    if len(temp)>10:
                        temp = funcs.find_cutoff(temp)
                        x = temp.ft_posx.to_numpy()
                        y = temp.ft_posy.to_numpy()
                        x0 = x[-1]
                        y0 = y[-1]
                        x = x-x0
                        y = y-y0
                        if np.abs(x[-1]-x[0])<1:
                            if np.mean(x)<0: # align insides to the right and outsides to the left
                                x = -x
                            t = np.arange(len(x))
                            t_common = np.linspace(t[0], t[-1], pts)
                            fx = interpolate.interp1d(t, x)
                            fy = interpolate.interp1d(t, y)
                            #axs.plot(fx(t_common), fy(t_common))
                            avg_x_in_temp.append(fx(t_common))
                            avg_y_in_temp.append(fy(t_common))
                avg_x_in.append(avg_x_in_temp)
                avg_y_in.append(avg_y_in_temp)
                averages_in[i] = avg_x_in, avg_y_in

        for i,condition in enumerate(conditions):
            avg_x_out, avg_y_out = averages_out[i]
            avg_x_in, avg_y_in = averages_in[i]
            all_x_in = []
            all_y_in = []
            all_x_out = []
            all_y_out = []
            for j in np.arange(len(avg_x_out)):
                axs[i].plot(np.mean(avg_x_out[j], axis=0), np.mean(avg_y_out[j], axis=0), 'k', alpha=0.2)
                all_x_out.append(np.mean(avg_x_out[j], axis=0))
                all_y_out.append(np.mean(avg_y_out[j], axis=0))
            for j in np.arange(len(avg_x_in)):
                axs[i].plot(np.mean(avg_x_in[j], axis=0), np.mean(avg_y_in[j], axis=0), 'r', alpha=0.2)
                all_x_in.append(np.mean(avg_x_in[j], axis=0))
                all_y_in.append(np.mean(avg_y_in[j], axis=0))
            axs[i].plot(np.mean(all_x_in, axis=0), np.mean(all_y_in, axis=0), 'r')
            axs[i].plot(np.mean(all_x_out, axis=0), np.mean(all_y_out, axis=0), 'k')
        fig.savefig(os.path.join(self.figurefol, 'average_trajectory.pdf'))
        #
        #     x0 = x_in[-1]
        #     y0 = y_in[-1]
        #     plot_individual=False
        #     if plot_individual:
        #         for j in np.arange(len(avg_x_in)):
        #             axs[i].plot(avg_x_in[j], avg_y_in[j], 'r', alpha=0.05)
        #         for j in np.arange(len(avg_x_out)):
        #             axs[i].plot(avg_x_out[j]+x0, avg_y_out[j]+y0, 'k', alpha=0.05)
        #     axs[i].plot(x_in, y_in, color='r')
        #     axs[i].plot(np.mean(avg_x_out, axis=0)+x0, np.mean(avg_y_out, axis=0)+y0, color='k')
        # fig.savefig(os.path.join(self.figurefol, 'average_trajectories.pdf'))
        return averages_in











    # def plot_linked_upwind_distance=(self):



%matplotlib
averages_in = fed_pam_chrimson().plot_average_traj()
