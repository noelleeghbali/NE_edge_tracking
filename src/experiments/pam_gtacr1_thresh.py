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
        self.sheet_id = '1ZGPrQSciyYYftOqrHcKMGQTu9jh7gb5OY0I0_W0EmW0'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
        self.logfol = os.path.join(self.cwd, 'data/pam_gtacr1_thresh/logs')
        self.picklefol = os.path.join(self.cwd, 'data/pam_gtacr1_thresh/pickles')
        self.picklesname = os.path.join(self.picklefol, 'pam_gtacr1_thresh.p')
        self.figurefol = os.path.join(self.cwd, 'figures/pam_gtacr1_thresh')

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

    def split_trajectories_select(self):
        select_logs = ['10292021-142308_PAMGTACR1_thresh_lights_LT_Fly2_002.log',
                      '10292021-143856_PAMGTACR1_thresh_lights_LT_Fly2_003.log',
                      '11102021_Fly1/11102021-104614_PAMGtACR1_thresh_lights_old_food_yeast_Fly1_001.log',
                      '11102021-110114_PAMGTACR1_thresh_lights_old_food_yeast_Fly1_002.log',
                      '11102021-134854_PAMGtACR1_thresh_lights_old_food_yeast_Fly2_001.log',
                      '11102021-140908_PAMGtACR1_thresh_lights_old_food_yeast_Fly2_002.log',
                      '11122021-151822_PAMGtACR1_Fly1_001.log',
                      '11122021-151822_PAMGtACR1_Fly1_001.log',
                      '11152021-150321_PAMGtACR1_Fly1_002.log',
                      '11152021-153105_PAMGtACR1_Fly1_003.log',
                      '11162021-104332_PAMGTACR1_Fly1_001.log',
                      '11162021-105012_PAMGtACR1_Fly1_002.log',
                      ]
        all_data = {}
        for i, log in enumerate(self.sheet.log):
            if log in select_logs:
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
        funcs.save_obj(all_data, os.path.join(self.picklefol, 'select_trajectories.p'))

    def load_trajectories(self):
        all_data = funcs.load_obj(self.picklesname)
        return all_data

    def load_trajectories_select(self):
        filename = os.path.join(self.picklefol, 'select_trajectories.p')
        all_data = funcs.load_obj(filename)
        return all_data

    def plot_individual_trajectories(self):
        all_data = self.load_trajectories()
        for key in list(all_data.keys()):
            df = all_data[key]['data']
            fig, axs = plt.subplots(1,1)
            axs = pl.plot_trajectory(df, axs)
            #axs = pl.plot_vertical_edges(df, axs)
            axs = pl.plot_trajectory_lights(df, axs)
            fig.suptitle(key)


    def plot_example_trajectory(self):
        example = ['11102021-134854_PAMGtACR1_thresh_lights_old_food_yeast_Fly2_001.log']


        all_data = self.load_trajectories_select()
        fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
        for i, log in enumerate(example):
            df = all_data[log]['data']
            di = all_data[log]['di']
            axs = pl.plot_vertical_edges(df, axs)
            axs = pl.plot_trajectory(df, axs)
            axs = pl.plot_trajectory_odor(df, axs)
            axs = pl.plot_trajectory_lights(df, axs)
        axs.axis('equal')

        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'example_trajectory.pdf'))

    def plot_average_traj(self):
        """
        plot the average inside and outside trajectories
        try x and y sem for the 2d trajectories
        """
        from scipy.stats import sem
        pts=10000

        all_data = self.load_trajectories_select()
        fig,axs = plt.subplots(1,2, sharex=True, sharey=True)
        avg_x_out_l, avg_y_out_l, avg_x_out_nl, avg_y_out_nl = [],[],[],[]
        avg_x_in_l, avg_y_in_l, avg_x_in_nl, avg_y_in_nl = [],[],[],[]
        for key in list(all_data.keys()):
            do = all_data[key]['do']
            di = all_data[key]['di']
            df = all_data[key]['data']
            avg_x_out_temp_l, avg_y_out_temp_l, avg_x_out_temp_nl, avg_y_out_temp_nl = [],[],[],[]
            avg_x_in_temp_l, avg_y_in_temp_l, avg_x_in_temp_nl, avg_y_in_temp_nl = [],[],[],[]
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
                        # where the lights on inside
                        if di[key-1].led1_stpt.min()==0.0:
                            avg_x_out_temp_l.append(fx(t_common))
                            avg_y_out_temp_l.append(fy(t_common))
                        else:
                            avg_x_out_temp_nl.append(fx(t_common))
                            avg_y_out_temp_nl.append(fy(t_common))
            avg_x_out_l.append(avg_x_out_temp_l)
            avg_y_out_l.append(avg_y_out_temp_l)
            avg_x_out_nl.append(avg_x_out_temp_nl)
            avg_y_out_nl.append(avg_y_out_temp_nl)
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
                        if di[key].led1_stpt.min()==0.0:
                            avg_x_in_temp_l.append(fx(t_common))
                            avg_y_in_temp_l.append(fy(t_common))
                        else:
                            avg_x_in_temp_nl.append(fx(t_common))
                            avg_y_in_temp_nl.append(fy(t_common))
            avg_x_in_l.append(avg_x_in_temp_l)
            avg_y_in_l.append(avg_y_in_temp_l)
            avg_x_in_nl.append(avg_x_in_temp_nl)
            avg_y_in_nl.append(avg_y_in_temp_nl)
            f,a = plt.subplots(1,3)
            a[0] = pl.plot_trajectory(df, a[0])
            a[0] = pl.plot_trajectory_lights(df, a[0])
            for i in np.arange(len(avg_x_in_temp_l)):
                a[1].plot(avg_x_in_temp_l[i], avg_y_in_temp_l[i], 'r')
            for i in np.arange(len(avg_x_out_temp_l)):
                a[1].plot(avg_x_out_temp_l[i], avg_y_out_temp_l[i], 'k')
            for i in np.arange(len(avg_x_in_temp_nl)):
                a[2].plot(avg_x_in_temp_nl[i], avg_y_in_temp_nl[i], 'r')
            for i in np.arange(len(avg_x_out_temp_nl)):
                a[2].plot(avg_x_out_temp_nl[i], avg_y_out_temp_nl[i], 'k')

        all_x_in_l, all_y_in_l, all_x_in_nl, all_y_in_nl = [],[],[],[]
        all_x_out_l, all_y_out_l, all_x_out_nl, all_y_out_nl = [],[],[],[]


        for i in np.arange(len(avg_x_in_l)):
            axs[0].plot(np.mean(avg_x_in_l[i], axis=0), np.mean(avg_y_in_l[i], axis=0), color='red', alpha=0.2)
            all_x_in_l.append(np.mean(avg_x_in_l[i], axis=0))
            all_y_in_l.append(np.mean(avg_y_in_l[i], axis=0))
        for i in np.arange(len(avg_x_out_l)):
            axs[0].plot(np.mean(avg_x_out_l[i], axis=0), np.mean(avg_y_out_l[i], axis=0), color='black', alpha=0.2)
            all_x_out_l.append(np.mean(avg_x_out_l[i], axis=0))
            all_y_out_l.append(np.mean(avg_y_out_l[i], axis=0))
        for i in np.arange(len(avg_x_in_nl)):
            axs[1].plot(np.mean(avg_x_in_nl[i], axis=0), np.mean(avg_y_in_nl[i], axis=0), color='red', alpha=0.2)
            all_x_in_nl.append(np.mean(avg_x_in_nl[i], axis=0))
            all_y_in_nl.append(np.mean(avg_y_in_nl[i], axis=0))
        for i in np.arange(len(avg_x_out_nl)):
            axs[1].plot(np.mean(avg_x_out_nl[i], axis=0), np.mean(avg_y_out_nl[i], axis=0), color='black', alpha=0.2)
            all_x_out_nl.append(np.mean(avg_x_out_nl[i], axis=0))
            all_y_out_nl.append(np.mean(avg_y_out_nl[i], axis=0))
        axs[0].plot(np.mean(all_x_in_l, axis=0), np.mean(all_y_in_l, axis=0), color='red')
        axs[0].plot(np.mean(all_x_out_l, axis=0), np.mean(all_y_out_l, axis=0), color='black')
        axs[1].plot(np.mean(all_x_in_nl, axis=0), np.mean(all_y_in_nl, axis=0), color='red')
        axs[1].plot(np.mean(all_x_out_nl, axis=0), np.mean(all_y_out_nl, axis=0), color='black')
        fig.savefig(os.path.join(self.figurefol, 'average_trajectories.pdf'))  

        return di, do


        # for i,condition in enumerate(conditions):
        #     avg_x_out, avg_y_out = averages_out[i]
        #     avg_x_in, avg_y_in = averages_in[i]
        #     all_x_in = []
        #     all_y_in = []
        #     all_x_out = []
        #     all_y_out = []
        #     for j in np.arange(len(avg_x_out)):
        #         axs[i].plot(np.mean(avg_x_out[j], axis=0), np.mean(avg_y_out[j], axis=0), 'k', alpha=0.2)
        #         all_x_out.append(np.mean(avg_x_out[j], axis=0))
        #         all_y_out.append(np.mean(avg_y_out[j], axis=0))
        #     for j in np.arange(len(avg_x_in)):
        #         axs[i].plot(np.mean(avg_x_in[j], axis=0), np.mean(avg_y_in[j], axis=0), 'r', alpha=0.2)
        #         all_x_in.append(np.mean(avg_x_in[j], axis=0))
        #         all_y_in.append(np.mean(avg_y_in[j], axis=0))
        #     axs[i].plot(np.mean(all_x_in, axis=0), np.mean(all_y_in, axis=0), 'r')
        #     axs[i].plot(np.mean(all_x_out, axis=0), np.mean(all_y_out, axis=0), 'k')
        # fig.savefig(os.path.join(self.figurefol, 'average_trajectory.pdf'))

%matplotlib
di, do = pam_inhibition().plot_average_traj()
do[55]

#all_data['11102021-134854_PAMGtACR1_thresh_lights_old_food_yeast_Fly2_001.log']
