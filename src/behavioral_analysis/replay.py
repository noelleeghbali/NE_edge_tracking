import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pingouin as pg
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
from src.utilities import imaging as im
from scipy import interpolate, stats
from scipy.optimize import minimize
# from numba import jit
import seaborn as sns
import time
importlib.reload(dr)
importlib.reload(im)
importlib.reload(pl)
importlib.reload(fn)


sns.set(font="Arial")
sns.set(font_scale=0.6)
sns.set_style("white")
sns.set_style("ticks")

class replay():
    """
    class for comparing constant concentration and gradient plumes
    """
    def __init__(self, directory='M1', experiment = 'constant'):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
        self.experiment = experiment

        # specify which Google sheet to pull log files from
        self.sheet_id = '1tWHbrcTY1zfFXA2TzyiPk_YG45T9m5plnpA7ZcLdgws'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/replay/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'tilted_replay.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/replay')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # download new log folders
        self.logfol = '/Volumes/Andy/logs'
        d = dr.drive_hookup()
        #d.download_logs_to_local('/Volumes/Andy/logs')

    def split_trajectories(self):
        # dict where all log files are stored
        all_data = {}
        accept = []
        for i, log in enumerate(self.sheet.logfile):
            # specify trial type
            trial_type = self.sheet.condition.iloc[i]
            # read in each log file
            data = fn.read_log(os.path.join(self.logfol, log))
            # if the tracking was lost, select correct segment

            mfcs = self.sheet.mfcs.iloc[i]
            if mfcs == '1':
                data['instrip'] = np.where(np.abs(data.mfc2_stpt)>0.01, True, False)
            # consolidate short in and short out periods
            data = fn.consolidate_in_out(data)

            # append speeds to dataframe
            data = fn.calculate_speeds(data)

            # annotate replay
            data['seconds'] = data['seconds']-data['seconds'].iloc[0]
            data.loc[data.seconds>840,'mode'] = 'replay'
            # split trajectories into inside and outside component
            d, di, do = fn.inside_outside(data)

            dict_temp = {"data": data,
                        "d": d,
                        "di": di,
                        "do": do,
                        "trial_type":trial_type
                        }
            all_data[log] = dict_temp
        # pickle everything
        fn.save_obj(all_data, self.picklesname)


    def load_trajectories(self):
        """
        open the pickled data stored from split_trajectories()
        """
        all_data = fn.load_obj(self.picklesname)
        return all_data

    def plot_individual_trajectories(self):
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('white')

        all_data = self.load_trajectories()
        for log in list(all_data.keys()):
            print(log)
            temp = all_data[log]
            data = temp['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(data, axs)
            pl.plot_trajectory_odor(data, axs)
            axs.axis('equal')
            fig.suptitle(log)
            fig.savefig(os.path.join(self.figurefol, 'trajectory_'+log.replace('.log', '.pdf')), transparent=True)
        return temp
    
    def entry_direction(self):
        def calc_entry_heading(df, t0):
            heading = fn.wrap(df['ft_heading'])
            t = df['seconds'].to_numpy()
            t = t-t[0]
            tmax = t[-1]
            if tmax<t0:
                heading = fn.circmean(heading)
            else:
                ix = np.argmin(np.abs(t-(tmax-t0)))
                heading = fn.circmean(heading[ix:])
            return heading
        
        def calc_entry_direction(df, t0):
            x = df.ft_posx.to_numpy()
            y = df.ft_posy.to_numpy()
            t = df['seconds'].to_numpy()
            t = t-t[0]
            tmax = t[-1]
            if tmax>t0:
                ix = np.argmin(np.abs(t-(tmax-t0)))
                x = x[ix:]
                y = y[ix:]
            delx = x[-1]-x[0]
            dely = y[-1]-y[0]
            direction = np.array([np.arctan2(dely,delx)])
            direction = fn.conv_cart_upwind(direction)
            return direction

        def calc_outside_heading(df):
            heading = fn.wrap(df['ft_heading'])
            heading = fn.circmean(heading)
            return heading
        
        def calc_displacement(df):
            x = df.ft_posx.to_numpy()
            y = df.ft_posy.to_numpy()
            delx = x[-1]-x[0]
            dely = y[-1]-y[0]
            angle = np.arctan2(dely,delx)
            angle = fn.conv_cart_upwind(np.array([angle]))
            return angle
        
        def find_color(df):
            if 'replay' in df['mode'].unique():
                color = 'm'
            else:
                color = 'r'
            return color
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')


        all_data = self.load_trajectories()


        for log in list(all_data.keys()):
            heading_entry = []
            heading_outside = []
            displacement_outside = []
            all_colors = []

            temp = all_data[log]
            do = temp['do']
            data = temp['data']
            fig, axs = plt.subplots(1,1)
            pl.plot_trajectory(df=data, axs=axs)
            pl.plot_trajectory_odor(df=data, axs=axs)
            for key in list(do.keys())[:-1]:
                
                df1 = do[key]
                df2 = do[key+2]

                heading_entry.append(calc_entry_direction(df1, t0=1))
                heading_outside.append(calc_outside_heading(df2))
                displacement_outside.append(calc_displacement(df2))
                all_colors.append(find_color(df1))

            savebase = log.replace('.log', '')
            fig, axs = plt.subplots(1,1)
            all_colors = np.array(all_colors)
            axs.plot(np.arange(len(heading_entry)), heading_entry)
            axs.plot(np.arange(len(heading_outside)), displacement_outside)
            num_live = len(all_colors[all_colors=='r'])
            axs.plot([num_live+0.5, num_live+0.5], [-np.pi, np.pi], 'k')
            axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs.set_ylabel('angle')
            axs.set_xlabel('bout number')
            fig.savefig(os.path.join(self.figurefol, savebase+'_vs_bout.pdf'))

            fig, axs = plt.subplots(1,1,figsize=(3,3))
            axs.scatter(heading_entry, displacement_outside, color=all_colors)
            axs.set_ylim(-np.pi, np.pi)
            axs.set_xlim(-np.pi, np.pi)
            axs.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
            axs.set_xlabel('entry heading (prev 5 sec.)')
            axs.set_ylabel('outside displacement vector')
            fig.tight_layout()
            fig.savefig(os.path.join(self.figurefol, 'scatter.pdf'))


        # heading_entry = np.array(heading_entry)
        # displacement_outside = np.array(displacement_outside)
        # all_colors = np.array(all_colors)

        # heading_entry = heading_entry[all_colors == 'm']
        # displacement_outside = displacement_outside[all_colors == 'm']
        # all_colors = all_colors[all_colors == 'm']

        # fig, axs = plt.subplots(1,1,figsize=(3,3))
        # axs.scatter(heading_entry, displacement_outside, color=all_colors)
        # axs.set_ylim(-np.pi, np.pi)
        # axs.set_xlim(-np.pi, np.pi)
        # axs.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
        # axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
        # axs.set_xlabel('entry heading (prev 5 sec.)')
        # axs.set_ylabel('outside displacement vector')
        # fig.tight_layout()
        # fig.savefig(os.path.join(self.figurefol, 'scatter.pdf'))

        return heading_entry, displacement_outside, all_colors

        