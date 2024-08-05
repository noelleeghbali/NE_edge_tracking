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

class closed_loop():
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
        self.sheet_id = '1lHuVaIXhwMbvTJFavWd_NmSSdKddslTRiRw26PHDTDE'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df

        # specify pickle folder and pickle name
        self.picklefol = os.path.join(self.cwd, 'data/closed_loop_trigger_switch/pickles')
        if not os.path.exists(self.picklefol):
            os.makedirs(self.picklefol)
        self.picklesname = os.path.join(self.picklefol, 'closed_loop.p')

        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/closed_loop_trigger_switch')
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
            # # consolidate short in and short out periods
            # data = fn.consolidate_in_out(data)

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
    
    def entry_direction(self, repeats):
        def calc_entry_heading(df, t0):
            df['ft_heading'] = fn.wrap(df['ft_heading'])
            df['seconds'] = df['seconds']-df['seconds'].iloc[0]
            t = df['seconds'].to_numpy()
            tmax = df['seconds'].iloc[-1]
            if tmax<t0:
                heading = fn.circmean(df.ft_heading)
            else:
                ix = np.argmin(np.abs(t-(tmax-t0)))
                heading = fn.circmean(df.ft_heading.iloc[ix:])
            return heading

        def calc_outside_heading(df):
            df['ft_heading'] = fn.wrap(df['ft_heading'])
            heading = fn.circmean(df.ft_heading)
            return heading
        
        def calc_displacement(df):
            delx = df['ft_posx'].iloc[-1]-df['ft_posx'].iloc[0]
            dely = df['ft_posy'].iloc[-1]-df['ft_posy'].iloc[0]
            angle = np.arctan2(dely,delx)
            angle = fn.conv_cart_upwind(np.array([angle]))
            return angle
    
        def switch_reshape(a, r):
            num = np.ceil(len(a)/r)*r
            num_add = int(num-len(a))
            a = np.append(a, np.full(num_add, np.nan))
            a = np.reshape(a, (-1,r))
            return fn.circmean(a, axis=1)
        
        
        sns.set(font="Arial")
        sns.set(font_scale=0.6)
        sns.set_style('ticks')


        all_data = self.load_trajectories()
        num_fly = 0

        # returns dfs
        dfs = []
        # scatter plot
        fig, axs = plt.subplots(1,1,figsize=(3,3))

        # plot the diffs
        fig2, axs2 = plt.subplots(1,1, figsize=(3,3))

        for log in list(all_data.keys()):
            df = self.sheet
            print(df.loc[df.logfile==log].condition.item())
            if df.loc[df.logfile==log].condition.item() == str(repeats):
                print('include +', log)
                heading_entry = []
                heading_outside = []
                displacement_outside = []
                all_colors = []
                temp = all_data[log]
                do = temp['do']
                dfs.append(temp['data'])
                for key in list(do.keys())[:-1]:
                    
                    df1 = do[key]
                    df2 = do[key+2]

                    heading_entry.append(calc_entry_heading(df1, t0=0.5))
                    heading_outside.append(calc_outside_heading(df2))
                    displacement_outside.append(calc_displacement(df2))


                savebase = log.replace('.log', '')

                heading_entry = switch_reshape(heading_entry, repeats)
                heading_outside = switch_reshape(heading_outside, repeats)
                displacement_outside = switch_reshape(displacement_outside, repeats)

                # heading_outside = heading_outside[(repeats-1)::repeats]
                # displacement_outside = displacement_outside[(repeats-1)::repeats]
                
                # make individual plots
                fig1, axs1 = plt.subplots(1,2, figsize=(10,4))
                # plot the trajectory
                pl.plot_trajectory(df = temp['data'], axs=axs1[0])
                pl.plot_trajectory_odor(df = temp['data'], axs=axs1[0])
                # plot the entry heading and outside heading
                axs1[1].plot(np.arange(len(heading_entry)), heading_entry, 'k', marker='.')
                axs1[1].plot(np.arange(len(heading_outside)), displacement_outside, 'm')
                axs1[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
                axs1[1].set_ylabel('angle')
                axs1[1].set_xlabel('bout number')
                fig1.savefig(os.path.join(self.figurefol, savebase+'_vs_bout.pdf'))
                

                # # plot the scatter entry heading vs outside heading
                # color = sns.color_palette()[num_fly]
                # print(color)
                # axs.scatter(heading_entry, heading_outside,color='k', alpha=0.4)
                # axs.set_ylim(-np.pi, np.pi)
                # axs.set_xlim(-np.pi, np.pi)
                # axs.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
                # axs.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=['-pi', '-pi/2', '0', 'pi/2', 'pi'])
                # axs.set_xlabel('entry heading (prev 5 sec.)')
                # axs.set_ylabel('outside displacement vector')

                # plot the delta
                delta_heading = np.diff(heading_outside)
                delta_heading1 = delta_heading[0::2]
                delta_heading2 = delta_heading[1::2]
                axs2.plot(np.zeros(len(delta_heading1))+np.random.normal(loc=0.0, scale=0.1, size=len(delta_heading1)), delta_heading1, '.', color='r')
                axs2.plot(np.zeros(len(delta_heading2))+np.random.normal(loc=0.0, scale=0.1, size=len(delta_heading2)), delta_heading2, '.', color='b')

                num_fly+=1
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol, 'scatter_'+str(repeats)+'.pdf'))

        return dfs