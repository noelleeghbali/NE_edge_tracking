import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from src.drive import drive as dr
from src.utilities import funcs
from src.utilities import plotting as pl
from scipy import signal as sg
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(funcs)
matplotlib.rcParams['pdf.fonttype'] = 42

sns.set(font="Arial")
sns.set(font_scale=0.6)
sns.set_style("white")
sns.set_style("ticks")


class left_right_antenna():
    """
    class for analyzing any potential timing differences between the left and right antenna
    """

    def __init__(self, directory = 'M1', download_data = False):
        d = dr.drive_hookup()
        # your working directory
        if directory == 'M1':
            self.cwd = os.getcwd()
        elif directory == 'LACIE':
            self.cwd = '/Volumes/LACIE/edge-tracking'
        elif directory == 'Andy':
            self.cwd = '/Volumes/Andy/GitHub/edge-tracking'
       
        # specify figure folder
        self.figurefol = os.path.join(self.cwd, 'figures/left_right_antenna')
        if not os.path.exists(self.figurefol):
            os.makedirs(self.figurefol)

        # specify google sheet
        d = dr.drive_hookup()
        sheet_id = '1IdAygm44ZxcISwsviu6zvADZ0Yxq293U8wNT7TPYN0Q'
        self.sheet = d.pull_sheet_data(sheet_id, 'Sheet1')

        # download log files
        log_id = '1Bh80I22xH9U0GCPh6DFRAnR2tEup3bBX'
        log_id_location = 'data/left_right_antenna/log'
        self.log_id_location = os.path.join(self.cwd,log_id_location)
        if not os.path.exists(self.log_id_location):
            os.makedirs(self.log_id_location)


        # download csv files
        csv_id = '1rzr45k6f9IMQf_yi-S3ZacK9PGoH9iOM'
        csv_id_location = 'data/left_right_antenna/csv'
        self.csv_id_location = os.path.join(self.cwd,csv_id_location)
        if not os.path.exists(self.csv_id_location):
            os.makedirs(self.csv_id_location)
        
        if download_data:
            d.download_folder(csv_id, csv_id_location)
            d.download_folder(log_id, log_id_location)

    def make_plot(self):
        fig, axs = plt.subplots(2, 3)

        df = self.sheet
        tube_positions = df.tube.unique()
        for i_subplot,pos in enumerate(tube_positions):
            for index in df[df.tube == pos].index:
                temp = df.loc[index]
                if temp.antenna == 'right':
                    color = 'red'
                else:
                    color = 'green'

                # read log file
                log_loc = temp.log
                log_loc = os.path.join(self.log_id_location, log_loc)
                df_log = funcs.read_log(log_loc)
                time = df_log.seconds
                mfc = df_log.mfc2_stpt
                signal = df_log.sig_status

                # find start of pid recording
                sig_idx, _ = sg.find_peaks(signal, height=0.8)
                start_ix = sig_idx[0]

                # crop time and mfc based on start points
                time = time.iloc[start_ix:]
                mfc_time = time-time.iloc[0]
                mfc_command = mfc.iloc[start_ix:]

                # read recording csv. which starts when sig goes high
                csv_loc = temp.csv
                csv_loc = os.path.join(self.csv_id_location, csv_loc)
                df_csv = pd.read_csv(csv_loc)
                pid_reading = df_csv[' Input 3']
                pid_time = df_csv['Time(ms)']/1000

                # create same time base between pid and mfc by upsampling
                mfc_command_up = np.interp(pid_time,mfc_time,mfc_command)
                mfc_command_up[mfc_command_up>0]=0.05


                # find where the odor pulse comes on
                ix_on, _ = sg.find_peaks(np.gradient(mfc_command_up), height=0.0005)
                ix_off, _ = sg.find_peaks(np.gradient(-mfc_command_up), height=0.0005)

                #find the number of points that the odor is on
                num_pts_on = int(np.round(np.mean(ix_off-ix_on)))

                all_pulses = []
                pre_ix = 1000
                post_ix = 2000
                for i, ix in enumerate(ix_on):
                    time = np.arange(-pre_ix,post_ix+num_pts_on)
                    odor = pid_reading[ix-pre_ix:ix+num_pts_on+post_ix]
                    all_pulses.append(odor.to_list())
                    # axs[0,i_subplot].plot(time, odor, color, alpha=0.1)
                    # axs[1,i_subplot].plot(time, odor, color, alpha=0.1)
                command = np.zeros(time.shape)
                pts_on = int(np.mean(ix_off-ix_on))
                command[pre_ix:pre_ix+pts_on] = 1
                #calculate average trace and baseline
                all_pulses = np.mean(np.array(all_pulses), axis=0)
                baseline = np.mean(all_pulses[0:1000])
                max = np.mean(all_pulses[4000:5000])
                all_pulses = (all_pulses-baseline)/(max-baseline)


                axs[0,i_subplot].plot(time/1000, all_pulses, color)
                axs[1,i_subplot].plot(time/1000, all_pulses, color)
                axs[1,i_subplot].plot(time/1000, command, 'k')

                axs[0,i_subplot].title.set_text(pos)

        fig.savefig(os.path.join(self.figurefol, 'pid.pdf'))

        fig, axs = plt.subplots(1,2)
        axs[0].plot(time/1000, all_pulses, color)
        axs[0].plot(time/1000, command, 'k')
        axs[0].plot([-1,7],[0.63, 0.63], 'k')
        axs[0].set_xlim(-1,1)

        axs[1].plot(time/1000, all_pulses, color)
        axs[1].plot(time/1000, command, 'k')
        axs[1].plot([-1,7],[0.32, 0.32], 'k')
        axs[1].set_xlim(4,6)

        fig.savefig(os.path.join(self.figurefol, 'pid_zoom.pdf'))

    def inter_antenna_timing(self):
        """
        make a schematic showing the inter-antennae timing differences
        as a function of wind speed.  Assumption: the two antennae are 250um apart.
        """
        fig, axs = plt.subplots(1,1, figsize = (1.5, 1.5))
        angles = np.linspace(-np.pi/2,np.pi/2)
        timing_func = (0.025)*np.sin(angles) # inter antenna distance in cm
        wind_speeds = [15, 25] #cm/s
        colors = ['lightgray', 'dimgray']
        for i,speed in enumerate(wind_speeds):
            timing_diff = timing_func/speed*1000 # units ms
            axs.plot(angles, timing_diff, color=colors[i])
        axs.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        axs.set_xticklabels([-90,-45,0,45,90])
        axs.set_xlabel('tube angle (o)')
        axs.set_yticks([-2,-1,0,1,2])
        axs.set_ylabel('L-R timing diff (s)')
        axs.spines[['right', 'top']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figurefol,'inter_antenna_timing.pdf'))

# # %%
# from scipy import signal as sg
# fig, axs = plt.subplots(1,1)
# axs.plot(command)
# axs.plot(all_pulses)
# axsr = axs.twinx()
# plt.plot(sg.deconvolve(all_pulses))
# conv = np.convolve(all_pulses, command)
# axsr.plot(conv)



# # %%
# from scipy.optimize import curve_fit
# fig, axs = plt.subplots(1,1)
# t1 = time[1380:4000]/1000
# t1 = t1-t1[0]
# p1 = all_pulses[1380:4000]
# p1_pts = 4000-1380

# t2 = time[6350:]/1000
# t2 = t2-t2[0]
# p2 = all_pulses[6350:]


# # def exp_func_up(x, k):
# #     return -1*pow(2, k*x)+1
# def exp_func_up(x, k):
#     return -1*np.exp(k*x)+1
# def exp_func_down(x, a1,k1,a2,k2):
#     return a1*np.exp(k1*x)+a2*np.exp(k2*x)
# params1, _ = curve_fit(exp_func_up, t1, p1, p0=[-3])
# params2, _ = curve_fit(exp_func_down, t2, p2, p0=[1,1,-1,-2])
# fit1 = exp_func_up(t1,*params1)
# fit2 = exp_func_down(t2,*params2)
# #fit_total = exp_func_up(t1,*params1)+exp_func_down(t1,*p

# fit = exp_func_up(time,*params1)-exp_func_down(time,*params2)
