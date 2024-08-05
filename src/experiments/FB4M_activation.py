import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs
from src.utilities import plotting as pl
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(funcs)
#%matplotlib

import pandas as pd

class fb4m_activation:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '12EvvUeiRs3mgQS4NiyGIeAFp7zjD_oKjMzkJWvwWh94'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df


        # load list of log files
        self.log_folder_id = '1h7JpUYj5PnqhjGjea78yslr5fBSbDM2Y'
        self.logfiles = d.get_files_in_folder(self.log_folder_id)

        # ensure that every entry in sheet has a corresponding log file on drive
        self.sheet = self.match_sheet_log()

        #check for duplicated logs in dataframe
        self.detect_duplicates()

        # define local site for log files. Only need this to do pre-processing of log files
        self.logfol = "/Volumes/LACIE/logs"

        # define drive location of pre-processed data
        self.preprocessed = '1iVnAPwK2WuZHfeW5uMRxVTtXDQ7BoFD1'

        self.savefolder = 'figures/FB4M'


    def match_sheet_log(self):
        """
        for each row in google sheet dataframe, matches datetime with a datetime
        of a log file.  updates the logfile name in the dataframe to match that
        of the .log name in drive.  prints all entries where there is not match
        for resolution.
        """
        df = self.sheet
        log_file_names = []
        for log in self.logfiles:
            log_file_names.append(log['name'])
        for i, log in enumerate(df.log):
            datetime = log.replace('-','')[0:14]
            match_found = False
            for name in log_file_names:
                if datetime == name.replace('-','')[0:14]:
                    df.iloc[i].log = name
                    if name == '04132022-114338_UAS-GtACR1_lights _on.log':
                        print('error')
                    match_found = True
                    break
            if not match_found:
                print('NO LOG FILE FOR ENTRY: ', df.iloc[i])
                print(datetime)
        return df

    def detect_duplicates(self):
        """
        detect duplicate log files in dataframe
        """
        duplicate = self.sheet[self.sheet.duplicated('log')]
        print(duplicate)

    def make_individual_plots(self):

        df = self.sheet
        for fly in df.fly.unique():
            # data frame for this fly
            df_fly = df[df.fly==fly]
            # how many trials
            num_trials = len(df_fly)
            num_rows = int(np.ceil(num_trials/2))
            # make a subplot to plot each trial
            fig, axs = plt.subplots(num_rows, 2, figsize=(4, 2*num_rows))
            axs = np.array(axs).reshape(-1)
            for ix, i in enumerate(df_fly.index):
                genotype = df_fly.loc[i].genotype
                filename = df_fly.loc[i].log
                trial_type = df_fly.loc[i].experiment
                file_loc = os.path.join(self.logfol, filename)
                log = funcs.read_log(file_loc)
                if trial_type == '0':
                    axs[ix] = pl.plot_vertical_edges(log, axs[ix])
                elif trial_type == '45':
                    axs[ix] = pl.plot_45_edges(log, axs[ix])
                elif trial_type == 'jump':
                    axs[ix] = pl.plot_jumping_edges(log, axs[ix])

                axs[ix] = pl.plot_trajectory(log, axs[ix])
                axs[ix] = pl.plot_trajectory_lights(log, axs[ix])


                # axs[ix].tick_params(axis='both', which='major', labelsize=5)
                # axs[ix].tick_params(axis='both', which='minor', labelsize=5)

        return axs, log

    def make_inside_outside_plots(self):

        df = self.sheet
        for fly in df.fly.unique():
            # data frame for this fly
            df_fly = df[df.fly==fly]
            # how many trials
            num_trials = len(df_fly)
            num_rows = int(np.ceil(num_trials/2))
            # make a subplot to plot each trial
            fig, axs = plt.subplots(num_rows, 2, figsize=(6, 3*num_rows))
            axs = np.array(axs).reshape(-1)
            for ix, i in enumerate(df_fly.index):
                genotype = df_fly.loc[i].genotype
                filename = df_fly.loc[i].log
                trial_type = df_fly.loc[i].experiment
                file_loc = os.path.join(self.logfol, filename)
                log = funcs.read_log(file_loc)
                print(filename)
                if 'instrip' not in log.columns:
                    continue
                d, di, do = funcs.inside_outside(log)
                print(len(di), len(do))
                do_lights = {}
                do_no_lights = {}
                for key in list(do.keys()):
                    df_temp = do[key]
                    if min(df_temp.led1_stpt)<0.1:
                        do_lights[key] = df_temp
                    else:
                        do_no_lights[key] = df_temp
                inside_avg = funcs.average_trajectory(di)
                outside_avg_lights = funcs.average_trajectory(do_lights)
                outside_avg_no_lights = funcs.average_trajectory(do_no_lights)


                # plot inside trajectories and the average trajectory
                axs[ix] = pl.plot_insides(di, axs[ix])
                axs[ix].plot(inside_avg[0], inside_avg[1], color = 'orange')
                axs[ix] = pl.plot_outsides(do_lights, axs[ix], color='red', alpha=0.05)
                axs[ix].plot(outside_avg_lights[0], outside_avg_lights[1], color = 'red')
                axs[ix] = pl.plot_outsides(do_no_lights, axs[ix], color='black', alpha=0.05)
                axs[ix].plot(outside_avg_no_lights[0], outside_avg_no_lights[1], color = 'black')

    def act_outside_plots(self):
        """
        plot trials when FB4M was activated outside.  select log files from
        sheet column 'act_outside_plots'.  Top row is trajectory, bottom
        row is aligned inside and outside trajectories
        """
        df = self.sheet
        df_select = df[df.act_outside_plot=='1']
        num_trials = len(df_select)
        fig, axs = plt.subplots(2,num_trials, figsize=(3.5*num_trials, 6))
        pathlengths = []
        for ix,filename in enumerate(df_select.log):
            file_loc = os.path.join(self.logfol, filename)
            log = funcs.read_log(file_loc)
            log = funcs.exclude_lost_tracking(log)
            d, di, do = funcs.inside_outside(log)
            do_lights = {}
            do_no_lights = {}

            for key in list(do.keys()):
                df_temp = do[key]
                if min(df_temp.led1_stpt)<0.1:
                    do_lights[key] = df_temp
                else:
                    do_no_lights[key] = df_temp
            inside_avg = funcs.average_trajectory(di, side='inside')
            outside_avg_lights = funcs.average_trajectory(do_lights, side='outside')
            outside_avg_no_lights = funcs.average_trajectory(do_no_lights, side='outside')
            # plot inside trajectories and the average trajectory
            _,len_lights = funcs.path_length(outside_avg_lights[0], outside_avg_lights[1])
            _,len_no_lights = funcs.path_length(outside_avg_no_lights[0], outside_avg_no_lights[1])
            pathlengths.append([len_no_lights, len_lights])
            axs[1,ix] = pl.plot_insides(di, axs[1,ix])
            axs[1,ix].plot(inside_avg[0], inside_avg[1], color = 'orange')
            axs[1,ix] = pl.plot_outsides(do_lights, axs[1,ix], color='red', alpha=0.05)
            axs[1,ix].plot(outside_avg_lights[0], outside_avg_lights[1], color = 'red')
            axs[1,ix] = pl.plot_outsides(do_no_lights, axs[1,ix], color='black', alpha=0.05)
            axs[1,ix].plot(outside_avg_no_lights[0], outside_avg_no_lights[1], color = 'black')

            axs[0,ix] = pl.plot_trajectory(log, axs[0,ix])
            axs[0,ix] = pl.plot_vertical_edges(log, axs[0,ix])
            axs[0,ix] = pl.plot_trajectory_lights(log, axs[0,ix])




        return np.array(pathlengths)





a = fb4m_activation()
a.make_individual_plots()
pathlengths = a.act_outside_plots()


pathlengths_x = np.array(pathlengths)
pathlengths_x[:,0]=0
pathlengths_x[:,1]=1

fig, axs = plt.subplots(1,1, figsize=(1,3))
for i in np.arange(len(pathlengths)):
    axs.plot(pathlengths_x[i,:], pathlengths[i,:], 'k')
    axs.plot(pathlengths_x[i,0], pathlengths[i,0], 'ko')
    axs.plot(pathlengths_x[i,1], pathlengths[i,1], 'ko')
