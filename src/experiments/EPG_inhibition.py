import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.drive import drive as dr
from src.utilities import funcs
from src.utilities import plotting as pl
matplotlib.rcParams['pdf.fonttype'] = 42
importlib.reload(dr)
importlib.reload(pl)
importlib.reload(funcs)


# SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
# SPREADSHEET_ID = '1-3sKJ-ecv06gX90KENOZM-xryVrt21iOa5JjVrzSu7E'
# DATA_TO_PULL = 'Sheet1'
# data = dr.pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL)
# df = pd.DataFrame(data[1:], columns=data[0])
# df
#
# topFolderId = '1G1K72dwZt3lCZgnex2bWWqtEFnfGjnWY' # Please set the folder of the top folder ID.

class epg_inhibition:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '1-3sKJ-ecv06gX90KENOZM-xryVrt21iOa5JjVrzSu7E'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        df.lights.loc[df.lights == 'TRUE'] = True
        df.lights.loc[df.lights == 'FALSE'] = False
        self.sheet = df


        # load list of log files
        self.log_folder_id = '1G1K72dwZt3lCZgnex2bWWqtEFnfGjnWY'
        self.logfiles = d.get_files_in_folder(self.log_folder_id)

        # ensure that every entry in sheet has a corresponding log file on drive
        self.sheet = self.match_sheet_log()

        # check for log file lights and google sheets lights consistency
        self.match_log_lights_boolean()

        #check for duplicated logs in dataframe
        self.detect_duplicates()

        # define local site for log files.
        self.logfol = "/Volumes/LACIE/logs"

        self.datafol = os.path.join(os.getcwd(), 'data', 'EPG-silencing')
        self.figure_folder = os.path.join(os.getcwd(),'figures', 'epg_silencing')

        # define drive location of pre-processed data
        self.preprocessed = '1vjJ6sGycvyVe0yVEFMO4683IS-8_Gtex'

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

    def match_log_lights_boolean(self):
        """
        for each row in google sheet dataframe, prints entries where log_file
        name doesn't match boolean lights entry.
        """
        df = self.sheet
        log_file_names = []
        for log in self.logfiles:
            log_file_names.append(log['name'])
        for i, log in enumerate(df.log):
            lights = None
            matches_off = ["lights_off", "no_lights", "no_light"]
            if any(x in log for x in matches_off):
                lights = False
            else:
                lights = True

            if lights != df.iloc[i].lights:
                print('LIGHTS MISMATCH:')
                print("lights = ", lights)
                print(df.iloc[i])

    def detect_duplicates(self):
        """
        detect duplicate log files in dataframe
        """
        duplicate = self.sheet[self.sheet.duplicated('log')]
        print(duplicate)

    def preprocess_data(self, overwrite = False):
        """
        possible function: is this useful? create a dictionary with the log file, inside trajectories (di) and
        outside trajectories (do). save to google drive folder
        """
        savename = os.path.join(self.datafol, 'preprocessed.p')

        if not os.path.exists(savename) or overwrite:
            preprocessed = {}
            for log in self.sheet.log:
                print(log)
                temp = {}
                #df = funcs.read_log(os.path.join(self.logfol, log))
                df = funcs.open_log_edit(os.path.join(self.logfol, log))
                df = df.fillna(method='pad')
                d, di, do = funcs.inside_outside(df)
                temp['data'] = df
                temp['di'] = di
                temp['do'] = do
                temp['d'] = d
                preprocessed[log] = temp
            funcs.save_obj(preprocessed, savename)
        else:
            preprocessed = funcs.load_obj(savename)
        return preprocessed

    def make_individual_plots(self):
        """
        make plots for individual animals
        """
        df = self.sheet
        for fly in df.Fly.unique():
            # data frame for this fly
            df_fly = df[df.Fly==fly]
            # how many trials
            num_trials = len(df_fly)
            num_rows = int(np.ceil(num_trials/2))
            # make a subplot to plot each trial
            fig, axs = plt.subplots(num_rows, 2, figsize=(4, 2*num_rows))
            axs = np.array(axs).reshape(-1)
            for ix, i in enumerate(df_fly.index):
                genotype = df_fly.loc[i].Genotype
                filename = df_fly.loc[i].log
                lights = df_fly.loc[i].lights
                trial_type = df_fly.loc[i].experiment
                file_loc = os.path.join(self.logfol, filename)
                log = funcs.open_log_edit(file_loc)
                if trial_type == '0':
                    axs[ix] = pl.plot_vertical_edges(log, axs[ix])
                elif trial_type == '45':
                    axs[ix] = pl.plot_45_edges(log, axs[ix])
                elif trial_type == 'jump':
                    axs[ix] = pl.plot_jumping_edges(log, axs[ix])

                if lights:
                    axs[ix] = pl.plot_trajectory(log, axs[ix], color='green')
                    axs[ix] = pl.plot_trajectory_odor(log, axs[ix])
                else:
                    axs[ix] = pl.plot_trajectory(log, axs[ix])
                    axs[ix] = pl.plot_trajectory_odor(log, axs[ix])

                axs[ix].set_title(genotype+' lights: '+str(lights), fontsize=5)
                axs[ix].tick_params(axis='both', which='major', labelsize=5)
                axs[ix].tick_params(axis='both', which='minor', labelsize=5)
            fig.savefig(os.path.join(self.figure_folder, str(fly)+'.pdf'))

    def returns_per_length(self, all_data, df):
        """
        find the number of returns per lengh and store them in all_data list
        """
        preprocessed = self.preprocess_data()
        # process each fly sequentially
        for fly in df.Fly.unique():
            # data frame for this fly
            df_fly = df[df.Fly==fly]
            for ix, i in enumerate(df_fly.index):
                #print(i)
                #fig, axs = plt.subplots(1,1)
                genotype = df_fly.loc[i].Genotype
                filename = df_fly.loc[i].log
                lights = df_fly.loc[i].lights
                trial_type = df_fly.loc[i].experiment
                file_loc = os.path.join(self.logfol, filename)
                temp = preprocessed[filename]
                log = temp['data']
                di = temp['di']
                do = temp['do']
                d = temp['d']
                num_returns = len(di)-1 #fist inside bout does not count as a return
                pathlength=0
                dist_up_plume = []
                outside_path_length = []
                del_y_return = []
                for key in list(d.keys())[1:]: #exclude first outside bout
                    x = d[key].ft_posx.to_numpy()
                    y = d[key].ft_posy.to_numpy()
                    _,l = funcs.path_length(x,y)
                    pathlength+=l

                for key in list(do.keys())[1:]:
                    temp = do[key]
                    temp = funcs.find_cutoff(temp)
                    x = temp.ft_posx.to_numpy()
                    y = temp.ft_posy.to_numpy()
                    if 19<np.abs(x[0]-x[-1])<21: # successful re-entry
                        _,l = funcs.path_length(x,y)
                        del_y_return.append(np.abs(y[-1]-y[0]))
                        outside_path_length.append(l)
                        #axs.plot(x-x[0], y-y[0])

                key_entry = list(di.keys())[0]
                entry_in = di[key_entry].iloc[0].ft_posy
                key_exit = list(di.keys())[-1]
                exit_in = di[key_exit].iloc[-1].ft_posy
                dist_up_plume = exit_in-entry_in

                #print('Length', pathlength, 'num_returns', num_returns)
                pathlength = pathlength/1000 # convert from mm to m
                outside_path_return = np.nanmean(outside_path_length)/1000
                #fig.suptitle(str(genotype)+'_'+str(lights)+'_'+str(outside_path_return)+'_'+str(num_returns)+'_'+str(len(di)))
                all_data.append({
                'returns_per_l': num_returns/pathlength,
                'num_returns': num_returns,
                'outside_path_return': outside_path_return,
                'del_y_return': np.nanmean(del_y_return),
                'dist_up_plume': dist_up_plume,
                'genotype': genotype,
                'lights': lights,
                'fly': fly
                })
        return all_data

    def num_returns(self, experiment):
        """
        for any plume, calculate the number of returns per total pathlength.
        For multiple trials per fly, average the number of returns per trial
        """
        df = self.sheet

        # dataframes for all control and experimental animals
        df_uas = df[(df['Genotype']=='UAS-GtACR1') & (df['experiment']==experiment)]
        df_ss = df[(df['Genotype']=='ss00096-GtACR1') & (df['experiment']==experiment)]

        # collect all calculated data in one place
        all_data = []
        all_data = self.returns_per_length(all_data, df_uas)
        all_data = self.returns_per_length(all_data, df_ss)
        return all_data

    def plot_returns_per_meter(self, metric = 'returns_per_l', experiment='jump'):
        """
        make a plot for the number of returns per meter
        """
        import scipy.stats as st
        log = self.num_returns(experiment)
        df = pd.DataFrame(log)
        # store average return per m for experimental and control animals
        exp_pairs = []
        control_pairs = []

        fig, axs = plt.subplots()

        # go through each animal and calculate returns/m for all genotypes
        for fly in df.fly.unique():
            df_fly = df[df.fly==fly]
            df_group = df_fly.groupby('lights')
            lights_off = df_group.get_group(False)[metric].mean()
            lights_on = df_group.get_group(True)[metric].mean()
            y=[lights_off, lights_on]
            print(fly, ' ',df_fly.genotype.iloc[0],' ', y)

            if df_fly.genotype.iloc[0]=='ss00096-GtACR1':
                exp_pairs.append(y)
                x=[0, 1]
                color='green'
            elif df_fly.genotype.iloc[0]=='UAS-GtACR1':
                control_pairs.append(y)
                x=[2, 3]
                color='black'

            axs.plot(x,y, color=color, alpha=0.3)
            if metric == 'returns_per_l':
                axs.set_ylabel('plume returns per meter')
            elif metric == 'del_y_return':
                axs.set_ylabel('|entry-exit y pos. (mm)|')

        exp_pairs = np.array(exp_pairs)
        control_pairs = np.array(control_pairs)
        exp_pairs = pd.DataFrame(exp_pairs).dropna().to_numpy()
        control_pairs = pd.DataFrame(control_pairs).dropna().to_numpy()
        #return exp_pairs, control_pairs

        axs.set_xticks([0,1,2,3], ['lights off', 'lights on', 'lights off', 'lights on'])

        # plot error bars
        exp_sem = st.sem(exp_pairs, axis=0)
        exp_mean = np.mean(exp_pairs, axis=0)
        control_sem = st.sem(control_pairs, axis=0)
        control_mean = np.mean(control_pairs, axis = 0)
        axs.errorbar([0,1], exp_mean, exp_sem, color='green', linewidth=2)
        axs.errorbar([2,3], control_mean, control_sem, color='grey', linewidth=2)

        # calculate paired t test
        _,p_exp = st.ttest_rel(exp_pairs[:,0], exp_pairs[:,1])
        _,p_control = st.ttest_rel(control_pairs[:,0], control_pairs[:,1])
        print('experimental p=', p_exp)
        print('control p=', p_control)
        fig.savefig(os.path.join(self.figure_folder, metric +'_paired_plot.pdf'))
        return log

    def interp_outside_trajectories(self, df, avg_x=[], avg_y=[], returns=True, pts=10000):

        """
        for a jumping plume log file, calculates the average trajectory.
        """
        from scipy import interpolate
        d,di,do = funcs.inside_outside(df)
        for key in list(do.keys())[1:]:
            temp = do[key]
            temp = funcs.find_cutoff(temp)
            x = temp.ft_posx.to_numpy()
            y = temp.ft_posy.to_numpy()
            x0 = x[0]
            y0 = y[0]
            x = x-x0
            y = y-y0
            if x0>di[key-1].ft_posx.iloc[-1]:
                x=-x
            t = np.arange(len(x))
            t_common = np.linspace(t[0], t[-1], pts)
            fx = interpolate.interp1d(t, x)
            fy = interpolate.interp1d(t, y)
            if returns:
                if 19<np.abs(x[0]-x[-1])<21:
                    avg_x.append(fx(t_common))
                    avg_y.append(fy(t_common))
            else:
                avg_x.append(fx(t_common))
                avg_y.append(fy(t_common))
        return avg_x, avg_y

    def plot_examples(self):
        preprocessed = self.preprocess_data()
        lights_off_example = '10142021-182446_ss00096GtACR1_Fly2_jump_no_lights_001.log'
        lights_off_log = preprocessed[lights_off_example]['data']

        lights_on_example = '10142021-183755_ss00096GtACR1_Fly2_jump_lights_002.log'
        lights_on_log = preprocessed[lights_on_example]['data']

        x_off = lights_off_log.ft_posx.to_numpy()
        y_off = lights_off_log.ft_posy.to_numpy()
        fig_off,axs_off = plt.subplots(1,1)
        axs_off.plot(x_off, y_off)
        axs_off = pl.plot_trajectory_odor(lights_off_log, axs_off)
        axs_off = pl.plot_jumping_edges(lights_off_log, axs_off)
        axs_off.axis('equal')
        fig_off.savefig(os.path.join(self.figure_folder, 'lights_off_example.pdf'))

        x_on = lights_on_log.ft_posx.to_numpy()
        y_on = lights_on_log.ft_posy.to_numpy()
        fig_on,axs_on = plt.subplots(1,1)
        axs_on.plot(x_on, y_on)
        axs_on = pl.plot_trajectory_odor(lights_on_log, axs_on)
        axs_on = pl.plot_jumping_edges(lights_on_log, axs_on)
        axs_on.axis('equal')
        fig_on.savefig(os.path.join(self.figure_folder, 'lights_on_example.pdf'))

    def plot_animal_averages(self, returns=True):
        preprocessed = self.preprocess_data()
        df = self.sheet
        # dataframes for all control and experimental animals
        df_off = df[(df['Genotype']=='ss00096-GtACR1') & (df['experiment']=='jump') & (df['lights']==False)]
        df_on = df[(df['Genotype']=='ss00096-GtACR1') & (df['experiment']=='jump') & (df['lights']==True)]
        # process each fly sequentially
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        for j, df in enumerate([df_off, df_on]):
            all_animals_x = []
            all_animals_y = []
            for fly in df.Fly.unique():
                # fly average trajectory
                avg_x = []
                avg_y = []
                # data frame for this fly
                df_fly = df[df.Fly==fly]
                for ix, i in enumerate(df_fly.index):
                    #print(i)
                    genotype = df_fly.loc[i].Genotype
                    filename = df_fly.loc[i].log
                    lights = df_fly.loc[i].lights
                    trial_type = df_fly.loc[i].experiment
                    file_loc = os.path.join(a.logfol, filename)
                    log = preprocessed[filename]['data']
                    avg_x, avg_y = self.interp_outside_trajectories(log, avg_x=avg_x, avg_y=avg_y, returns=returns)
                if avg_x:
                    avg_x = np.nanmean(np.array(avg_x), axis=0)
                    avg_y = np.nanmean(np.array(avg_y), axis=0)
                    axs[j].plot(avg_x, avg_y, 'k', alpha = 0.1)
                    all_animals_x.append(avg_x)
                    all_animals_y.append(avg_y)
            axs[j].plot(np.nanmean(np.array(all_animals_x), axis=0), np.nanmean(np.array(all_animals_y), axis=0), 'k')
            axs[j].plot([20,20], [0,100], 'k')
            #fig.savefig(os.path.join(self.figure_folder, 'average_trajectories_returns_'+str(returns)+'.pdf'))
            #axs[j].axis('equal')
        return all_animals_x, all_animals_y

    def plot_animal_traces(self, returns=True):
        preprocessed = self.preprocess_data()
        df = self.sheet
        # dataframes for all control and experimental animals
        df_off = df[(df['Genotype']=='ss00096-GtACR1') & (df['experiment']=='jump') & (df['lights']==False)]
        df_on = df[(df['Genotype']=='ss00096-GtACR1') & (df['experiment']=='jump') & (df['lights']==True)]
        # process each fly sequentially
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        for j, df in enumerate([df_off, df_on]):
            all_animals_x = []
            all_animals_y = []
            for fly in df.Fly.unique():
                # fly average trajectory
                avg_x = []
                avg_y = []
                # data frame for this fly
                df_fly = df[df.Fly==fly]
                for ix, i in enumerate(df_fly.index):
                    #print(i)
                    genotype = df_fly.loc[i].Genotype
                    filename = df_fly.loc[i].log
                    lights = df_fly.loc[i].lights
                    trial_type = df_fly.loc[i].experiment
                    file_loc = os.path.join(a.logfol, filename)
                    log = preprocessed[filename]['data']
                    d,di,do = funcs.inside_outside(log)
                    for key in list(do.keys())[1:]:
                        temp = do[key]
                        temp = funcs.find_cutoff(temp)
                        x = temp.ft_posx.to_numpy()
                        y = temp.ft_posy.to_numpy()
                        x0 = x[0]
                        y0 = y[0]
                        x = x-x0
                        y = y-y0
                        if x0>di[key-1].ft_posx.iloc[-1]:
                            x=-x
                        axs[j].plot(x, y, 'k', alpha=0.1)

            axs[j].plot([20,20], [0,100], 'k')
        #axs[j].axis('equal')
        fig.savefig(os.path.join(self.figure_folder, 'all_traces.pdf'))











#log = '10142021-185250_ss00096GtACR1_Fly2_jump_no_lights_003.log'
%matplotlib
a = epg_inhibition()
a.make_individual_plots()
# %%
df = pd.DataFrame(log)
df
exp_pairs = []
control_pairs = []
metric = 'outside_path_return'
for fly in df.fly.unique():
    df_fly = df[df.fly=='20']
    df_group = df_fly.groupby('lights')
    lights_off = df_group.get_group(False)[metric].mean()
    lights_on = df_group.get_group(True)[metric].mean()
    y=[lights_off, lights_on]


print(df_fly)
print(y)
