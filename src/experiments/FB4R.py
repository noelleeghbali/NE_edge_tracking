from src.utilities import imaging as im
from src.utilities import plotting as pl
import pandas as pd
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
import seaborn as sns
from src.drive import drive as dr
from src.utilities import funcs as fn
from src.utilities import plotting as pl
import scipy.signal as sg
from scipy import stats

importlib.reload(dr)
importlib.reload(pl)
importlib.reload(fn)
importlib.reload(im)

class fb4r:
    def __init__(self):
        d = dr.drive_hookup()

        # load google sheet with experiment information
        self.sheet_id = '12fcgLhDgC0G05UHSWR4oMhjFRAWIR1fDx6u280hnEao'
        df = d.pull_sheet_data(self.sheet_id, 'Sheet1')
        self.sheet = df
        df.analyze.loc[df.analyze == 'TRUE'] = True
        df.analyze.loc[df.analyze == 'FALSE'] = False

        # define local folder for data
        self.datafol = "/Volumes/LACIE/Andy/PAM_Chr_47H09_GCaMP"


    def register(self):
        """
        register all images
        """
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['analyze']:
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                ex = im.fly(name, folderloc)
                ex.register_all_images(overwrite=False)

    def processing(self):
        """

        """
        for i, row in self.sheet.iterrows():
            d = row.to_dict()
            if d['analyze']:

                # do pre and post processing
                name = d['folder']
                folderloc = os.path.join(self.datafol, name)
                kwargs = self.sheet.iloc[i].to_dict()
                ex = im.fly(name, folderloc, **kwargs)

                ex.save_preprocessing()
                ex.save_postprocessing()

class fb4r_example():
    def __init__(self, dict, datafol):

        for key in list(dict.keys()):
            setattr(self,key, dict[key])

        # specify where the data is stored
        self.datafol = os.path.join(datafol, self.folder)

        self.ex = im.fly(self.folder, self.datafol)

    def plot_trace(self):
        pv2, ft, ft2, ix = self.ex.load_postprocessing()
        time = pv2.seconds-pv2.seconds.iloc[0]
        ft_speed = fn.calculate_speeds(ft2)

        # replace values where shutter is closed with nans
        trace = pv2.fb4r_dff.to_numpy()
        trace_diff = np.gradient(trace)
        trace_diff_z = np.abs(stats.zscore(trace_diff))
        shutter = np.where(trace_diff_z>3)
        trace[shutter] = np.nan
        trace[trace<-0.2] = np.nan


        # interpolate values
        trace_fill = pd.Series(trace).interpolate(method='polynomial',order=3).to_numpy()

        fig, axs = plt.subplots(4,1, sharex=True)
        #axs[0].plot(trace_fill)
        axs[0].plot(time, trace_fill)
        axs[1].plot(time, ft2.mfc2_stpt, color='black')
        axs[2].plot(time, -ft2.led1_stpt, color='red')
        axs[3].plot(time, ft_speed.speed, color='black')
        return fig

    def plot_trajectory(self):
        pv2, ft, ft2, ix = self.ex.load_postprocessing()
        time = pv2.seconds-pv2.seconds.iloc[0]
        fig, axs = plt.subplots(1,1)
        axs.plot(ft.ft_posx, ft.ft_posy)
        return fig

    def plot_triggered_average(self):
        pv2, ft, ft2, ix = self.ex.load_postprocessing()
        time = pv2.seconds-pv2.seconds.iloc[0]
        del_t = np.mean(np.gradient(ft2.seconds))

        # replace values where shutter is closed with nans
        trace = pv2.fb4r_dff.to_numpy()
        trace_diff = np.gradient(trace)
        trace_diff_z = np.abs(stats.zscore(trace_diff))
        shutter = np.where(trace_diff_z>3)
        trace[shutter] = np.nan
        trace[trace<-0.2] = np.nan

        # interpolate values
        trace_fill = pd.Series(trace).interpolate(method='polynomial',order=3).to_numpy()

        # plot triggered average, background subtract each trace
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,5))
        buf = 3
        buf_pts = int(np.ceil(buf/del_t))

        shutter_t = 5
        n_pts = int(np.ceil(shutter_t/del_t))

        # time vector
        t_pts = 2*buf_pts+n_pts
        t_vec = np.linspace(-buf, buf+shutter_t, t_pts)

        ft2.loc[ft2.mfc2_stpt<0.05, 'mfc2_stpt']=0
        ft2.loc[ft2.mfc2_stpt>0.049, 'mfc2_stpt']=1
        ft2.loc[ft2.led1_stpt>0.9, 'led1_stpt']=1
        ft2.loc[ft2.led1_stpt<0.9, 'led1_stpt']=0
        peaks_led, properties = sg.find_peaks(np.gradient(-ft2.led1_stpt))
        peaks_odor, properties = sg.find_peaks(np.gradient(ft2.mfc2_stpt))
        if min(peaks_led-peaks_odor)==-1:
            peaks_odor = peaks_odor-1
        elif min(peaks_led-peaks_odor)==1:
            peaks_odor = peaks_odor+1
        print(peaks_led)
        print(peaks_odor)

        peaks_both = np.intersect1d(peaks_led, peaks_odor)
        peaks_led = np.setdiff1d(peaks_led, peaks_both)
        peaks_odor = np.setdiff1d(peaks_odor, peaks_both)


        all_peaks = [peaks_odor, peaks_led, peaks_both]
        for i, peaks in enumerate(all_peaks):
            all_t = []
            for j, peak in enumerate(peaks):
                baseline = np.mean(trace[peak-buf_pts:peak])
                ti = trace_fill[peak-buf_pts:peak+buf_pts+n_pts]-baseline
                all_t.append(ti)
                if j==0:
                    axs[i].plot(t_vec, ti, 'red', alpha = 0.3)
                else:
                    axs[i].plot(t_vec, ti, 'k', alpha = 0.3)

            # plot average
            all_t = np.array(all_t)
            axs[i].plot(t_vec, np.nanmean(all_t, axis=0), linewidth=2, color='k')

        return fig

    def plot_triggered_average_speed(self):
        pv2, ft, ft2, ix = self.ex.load_postprocessing()
        time = pv2.seconds-pv2.seconds.iloc[0]
        del_t = np.mean(np.gradient(ft2.seconds))

        ft_speed = fn.calculate_speeds(ft2)
        speed = ft_speed.speed

        # plot triggered average, background subtract each trace
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,5))
        buf = 3
        buf_pts = int(np.ceil(buf/del_t))

        shutter_t = 5
        n_pts = int(np.ceil(shutter_t/del_t))

        # time vector
        t_pts = 2*buf_pts+n_pts
        t_vec = np.linspace(-buf, buf+shutter_t, t_pts)

        ft2.loc[ft2.mfc2_stpt<0.05, 'mfc2_stpt']=0
        ft2.loc[ft2.mfc2_stpt>0.049, 'mfc2_stpt']=1
        ft2.loc[ft2.led1_stpt>0.9, 'led1_stpt']=1
        ft2.loc[ft2.led1_stpt<0.9, 'led1_stpt']=0
        peaks_led, properties = sg.find_peaks(np.gradient(-ft2.led1_stpt))
        peaks_odor, properties = sg.find_peaks(np.gradient(ft2.mfc2_stpt))
        if min(peaks_led-peaks_odor)==-1:
            peaks_odor = peaks_odor-1
        elif min(peaks_led-peaks_odor)==1:
            peaks_odor = peaks_odor+1
        print(peaks_led)
        print(peaks_odor)

        peaks_both = np.intersect1d(peaks_led, peaks_odor)
        peaks_led = np.setdiff1d(peaks_led, peaks_both)
        peaks_odor = np.setdiff1d(peaks_odor, peaks_both)


        all_peaks = [peaks_odor, peaks_led, peaks_both]
        for i, peaks in enumerate(all_peaks):
            all_t = []
            for j, peak in enumerate(peaks):
                baseline = np.mean(speed[peak-buf_pts:peak])
                ti = speed[peak-buf_pts:peak+buf_pts+n_pts]-baseline
                all_t.append(ti)
                if j==0:
                    axs[i].plot(t_vec, ti, 'k', alpha = 0.3)
                else:
                    axs[i].plot(t_vec, ti, 'k', alpha = 0.3)

            # plot average
            all_t = np.array(all_t)
            axs[i].plot(t_vec, np.nanmean(all_t, axis=0), linewidth=2, color='k')

        return fig


%matplotlib
fbs = fb4r()
animal = 2
d = fbs.sheet.iloc[animal]
datafol = fbs.datafol

example = fb4r_example(d, datafol)
example.plot_trace()
fig = example.plot_trajectory()
# fig = example.plot_trace()
fig.savefig('/Users/andrewsiliciano/Documents/GitHub/edge-tracking/notebooks/figures/trajectory_'+str(animal)+'.pdf')


# np.intersect1d(peaks_led, peaks_odor)
# np.setdiff1d(peaks_led,np.intersect1d(peaks_led, peaks_odor))
