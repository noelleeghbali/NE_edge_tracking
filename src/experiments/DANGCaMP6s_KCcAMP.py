trials = {}
trials['20190129_58E02GCaMP6s_16A06mpcADDisRU_Fly1-005'] = {
'fileloc': '/Volumes/LACIE/Andy/DAN-cAMP_imaging_resurrected/20190129_58E02GCaMP6s_16A06mpcADDisRU_Fly1-005',
'cell_type': 'KC_DAN',
'trial_type': 'closed-loop',
'linked':None,
'z_slices':[1],
'crop':False,
'dual_color_old':True
}

import skimage
from skimage import io, data, registration, filters, measure
from scipy import stats, signal
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utilities import imaging as im
from src.utilities import funcs as fn
from src.utilities import plotting as pl

import importlib
import glob
import pandas as pd
import seaborn as sns
import importlib
importlib.reload(im)
importlib.reload(fn)
importlib.reload(pl)


# %%
# register the data
%matplotlib
importlib.reload(im)
from pystackreg import StackReg
for trial in trials.keys():
    # pull trial from trials dict
    name = trial
    folder = trials[name]['fileloc']
    trial_type = trials[name]['trial_type']
    slices = trials[name]['z_slices']
    kwargs = trials[name]

    # create imaging object
    ex = im.fly(name, folder, **kwargs)

    # register images
    #ex.register_all_images(overwrite=True)

# %%
%matplotlib
figurefol = os.path.join(ex.fileloc, 'figures')
dir(ex)
# load the data
df = pd.read_csv(os.path.join(ex.regfol, 'rois.csv'))
df = df.apply(fn.lnorm).add_suffix('_dff')
t = ex.read_image_xml().absolute_time.to_numpy()
t = t-t[0]
df['time']=t
t1 = 60
t2 = 110

df['g4camp_sub_dff'] = fn.savgolay_smooth(df['g4camp_sub_dff'])
df['g5camp_sub_dff'] = fn.savgolay_smooth(df['g5camp_sub_dff'])

df4 = df[['g4dan_dff','g4camp_sub_dff']]
df4.corr()
df5 = df[['g5dan_dff','g5camp_dff']]
df5.corr()
# remove first 100 slices
# df = df.iloc[100:]


# plot camp vs dan for G4 and G5 compartments.  Using the photobleaching-corrected movie here (bleach correction, exponential)
fig, axs = plt.subplots(2,1, figsize = (3,6))
axs[0].plot(df.g4dan_dff, df.g4camp_sub_dff, '.', alpha=0.2)
axs[0] = pl.plot_linear_best_fit(axs[0], df.g4dan_dff.to_numpy(), df.g4camp_sub_dff.to_numpy())
slope, intercept, r_value, p_value, std_err = stats.linregress(df.g4dan_dff.to_numpy(), df.g4camp_sub_dff.to_numpy())
print('p=',p_value,' r-squared=', r_value**2)
axs[1].plot(df.g5dan_dff, df.g5camp_sub_dff, '.', alpha=0.2)
axs[1] = pl.plot_linear_best_fit(axs[1], df.g5dan_dff.to_numpy(), df.g5camp_sub_dff.to_numpy())
axs[0].axis('equal')
axs[1].axis('equal')
for ax in axs:
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
fig.savefig(os.path.join(figurefol, 'scatter.pdf'))

# plot the traces
fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(df.time, df.g4camp_sub_dff)
axs[0].plot(df.time, df.g4dan_dff)
axs[1].plot(df.time, df.g5camp_sub_dff)
axs[1].plot(df.time, df.g5dan_dff)
fig.savefig(os.path.join(figurefol, 'traces.pdf'))
# axs[1].set_xlim(t1,t2)



corr = signal.correlate(df.g4camp_sub_dff, df.g4dan_dff)
lags = signal.correlation_lags(len(df.g4dan_dff), len(df.g4camp_sub_dff))
corr /= np.max(corr)

fig, axs = plt.subplots(1,1)
axs.plot(lags, corr)
