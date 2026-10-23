
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
import sys
import pickle
from sklearn.linear_model import LinearRegression


folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/FB4R/ne/replay'
neuron = 'fsbtn'
for filename in os.listdir(folder):
    if filename.endswith('.pkl'):
        img_dict = open_pickle(f'{folder}/{filename}')
        pv2, ft2 = process_pickle(img_dict, neuron)
        fig, axs, markc = configure_white_plot(size=(15,6), xaxis=True, nrows=2)
    
        FF = pv2[f'0_{neuron}'].to_numpy()
        time = pv2['relative_time'].to_numpy()
        odor2_on = ft2['mfc4_stpt'].to_numpy()
        odor1_on = ft2['mfc3_stpt'].to_numpy()
        loc = ft2['speed'].to_numpy()
        n = min(len(time), len(FF), len(odor1_on), len(odor2_on))
        time = time[:n]
        FF = FF[:n]
        odor1_on = odor1_on[:n]
        odor2_on = odor2_on[:n]
        loc = loc[:n]

        axs[0].plot(time, FF, color=markc, linewidth=.75)
        axs[1].plot(time, loc, color=markc, linewidth=.75)

        for ax in axs:
            if 'mch' in filename:
                shade_on_intervals(ax, time, odor1_on, mode='span', alpha=0.65, color="#fbceb1")
            else:
                shade_on_intervals(ax, time, odor1_on, mode='span', alpha=0.65, color="#fbceb1")
            shade_on_intervals(ax, time, odor2_on, mode='span', alpha=0.65, color="#7ed4e6")
       
        plt.suptitle(filename, size=16)
        axs[0].set_ylabel(r'$\Delta$F/F', size=16)
        axs[1].set_ylabel('speed', size=16)
        plt.xlabel('time (s)', size=16)
        plt.show()

#%%





# %%
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
import sys
import pickle

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/PAM_MBON09/picklez'
neuron = 'mbon09'
color = 'white'

# triggered_zFF_PAM(figure_folder,neuron,color,tbef=2,taf=5,event_type='entry',first=False,zscore_mode='raw')
# triggered_upwind_speed(figure_folder,neuron,color,tbef=2,taf=5,event_type='entry',first=False,norm_mode='none')

# triggered_FF_raw(figure_folder,neuron,color,tbef=0,taf=5,odor_channel='mfc3', event_type='exit',first=False,linewidth=0.8,alpha=0.25,plot_mean=True,mean_linewidth=2.0)
# triggered_zFF_exit_minOFF_all(figure_folder,neuron,color,taf=5,min_off_bout_dur_s=5.0,which_odor="mfc3",zscore_mode="trial",first=False)


# %%
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
import sys
import pickle
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/FB4R/ne/replay/oct'
lobes = ['fsbtn']
neuron = ['fsbtn']
colors = ['white']

triggered_zFF_new(figure_folder, lobes, tbef=2, taf=5, event_type="entry", first=False,colors=colors, odor_source="auto", max_events=None, zscore_mode="baseline")
# avg_trace(figure_folder, lobes, event_type='entry', odor_source='auto', n_pts=200, min_bout_s=1.0, max_bouts=None, plot_per_file=True, mode='baseline_z', baseline_frac=0.1, eps=1e-12)
# trace_FF(figure_folder, neuron, window=(300,740), shade_mode='span')

# triggered_zFF_PAM(figure_folder,neuron,color='white',tbef=5,taf=10,event_type='entry',first=False,zscore_mode='raw')
# triggered_upwind_speed(figure_folder,neuron,color='white',tbef=2,taf=5,event_type='entry',first=False,norm_mode='none')
# %%
