
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

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez/acv_pulses'
neuron = 'mbon21'
#rebaseline(figure_folder, neuron, span=500)


#Fluorescence trace over experiment
#trace_FF_bw(figure_folder, neuron, window=None)
# trace_FF_bw_bouts(figure_folder, neuron, pre_post_time=2, separation=2, thresh=30)
# dF_entries_bw(figure_folder, neuron, thresh=30, sign='pos')
# dF_entries_time_bw(figure_folder, neuron, thresh=30, sign='pos')
# entry_auc_bw(figure_folder, neuron, unit_time=True, thresh=40)
# interp_jump_FF(figure_folder, neuron)
stacked_FF(figure_folder, neuron)

# Triggered average fluorescence at entry and exit
# triggered_FF(figure_folder, neuron, tbef=10, taf=10, event_type='entry', first=True)
# triggered_FF(figure_folder, neuron, tbef=10, taf=10, event_type='entry', first=False)
# triggered_zFF(figure_folder, neuron, tbef=10, taf=10, event_type='entry', first=True)
# triggered_zFF(figure_folder, neuron, tbef=10, taf=10, event_type='entry', first=False)
# triggered_zFF(figure_folder, neuron, tbef=10, taf=10, event_type='exit', first=True)
# triggered_zFF(figure_folder, neuron, tbef=10, taf=10, event_type='exit', first=False)



# Trajectory w/ fluorescence colormap
# traj_FF(figure_folder, neuron)
# FF_tuning(figure_folder, neuron, 'traveling direction')
# FF_time_corr(figure_folder, neuron)
#FF_xpos_corr(figure_folder, neuron)
#inbound_outbound_FF(figure_folder, neuron, window=1)
#FF_peaks(figure_folder, neuron)

# savedir = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez/jumping/rb'
# datadirs = os.listdir(savedir)
# neuron = 'mbon21'
# regchoice = ['odour onset', 'odour offset', 'in odour', 'cos heading pos', 'cos heading neg', 'sin heading pos', 'sin heading neg',
#              'angular velocity', 'translational velocity', 'ramp down since exit', 'ramp to entry']
# d_R2s = np.zeros((len(datadirs), len(regchoice)))
# coeffs = np.zeros((len(datadirs), len(regchoice)))
# rsq = np.zeros(len(datadirs))
# rsq_t_t = np.zeros((len(datadirs), 2))

# for i, filename in enumerate(datadirs):
#     if filename.endswith('.pkl'):
#         savename, extension = os.path.splitext(filename)
#         img_dict = open_pickle(os.path.join(savedir, filename))
#         pv2, ft2 = process_pickle(img_dict, neuron)
#         fc = fci_regmodel(pv2[[f'0_{neuron}']].to_numpy().flatten(), ft2, pv2)
#         fc.run(regchoice, partition='pre_air')
#         fc.run_dR2(20, fc.xft)
#         d_R2s[i, :] = fc.dR2_mean
#         coeffs[i, :] = fc.coeff_cv[:-1]
#         rsq[i] = fc.r2
#         rsq_t_t[i, 0] = fc.r2_part_train
#         rsq_t_t[i, 1] = fc.r2_part_test

# # Summary Plot: delta R2
# plt.figure()
# plt.plot(d_R2s.T, color='k', alpha=0.3)  # plot all individual delta R2 curves
# plt.plot(np.mean(d_R2s, axis=0), color='r', linewidth=2)  # plot the mean
# plt.plot([0, len(regchoice) - 1], [0, 0], color='k', linestyle='--')
# plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
# plt.subplots_adjust(bottom=0.4)
# plt.ylabel('delta R2')
# plt.xlabel('Regressor name')
# plt.title('Delta R2 across all files')
# plt.savefig(os.path.join(savedir, 'summary_dR2.png'))
# plt.show()

# # Summary Plot: Coefficients
# plt.figure()
# plt.plot(coeffs.T, color='k', alpha=0.3)  # plot all individual coefficient curves
# plt.plot(np.mean(coeffs, axis=0), color='r', linewidth=2)  # plot the mean
# plt.plot([0, len(regchoice) - 1], [0, 0], color='k', linestyle='--')
# plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
# plt.subplots_adjust(bottom=0.4)
# plt.ylabel('Coefficient weight')
# plt.xlabel('Regressor name')
# plt.title('Coefficients across all files')
# plt.savefig(os.path.join(savedir, 'summary_Coeffs.png'))
# plt.show()

# # Summary Plot: R2 train vs test
# plt.figure()
# plt.scatter(rsq_t_t[:, 0], rsq_t_t[:, 1], color='k')
# plt.plot([np.min(rsq_t_t[:]), np.max(rsq_t_t[:])], [np.min(rsq_t_t[:]), np.max(rsq_t_t[:])], color='k', linestyle='--')
# plt.xlabel('R2 pre air')
# plt.ylabel('R2 live air')
# plt.title('Model trained on pre air period')
# plt.savefig(os.path.join(savedir, 'summary_R2_train_vs_test.png'))
# plt.show()