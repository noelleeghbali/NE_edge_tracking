# %%
from imaging_analysis import *
from behavior_analysis import *
from analysis_funs.regression import *
from analysis_funs.optogenetics import opto
from analysis_funs.CX_imaging import CX
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
import statsmodels.api as sm
import sys
import pickle

datadir = os.path.join('/Volumes/LaCie/noelle_imaging/MBON21/241010/F1/T3')
savedir = os.path.join('/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez')
d = datadir.split(os.path.sep)
name = d[-3] + '_' + d[-2] + '_' + d[-1]
savename = f'{name}.pkl'

# %% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
# %% Masks for ROI drawing
ex.mask_slice = {'All': [1, 2, 3, 4]}
ex.t_projection_mask_slice()
# %%
cx = CX(name, ['mbon21'], datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()
zeros = pv2[pv2['0_mbon21']==0].index.min()
if pd.notna(zeros):
    pv2 = pv2.iloc[:zeros]
    ft2 = ft2.iloc[:zeros]
img_dict = {'pv2': pv2, 'ft2': ft2}
pickle_obj(img_dict, savedir, savename)
# %%
fc = fci_regmodel(pv2[['0_mbon21']].to_numpy().flatten(), ft2, pv2)
fc.rebaseline(span=500, plotfig=True)

# %%
rebaseline(savedir, 'mbon21', span=500)

# %%
y = fc.ca
#plt.plot(y)
#plt.plot(ft2['instrip'], color='k')

#fc = fci_regmodel(y, ft2, pv2)
#fc.example_trajectory(cmin=0, cmax=0.5)
#fc.mean_traj_nF_jump(y, plotjumps=True)

# %%
# fc = fci_regmodel(pv2[['0_mbon21']].to_numpy().flatten(), ft2, pv2)
# #fc.rebaseline(span=500, plotfig=True)
# regchoice = ['odour onset', 'odour offset', 'in odour',
#              'cos heading pos', 'cos heading neg', 'sin heading pos', 'sin heading neg',
#                                 'angular velocity pos', 'angular velocity neg', 'translational velocity', 'ramp down since exit', 'ramp to entry']
# fc.run(regchoice)
# fc.run_dR2(20, fc.xft)
# plt.figure()
# plt.plot(fc.dR2_mean)
# plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
# plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
# plt.subplots_adjust(bottom=0.4)
# plt.ylabel('delta R2')
# plt.xlabel('Regressor name')
# plt.show()
# plt.figure()
# plt.plot(fc.coeff_cv[:-1])
# plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
# plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
# plt.subplots_adjust(bottom=0.4)
# plt.ylabel('Coefficient weight')
# plt.xlabel('Regressor name')
# plt.show()



# # %% Summarise all
# # filename = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/240725_Fly1_T1.pkl'
# # figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21'
# # img_dict = open_pickle(filename)

# # pv2 = img_dict['pv2']
# # ft2 = img_dict['ft2']
# # ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
# # ft2['mbon21'] = pv2['0_mbon21']
# # ft2['relative_time'] = pv2['relative_time']
# # print(ft2)

# # FF = ft2['mbon21']
# # time = ft2['relative_time']

# # fig, axs = plt.subplots(1, 1, figsize=(15, 5))
# # axs.plot(time, FF, color='black', linewidth=1)
# # #axs.plot(time, ft2['net_motion'], color='blue', linewidth=1)

# # d, di, do = inside_outside(ft2)
# # for key, df in di.items():
# #     time_on = df['relative_time'].iloc[0]
# #     time_off = df['relative_time'].iloc[-1]
# #     timestamp = time_off - time_on
# #     rectangle = patches.Rectangle((time_on, FF.min()), timestamp, FF.max() + 0.5, facecolor='#ff7f24', alpha=0.3)
# #     axs.add_patch(rectangle)

# # # # Set title with fly number
# # #title = f'fly {fly_number}'
# # #plt.suptitle(title)
# # plt.xlim(1600, 1700)
# # plt.xlabel('Time')
# # plt.show()

# # savename='odor_FF.pdf'
# # #fig.savefig(os.path.join(figure_folder, savename))



# %%
filename = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON09/picklez/jumping/241007_F3_T3_rb.pkl'
img_dict = open_pickle(filename)
pv2 = img_dict['pv2']
ft2 = img_dict['ft2']
ft2['instrip'] = ft2['instrip'].replace({1: True, 0: False})
ft2['FF'] = pv2['0_mbon09']
ft2['relative_time'] = pv2['relative_time']
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
first_on_index = ft2[ft2['instrip']].index[0]
exp_df = ft2.loc[first_on_index:] # This filters the dataframe
xo = exp_df.iloc[0]['ft_posx']
yo = exp_df.iloc[0]['ft_posy']
# Assign FF (fluorescence) to the correct df column
FF = exp_df['FF']
smoothed_FF = FF.rolling(window=10, min_periods=1).mean()
cmap = plt.get_cmap('coolwarm')

# Normalize FF to [0, 1] for colormap
min_FF = smoothed_FF.min()
max_FF = smoothed_FF.max()
range_FF = max_FF - min_FF
norm = colors_mod.Normalize(vmin=min_FF - 0.1 * range_FF, vmax=max_FF + 0.1 * range_FF)

# Plot the trajectory on the corresponding subplot
axs.scatter(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, c=smoothed_FF, cmap=cmap, norm=norm, s=1)
axs.add_patch(patches.Rectangle((-10 / 2, 0), 10, 1000, facecolor='lightgrey', edgecolor='lightgrey', alpha=0.3))
# Set axes, labels, and title
axs.set_xlim(-250,250)
axs.set_ylim(0, 500)
axs.set_xlabel('x position', fontsize=14)
axs.set_ylabel('y position', fontsize=14)
#axs.set_title(f'{title} {lobe} lobe', fontsize=14)

# Further customization
axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
for pos in ['right', 'top']:
    axs.spines[pos].set_visible(False)

for _, spine in axs.spines.items():
    spine.set_linewidth(2)
for spine in axs.spines.values():
    spine.set_edgecolor('black')

# Apply tight layout to the entire figure
fig.tight_layout()
# %%
savedir = '/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez/rb'
figure_folder = ('/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON21/picklez/rb')
neuron = 'mbon21'
regchoice = ['odour onset', 'odour offset', 'in odour',
             'cos heading pos', 'cos heading neg', 'sin heading pos', 'sin heading neg',
             'angular velocity pos', 'angular velocity neg', 'translational velocity', 'ramp down since exit', 'ramp to entry']
d_R2s = np.zeros((len(datadirs), len(regchoice)))
coeffs = np.zeros((len(datadirs), len(regchoice)))
rsq = np.zeros(len(datadirs))
rsq_t_t = np.zeros((len(datadirs), 2))

for filename in os.listdir(figure_folder):
    if filename.endswith('.pkl'):
        print(i)
        savename, extension = os.path.splitext(filename)
        img_dict = open_pickle(f'{figure_folder}/{filename}')
        pv2, ft2 = process_pickle(img_dict, neuron)
        fc = fci_regmodel(pv2[[f'0_{neuron}']].to_numpy().flatten(), ft2, pv2)
        #fc.rebaseline(span=400, plotfig=True)
        fc.run(regchoice, partition='pre_air')
        fc.run_dR2(20, fc.xft)
        d_R2s[i, :] = fc.dR2_mean
        coeffs[i, :] = fc.coeff_cv[:-1]
        rsq[i] = fc.r2
        rsq_t_t[i, 0] = fc.r2_part_train
        rsq_t_t[i, 1] = fc.r2_part_test


fc.plot_mean_flur('odour_onset')
fc.plot_example_flur()
plt.figure()
plt.plot(d_R2s.T, color='k')
plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()
plt.savefig(os.path.join(savedir, 'dR2.png'))


plt.figure()
plt.plot(coeffs.T, color='k')
plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()
plt.savefig(os.path.join(savedir, 'Coeffs.png'))

plt.figure()
plt.scatter(rsq_t_t[:, 0], rsq_t_t[:, 1], color='k')
plt.plot([np.min(rsq_t_t[:]), np.max(rsq_t_t[:])], [
         np.min(rsq_t_t[:]), np.max(rsq_t_t[:])], color='k', linestyle='--')
plt.xlabel('R2 pre air')
plt.ylabel('R2 live air')
plt.title('Model trained on pre air period')


# %% 
