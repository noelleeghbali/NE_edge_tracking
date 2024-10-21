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

datadir = os.path.join('/Volumes/LaCie/noelle_imaging/MBON30/240810/F2/T1')
savedir = os.path.join('/Users/noelleeghbali/Desktop/exp/imaging/noelle_imaging/MBON30/picklez')
d = datadir.split(os.path.sep)
name = d[-3] + '_' + d[-2] + '_' + d[-1]
savename = f'{name}.pkl'

# %% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
# %% Masks for ROI drawing
ex.mask_slice = {'All': [1, 2, 3, 4, 5]}
ex.t_projection_mask_slice()
# %%
cx = CX(name, ['mbon30'], datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()
img_dict = {'pv2': pv2, 'ft2': ft2}
pickle_obj(img_dict, savedir, savename)

# %%
fc = fci_regmodel(pv2[['0_mbon30']].to_numpy().flatten(), ft2, pv2)
fc.rebaseline(span=600, plotfig=True)


# %%
y = pv2['0_mbon30']
plt.plot(ft2['instrip'].to_numpy(), color='grey')
plt.plot(y)
fc = fci_regmodel(y, ft2, pv2)
fc.example_trajectory(cmin=0, cmax=0.5)


# %%
rebaseline(savedir, 'mbon30', span=500)

# %%
fc = fci_regmodel(pv2[['0_mbon30']].to_numpy().flatten(), ft2, pv2)
fc.rebaseline(span=500, plotfig=True)
regchoice = ['odour onset', 'odour offset', 'in odour',
             'cos heading pos', 'cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos', 'angular velocity neg', 'x pos', 'x neg', 'y pos', 'y neg', 'ramp down since exit', 'ramp to entry']
fc.run(regchoice)
fc.run_dR2(20, fc.xft)

fc.plot_mean_flur('odour_onset')
fc.plot_example_flur()
plt.figure()
plt.plot(fc.dR2_mean)
plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(fc.coeff_cv[:-1])
plt.plot([0, len(regchoice)], [0, 0], color='k', linestyle='--')
plt.xticks(np.arange(0, len(regchoice)), labels=regchoice, rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()


# %% Summarise all
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

print(rsq)
print(rsq_t_t)

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
