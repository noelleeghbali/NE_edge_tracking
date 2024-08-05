# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:02:41 2024

@author: dowel
"""

#%% 

# Notebook acts as a processing pipeline for PFL3 neurons. 
# Protocerebral bridge only
#%%
from analysis_funs.regression import fci_regmodel
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX

#%% Imaging test for PFL3 neurons


datadir =os.path.join("Y:\Data\FCI\Hedwig\\SS60291_PFL3_60D05_sytGC7f\\240314\\f1\\Trial2")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#%% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%% Masks for ROI drawing
ex.mask_slice = {'All': [1,2]}
ex.t_projection_mask_slice()
#%% Use MatLab Gui to draw ROIs

#%% Extract fluorescence and save
# Set up CX object
cx = CX(name,['LAL'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()
#%%
phase,phase_offset,amp = cx.unyoked_phase('fsb')
pdat = cx.phase_yoke('eb',['fsb'])
#%%
pon = ft2['instrip'].to_numpy(dtype=int)
x =180* pdat['offset_eb_phase']/np.pi
y = 180*pdat['offset_fsb_phase']/np.pi
plt.scatter(x[pon==0],y[pon==0],color='k',s=2)
plt.scatter(x[pon==1],y[pon==1],color='r',s=5)
plt.xlabel('Elipsoid body phase')
plt.ylabel('FSB phase')
#%%

pon = ft2['instrip'].to_numpy(dtype=int)
y =180* phase/np.pi
x = 180*ft2['ft_heading']/np.pi
plt.scatter(x[pon==0],y[pon==0],color='k',s=amp[pon==0]*100)
plt.scatter(x[pon==1],y[pon==1],color='r',s=amp[pon==1]*100)
plt.xlabel('Heading')
plt.ylabel('FSB phase')
# %% EB to FSB transformation 
# Look in hdelta C to find it
from analysis_funs.regression import fci_regmodel
# %% bump fitting


#%% Plot data
plt.close('all')
ebs = []
for i in range(16):
    ebs.append(str(i) +'_eb')

plt.figure()
eb = pv2[ebs]
plt.imshow(eb, interpolation='None',aspect='auto')
plt.show()

#%%
fbs = []
for i in range(16):
    fbs.append(str(i) + '_fsb')

fb = pv2[fbs].to_numpy()
plt.figure()
plt.imshow(fb, interpolation='None',aspect='auto')

t = np.arange(0,len(fb))
new_phase = np.interp(phase, (phase.min(), phase.max()), (-0.5, 15.5))
plt.plot(new_phase,t,color='r',linewidth=0.5)
plt.show()

y  = np.max(fb,axis=1)

fc = fci_regmodel(pv2['0_lal'].to_numpy(),ft2,pv2)
fc.example_trajectory(cmin=-0.25,cmax=0.25)
fc.plot_mean_flur('odour_onset')
fc.run()
#%%
fc = fci_regmodel(pv2['0_lal'].to_numpy(),ft2,pv2)
fc.rebaseline()
yL = fc.ca
fc = fci_regmodel(pv2['1_lal'].to_numpy(),ft2,pv2)
fc.rebaseline()
yR = fc.ca

plt.plot(yL,color='b')
plt.plot(yR,color='r')
plt.plot(yR-yL,color='m')
plt.plot(ft2['instrip'],color='k')
plt.show()

fc = fci_regmodel(yR,ft2,pv2)
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry']



fc.run(regchoice)
fc.run_dR2(20,fc.xft)
plt.figure()
plt.plot(fc.dR2_mean)
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(fc.coeff_cv[:-1])
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()

# Old #################################
#%% Get PB glomerulous 
pb = im.PB2(name,datadir,[1,2,3])
glms = pb.get_gloms()
# get phase
#%% Get FSB
fb = im.FSB2(ex.name, datadir, [1,2,3,4,5])
rois16 = fb.get_wedges()
#%% Get EB

#%% pb alignment checking
plt.close('all')
l_pb = glms[:,:8]
r_pb = glms[:,8:]
glm_order = np.array([0,1,2,3,4,5,6,7],dtype='int')
offsets = []
for i in range(len(glm_order)):
    dx = np.append(glm_order[-i:], glm_order[:-i])
    m = l_pb*r_pb[:,dx]
    mx = np.nansum(m[:])
    print(i,'Max: ', mx)
    offsets.append(mx)
offset = np.argmax(offsets)
print('Pb offset: ',offset)
if offset!=1:
    print('May want to check mask')
# offset should be 1 if not 

#%%
num = []
for i in range(3, 15, 2):
	num.append(i)
    
#%% Save pre processing and post-processing data
# 
ex.crop = False
ex.save_preprocessing()
# Work on below - there may be an issue with how the signal is received. 
# Looks like it may not have been recorded for the entire exp
ex.save_postprocessing()
#%% Load data - all sampled onto a 10Hz timebase
pv2,ft,ft2,ix = ex.load_postprocessing()

#%% Fit bump
pb = im.PB2(name,datadir,[1,2,3])

# 1. interpolate over missing gloms

glms = pb.get_gloms()
#%%
l_pb = pv2[['0_pb',
'1_pb',
'2_pb',
'3_pb',
'4_pb',
'5_pb',
'6_pb',
'7_pb']].values
r_pb = pv2[['8_pb',
'9_pb',
'10_pb',
'11_pb',
'12_pb',
'13_pb',
'14_pb',
'15_pb']].values
l_interp = np.zeros_like(l_pb)

r_interp = np.zeros_like(r_pb)
rows = np.shape(l_pb)[0]
x = np.array([0,1,2,3,4,5,6,7,8,9],dtype='int')
for i in range(rows):
    t_r = l_pb[i,:]
    nchk = np.isnan(t_r)
    if np.sum(nchk)==len(nchk):
        continue
    t_r = np.append(t_r,t_r[0])
    t_r = np.append(t_r[-2],t_r)
    nn = np.isnan(t_r)
    t_r[nn] = np.interp(x[nn],x[~nn],t_r[~nn])
    l_interp[i,:] = t_r[1:-1]
    
    t_r = r_pb[i,:]
    t_r = np.append(t_r,t_r[0])
    t_r = np.append(t_r[-2],t_r)
    
    nn = np.isnan(t_r)
    t_r[nn] = np.interp(x[nn],x[~nn],t_r[~nn])
    r_interp[i,:] = t_r[1:-1]
    
glm_int = np.zeros([np.shape(r_interp)[0], np.shape(r_interp)[1]*2])
glm_int[:,:8] = l_interp
glm_int[:,8:] = r_interp
plt.imshow(glm_int,aspect='auto',interpolation='none')
# %% lets do a rough plot
# Heading versus left pb glom and right pb glom peak
lmx = np.argmax(glm_int[:,:8],axis=1)
lmx = sg.savgol_filter(lmx,20,2)
rmx = np.argmax(glm_int[:,[14, 15,8,9,10,11,12,13]],axis=1)
rmx = sg.savgol_filter(rmx,20,2)
t = pv2['relative_time']
plt.plot(t,ft2['ft_heading'])
lmn = (lmx+rmx)/2
plt.plot(t,-lmn)
#plt.plot(t,-rmx)
plt.plot(t,i_st,color='k')
# %% Do rough plot of argdiff
l_val = np.max(glm_int[:,:8],axis=1)
r_val = np.max(glm_int[:,8:],axis=1)
i_st = ft2['instrip']
plt.plot(t,ft2['ft_heading'])

plt.plot(t,l_val-r_val)
#plt.plot(t,r_val,color='b')
#plt.plot(t,l_val,color='r')
plt.plot(t,i_st,color='k')
#%% Check log vs dat files to see what is needed
dat_path = "Y:\Data\FCI\Test_dat_log\\240207\\f2\\Trial2\\fictrac-20240207_143243.dat"
log_path = "Y:\Data\FCI\Test_dat_log\\240207\\f2\\Trial2\\02072024-143249.log"
names = [
      'frame',
      'del_rot_cam_x',
      'del_rot_cam_y',
      'del_rot_cam_z',
      'del_rot_error',
      'df_pitch',
      'df_roll',
      'df_yaw',
      'abs_rot_cam_x',
      'abs_rot_cam_y',
      'abs_rot_cam_z',
      'abs_rot_lab_x',
      'abs_rot_lab_y',
      'abs_rot_lab_z',
      'ft_posx',
      'ft_posy',
      'ft_heading',
      'ft_movement_dir',
      'ft_speed',
      'forward_motion',
      'side_motion',
      'timestamp',
      'sequence_counter',
      'delta_timestep',
      'alt_timestep'
]
df1 = pd.read_table(dat_path, delimiter='[,]', names = names, engine='python')
df1.ft_posx = -3*df1.ft_posx # flip x and y for mirror inversion
df1.ft_posy = -3*df1.ft_posy
df1.ft_speed = 3*df1.ft_speed
df1['seconds'] = (df1.timestamp-df1.timestamp.iloc[0])/1000


df = pd.read_table(log_path, delimiter='[,]', engine='python')
#split timestamp and motor into separate columns
new = df["timestamp -- motor_step_command"].str.split("--", n = 1, expand = True)
df["timestamp"]= new[0]
df["motor_step_command"]=new[1]
df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
df['seconds'] = 3600*df.timestamp.dt.hour+60*df.timestamp.dt.minute+df.timestamp.dt.second+10**-6*df.timestamp.dt.microsecond

# motor command sent to arduino as string, need to convert to numeric
df['motor_step_command'] = pd.to_numeric(df.motor_step_command)

# CAUTION: invert x. Used to correct for mirror inversion, y already flipped in voe
df['ft_posx'] = -df['ft_posx']

#calculate when fly is in strip
df['instrip'] = np.where(df['mfc2_stpt']>0.0, True, False)


#%% LAL activity
fc = fci_regmodel(pv2['1_lal']-pv2['0_lal'],ft2,pv2)

fc.example_trajectory(-0.25,0.25)
#%% In odour entry/exit average response and trajectory
from scipy.signal import savgol_filter
plt.plot(ft2['instrip'])
y = pv2['1_lal']-pv2['0_lal']
plt.plot(y,color='r')
y = savgol_filter(y,50,5)
plt.plot(y,color='k')

fc = fci_regmodel(y,ft2,pv2)

fc.example_trajectory(-0.5,0.5)