# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:57:07 2024

@author: dowel
"""

from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from analysis_funs.utilities import funcs as fn
import matplotlib as mpl
import pickle
from matplotlib import cm
from analysis_funs.regression import fci_regmodel
#%%
class CX_tan:
    def __init__(self,datadir,tnstring='0_fsbtn',Andy=False,span=500):
        
        
        d = datadir.split("\\")
        self.name = d[-3] + '_' + d[-2] + '_' + d[-1]
        if Andy==False:
            cx = CX(self.name,['fsbTN'],datadir)
            self.pv2, self.ft, self.ft2, self.ix = cx.load_postprocessing()
        else:
            post_processing_file = os.path.join(datadir,'postprocessing.h5')
            self.pv2 = pd.read_hdf(post_processing_file, 'pv2')
            self.ft2 = pd.read_hdf(post_processing_file, 'ft2')
        self.fc = fci_regmodel(self.pv2[[tnstring]].to_numpy().flatten(),self.ft2,self.pv2)
        self.fc.rebaseline(span=span,plotfig=False)
        
    def mean_traj_nF(self,use_rebase = True,tnstring='0_fsbtn'):
        """
        Function outputs mean trajectory of animal entering and exiting the plume
        alongside the mean fluorescence

        Returns
        -------
        None.

        """
        plume_centres = [0,210,420]
        if use_rebase:
            ca = self.fc.ca
        else:
            ca = self.pv2[tnstring]
            
        ft2 = self.ft2
        pv2 = self.pv2
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fc.fictrac_repair(x,y)
        expst = np.where(ins==1)[0][0]
        x = x-x[expst]
        y = y-y[expst]
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1
        ents_O = ents.copy()
        exts_O = ents.copy()
        ents = ents[1:]
        exts = exts[1:]
        # Need to pick a side
        if len(ents)>len(exts):
            ents = ents[:-1]
        
        ent_x = np.round(x[ents])
        ex_x = np.round(x[exts])
        sides = np.zeros(len(ent_x))
        plume_centre = np.zeros(len(ent_x))
        
        
        # 1 -1 indicates sides of entry/exit
        # -0.9 0.9 indicates crossing over from left and right
        for i,x1 in enumerate(ent_x):
            x2 = ex_x[i]
            pcd = plume_centres-np.abs(x1)
            
            pi = np.argmin(np.abs(pcd))
            pc = plume_centres[pi]*np.sign(x1)
            plume_centre[i] = pc
            s_en = np.sign(x1-pc)
            s_ex = np.sign(x2-pc)
            if s_en==s_ex:
                sides[i] = s_en
            elif s_en!=s_ex:
               
                sides[i] = s_en+(s_ex/10)
                
                    
        v,c = np.unique(sides,return_counts=True)
        
        ts = v[np.argmax(c)]
        #print(ts)
        t_ents = ents[sides==ts]
        t_exts = exts[sides==ts]
        t_pc = plume_centre[sides==ts]
        # initialise arrays
        trajs = np.empty((100,2,len(t_ents)))
        Ca = np.empty((100,len(t_ents)))
        trajs[:] = np.nan
        Ca[:] = np.nan
        for i,en in enumerate(t_ents):
            prior_ex = exts_O-en
            prior_ex = prior_ex[prior_ex<0]
            pi = np.argmax(prior_ex)
            ex1 = exts_O[pi]
            dx1 = np.arange(ex1,en,dtype=int)
            dx2 = np.arange(en,t_exts[i],dtype=int)
            x1 = x[dx1]-t_pc[i]
           # print(t_pc[i])
            y1 = y[dx1]
            ca1 = ca[dx1]
            # print(x[dx2[0]]-t_pc[i])
            # print(x[dx2[-1]]-t_pc[i])
            # print(x[dx1[-1]]-t_pc[i])
            
            if np.sign(x[dx1[0]]-t_pc[i])!=np.sign(x[dx1[-1]]-t_pc[i]):
                continue
            # print('aaa')
            x2 = x[dx2]-t_pc[i]
            y2 = y[dx2]
            ca2 = ca[dx2]
            y2 = y2-y1[0]
            y1 = y1-y1[0]
            
            x1 = x1*ts
            x2 = x2*ts
            
            x1d = 5-x1[0]
            x1 = x1
            x2 = x2
            #Interpolate onto timebase: return
            old_time = dx1-dx1[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x1)
            y_int = np.interp(new_time,old_time,y1)
            ca_int = np.interp(new_time,old_time,ca1)
            trajs[:50,0,i] = x_int
            trajs[:50,1,i] = y_int
            Ca[:50,i] = ca_int
            
            #Interpolate onto timebase: in plume
            old_time = dx2-dx2[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x2)
            y_int = np.interp(new_time,old_time,y2)
            ca_int = np.interp(new_time,old_time,ca2)
            trajs[50:,0,i] = x_int
            trajs[50:,1,i] = y_int
            Ca[50:,i] = ca_int
            
        traj_mean = np.nanmean(trajs,axis=2)
        Ca_mean = np.nanmean(Ca,axis=1)
        return traj_mean,Ca_mean
    def mean_traj_heat(self,xoffset=0,set_cmx =False,cmx=1):
        trj,ca = self.mean_traj_nF()
        colour = ca
        if set_cmx==False:
            cmx = np.max(np.abs(ca))
        c_map = plt.get_cmap('coolwarm')
        cnorm = mpl.colors.Normalize(vmin=-cmx, vmax=cmx)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        yrange = np.array([min(trj[:,1]),max(trj[:,1])])
        plt.fill([-5+xoffset,5+xoffset,5+xoffset,-5+xoffset],yrange[[0,0,1,1]],color=[0.7,0.7,0.7])
        
        for i in range(len(ca)-1):
            x = trj[i:i+2,0]
            y = trj[i:i+2,1]
            #ca = np.mean(ca[i:i+2])
            plt.plot(x+xoffset,y,color=c_map_rgb[i,:])
            
            
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        