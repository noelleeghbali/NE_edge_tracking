# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:04:07 2024

@author: dowel

This class object will model the phase of columnar neurons from given parameters

Construction notes:
    
    1. Model 1 - fit phase
    - Tricky because: circular variables, therefore have to use a custom function
    rather than linear regression. As far as I know.
    2. Model 2 - fit each column
    - Again a little tricky but should be possible
    

"""
import sklearn.linear_model as lm
from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from analysis_funs.utilities import funcs as fn
import pickle
from scipy.optimize import curve_fit,minimize
#%%
class CX_phase_modelling:
    def __init__(self,cxa):
        self.cxa = cxa
        self.popt = None
        self.pcov = None
        
    def fit_in_parts(self,x,phase,parts):
        ft2 = self.cxa.ft2
        popt_array = np.zeros((np.shape(x)[0],len(parts)))
        for i,p in enumerate(parts):
            dx = self.output_time_epochs(ft2,p)
            self.fit_phase_function(x[:,dx],phase[dx])
            
            popt_array[:,i] = self.results.x
                
            plt.figure()
            print(self.results.x)
            err = self.loss_fun(self.phase_function(x,*self.results.x),phase)
            print(err)
            plt.plot(phase[dx])
            plt.plot(self.phase_function(x[:,dx],*self.results.x))
        self.popt_array = popt_array
        
    def reg_in_parts(self,x,phase,parts):
        ft2 = self.cxa.ft2
        for i,p in enumerate(parts):
            dx = self.output_time_epochs(ft2,p)
            self.fit_reg_model(x[:,dx],phase[dx])
            
            
            print(self.r2)
            print(self.reg.coef_)
            plt.figure()
            plt.plot(self.y)
            plt.plot(self.yp)
            
    def output_time_epochs(self,ft2,epoch):
        inplume = ft2['instrip']
        ids = np.diff(inplume)
        pon = np.where(ids>0)[0]+1
        poff = np.where(ids<0)[0]+1
        times = self.cxa.pv2['relative_time']
        
        if epoch =='Pre Air':
            est = np.where(inplume>0)[0][0]
            index = np.arange(0,est,1,dtype=int)
            
        elif epoch == 'Returns':
            index = np.array([],dtype=int)
            poff2 = poff[:-1]
            for i,p in enumerate(poff2):
                dx = np.arange(p,pon[i+1])
                index = np.append(index,dx)
                
        elif epoch == 'Jump Returns':
            jumps = ft2['jump']
            jd = np.diff(jumps)
            jn = np.where(np.abs(jd)>0)[0]
            jkeep = np.where(np.diff(jn)>1)[0]
            jn = jn[jkeep]
            jns = np.sign(jd[jn])
            time_threshold = 60
            # Pick the most common side
            v,c = np.unique(jns,return_counts=True)
            side = v[np.argmax(c)]
            self.side = side
            # Get time of return: choose quick returns
            dt = []
            for i,j in enumerate(jn):
                ex = poff-j
                ie = np.argmin(np.abs(ex))
                t_ent = ie+1
                sub_dx = poff[ie]
                tdx = np.arange(pon[ie],poff[t_ent],step=1,dtype='int')
                dt.append(times[tdx[-1]]-times[sub_dx])
            this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]
            
            index = np.array([],dtype=int)
            for i,j in enumerate(this_j):
                ex = poff-j
                ie = np.argmin(np.abs(ex))
                t_ent = ie+1
                sub_dx = poff[ie]
               
                ipdx = np.arange(sub_dx,pon[t_ent],step=1,dtype=int)
                index = np.append(index,ipdx)
        elif epoch == 'Jump All':
            jumps = ft2['jump']
            jd = np.diff(jumps)
            jn = np.where(np.abs(jd)>0)[0]
            jkeep = np.where(np.diff(jn)>1)[0]
            jn = jn[jkeep]
            jns = np.sign(jd[jn])
            time_threshold = 60
            # Pick the most common side
            v,c = np.unique(jns,return_counts=True)
            side = v[np.argmax(c)]
            # Get time of return: choose quick returns
            dt = []
            for i,j in enumerate(jn):
                ex = poff-j
                ie = np.argmin(np.abs(ex))
                t_ent = ie+1
                sub_dx = poff[ie]
                tdx = np.arange(pon[ie],poff[t_ent],step=1,dtype='int')
                dt.append(times[tdx[-1]]-times[sub_dx])
            this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]
            
            index = np.array([],dtype=int)
            for i,j in enumerate(this_j):
                ex = poff-j
                ie = np.argmin(np.abs(ex))
                t_ent = ie+1
                sub_dx = pon[ie]
               
                ipdx = np.arange(sub_dx,pon[t_ent],step=1,dtype=int)
                index = np.append(index,ipdx)
        elif epoch == 'In Plume':
            index = np.where(inplume>0)[0]
            
        return index
    def objective_fun(self,weights,x,phase):#May need to reorder this
        
        err = self.loss_fun(self.phase_function(x,weights[0],weights[1],weights[2]),phase)
        return err      
    
    def loss_fun(self,phase_p,phase):
        pdiff = phase_p-phase
        err = np.mean(np.abs(np.cos(pdiff)-1))
        return err
    def fit_reg_model(self,x,phase):
        y = np.cos(phase)+np.sin(phase)
        xft = np.cos(x)+np.sin(x)
        xft = np.atleast_2d(xft).T
        reg = lm.LinearRegression(fit_intercept=False)
        reg.fit(xft,y)
        self.reg = reg
        self.y = y
        self.xft =xft
        self.yp = np.matmul(xft,reg.coef_)
        self.r2 = reg.score(xft,y)
    def fit_phase_function(self,x,phase):
        
        #self.popt, self.pcov = curve_fit(self.phase_function,x,phase)
        #self.popt, self.pcov = curve_fit(self.phase_funtest,x,phase)
        weight_init = np.array([1]*x.shape[0])
        self.results = minimize(self.objective_fun,weight_init,args=(x,phase))
        
    
    def phase_function(self,x,w1,w2,w3): 

        weights = np.array([w1,w2,w3],dtype=float)
        
        weights = np.atleast_2d(weights).T
        xcos = np.sum(np.cos(x)*weights,axis=0)
        xsin = np.sum(np.sin(x)*weights,axis=0)
        phase = np.arctan2(xsin,xcos)
        return phase
    
    def plume_memory(self):
        ft2 = self.cxa.ft2
        ins = ft2['instrip']
        heading = ft2['ft_heading'].to_numpy()
        plume_mem = np.zeros_like(heading)
        
        ind = np.diff(ins)
        pon = np.where(ind>0)[0]
        pon_heading  = heading[pon]
        for i,p in enumerate(pon):
            if i<len(pon)-1:
                plume_mem[p:pon[i+1]] = pon_heading[i]
            else:
                plume_mem[p:] = pon_heading[i]
        return plume_mem
    def mean_phase_polar(self,phase,succession=False,part='Jump Returns'):
        
        dx = self.output_time_epochs(self.cxa.ft2,part)
        phase = phase*self.side*-1 
        
        phase_eb = self.cxa.pdat['offset_eb_phase']*self.side*-1
        xm = self.plume_memory()*self.side*-1

        ddiff = np.diff(dx)
        e_end = np.where(ddiff>1)[0]
        e_end = np.append(e_end,len(ddiff)-1)
        estart = np.append(1,e_end[:-1]+1)
        endx = dx[e_end]
        stdx = dx[estart]
        
        times = self.cxa.pv2['relative_time'].to_numpy()
       
        #amp = self.cxa.pdat['amp_fsb_upper']
       #pamp = np.percentile(amp,99)
        #amp[amp>0.2] = 0.2 
        #amp = amp/0.2
        
        fit_array = np.zeros((50,3,len(endx)))
        for i,e in enumerate(endx):
            print(i)
            tdc = np.arange(stdx[i],e,1,dtype=int)
            t = times[tdc]
            old_time = t-t[0]
            tp = phase[tdc]
            tpe = phase_eb[tdc]
            new_time = np.linspace(0,max(old_time),50)
            tp_int = np.interp(new_time,old_time,tp)
            tpe_int = np.interp(new_time,old_time,tpe)
            fit_array[:,0,i] = tp_int
            fit_array[:,1,i] = tpe_int
            fit_array[:,2,i] = xm[tdc[0]]
        pltmn = circmean(fit_array,high=np.pi, low=-np.pi,axis=2)
        if succession==False:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            fig = succession[0]
            ax = succession[1]
        tstandard = np.linspace(0,49,50)
        colours = np.array([[0.2,0.2,0.8],[0, 0, 0],[1,0,0]])
        for i in range(3):
            ax.plot(pltmn[:,i],tstandard,color=colours[i,:])    
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        return pltmn
    def phase_memory_scatter(self,phase,part='Jump Returns'):
        dx = self.output_time_epochs(self.cxa.ft2,part)
        phase = phase*self.side*-1 
        
        xm = self.plume_memory()*self.side*-1

        ddiff = np.diff(dx)
        e_end = np.where(ddiff>1)[0]
        e_end = np.append(e_end,len(ddiff)-1)
        estart = np.append(1,e_end[:-1]+1)
        endx = dx[e_end]
        stdx = dx[estart]
        
        times = self.cxa.pv2['relative_time'].to_numpy()
       
        #amp = self.cxa.pdat['amp_fsb_upper']
       #pamp = np.percentile(amp,99)
        #amp[amp>0.2] = 0.2 
        #amp = amp/0.2
        p_scat = np.zeros((len(endx),2))
        for i,e in enumerate(endx):
            print(i)
            tdc = np.arange(stdx[i],e,1,dtype=int)
            t = times[tdc]
            old_time = t-t[0]
            tp = phase[tdc]
            tmem = xm[tdc]
            ttdx = (old_time-max(old_time))>=-1
            p_scat[i,0] = circmean(tp[ttdx],high=np.pi,low=-np.pi)
            p_scat[i,1] = tmem[-1]
        return p_scat

        
        
        
        
        
        