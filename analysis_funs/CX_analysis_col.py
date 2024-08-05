# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:35:43 2024

@author: dowel
"""
#%%
from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
#%%
class CX_a:
    def __init__(self,datadir,regions =['eb','fsb']):
        # Will need to edit more if yoking to PB and multiple FSB layers
        
        self.stab = regions[0]
        d = datadir.split("\\")
        name = d[-3] + '_' + d[-2] + '_' + d[-1]
        self.cx = CX(name,regions,datadir)
        self.pv2, self.ft, self.ft2, ix = self.cx.load_postprocessing()
        self.phase,self.phase_offset,self.amp = self.cx.unyoked_phase(regions[1])
        self.phase_eb,self.phase_offset_eb,self.amp_eb = self.cx.unyoked_phase(self.stab)
        self.pdat = self.cx.phase_yoke(self.stab,regions[1:])
    def simple_raw_plot(self,plotphase=False):
        plt.figure(figsize=(5,10))
        phase_eb = self.phase_eb.copy()
        phase = self.phase.copy()
        ebs = []
        for i in range(16):
            ebs.append(str(i) +'_eb')
        for i in range(16):
            ebs.append(str(i) +'_fsb')
        
        eb = self.pv2[ebs]
        t = np.arange(0,len(eb))
        plt.imshow(eb, interpolation='None',aspect='auto',cmap='Blues',vmax=np.percentile(eb[:],97),vmin=np.percentile(eb[:],5))
        new_phase = np.interp(phase_eb, (phase_eb.min(), phase_eb.max()), (-0.5, 15.5))
        if plotphase:
            plt.plot(new_phase,t,color='r',linewidth=0.5)
        plt.plot([15.5,15.5],[min(t), max(t)],color='w')
        plt.xticks([0, 7, 15, 16,23, 31,32,40,48],
                   labels=['eb:1', 'eb:8', 'eb:16','fsb:1','fsb:8','fsb16','-$\pi$','0','$\pi$'],rotation=45)
        new_phase = np.interp(phase, (phase.min(), phase.max()), (15.5, 31.5))
        if plotphase:
            plt.plot(new_phase,t,color='r',linewidth=0.5)

        new_heading = self.ft2['ft_heading'].to_numpy()
        new_heading = np.interp(new_heading, (new_heading.min(), new_heading.max()), (32, 48))
        plt.plot(new_heading,t,color='k')
        sdiff = np.diff(self.ft2['instrip'].to_numpy())
        son = np.where(sdiff>0)[0]+1 
        soff = np.where(sdiff<0)[0]+1 
        for i,s in enumerate(son):
            plt.plot([32, 48],[s, s],color=[1,0.5,0.5])
            plt.plot([32, 48],[soff[i], soff[i]],color=[1,0.5,0.5])
            plt.plot([32,32],[s,soff[i]],color=[1,0.5,0.5])
            plt.plot([48,48],[s,soff[i]],color=[1,0.5,0.5])
        yt = np.arange(0,max(t),600)
        plt.yticks(yt,labels=yt/10)
        plt.ylabel('Time (s)')
        plt.show()
    def entry_exit_phase(self):
        mult = float(180)/np.pi
        eb_phase = self.pdat['offset_eb_phase']*mult
        fsb_phase = self.pdat['offset_fsb_phase']*mult
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        son = np.where(sdiff>0)[0]+1
        soff = np.where(sdiff<0)[0]+1
        
        for i,s in enumerate(son):
            plt.figure()
            pdx = np.arange(s-1,soff[i],dtype='int')
            bdx = np.arange(s-50,s,dtype='int')
            #print(bdx)
            
            plt.scatter(eb_phase[bdx],fsb_phase[bdx],color='k',s=10)
            plt.scatter(eb_phase[pdx],fsb_phase[pdx],color='r',s=10)
            plt.plot(eb_phase[bdx],fsb_phase[bdx],color='k')
            plt.plot(eb_phase[pdx],fsb_phase[pdx],color='k')
            plt.xlim([-180,180])
            plt.ylim([-180,180])
            plt.plot([-180,180],[-180,180],color='k',linestyle='--')
            plt.xticks([-180,-90,0,90,180])
            plt.yticks([-180,-90,0,90,180])
            plt.ylabel('FSB phase')
            plt.xlabel('EB phase')
            plt.show()
    def simple_traj(self):
        plt.figure()
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        strip = self.ft2['instrip'].to_numpy()
        plt.plot(x,y,color='k')
        plt.scatter(x[strip>0],y[strip>0],c='r',s=15)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def mean_phase_trans(self,tbef=5,taf=5):
        eb_phase = self.pdat['offset_eb_phase']
        fsb_phase = self.pdat['offset_fsb_phase']
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        son = np.where(sdiff>0)[0]+1
        idx_bef = int(np.round(float(tbef)/tinc))
        idx_af = int(np.round(float(taf)/tinc))
        son = son[1:]
        mn_mat = np.zeros((len(son),idx_bef+idx_af+1))
        
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(eb_phase):
                nsum = np.sum(idx_array>len(eb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat[i,:-nsum] = eb_phase[idx_array]
            else:
                mn_mat[i,:] = eb_phase[idx_array]
        plt_mn = circmean(mn_mat,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        plt.figure()
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 0.6],zorder=0,alpha = 0.3)
        plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        plt.plot(t,plt_mn,color='k',zorder=1)
        
        mn_mat2 = np.zeros((len(son),idx_bef+idx_af+1))
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(fsb_phase):
                nsum = np.sum(idx_array>len(fsb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat2[i,:-nsum] = fsb_phase[idx_array]
            else:
                mn_mat2[i,:] = fsb_phase[idx_array]
                
        plt_mn = circmean(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 1],zorder=3,alpha = 0.3)
        plt.plot(t,plt_mn,color=[0.3,0.3,0.8],zorder=4)
        plt.plot([-tbef,taf],[0,0],color='k',linestyle='--',zorder=5)
        mn = -np.pi
        mx = np.pi
        plt.plot([0,0],[mn,mx],color='k',linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase')
        plt.yticks([mn,mn/2,0,mx/2,mx],labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
        plt.ylim([mn,mx])
        plt.show()
        
        plt.figure()
        plt.plot(t,np.transpose(mn_mat),color='k',alpha=0.5)
        plt.show()
        plt.figure()
        plt.plot(t,np.transpose(mn_mat2),color='b',alpha=0.5)
        plt.show()
    def plot_traj_arrow(self,phase,amp,a_sep= 20):
        phase_eb = self.pdat['offset_eb_phase']
        phase = self.phase_eb
        amp_eb = self.amp_eb
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        instrip = self.ft2['instrip'].to_numpy()
        
        dist = np.sqrt(x**2+y**2)
        dist = dist-dist[0]
        plt.figure()
        plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        plt.plot(x,y,color='k')
        t_sep = a_sep
        for i,d in enumerate(dist):
            if d-t_sep>0:
                t_sep = t_sep+a_sep
                
                xa = 50*amp_eb[i]*np.sin(phase_eb[i])
                ya = 50*amp_eb[i]*np.cos(phase_eb[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
                
                xa = 50*amp[i]*np.sin(phase[i])
                ya = 50*amp[i]*np.cos(phase[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color='r')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def plot_taj_cond(self,phase,amp,cond):
        phase_eb = self.pdat['offset_eb_phase']
        
        amp_eb = self.amp_eb
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        instrip = self.ft2['instrip'].to_numpy()
        
        dist = np.sqrt(x**2+y**2)
        dist = dist-dist[0]
        plt.figure()
        plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        plt.plot(x,y,color='k')
        if cond=='entry_exit':
            dstrip = np.diff(instrip)
            dx = np.where(np.abs(dstrip)>0)[0]+1
        
        for i in dx:
            
                
            xa = 50*amp_eb[i]*np.sin(phase_eb[i])
            ya = 50*amp_eb[i]*np.cos(phase_eb[i])
            plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
            
            xa = 50*amp[i]*np.sin(phase[i])
            ya = 50*amp[i]*np.cos(phase[i])
            plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color='r')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def cond_polar(self,cond,phase,amp):
        plt.figure()
        instrip = self.ft2['instrip'].to_numpy()
        if cond=='entry_exit':
            
            dstrip = np.diff(instrip)
            dxe = np.where(dstrip>0)[0]+1 
            dxex = np.where(dstrip<0)[0]+1 
        for i in dxe:
            xa = amp[i]*np.sin(phase[i])
            ya = amp[i]*np.cos(phase[i])
            plt.arrow(0,0,xa,ya,length_includes_head=True,head_width=0.005,color=[1,0.3,0.3],alpha=0.25)
        for i in dxex:
            xa = amp[i]*np.sin(phase[i])
            ya = amp[i]*np.cos(phase[i])
            plt.arrow(0,0,xa,ya,length_includes_head=True,head_width=0.005,color=[0.3,0.3,1],alpha=0.25)
            
        xall = amp[dxe]*np.sin(phase[dxe])
        xm = xall.mean()
        yall = amp[dxe]*np.cos(phase[dxe])
        ym = yall.mean()
        plt.arrow(0,0,xm,ym,length_includes_head=True,head_width=0.005,color=[1,0.1,0.1],alpha=1)
        
        xall = amp[dxex]*np.sin(phase[dxex])
        xm = xall.mean()
        yall = amp[dxex]*np.cos(phase[dxex])
        ym = yall.mean()
        plt.arrow(0,0,xm,ym,length_includes_head=True,head_width=0.005,color=[0.3,0.3,1],alpha=1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()  
            
    def mean_in_plume(self):
        plt.figure()
        eb_phase = self.pdat['offset_eb_phase']
        fsb_phase = self.pdat['offset_fsb_phase']
        #eb_phase = self.phase_eb
        #fsb_phase = self.phase
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        son = np.where(sdiff>0)[0]+1
        soff = np.where(sdiff<0)[0]+1
        son = son[1:]
        soff = soff[1:]
        mn_mat = np.zeros((len(son),20))
        mn_mat2 = np.zeros((len(son),20))
        new_time = np.arange(0,20,dtype=float)
        for i,s in enumerate(son):
            idx_array = np.arange(s-1,soff[i],dtype= int)
            old_time = (idx_array-idx_array[0])
            tp = fsb_phase[idx_array]
            tpi = np.interp(new_time,old_time,tp)
            mn_mat2[i,:] = tpi
            
            tp2 = eb_phase[idx_array]
            tpi_2 = np.interp(new_time,old_time,tp2)
            mn_mat[i,:] = tpi_2
        
        plt_mn = circmean(mn_mat,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat,axis=0,high=np.pi,low=-np.pi)
        t = new_time
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 0.6],zorder=0,alpha = 0.3)
        plt.plot(t,plt_mn,color=[0.3,0.3,0.3],zorder=1)
        
        plt_mn = circmean(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 1],zorder=3,alpha = 0.3)
        plt.plot(t,plt_mn,color=[0.3,0.3,0.8],zorder=4)
        plt.xlim([0,19])
        mn = -np.pi
        mx = np.pi
        plt.ylim([-np.pi,np.pi])
        plt.ylabel('Phase')
        plt.yticks([mn,mn/2,0,mx/2,mx],labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
        
        plt.plot([0,19],[0,0],color='k',linestyle='--')
        plt.show()
        plt.ylabel('Phase')
        plt.xlabel('In plume time (AU)')
    def polar_movie(self,phase,amp):
        import matplotlib as mpl
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
        import networkx as nx

        #mpl.use("TkAgg") 
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
        # Your specific x and y values
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        instrip = self.ft2['instrip'].to_numpy()
        # Create initial line plot

        fig, ax = plt.subplots(figsize=(10,10))
        line2, = ax.plot([],[],lw=2,color=[0.2,0.2,0.2])
        line, = ax.plot([], [], lw=2,color=[0.2,0.4,1])  # Empty line plot with line width specified
        sc = ax.scatter([],[],color=[0.5,0.5,0.5])

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')
        # Set axis limits
        #ax.set_xlim(xrange[0], xrange[1])
        

        # Animation update function
        def update(frame):
            # Update the line plot with the current x and y values
            line2.set_data(x[:frame], y[:frame])
            if frame>100:
                line.set_data(x[frame-100:frame], y[frame-100:frame])
            else:
                line.set_data(x[:frame], y[:frame])
            
            if instrip[frame]>0:
                sc.set_offsets(np.column_stack((x[frame],y[frame])))
            
        # Create animation
        anim = mpl.animation.FuncAnimation(fig, update, frames=len(x), interval=0.01)
        plt.show()
        
        