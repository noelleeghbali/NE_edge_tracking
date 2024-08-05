# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:39:38 2023

@author: dowel
"""

#%% Regression modelling test ground
import sklearn.linear_model as lm
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import statsmodels.api as sm
from src.utilities import funcs as fn
#%%
class fci_regmodel:
    def __init__(self,y,ft2,pv2):
        self.reg_type = 'Ridge'
        self.generation = 'one'
        self.ca = y
        self.ft2 = ft2 #behaviour
        self.pv2 = pv2 #additional data including timestamps
        self.ts = pv2['relative_time']
    def rebaseline(self,span=400,plotfig=False):
        y = self.ca
        frac = float(span)/np.max(self.ts)
        lowess = sm.nonparametric.lowess

        # Apply lowess smoothing
        yf = lowess(y, np.arange(0, len(y)), frac=frac)

        if len(y) != len(yf[:, 1]):
            print(f"Warning: Lengths differ: y ({len(y)}), yf[:, 1] ({len(yf[:, 1])}). Correcting lengths.")
            min_len = min(len(y), len(yf[:, 1]))
            y = y[:min_len]
            yf = yf[:min_len, :]

        self.ca_original = y
        self.ca = y-yf[:,1]

        # if len(y) != len(yf[:,1]):
        #     # Option to pad yf with last value
        #     yf_padded = np.append(yf[:,1], yf[-1,1])
        #     self.ca = y - yf_padded
        # else:
        #     self.ca = y - yf[:,1]
            
        if plotfig:
            plt.figure()
            plt.plot(y)
            plt.plot(yf[:,1])
            plt.show()
    def set_up_regressors(self,regchoice,cirftau =[0.3,0.01]):
        #Cirf is very approximate and should be verified
        
        xs = np.shape(self.ft2['mfc2_stpt'])
        self.regchoice =regchoice
        regmatrix = np.ones([xs[0],len(regchoice)+1],dtype = float)
        # define regressors
        for i,r in enumerate(regchoice):
            if r=='odour onset':
                x = self.ft2['instrip'].copy()
                x = np.diff(x)>0
                x = np.append([0],x)
                
            elif r=='odour offset':  
                x = self.ft2['instrip'].copy()
                x = np.diff(x)<0
                x = np.append([0],x)
                
            elif r=='in odour':
                x = self.ft2['instrip'].copy()
                x = x>0
                
            elif r == 'cos heading pos':
                x = np.cos(self.ft2['ft_heading'].copy())
                x[x<0] = 0 
                
            elif r =='cos heading neg':
                x = -np.cos(self.ft2['ft_heading'].copy())
                x[x<0] = 0
                
            elif r == 'sin heading pos':
                x = np.sin(self.ft2['ft_heading'].copy())
                x[x<0] = 0 
                
            elif r == 'sin heading neg':
                x = -np.sin(self.ft2['ft_heading'].copy())
                x[x<0] = 0 
                
            elif r == 'angular velocity pos':
                x = pd.Series.to_numpy(self.ft2['ang_velocity'].copy())  
                x[x<0] = 0 
                
            elif r == 'angular velocity neg':
                x = -pd.Series.to_numpy(self.ft2['ang_velocity'].copy())
                x[x<0] = 0
                
            elif r== 'x pos':
                #x = pd.Series.to_numpy(self.ft2['x_velocity'])
                x = pd.Series.to_numpy(self.ft2['ft_posx'].copy())
                x = np.diff(x)
                xp = np.percentile(np.abs(x),99)
                x[x>xp] = xp
                x[x<-xp] = -xp
                x = np.append([0],x)
                x[x<0] = 0
            elif r== 'x neg':
                #x = -pd.Series.to_numpy(self.ft2['x_velocity'])
                x = pd.Series.to_numpy(self.ft2['ft_posx'].copy())
                x = -np.diff(x)
                x = np.append([0],x)
                xp = np.percentile(np.abs(x),99)
                x[x>xp] = xp
                x[x<-xp] = -xp
                x[x<0] = 0
                
            elif r == 'y pos':
                #x = pd.Series.to_numpy(self.ft2['y_velocity'])
                #x[x<0] = 0
                x = pd.Series.to_numpy(self.ft2['ft_posy'].copy())
                x = np.diff(x)
                xp = np.percentile(np.abs(x),99)
                x[x>xp] = xp
                x[x<-xp] = -xp
                x = np.append([0],x)
                x[x<0] = 0
            elif r =='y neg':
                #x = -pd.Series.to_numpy(self.ft2['y_velocity'])
                x = pd.Series.to_numpy(self.ft2['ft_posy'].copy())
                x = -np.diff(x)
                xp = np.percentile(np.abs(x),99)
                x[x>xp] = xp
                x[x<-xp] = -xp
                x = np.append([0],x)
                x[x<0] = 0
            elif r == 'stationary':
                x1 = pd.Series.to_numpy(self.ft2['x_velocity'].copy())
                x2 = pd.Series.to_numpy(self.ft2['y_velocity'].copy()) 
                x = x1==0&x2==0
            
            elif r == 'ramp to entry':
                x1 = (pd.Series.to_numpy(self.ft2['mfc2_stpt'])>0).astype(float)
                x1 = np.diff(x1)<0
                
                x1 = np.append([0],x1)
                x = np.zeros_like(x1,dtype='float')
                wx = [ ir for ir, xi in enumerate(x1) if xi>0]
                
                x2 = (pd.Series.to_numpy(self.ft2['mfc2_stpt'])>0).astype(float)
                x2 = np.diff(x2)>0
                x2 = np.append([0],x2)
                we = [ ir for ir, xi in enumerate(x2) if xi>0]

                ws = len(wx)
               
                
                for v in range(ws-1):
                    add = 1
                    xln = we[v+add]-wx[v]
                    while xln<0:   
                        add = add+1
                        xln = we[v+add]-wx[v]
                              
                    xin = np.linspace(0,1,xln)
                    
                    x[wx[v]:we[v+add]] = xin
                
            elif r == 'ramp down since exit':
                x1 = (pd.Series.to_numpy(self.ft2['instrip'])>0).astype(float)
                x1 = np.diff(x1)<0
                
                x1 = np.append([0],x1)
                x = np.zeros_like(x1,dtype='float')
                wx = [ ir for ir, xi in enumerate(x1) if xi>0]
                
                x2 = (pd.Series.to_numpy(self.ft2['instrip'])>0).astype(float)
                x2 = np.diff(x2)>0
                x2 = np.append([0],x2)
                we = [ ir for ir, xi in enumerate(x2) if xi>0]

                ws = len(wx)
                for v in range(ws-1):
                    add = 1
                    xln = we[v+add]-wx[v]
                    while xln<0:   
                        add = add+1
                        xln = we[v+add]-wx[v]
                              
                    xin = np.linspace(1,0,xln)
                    
                    x[wx[v]:we[v+add]] = xin
                    
                
                
            x[np.isnan(x)] = 0
            regmatrix[:,i] = x
             
            
        regmatrix_preconv = regmatrix.copy()    
        # convolve with Ca response kernel
        ts = self.pv2['relative_time'].copy()
        cirf = np.exp(-ts[0:1000]/cirftau[0]) - np.exp(-ts[0:1000]/cirftau[1])
        zpad = np.zeros(100)
        #plt.plot(regmatrix[:,0])
        for i in range(len(regchoice)):
            x = regmatrix[:,i]
            #print(np.shape(x))
            
            x = np.concatenate((zpad,x,zpad),0)
            c_conv = np.convolve(x,cirf)
            #(np.shape(c_conv))
            c_conv = c_conv[99:-1100]
            regmatrix[:,i] = c_conv
        #plt.plot(regmatrix[:,0])
        
        #plt.show()
        # normalise by standard deviation
        regmatrix = regmatrix/np.std(regmatrix,0)
        regmatrix[np.isnan(regmatrix)] = 0# deals with divide by zero for when animal does not do the behaviour
        regmatrix[:,-1] = 1
        return regmatrix, regmatrix_preconv
    def run(self,regchoice,partition=False):
        
        # Set up regessors
        print('Determining regressors')
        regmatrix, regmatrix_preconv = self.set_up_regressors(regchoice)
        
        self.regmatrix = regmatrix
        self.regmatrix_preconv = regmatrix_preconv.copy()
        # regression engine
        y = self.ca
        x = regmatrix
        yn = ~np.isnan(y)
        y = y[yn]
        x = x[yn,:]
        ts_2 = self.ts.copy()
        self.yn = yn.copy()
        self.ts_y = ts_2[yn]
        # determine temporal offset
        xs = np.shape(x)
        xpad = np.zeros([20,xs[1]])
        x_p = np.concatenate((xpad,x,xpad),axis= 0)
        
        r2forward = np.zeros(20)
        reg = lm.LinearRegression(fit_intercept=False)
        for i in range(20):
            xft = x_p[20-i:-20-i,:]
            reg.fit(xft,y)
            r2forward[i] = reg.score(xft,y)
            
        r2backward = np.zeros(20)
        
        for i in range(20):
            xft = x_p[20+i:-20+i,:]
            reg.fit(xft,y)
            r2backward[i] = reg.score(xft,y)    
            
        isfor = max(r2forward)>max(r2backward)
        
        if isfor:
            i = np.argmax(r2forward)
            xft = x_p[20-i:-20-i,:]
        else:
            i = np.argmax(r2backward)
            xft = x_p[20-i:-20-i,:]
        print(np.shape(xft))
        print(np.shape(y))
        self.y = y
        self.isfor = isfor
        self.delay = i
        self.xft = xft
        self.r2backward = r2backward
        self.r2forward = r2forward
        
        
        # Run ridge and determine alphas
        alphaz = [0.25, 0.1, 0.05, 0.025, 0.01, 0.001, 0.0001,]
        
        r2alphas = np.zeros_like(alphaz)
        group_kfold = GroupKFold(n_splits=10)
        groups = np.random.randint(0,10,len(y))
        group_kfold.get_n_splits(xft, y,groups)
        for i, a in enumerate(alphaz):
            reg = lm.Ridge(a,fit_intercept=False)
            r2s = np.zeros(10)
            # Need to do cross validation
            for i2, (train_index, test_index) in enumerate(group_kfold.split(xft,y,groups)):
                reg.fit(xft[train_index,:],y[train_index])
                r2s[i2] = reg.score(xft[test_index,:],y[test_index])
            r2alphas[i] = np.mean(r2s)
        
        i = np.argmax(r2alphas)
        self.alpha = alphaz[i]
        self.r2 = r2alphas[i]
        # Run ridge with preferred alpha with cross validation
        reg = lm.Ridge(alphaz[i],fit_intercept=False)
        self.lm = reg
        
        coeffs = np.zeros([10,xs[1]])
        for i, (train_index, test_index) in enumerate(group_kfold.split(xft,y,groups)):
            reg.fit(xft[train_index,:],y[train_index])
            coeffs[i,:] = reg.coef_
            
        self.coeffs = coeffs
        self.coeff_cv = np.mean(coeffs,0)
        
        self.predy = np.matmul(xft,self.coeff_cv)
        # cross validate model 10 fold
        if partition =='pre_air':
            self.partname = partition
            son = self.ft2['instrip'].to_numpy()
            son = np.where(son)[0][0]
            dx_train = np.arange(0,son,dtype='int')
            dx_test = np.arange(son,len(y),dtype='int')
            train_parts = True
        elif partition ==False:
            train_parts = False
        
        if train_parts:
            r2_alphas = np.zeros_like(alphaz)
            for i, a in enumerate(alphaz):
                reg = lm.Ridge(a,fit_intercept=False)
                r2s = np.zeros(10)
                # Need to do cross validation
                reg.fit(xft[dx_train,:],y[dx_train])
                r2_alphas[i] = reg.score(xft[test_index,:],y[test_index])
            i = np.argmax(r2alphas)
            self.alpha_part = alphaz[i]
            #self.r2_part_train = r2alphas[i]
            
            reg = lm.Ridge(alphaz[i],fit_intercept=False)
            reg.fit(xft[dx_train,:],y[dx_train])
            self.lm_part = reg
            self.coeffs_part = reg.coef_
            self.predy_part = np.matmul(xft,self.coeffs_part)
            self.r2_part_test = metrics.r2_score(y[dx_test],self.predy_part[dx_test])
            self.r2_part_train = metrics.r2_score(y[dx_train],self.predy_part[dx_train])
        
    def run_dR2(self,iterations,x):
        # run unique contribution model to output dR2
        y = self.y
        
        alpha = self.alpha
        beta = self.coeffs
        #
        
        group_kfold = GroupKFold(n_splits=10)
        groups = np.random.randint(0,10,len(y))
        
        xft = x
        group_kfold.get_n_splits(xft, y,groups)
        
        # get cvr2
        reg = lm.Ridge(alpha,fit_intercept=False)
        r2s = np.zeros(10)
        for i2, (train_index, test_index) in enumerate(group_kfold.split(xft,y,groups)):
            reg.fit(xft[train_index,:],y[train_index])
            r2s[i2] = reg.score(xft[test_index,:],y[test_index])
            
        cvr2 = np.mean(r2s)
        self.cvR2 = cvr2
        dR2 = np.zeros([iterations,len(beta[0])-1])
        ttest = np.zeros(len(beta[0])-1)
        # get dR2
        xi = np.linspace(0,len(y)-1,len(y),dtype='int')
        for b in range(len(beta[0])-1):
            print(b)
            for i in range(iterations):
                xft2 = x.copy()
                cp = np.random.randint(len(y))
                xiperm = np.append(xi[cp:],xi[:cp])
                xft2[:,b] = xft2[xiperm,b]
                
                r2s = np.zeros(10)
                reg = lm.Ridge(alpha,fit_intercept=False)
                for i2, (train_index, test_index) in enumerate(group_kfold.split(xft2,y,groups)):
                    reg.fit(xft2[train_index,:],y[train_index])
                    r2s[i2] = reg.score(xft2[test_index,:],y[test_index])
                cvr = np.mean(r2s)
                
                del reg
                print(cvr)
                dR2[i,b] = cvr-cvr2
            #plt.plot(x[:,5]+b)    
            O = stats.ttest_1samp(dR2[:,b],0,alternative='less') # ttest < zero
            ttest[b] = O.pvalue
            
        #plt.show()
        self.dR2_All = dR2
        self.dR2_mean = np.mean(dR2,axis=0)
        self.dR2_ttest = ttest
        
    def plot_example_flur(self):
        plt.figure(figsize=(18,8))
        plt.plot(self.ts,self.ca,color='k')
        plt.plot(self.ts_y,self.predy,color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('dF/F')
        plt.show()
        
    def plot_flur_w_regressors(self,regchoice):
        plt.figure(figsize=(18,8))
        plt.plot(self.ts,self.ca,color='k')
        R = self.regmatrix_preconv[:,:-1]
        for r in regchoice:
            rdx = np.in1d(self.regchoice,r)
            y = R[:,rdx]
            y = y/np.max(y)
            y = y*np.max(self.ca)
            plt.plot(self.ts,y)
            
        plt.xlabel('Time (s)')
        plt.ylabel('dF/F')
        plt.show()
        
    def plot_all_regressors(self,regchoice):
        plt.figure(figsize=(18,15))
        R = self.regmatrix[:,:-1]
        for i,r in enumerate(regchoice):
            rdx = np.in1d(self.regchoice,r)
            y = R[:,rdx]
            y = y/np.max(y)
            y = y+float(i)
            plt.plot(self.ts,y,color='k')
                    
        plt.xlabel('Time (s)')
        plt.yticks(np.linspace(0,i,i+1),labels =regchoice)
        plt.show()
        
    def example_trajectory(self,cmin=0,cmax=1):
        colour = self.ca
        x = self.ft2['ft_posx']
        y = self.ft2['ft_posy']
        x,y = self.fictrac_repair(x,y)
        xrange = np.max(x)-np.min(x)
        yrange = np.max(y)-np.min(y)
        mrange = np.max([xrange,yrange])+100
        y_med = yrange/2
        x_med = xrange/2
        ylims = [y_med-mrange/2, y_med+mrange/2]
   
        xlims = [x_med-mrange/2, x_med+mrange/2]

        acv = self.ft2['instrip']
        inplume = acv>0
        c_map = plt.get_cmap('coolwarm')
        if cmin==cmax:
            cmax = np.round(np.percentile(colour[~np.isnan(colour)],97.5),decimals=1)
        cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        x = x-x[0]
        y = y -y[0]
        plt.rcParams['pdf.fonttype'] = 42 
        plt.rcParams['ps.fonttype'] = 42 
        fig = plt.figure(figsize=(15,15))

        ax = fig.add_subplot(111)
        ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
        for i in range(len(x)-1):
            ax.plot(x[i:i+2],y[i:i+2],color=c_map_rgb[i+1,:3])
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel('x position (mm)')
        plt.ylabel('y position (mm)')
        plt.title('Flur range 0 - ' + str(cmax))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.ion()
        plt.show()
        
    def entries_in_a_row(self):
        # Plots entries all in same strip
        colour = self.ca
        x = self.ft2['ft_posx']
        y = self.ft2['ft_posy']
        x,y = self.fictrac_repair(x,y)
        pon = self.ft2['instrip'].to_numpy()
        pon_i = np.where(pon)[0]
        
        
    def fictrac_repair(self,x,y):
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        lrgx = dx>5 
        lrgy = dy>5 
        bth = np.logical_or(lrgx, lrgy)
        
        fixdx =[i+1 for i,b in enumerate(bth) if b]
        for i,f in enumerate(fixdx):
            
            x[f:] =x[f:]- (x[f]-x[f-1])
            
            y[f:] = y[f:]- (y[f]-y[f-1])
        return x, y
    def plot_mean_flur(self,alignment,tbef=30,taf=30,output=False,plotting=True):
        if alignment=='odour_onset':
            td = self.ft2['instrip'].to_numpy()
            tdiff= np.diff(td)
            son = np.where(tdiff>0)[0]+1
        tinc = np.mean(np.diff(self.ts))
        idx_bef = int(np.round(float(tbef)/tinc))
        ca = self.ca
        print(idx_bef)
        idx_af = int(np.round(float(taf)/tinc))
        mn_mat = np.zeros((len(son),idx_bef+idx_af+1))
        for i,s in enumerate(son):
            print(i)
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(ca):
                print('Bang')
                nsum = np.sum(idx_array>len(ca))
                idx_array = idx_array[idx_array<len(ca)]
                mn_mat[i,:-nsum] = ca[idx_array]
            else:
                mn_mat[i,:] = ca[idx_array]
        
        plt_mn = np.mean(mn_mat,axis=0)
        std = np.std(mn_mat,axis = 0)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        if plotting:
            plt.figure()
            plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 0.6])
            plt.plot(t,plt_mn,color='k')
            mn = np.min(plt_mn-std)
            mx = np.max(plt_mn+std)
            plt.plot([0,0],[mn,mx],color='k',linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('dF/F')
            plt.show()
            plt.figure()
            plt.plot(t,np.transpose(mn_mat),color='k',alpha=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('dF/F')
            plt.plot([0,0],[mn,mx],color='k',linestyle='--')
            plt.show()
        if output:
            return plt_mn,t
        
    def plot_dR2_coeffs(self,regchoice):
        plt.figure()
        plt.plot(self.dR2_mean)
        plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
        plt.subplots_adjust(bottom=0.4)
        plt.ylabel('delta R2')
        plt.xlabel('Regressor name')
        plt.show()
        
        plt.figure()
        plt.plot(self.coeff_cv[:-1])
        plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
        plt.subplots_adjust(bottom=0.4)
        plt.ylabel('Coefficient weight')
        plt.xlabel('Regressor name')
        plt.show()
    def bump_jump_tune(self,ft,intwin=3):
        # Aim of code is to produce a tuning curve of bump jump fluorescence changes
        
        bumps = self.ft2['bump'].to_numpy()
        bumps_full = ft['bump'].to_numpy()
        bump_size = bumps_full[np.abs(bumps_full)>0]
        bdx = np.abs(bumps)>0
        bdx_w = np.where(bdx)[0]
        ti = np.mean(np.diff(self.ts))
        tnum = np.round(intwin/ti)
        tnum = int(tnum)
        bca = np.zeros(len(bump_size))
        for i,idx in enumerate(bdx_w):
            bca[i] = np.mean(self.ca[idx:(idx+tnum)])
            
        si = np.argsort(bump_size)    
        plt.plot(bump_size[si],bca[si],color='k')
        
        
        
        
        