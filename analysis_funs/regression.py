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
    def rebaseline(self,span=500,plotfig=False):
        y = self.ca
        y[np.isnan(y)] = 0
        
        frac = float(span)/np.max(self.ts)
        frac = min(frac,1)
        lowess = sm.nonparametric.lowess 
        yf = lowess(y,np.arange(0,len(y)),frac=frac)
        self.ca_original = y
        self.ca = y-yf[:,1]
        if plotfig:
            plt.figure()
            plt.plot(y)
            plt.plot(yf[:,1])
            plt.show()
    def set_up_regressors(self,regchoice,cirftau =[0.3,0.01]):
        #Cirf is very approximate and should be verified
        
        xs = np.shape(self.ft2['instrip'])
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
            elif r == 'angular velocity abs':
                x = np.abs(pd.Series.to_numpy(self.ft2['ang_velocity'].copy()))
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
            elif r == 'translational vel':
                x1 = self.ft2['ft_posx'].copy()
                y1 = self.ft2['ft_posy'].copy()
                dx = np.diff(x1)
                dy = np.diff(y1)
                trans_diff = np.sqrt(dx**2+dy**2)
                x = np.append([0],trans_diff)
                xp = np.percentile(np.abs(x),99)
                x[x>xp] = xp
                x[x<-xp] = -xp
            elif r == 'stationary':
                # x1 = pd.Series.to_numpy(self.ft2['x_velocity'].copy())
                # x2 = pd.Series.to_numpy(self.ft2['y_velocity'].copy()) 
                # x = x1==0&x2==0
                x1 = self.ft2['ft_posx'].copy()
                y1 = self.ft2['ft_posy'].copy()
                dx = np.diff(x1)
                dy = np.diff(y1)
                trans_diff = np.sqrt(dx**2+dy**2)
                x2 = np.append([0],trans_diff)
                xp = np.percentile(np.abs(x2),1)
                x = np.zeros_like(x2)
                x[np.abs(x2)<0.05] = 1

            
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
    def run_pearson(self,regchoice):
        print('Determining regressors')
        regmatrix, regmatrix_preconv = self.set_up_regressors(regchoice)
        self.regmatrix = regmatrix
        self.regmatrix_preconv = regmatrix_preconv.copy()
        
        y = self.ca
        # Iterate through regressors getting pearson corr
        rhos = np.zeros(len(regchoice))
        ps = np.zeros(len(regchoice))
        print(len(regchoice))
        print(np.shape(rhos))
        for r in range(len(regchoice)):
            st = stats.pearsonr(y,regmatrix[:,r])
            rhos[r] = st.statistic
            ps[r] = st.pvalue
        self.pearson_rho = rhos
        self.pearson_p = ps
        
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
        plt.plot(self.ts,self.ft2['instrip']*np.max(self.ca),color=[0.2,0.2,1])
        plt.xlabel('Time (s)')
        plt.ylabel('dF/F')
        plt.show()
        
    def plot_flur_w_regressors(self,regchoice,cacol='k'):
        #plt.figure(figsize=(18,8))
        plt.plot(self.ts,self.ca,color=cacol)
        R = self.regmatrix_preconv[:,:-1]
        for ir, r in enumerate(regchoice):
            rdx = np.in1d(self.regchoice,r)
            y = R[:,rdx]
            y = y/np.max(y)
            #y = y*np.max(self.ca)
            plt.plot(self.ts,y-ir-1,color='k')
            
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
        
    def simple_trajectory(self,dx):
        x = self.ft2['ft_posx']
        y = self.ft2['ft_posy']
        x,y = self.fictrac_repair(x,y)
        acv = self.ft2['instrip'].to_numpy()
        
        inplume = acv>0
        st  = np.where(inplume)[0][0]
        x = x-x[st]
        y = y-y[st]
        
        x = x[dx]
        y = y[dx]
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        #ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
        ax.fill([-5,-5,5,5],[0,max(y),max(y),0],color=[0.7,0.7,0.7])
        mnx = min(x)
        mxx = max(x)
        if mnx<-210:
            prange = mnx
            m_mod = np.mod(mnx,-210)
            m_add = mnx-m_mod
            ax.fill([-5,-5,5,5]+m_add,[0,max(y),max(y),0],color=[0.7,0.7,0.7])
        if mxx>210:
            prange = mxx
            m_mod = np.mod(mxx,210)
            m_add = mxx-m_mod
            ax.fill([-5,-5,5,5]+m_add,[0,max(y),max(y),0],color=[0.7,0.7,0.7])
            
        acv = acv[dx]
        isd =np.diff(acv)
        st = np.where(isd>0)[0]+1
        se = np.where(isd<0)[0]+1
        
        if se[0]<st[0]:
            print('True')
            st = np.append(0,st)
        if st[-1]>se[-1]:
            se = np.append(se,len(x))
        plt.plot(x[:st[0]],y[:st[0]],color='k')
        plt.plot(x[se[-1]-1:],y[se[-1]-1:],color='k')
        print(st)
        print(se)
        for i,s in enumerate(st):
            plt.plot(x[(st[i]-1):se[i]],y[(st[i]-1):se[i]],color='r')
            if i<(len(st)-1):
                plt.plot(x[(se[i]-1):st[i+1]],y[(se[i]-1):st[i+1]],color='k')
        
        
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        plt.ylim([min(y), max(y)])
    def example_trajectory(self,cmin=0,cmax=1):
        colour = self.ca
        x = self.ft2['ft_posx']
        y = self.ft2['ft_posy']
        
        
        
        x,y = self.fictrac_repair(x,y)
        acv = self.ft2['instrip'].to_numpy()
        inplume = acv>0
        st  = np.where(inplume)[0][0]
        x = x-x[st]
        y = y-y[st]
        
        xrange = np.max(x)-np.min(x)
        yrange = np.max(y)-np.min(y)
        
        mrange = np.max([xrange,yrange])+100
        y_med = yrange/2
        x_med = xrange/2
        ylims = [y_med-mrange/2, y_med+mrange/2]
   
        xlims = [x_med-mrange/2, x_med+mrange/2]

        c_map = plt.get_cmap('coolwarm')
        if cmin==cmax:
            cmax = np.round(np.percentile(colour[~np.isnan(colour)],97.5),decimals=1)
        cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        #x = x-x[0]
       # y = y -y[0]
        plt.rcParams['pdf.fonttype'] = 42 
        plt.rcParams['ps.fonttype'] = 42 
        fig = plt.figure(figsize=(15,15))
        
        ax = fig.add_subplot(111)
        #ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
        ax.fill([-5,-5,5,5],[0,max(y),max(y),0],color=[0.7,0.7,0.7])
        mnx = min(x)
        mxx = max(x)
        if mnx<-210:
            prange = mnx
            m_mod = np.mod(mnx,-210)
            m_add = mnx-m_mod
            ax.fill([-5,-5,5,5]+m_add,[0,max(y),max(y),0],color=[0.7,0.7,0.7])
            ax.fill([-5,-5,5,5]-np.array(210),[0,max(y),max(y),0],color=[0.7,0.7,0.7])
        if mxx>210:
            prange = mxx
            m_mod = np.mod(mxx,210)
            m_add = mxx-m_mod
            ax.fill([-5,-5,5,5]+np.array(210),[0,max(y),max(y),0],color=[0.7,0.7,0.7])
            ax.fill([-5,-5,5,5]+m_add,[0,max(y),max(y),0],color=[0.7,0.7,0.7])
        
        for i in range(len(x)-1):
            ax.plot(x[i:i+2],y[i:i+2],color=c_map_rgb[i+1,:3])
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel('x position (mm)')
        plt.ylabel('y position (mm)')
        plt.title('Flur range 0 - ' + str(cmax))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    def example_trajectory_jump(self,cmin=0,cmax=1,xcent= 0):    
        colour = self.ca
        x = self.ft2['ft_posx']
        y = self.ft2['ft_posy']
        jumps = self.ft2['jump']
        jumps = jumps-np.mod(jumps,3)
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]+1
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        
        x,y = self.fictrac_repair(x,y)
        acv = self.ft2['instrip'].to_numpy()
        inplume = acv>0
        st  = np.where(inplume)[0][0]
        x = x-x[st]
        y = y-y[st]
        
        xrange = np.max(x)-np.min(x)
        yrange = np.max(y)-np.min(y)
        
        mrange = np.max([xrange,yrange])+100
        y_med = yrange/2
        x_med = xrange/2
        ylims = [y_med-mrange/2, y_med+mrange/2]
   
        xlims = [x_med-mrange/2, x_med+mrange/2]

        c_map = plt.get_cmap('coolwarm')
        if cmin==cmax:
            cmax = np.round(np.percentile(colour[~np.isnan(colour)],97.5),decimals=1)
        cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        #x = x-x[0]
       # y = y -y[0]
        plt.rcParams['pdf.fonttype'] = 42 
        plt.rcParams['ps.fonttype'] = 42 
        fig = plt.figure(figsize=(15,15))
        
        ax = fig.add_subplot(111)
        #ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
        
        yj = y[jn].to_numpy()
        yj = np.append(yj,y.to_numpy()[-1])
        plt.fill()
        tj = 0
        x1 = xcent+5+tj
        x2 = xcent-5+tj
        y1 = 0
        y2 = yj[0]
        xvec = np.array([x1,x2,x2,x1])
        yvec = [y1,y1,y2,y2]
        
        cents = [-630,-420,-210, 0,210,420,630]
        
        plt.fill(xvec,yvec,color=[0.7,0.7,0.7])
        for c in cents:
            plt.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
        for i,j in enumerate(jn):
            tj = jumps[j]
            x1 = xcent+5+tj
            x2 = xcent-5+tj
            y1 = yj[i]
            y2 = yj[i+1]
            xvec = np.array([x1,x2,x2,x1])
            yvec = [y1,y1,y2,y2]
            for c in cents:
                plt.fill(xvec+c,yvec,color=[0.7,0.7,0.7])

        
        for i in range(len(x)-1):
            ax.plot(x[i:i+2],y[i:i+2],color=c_map_rgb[i+1,:3])
        #plt.scatter(x[inplume],y[inplume],color='b')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel('x position (mm)')
        plt.ylabel('y position (mm)')
        plt.title('Flur range 0 - ' + str(cmax))
        x1 = np.min(x)-10
        x2 = np.max(x)+10
        plt.xlim([x1,x2])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
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
    def plot_mean_flur(self,alignment,tbef=5,taf=5,output=False,plotting=True):
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
        
    def mean_traj_nF_jump(self,ca,plotjumps=False,cmx=False,offsets=20):
        ft2 = self.ft2
        pv2 = self.pv2
        
        jumps = ft2['jump']
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        #x,y = self.fictrac_repair(x,y)
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1 
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
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
            dt.append(times[tdx[-1]]-times[sub_dx])
        this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]
        
        # Initialise arrays
        inplume_traj = np.zeros((100,len(this_j),2))
        outplume_traj = np.zeros((100,len(this_j),2))
        inplume_amp = np.zeros((100,len(this_j)))
        outplume_amp = np.zeros((100,len(this_j)))
        side_mult = side*-1
        x = x*side_mult
        amp = ca
        off = 0
        for i,j in enumerate(this_j):
            print(i)
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            # in plume    
            ipdx = np.arange(ents[ie],sub_dx,step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[-1]
            ip_y = ip_y-ip_y[-1]
            new_time = np.linspace(0,max(old_time),100)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            a_int = np.interp(new_time,old_time,amp[ipdx])
            inplume_traj[:,i,0] = x_int
            inplume_traj[:,i,1] = y_int
            inplume_amp[:,i] = a_int
            
            # out plume
            ipdx = np.arange(sub_dx,ents[t_ent],step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[0]
            ip_y = ip_y-ip_y[0]
            new_time = np.linspace(0,max(old_time),100)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            a_int = np.interp(new_time,old_time,amp[ipdx])
            outplume_traj[:,i,0] = x_int
            outplume_traj[:,i,1] = y_int
            outplume_amp[:,i] = a_int
            
            if plotjumps:
                tj = np.append(inplume_traj[:,i,:],outplume_traj[:,i,:],axis=0)
                tjca = np.append(inplume_amp[:,i],outplume_amp[:,i],axis=0)
                self.jump_heat(tj,tjca,xoffset=off,set_cmx=cmx)
            off = off+offsets
            
            
        inmean_traj = np.mean(inplume_traj,axis=1)
        outmean_traj = np.mean(outplume_traj,axis=1)
        inmean_amp = np.mean(inplume_amp,axis=1)
        outmean_amp = np.mean(outplume_amp,axis=1)
        traj = np.append(inmean_traj,outmean_traj,axis=0)
        tca = np.append(inmean_amp,outmean_amp,axis=0)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        return traj, tca
    def jump_heat(self,traj,ca,xoffset,set_cmx=False):
        
        yrange = [np.min(traj[:,1]),np.max(traj[:,1])]
        xfl = np.array([-10,0,0,-10,-10])+xoffset
        yfl = np.array([yrange[0],yrange[0],0,0,yrange[0]])
        yfl2 = np.array([0,0,yrange[1],yrange[1],0])
        plt.fill(xfl,yfl,color=[0.7,0.7,0.7])
        plt.fill(xfl-3,yfl2,color=[0.7,0.7,0.7])
        plt.plot(np.array([0,0])+xoffset,[yrange[0],0],color='k',linestyle='--')
        plt.plot(np.array([-3,-3])+xoffset,[yrange[1],0],color='k',linestyle='--')
        
        colour = ca
        if set_cmx==False:
            cmx = np.max(np.abs(ca))
        else:
            cmx = set_cmx
        c_map = plt.get_cmap('coolwarm')
        cnorm = mpl.colors.Normalize(vmin=-cmx, vmax=cmx)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        
        for i in range(len(ca)-1):
            x = traj[i:i+2,0]
            y = traj[i:i+2,1]
            #ca = np.mean(ca[i:i+2])
            plt.plot(x+xoffset,y,color=c_map_rgb[i,:])
    def mean_traj_heat_jump(self,CA,xoffset=0,set_cmx =False,cmx=1):
        traj,ca = self.mean_traj_nF_jump(CA)
        self.jump_heat(traj,ca,xoffset)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
    def mean_traj_nF(self,use_rebase = True,tnstring='0_fsbtn'):
        """
        Function outputs mean trajectory of animal entering and exiting the plume
        alongside the mean fluorescence

        Returns
        -------
        None.

        """
        plume_centres = np.array([0,210,420])
        if use_rebase:
            ca = self.ca
        else:
            ca = self.pv2[tnstring]
            
        ft2 = self.ft2
        pv2 = self.pv2
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fictrac_repair(x,y)
        expst = np.where(ins==1)[0][0]
        x = x-x[expst]
        y = y-y[expst]
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1
        ents_O = ents.copy()
        exts_O = exts.copy()
        ents = ents[1:]
        exts = exts[1:]
        print(ents,exts)
        # Need to pick a side
        if len(ents)>len(exts):
            ents = ents[:-1]
        
        ent_x = np.round(x[ents])
        ex_x = np.round(x[exts])
        sides = np.zeros(len(ent_x))
        plume_centre = np.zeros(len(ent_x))
        try:
            jumps = self.ft2['jump'].to_numpy()
        except :
            jumps = np.zeros_like(x)
        
        jumps = jumps-np.mod(jumps,3)
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]+1
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        
        ent_j = np.round(jumps[ents])
        
        
        # 1 -1 indicates sides of entry/exit
        # -0.9 0.9 indicates crossing over from left and right
        for i,x1 in enumerate(ent_x):
            x2 = ex_x[i]
            pcd = plume_centres-np.abs(x1)+np.abs(ent_j[i])
            
            pi = np.argmin(np.abs(pcd))
            pc = (plume_centres[pi]+ent_j[i] )*np.sign(x1)
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
            print('TPC', t_pc[i])
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
            
            if max(old_time)<20 or dx2[-1]-dx2[0]<20 :
                #print(max(old_time))
                continue
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x1)
            y_int = np.interp(new_time,old_time,y1)
            ca_int = np.interp(new_time,old_time,ca1)
            # plt.figure()
            # plt.plot(new_time,ca_int,color='r')
            # plt.plot(old_time,ca1,color='k')
            mn = max(old_time)
            trajs[:50,0,i] = x_int
            trajs[:50,1,i] = y_int
            Ca[:50,i] = ca_int
            
            #Interpolate onto timebase: in plume
            old_time = dx2-dx2[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x2)
            y_int = np.interp(new_time,old_time,y2)
            ca_int = np.interp(new_time,old_time,ca2)
            # plt.figure()
            # plt.plot(new_time+mn,ca_int,color='r',linestyle='--')
            # plt.plot(old_time+mn,ca2,color='k',linestyle='--')
            # plt.title(str(dx1[0]) +' -' + str(dx2[-1]))
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
        
        
        
        
        