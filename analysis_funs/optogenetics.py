# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:02:48 2024

@author: dowel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
class opto:
    def __init__(self):
        self.name = 'opto'
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        idx =idx
        return idx
    def plot_plume(self,meta_data,df):
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        
        #pon = pd.Series.to_numpy(df['mfc2_stpt']>0)
        pon = pd.Series.to_numpy(df['instrip']>0)
        pw = np.where(pon)
        x = x-x[pw[0][0]]
        y = y-y[pw[0][0]]
        plt.figure(figsize=(8,16))
        
        yrange = [min(y), max(y)]
        xrange = [min(x), max(x)]
        
        
        # Plt plume
        pi = np.pi
        psize =meta_data['PlumeWidth']
        pa = meta_data['PlumeAngle']
        xmplume = yrange[1]/np.tan(pi*(pa/180))
        xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
        
        pan = meta_data['PlumeAngle']
        
        
        # Plt opto
        lo = meta_data['ledONy']
        loff = meta_data['ledOffy']
        yo = [lo,lo,loff,loff,lo]
        lin = meta_data['LEDinplume']
        lout = meta_data['LEDoutplume']
        ym = yrange[1]
        yplus = float(0)
        xsub_old =0
        while ym>1000:
            
            if lout and not lin:
                xo = [-500,500,500,-500, -500 ]
                plt.fill(xo,np.add(yo,yplus),color = [1,0.8,0.8])
                #plt.fill(np.multiply(-1,xo),np.add(yo,yplus),color = [1,0.8,0.8])
            yp = [yplus, yplus+1000, yplus+1000,yplus,yplus] 
            xsub = self.find_nearest(y,yplus)
            print(yplus)
            print(y[xsub])
            if yplus>0 :
                
                plt.fill(xp+x[xsub],yp,color =[0.8,0.8,0.8])
            
            else:
                plt.fill(xp,yp,color =[0.8,0.8,0.8])
            xsub_old = xsub    
            ym = ym-float(1000)
            yplus = yplus+float(1000)
            
        plt.ylim([0,1000])
        plt.xlim([-500,500])
        plt.plot(x,y,color='k')
        plt.scatter(x[pon],y[pon], color = [0.8, 0.8 ,0.2])
    def plot_traj_scatter(self,df):
        # Simple plot to show tajectory and scatter of when experiencing odour 
        # and LED
        # Useful for when there is not much information about the experiment
        plt.figure(figsize=(16,16))
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        led_on = df['led1_stpt']==0
        in_s = df['instrip']
        x,y = self.fictrac_repair(x,y)
        plt.plot(x,y,color = 'k')
        plt.scatter(x[in_s],y[in_s],color=[0.5, 0.5, 0.5])
        plt.scatter(x[led_on],y[led_on],color= [1,0.5,0.5],marker='+')
        plt.gca().set_aspect('equal')
        plt.show()
    def plot_plume_simple(self,meta_data,df):
        
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        led_on = df['led1_stpt']==0
        
        x,y = self.fictrac_repair(x,y)
        s_type = meta_data['stim_type']
        plt.figure(figsize=(16,16))
        if s_type =='plume':
        
            #pon = pd.Series.to_numpy(df['mfc2_stpt']>0)
            pon = pd.Series.to_numpy(df['instrip']>0)
            pw = np.where(pon)
            x = x-x[pw[0][0]]
            y = y-y[pw[0][0]]
            
            
            yrange = [min(y), max(y)]
            xrange = [min(x), max(x)]
            xlm = np.max(np.abs(xrange))
            
            # Plt plume
            pi = np.pi
            psize =meta_data['PlumeWidth']
            pa = meta_data['PlumeAngle']
            a_s = meta_data['act_inhib']
            if a_s=='act':
                led_colour = [1,0.8,0.8]
            elif a_s=='inhib':
                led_colour = [0.8, 1, 0.8]
            
            #xmplume = yrange[1]/np.tan(pi*(pa/180))
            
            if pa ==90:
                xp = [xrange[0], xrange[1],xrange[1], xrange[0]]
                yp = [psize/2, psize/2,-psize/2,-psize/2]
                xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            else :
                    xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
                    yp = [yrange[0], yrange[1], yrange[1],yrange[0],yrange[0]]
                    xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            #pan = meta_data['PlumeAngle']
            
             
            
            
            if meta_data['ledOny']=='all':
                lo = yrange[0]
            else:
                lo = meta_data['ledOny']
            
            if meta_data['ledOffy']=='all':
                loff = yrange[1]
                
            else:
                loff = meta_data['ledOffy']
            yo = [lo,lo,loff,loff,lo]
            if meta_data['LEDoutplume']:
                plt.fill(xo,yo,color = led_colour,alpha =0.5)
            
            if loff<yrange[1]:
                while loff<yrange[1]:
                    loff = loff+1000
                    lo = lo+1000
                    yo = [lo,lo,loff,loff,lo]
                    plt.fill(xo,yo,color = led_colour,alpha=0.5)
            
            plt.fill(xp,yp,color =[0.8,0.8,0.8])
            # Add in extra for repeated trials
            plt.plot(x[pw[0][0]:],y[pw[0][0]:],color='k')
            plt.plot(x[0:pw[0][0]],y[0:pw[0][0]],color=[0.5,0.5,0.5])
            if meta_data['LEDinplume']:
                plt.fill(xo,yo,color = led_colour,alpha= 0.5)
            
               # plt.scatter(x[led_on],y[led_on], color = [0.8, 0.8 ,0.2])
            #yxlm = np.max(np.abs(np.append(yrange,xrange)))
            #ymn = np.mean(yrange)
            plt.ylim([np.min(y),np.max(y)])
            #plt.ylim([ymn-(yxlm/2), ymn+(yxlm/2)])
            #plt.xlim([-1*(yxlm/2), yxlm/2])
        elif s_type == 'pulse':
            led = df['led1_stpt']==0
            plt.scatter(x[led],y[led],color=led_colour)
            plt.plot(x,y,color='k')
        plt.gca().set_aspect('equal')
        plt.show()
        
    def plot_plume_horizontal(self,meta_data,df):
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        
        
        x,y = self.fictrac_repair(x,y)
        
        pon = pd.Series.to_numpy(df['instrip']>0)
        pw = np.where(pon)
        x = x-x[pw[0][0]]
        y = y-y[pw[0][0]]
        
        pa = meta_data['PlumeWidth']
        plt.figure(figsize=(16,16))
        yrange = [min(y), max(y)]
        xrange = [min(x), max(x)]
        x_plm = [xrange[0], xrange[0], xrange[1],xrange[1]]
        y_plm = [-pa/2, pa/2, pa/2, -pa/2]
        
        plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])
        x_on = meta_data['ledOnx']
        x_off = meta_data['ledOffx']
        y_on = meta_data['ledOny']
        y_off= meta_data['ledOffy']
        y_stm = [y_on, y_off, y_off,y_on]
        rep_int = meta_data['RepeatInterval']
        
        a_s = meta_data['act_inhib']
        if a_s=='act':
            led_colour = [1,0.8,0.8]
        elif a_s=='inhib':
            led_colour = [0.8, 1, 0.8]
        if xrange[0]<0:
            
            xr = -np.arange(0,np.abs(xrange[0]),rep_int)
            print(xr)
            for i in xr:
                
                x_stm = [i-x_on, i-x_on,i-x_off,i-x_off]
                plt.fill(x_stm,y_stm,color=led_colour)
        if xrange[1]>0:
            
            xr = np.arange(0,np.abs(xrange[1]),rep_int)
            print(xr)
            for i in xr:
                
                x_stm = [i+x_on, i+x_on,i+x_off,i+x_off]
                plt.fill(x_stm,y_stm,color=led_colour)
        led_on = df['led1_stpt']<1
        plt.scatter(x[led_on],y[led_on],color='r')    
        plt.plot(x,y,color='k')
        
        
        plt.gca().set_aspect('equal')
        plt.show()
    
        
    def light_pulse_pre_post(self,meta_data,df):
        plt.figure(figsize=(10,10))
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        t = self.get_time(df)
        
        led = df['led1_stpt']
        led_on = np.diff(led)<0 
        led_off = np.diff(led)>0 
        lo_dx = [i+1 for i,ir in enumerate(led_on) if ir]
        loff_dx = [i+1 for i,ir in enumerate(led_off) if ir]
        if len(loff_dx)<len(lo_dx):
            lo_dx = lo_dx[:-1]
        tbef = 2 
        tdx = np.sum(t<tbef)
        ymax = 0
        for i,on in enumerate(lo_dx):
            st = on-tdx
            st_o = loff_dx[i]
            y_b = y[st:on]
            x_b = x[st:on]
            x_vec = x_b[-1]-x_b[0]
            y_vec = y_b[-1]-y_b[0]
            theta = -np.arctan(y_vec/x_vec)-np.pi
            hyp = np.sqrt(x_vec**2+y_vec**2)
            cos_thet = np.cos(theta)
            sin_thet = np.sin(theta)
            rotmat = np.array([[cos_thet, -sin_thet],[sin_thet, cos_thet]])
            plt_x = x[st:st_o]-x[on]
            plt_y = y[st:st_o]-y[on]
            xymat = np.array([plt_x,plt_y])
            
            rot_xy = np.matmul(rotmat,xymat)
            # need to flip x axis if negative
            if rot_xy[0,0]>0:
                rot_xy[0,:] = -rot_xy[0,:]
            plt.plot(rot_xy[0,:tdx],rot_xy[1,:tdx],color= [0.8,0.8,0.8])
            plt.plot(rot_xy[0,tdx+1:],rot_xy[1,tdx+1:],color='k')
            ymax = np.max(np.abs(np.append(rot_xy[1,:],ymax)))   
        plt.ylim([-ymax,ymax])
        plt.gca().set_aspect('equal')
        plt.show()
    def get_time(self,df):
        t = pd.Series.to_numpy(df['timestamp'])
        t = np.array(t,dtype='str')
        t_real = np.empty(len(t),dtype=float)
        for i,it in enumerate(t):
            tspl = it.split('T')
            tspl2 = tspl[1].split(':')
            t_real[i] = float(tspl2[0])*3600+float(tspl2[1])*60+float(tspl2[2])
        t_real = t_real-t_real[0]
        return t_real
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
    def extract_stats(self,meta_data,df):
        
        # Stats to get: med max dist from plume, med time outside plume, length traj outside,
        # med vel outside
        plume_an =  np.pi*meta_data['PlumeAngle']/180
        edge_vec = np.array([np.cos(plume_an),np.sin(plume_an)])
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        t = self.get_time(df)
        led = df['led1_stpt']

        strip_on = pd.Series.to_numpy(df['strip_thresh'])
        strip_on =np.isnan(strip_on)
        st_on = np.where(~strip_on)[0][0]
        instrip = pd.Series.to_numpy(df['instrip'])
        adapt_cent = pd.Series.to_numpy(df['adapted_center'])
        l_edge = y*np.sin((plume_an))-instrip+adapt_cent
        r_edge = y*np.sin((plume_an))+instrip+adapt_cent
        # Ignore pre period for now
        x = x[st_on:]
        y = y[st_on:]
        t = t[st_on:]
        l_edge = l_edge[st_on:]
        r_edge = r_edge[st_on:]
        instrip = instrip[st_on:].astype(int)
        led = led[st_on:]
        exits = np.where(np.diff(instrip)==-1)[0]+1
        entries = np.where(np.diff(instrip)==1)[0]+1
        
        if (len(exits)-len(entries))==1:
            exits = exits[:-1]
        
        # remove first entry and exit as will not be representative of tracking:
        # May want to make this larger in future iterations
        t_ex = exits[1:]
        t_ent = entries[1:]
        data_array = np.empty([len(t_ex),5])    
        for i,ex in enumerate(t_ex):
            en = t_ent[i]
            ys = y[ex:en]
            xs = x[ex:en]
            ts = t[ex:en]
            tled = led[ex:en]
            xz = xs-xs[0]
            yz = ys-ys[0]
            tz = ts-ts[0]
            xz = xz[:, np.newaxis]
            yz = yz[:, np.newaxis]
            xy = np.append(xz,yz,axis=1)
            edist = np.matmul(xy,edge_vec)
            ledsum = np.sum(tled)
            
            if ledsum==len(tled):
                data_array[i,0] = 0
            elif ledsum==0:
                data_array[i,0] =1 
            else :
                data_array[i,0] = 2
                
            # Max distance from plume: check for slanted/jumping plumes    
            data_array[i,1] = np.max(edist)
            # Time out of plume
            data_array[i,2] = ts[-1]-ts[0]
            # Path length
            dxy = np.diff(xy,axis =0)
            mxy = np.sqrt(np.sum(dxy**2,axis=1))
            data_array[i,3] = np.sum(mxy)
            # Median speed
            data_array[i,4] = np.median(mxy)
            
        min_time = 0.5 
        keep = data_array[:,2]>min_time
        data_array = data_array[keep,:]
        mdn_data = np.empty([3,5],dtype=float)
        mn_data = np.empty([3,5],dtype=float)
        for i in range(3):
            dx = data_array[:,0].astype(int)==i
            mn_data[i,:] = np.mean(data_array[dx,:],axis=0)
            mdn_data[i,:] = np.median(data_array[dx,:],axis=0)
        out_dict = {'all data': data_array,
                    'mean data': mn_data,
                    'median data': mdn_data
            }
        return out_dict
    
    
