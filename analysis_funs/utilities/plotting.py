import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['pdf.fonttype'] = 42
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from src.utilities import funcs
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import seaborn as sns
sns.set(font="Arial")

#inside_color = '#F7941D' # orange color used in thesis
inside_color = '#ED2224'
outside_color = '#6ACFF6'
grey_color = '#808080'
def plot_trajectory(df, axs, color = outside_color, linewidth=1):
    """
    take an axis and plot a trajectory .
    Return the axis
    """
    axs.plot(df.ft_posx, df.ft_posy, color, linewidth=linewidth)
    return axs

def plot_trajectory_odor(df, axs, color = inside_color, linewidth=1):
    """
    take an axis and plot a trajectory only where the fly is in odor.
    Return the axis
    """
    df_instrip = df.mask(df.instrip==False)
    axs.plot(df_instrip.ft_posx, df_instrip.ft_posy, color=color, linewidth=1)
    return axs

def plot_trajectory_lights(df, axs, color = 'pink'):
    """
    take an axis and plot a trajectory only where the fly is in odor.
    Return the axis
    """
    df_lights = df.mask(df.led1_stpt>0.1)
    axs.plot(df_lights.ft_posx, df_lights.ft_posy, 'red')
    return axs

def plot_vertical_edges(df, axs, width=50, color = 'grey'):
    """
    plot vertical plume edges from point of entry
    mark odor onset with red dot
    """
    x = df.ft_posx
    y = df.ft_posy
    ix0 = df.index[df.instrip==True].tolist()[0]
    x0 = x[ix0] # First point where the odor turn on
    y0 = y[ix0] # First point where the odor turn on
    axs.plot([x0-width/2, x0-width/2], [min(y), max(y)], color)
    axs.plot([x0+width/2, x0+width/2], [min(y), max(y)], color)
    axs.plot(x0, y0, 'ro')
    return axs

def plot_45_edges(df, axs, width=50, color='grey'):
    """
    plot 45 degreee plume edges from point of entry
    mark odor onset with red dot
    """
    x = df.ft_posx
    y = df.ft_posy
    ix0 = df.index[df.instrip==True].tolist()[0]
    x0 = x[ix0] # First point where the odor turn on
    y0 = y[ix0] # First point where the odor turn on
    maxy = max(y)
    axs.plot([-width/2, maxy-width/2]+x0, [0,maxy]+y0, color='black')
    axs.plot([width/2, maxy+width/2]+x0, [0,maxy]+y0, color='black')
    axs.plot(x0, y0, 'ro')
    return axs

def plot_jumping_edges_old(df, axs, width=50, color = 'grey'):
    """
    plt edges of a plume that jumps.
    """
    try:
        df.left_border
    except:
        df['left_border'] = df['left_edge']
        df['right_border'] = df['right_edge']
    # where does the odor turn on
    ix0 = df[df.mfc2_stpt>0.01].index.to_list()[0]
    x0 = df.ft_posx.to_numpy()[ix0]
    y0 = df.ft_posy.to_numpy()[ix0]
    # find all points in the dataframe where the plume jumps
    ix, _ = sg.find_peaks(np.abs(df['left_border'].diff()))
    ixs = np.append([ix0],ix, len(df))
    for i in np.arange(len(ixs)-1):
        start_ix = ixs[i]
        end_ix = ixs[i+1]
        df_sub = df.iloc[start_ix:end_ix]
        # plot left border
        x_left = [df_sub.left_border.iloc[0], df_sub.left_border.iloc[0]]+x0
        y_left = [df_sub.ft_posy.iloc[0], df_sub.ft_posy.iloc[-1]]
        # plot right border
        x_right = [df_sub.right_border.iloc[0], df_sub.right_border.iloc[0]]+x0
        y_right = [df_sub.ft_posy.iloc[0], df_sub.ft_posy.iloc[-1]]
        axs.plot(x_left, y_left, 'grey')
        axs.plot(x_right, y_right, 'silver')
        plt.plot(x0,y0, 'o')
    return axs

def plot_jumping_edges(df, axs, width=50, color = 'grey'):
    """
    plt edges of a plume that jumps.
    """
    _, di, do=funcs.inside_outside(df)
    strip_width = width
    print(di.keys)
    first = di[list(di.keys())[0]]
    exit_x = first.ft_posx.iloc[-1]
    out = do[list(di.keys())[0]+1]
    if out.ft_posx.iloc[0]<exit_x:
        border = exit_x+strip_width
    else:
        border = exit_x-strip_width
    axs.plot([exit_x, exit_x],[first.ft_posy.iloc[0], first.ft_posy.iloc[-1]], 'k', alpha=0.5)
    axs.plot([border, border],[first.ft_posy.iloc[0], first.ft_posy.iloc[-1]], 'k', alpha=0.5)

    for i in np.arange(1,len(di)):
        temp_in = di[list(di.keys())[i]]
        temp_out = do[list(di.keys())[i]-1]
        if temp_out.ft_posx.iloc[-1]>temp_in.ft_posx.iloc[0]:
            border = temp_in.ft_posx.iloc[0]-strip_width
        else:
            border = temp_in.ft_posx.iloc[0]+strip_width
        axs.plot([temp_in.ft_posx.iloc[0], temp_in.ft_posx.iloc[0]],[temp_in.ft_posy.iloc[0], temp_in.ft_posy.iloc[-1]], 'k', alpha=0.5)
        axs.plot([border, border],[temp_in.ft_posy.iloc[0], temp_in.ft_posy.iloc[-1]], 'k', alpha=0.5)
    return axs

def plot_plume_corridor(ax, x0=0,y0=0,width=50, height=1000, type='constant', cmap = cm.Greys):
    """
    function for plotting a vertical plume shaded by colormap with options for plotting a gradient or constant plume
    """ 

    w = width/2

    # plot increasing gradient strip
    if type == 'increasing':
        ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1),cmap=cmap,extent=[x0-w, x0+w, y0, y0+height], origin='lower')
    # plot decreasing gradient strip
    elif type == 'decreasing':
       ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1),cmap=cmap,extent=[x0-w, x0+w, y0, y0+height], origin='upper')
    # plot constant plume
    elif type == 'constant':
        color = cmap(np.arange(0,1000, 1))[85]
        cmap = colors.ListedColormap(color)
        ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), cmap = cmap, extent=[x0-w, x0+w, y0, y0+height], origin='lower')
    else:
        print('invalid plume type entered.  Options are increasing, decreasing and constant')
        return
    
    return ax

def plot_insides(di, axs, color = 'orange', alpha=0.2, offset = [0,0]):
    """
    plot all inside trajectories aligned to the point of entry
    """
    for key in list(di.keys()):
        x = di[key].ft_posx
        y = di[key].ft_posy
        x0 = x.iloc[0]
        y0 = y.iloc[0]
        x = x - x0 + offset[0]
        y = y - y0 + offset[1]
        axs.plot(x, y, color, alpha=alpha)
    return axs

def plot_outsides(do, axs, color = 'black', alpha=0.2, offset = [0,0]):
    """
    plot all inside trajectories aligned to the point of entry
    """
    for key in list(do.keys()):
        x = do[key].ft_posx
        y = do[key].ft_posy
        x0 = x.iloc[0]
        y0 = y.iloc[0]
        x = x - x0 + offset[0]
        y = y - y0 + offset[1]
        axs.plot(x, y, color, alpha=alpha)
    return axs

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments

# Interface to LineCollection:

def colorline(axs, x, y, z=None, segmented_cmap=True, cmap=plt.get_cmap('twilight'), norm=plt.Normalize(-np.pi, np.pi), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    def make_segmented_cmap():
        import matplotlib.colors as col
        white = '#ffffff'
        black = '#000000'
        red = '#ff0000'
        blue = '#0000ff'
        anglemap = col.LinearSegmentedColormap.from_list(
            'anglemap', [black, red, white, blue, black], N=256, gamma=1)
        return anglemap
    if segmented_cmap:
        cmap = make_segmented_cmap()

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    # axs.set_xlim(x.min()-10, x.max()+10)
    # axs.set_ylim(y.min()-10, y.max()+10)
    axs.add_collection(lc)
    return axs

def colorline_specify_colors(axs, x, y, colors, linewidth=3, alpha=1.0):
        '''
        Plot a colored line with coordinates x and y
        explicitly specify colors
        '''
        segments = make_segments(x, y)
        lc = LineCollection(segments, colors=colors, linewidth=linewidth, alpha=alpha)
        axs.set_xlim(x.min()-10, x.max()+10)
        axs.set_ylim(y.min()-10, y.max()+10)
        axs.add_collection(lc)
        return axs

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, remove_x_labels=False, color='k', label=''):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    x=-x # need to flip angle to match our coordinate system where negative angles are pointing left, positive angles are pointing right

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)


    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area

    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    #print(bins, len(bins), len(radius))

    x = []
    y = []
    angles = (bins[0:-1]+bins[1:])/2
    for i in np.arange(len(radius)):
        x.append(radius[i]*np.cos(angles[i]))
        y.append(radius[i]*np.sin(angles[i]))
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    r_avg = np.sqrt(x_avg**2+y_avg**2)
    ang_avg = np.arctan2(y_avg, x_avg)

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, facecolor=color, fill=True, alpha=0.5, linewidth=0.0)

    # average angle
    #ang_avg = funcs.circmean(x)
    # ax.plot([ang_avg, ang_avg], [0,max(radius)], color=color, linewidth=2)
    # ax.plot(ang_avg, max(radius), 'o',color=color, linewidth=2)

    ax.plot([ang_avg, ang_avg], [0,r_avg], color=color, linewidth=2, label=label)
    ax.plot(ang_avg, r_avg, 'o',color=color, linewidth=2, markersize=3)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4], labels=[0,-45,-90,-135,180,135,90,45])

    # Remove xlabels
    if remove_x_labels:
        ax.set_xticks([])

    return ang_avg, r_avg

def circular_hist2(ax, x, bins=16, density=True, offset=0, gaps=True,
                    edgecolor='w', facecolor=[0.7]*3, alpha=0.7, lw=0.5):
    """
    Produce a circular histogram of angles on ax.
    From: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').
    x : array
        Angles to plot, expected in units of radians.
    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.
    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.
    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.
    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.
    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.
    bins : array
        The edges of the bins.
    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        #bins = np.linspace(-np.pi, np.pi, num=bins+1)
        bins = np.linspace(0, 2*np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, width=widths, #align='edge',
                     edgecolor=edgecolor, fill=True, linewidth=lw, facecolor=facecolor,
                    alpha=alpha)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
        #ax.tick_params(which='both', axis='both', size=0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # theta increasing clockwise

    return n, bins, patches

def plot_linear_best_fit(axs, x, y):
    axs.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'k')
    return axs

def paired_plot(axs,obs1,obs2,color1='r',color2='b',line_color='k', log=False,
    alpha=0.3, mean_line=True, scatter_scale=0.05, print_stat=False,
    mean_std=True, markersize=5, indiv_pts=False, error=True, indiv_markers=True):
    """
    create a paired plot between observation 1 and observation 2

    perform statistics
    """
    if len(obs1) != len(obs2):
        print('observations not paired')
        return
    print(len(obs1), ' observations')

    # stats -- test normality
    if log:
        obs_stat1 = np.log10(obs1)
        obs_stat2 = np.log10(obs2)
    else:
        obs_stat1 = obs1
        obs_stat2 = obs2
    p_values = [stats.shapiro(obs1)[1], stats.shapiro(obs2)[1]]
    if max(p_values)>0.05:
        print('paired data is not normal, performing Wilcoxon signed rank test:')
        stat = stats.wilcoxon(obs_stat1, obs_stat2)
        print('Wilcoxon test results: ', stat)
    else:
        print('data is normal, performing paired t test')
        stat = stats.ttest_rel(obs_stat1, obs_stat2)
        print('paired t test results: ', stat)

    # plot data
    if indiv_pts:
        for i in np.arange(len(obs1)):
            x = np.array([0,1])+np.random.normal(loc=0.0, scale=scatter_scale, size=2)
            y = np.array([obs1[i], obs2[i]])
            axs.plot(x,y,color=line_color, alpha=alpha, linewidth=0.5)
            if indiv_markers:
                axs.plot(x[0],y[0],marker='o',color=color1, alpha=1, markeredgewidth=0, markersize=markersize)
                axs.plot(x[1],y[1],marker='o',color=color2, alpha=1, markeredgewidth=0, markersize=markersize)
    if mean_line:
        axs.plot([0,1], [np.mean(obs1), np.mean(obs2)], color=line_color)
    if mean_std:
        axs.plot([-0.1,0.1], [np.mean(obs1), np.mean(obs1)], color=color1, linewidth=2)
        axs.plot([0.9,1.1], [np.mean(obs2), np.mean(obs2)], color=color2, linewidth=2)
    if error:
        obs1_error = stats.t.interval(0.95, len(obs1)-1, loc=np.mean(obs1), scale=stats.sem(obs1))
        obs2_error = stats.t.interval(0.95, len(obs2)-1, loc=np.mean(obs2), scale=stats.sem(obs2))
        axs.plot([0,0], obs1_error, color=color1,linewidth=2)
        axs.plot([1,1], obs2_error, color=color2,linewidth=2)
    if log:
        axs.set_yscale('log')
    if print_stat:
        axs.text(0.5, np.max([obs1,obs2]), "{:.5f}".format(stat[1]))
    return axs

def despine(axs):
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    return axs

from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from matplotlib import rcParams
import matplotlib
from matplotlib.ticker import StrMethodFormatter
from scipy import stats

import os, random, math


# redorange = (.96, .59, .2)
redorange = (244/255., 118/255., 33/255.)
ddorange = (102/255., 51/255., 0/255.)
borange = (1, .5, 0)
purple = (0.5, 0.2, .6)
lpurple = (0.6, 0.3, .8)
dpurple = (0.2, 0.2, .6)
dblue = (0, .2, .4)
nblue = (.11, .27, .49)
blue = (0.31, 0.47, 0.69)
bblue = (0.51, 0.67, 0.89)

dgreen = (0.3, 0.55, 0.22)
ddgreen = (0.1, 0.35, 0.02)
bgreen = (.34, .63, .56)
dred_pure = (.4, 0,0)
dblue_pure = (0,0,.5)
blue_green = (0,.3,.8)
blue_red = (0,0,1)
green_red = (.7,.7,0)
brown = '#D2691E'

odor_color = redorange
air_color = bblue

yellow = (.95, .74, 22)
nyellow = (.3, .3, 0)

magenta = (175/255.,101/255.,168/255.)
dmagenta = (127/255.,22/255.,122/255.)
orange = (244/255., 135/255., 37/255.)
dorange = (232/255., 143/255., 70/255.)
red = (230/255., 0/255., 0/255.)
dred = (131/255., 35/255., 25/255.)
dimred = (221/255., 49/255., 49/255.)
blue = (37/255.,145/255.,207/255.)
dblue = (27/255.,117/255.,187/255.)
lgreen = (128/255.,189/255.,107/255.)
rose = (194/255.,52/255.,113/255.)
pink = (221/255., 35/255., 226/255.)
green = (0.5, 0.75, 0.42)
ngreen = (0.01, 0.3, 0.01)
brown = (71/255., 25/255., 2/255.)

black = (0., 0., 0.)
grey9 = (.1, .1, .1)
grey8 = (.2, .2, .2)
grey7 = (.3, .3, .3)
grey6 = (.4, .4, .4)
grey5 = (.5, .5, .5)
grey4 = (.6, .6, .6)
grey3 = (.7, .7, .7)
grey2 = (.8, .8, .8)
grey15 = (.85, .85, .85)
grey1 = (.9, .9, .9)
grey05 = (.95, .95, .95)
grey03 = (.97, .97, .97)
white = (1., 1., 1.)
lgrey = grey3
grey = grey5
dgrey = grey7

def create_linear_cm(RGB=[244,118,33]):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(RGB[0]/255., 1, N)[::-1]
    vals[:, 1] = np.linspace(RGB[1]/255., 1, N)[::-1]
    vals[:, 2] = np.linspace(RGB[2]/255., 1, N)[::-1]
    newcmp = ListedColormap(vals)
    return newcmp

Magentas = create_linear_cm(RGB=[76,0,76])
Roses = create_linear_cm(RGB=[178,0,76])
# Pinks = create_linear_cm(RGB=[221,35,226])
Pinks = create_linear_cm(RGB=[203,45,211])
Reds = create_linear_cm(RGB=[204,30,30])
Browns = create_linear_cm(RGB=[71,25,2])

colors_black_red = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
cm_black_red = LinearSegmentedColormap.from_list(
        "Custom", colors_black_red, N=200)

mb_color = {
    'g5_mb':dmagenta,
    'g4_mb':green,
    'g3_mb':rose,
    'g2_mb':blue,
    'bp1_mb':orange,
    'bp2_mb':brown,
    'b1_mb':orange,
    'b2_mb':brown
}
