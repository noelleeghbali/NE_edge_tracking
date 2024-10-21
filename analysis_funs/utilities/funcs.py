import pickle
import os
import pandas as pd
import rdp
import numpy as np
import copy
import scipy.signal as sg
import random
from scipy import stats, interpolate
import matplotlib.pyplot as plt
import numpy.fft as fft
from rdp import rdp
from scipy.optimize import curve_fit



def save_obj(obj, name):
    """
    Pickle an object

    save something of interest

    Parameters
    ----------
    obj:
        obj to be saved
    name: str
        file location, end with .p or .pickle
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Load pickle

    load pickled object

    Parameters
    ----------
    name: str
        file location of picle

    Returns
    -------

    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def read_log(fileloc):
    """
    Read experimental log file

    For a closed-loop log file, read in all columns into a dataframe and convert
    the timestamp to seconds.

    Parameters
    ----------
    fileloc: str
        file location containing the .log file

    Returns
    -------
    df: Pandas DataFrame
        Dataframe of .log file
    """
    if fileloc.endswith('.log'):
        file_path = fileloc
    # else:
    #     for file in os.scandir(fileloc):
    #         if file.name.endswith('.log'):
    #             file_path = os.path.join(datafol, file.name)
    df = pd.read_table(file_path, delimiter='[,]', engine='python')
    #data = pd.xread_csv(logfile, sep='[--,]', engine='python')
    new = df["timestamp -- motor_step_command"].str.split("--", n = 1, expand = True)
    df["timestamp"]= new[0]
    df["motor_step_command"]=new[1]
    df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
    df['seconds'] = 3600*df.timestamp.dt.hour+60*df.timestamp.dt.minute+df.timestamp.dt.second+10**-6*df.timestamp.dt.microsecond
    return df

def open_log_edit(logfile):
    """
    Read experimental log file.

    This variant is needed to read some of the EPG-silencing log files.

    For a closed-loop log file, read in all columns into a dataframe and convert
    the timestamp to seconds.

    Parameters
    ----------
    fileloc: str
        file location containing the .log file

    Returns
    -------
    df: Pandas DataFrame
        Dataframe of .log file
    """
    names = ['timestamp -- motor_step_command','mfc1_stpt','mfc2_stpt','mfc3_stpt','led1_stpt','led2_stpt','sig_status','ft_posx','ft_posy','ft_frame','ft_error','ft_roll','ft_pitch','ft_yaw','ft_heading','instrip','left_edge', 'right_edge']
    df = pd.read_table(logfile, delimiter='[,]', names=names,engine='python')
    #data = pd.xread_csv(logfile, sep='[--,]', engine='python')
    new = df["timestamp -- motor_step_command"].str.split("--", n = 1, expand = True)
    df["timestamp"]= new[0]
    df["motor_step_command"]=new[1]
    df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
    df['seconds'] = 3600*df.timestamp.dt.hour+60*df.timestamp.dt.minute+df.timestamp.dt.second+10**-6*df.timestamp.dt.microsecond
    df = df.drop(0).reset_index()
    df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    df['instrip'] = np.where(df['mfc2_stpt']>0.01, True, False)
    return df

def path_length(x, y):
    """
    Calculate path length of a trajectory.

    For a continuous (x,y) trajectory, calculates a cumulative pathlength and a
    total pathlength.  Will calculate length in units provided

    Parameters
    ----------
    x: array-like
        x position
    y: array-like
        y position

    Returns
    -------
    lt: float
        cumulative pathlenth over trajectory
    L: float
        total pathlength of the trajectory
    """
    n = len(x)
    lv = [np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range (1,n)]
    lt = np.cumsum(lv)
    L = sum(lv)
    return lt, L

def inside_outside(data):
    """
    Split a trajectory into inside and outside segments

    For an experiment dataframe, breaks the trajectory apart into inside and
    outside components.

    Parameters
    ----------
    data: Pandas dataframe
        experimental dataframe. dataframe must contain 'instip' field

    Returns
    -------
    di: dict
        dictionary of dataframes, each of which is an inside trajectory
    do: dict
        dictionary of dataframes, each of which is an outside trajtory
    d: dict
        di and do interleaved
    """
    di = dict()
    do = dict()
    d = dict([*data.groupby(data['instrip'].ne(data['instrip'].shift()).cumsum())])
    for bout in d:
        if d[bout].instrip.any():
            di[bout]=d[bout]
        else:
            do[bout]=d[bout]
    return d, di, do

def dict_stops(df):
    """
    create a dict of dataframes for periods of moving and stopping
    """
    _,df = find_stops(df)
    df['stop'] = df.stop.fillna(0)
    dstop = dict()
    dmove = dict()
    d = dict([*df.groupby(df['stop'].ne(df['stop'].shift()).cumsum())])
    for bout in d:
        if d[bout].stop.any():
            dmove[bout]=d[bout]
        else:
            dstop[bout]=d[bout]
    return d, dmove, dstop

def find_stops(df, roll_time=0.5, speed_thresh=2.0, plot=False):
    """
    Find where the fly stops

    For an experiment dataframe, find epochs where the animal stops.

    Parameters
    ----------
    df: Pandas DataFrame
        experimental dataframe.
    ball_diameter: float
        diameter of foam ball
    roll_time: float
        time in s over which to average velocity
    speed_thresh: float
        speed in mm/s below which the animal is stopped

    Returns
    -------
    stops: Pandas DataFrame
        stops
    """
    del_t = np.mean(np.diff(df.seconds))
    effective_rate = 1/del_t
    roll_pts = int(np.round(roll_time/del_t))
    df = calculate_speeds(df)
    df['roll']=df['speed'].rolling(roll_pts).mean()
    df['stop'] = np.where(df['roll']>speed_thresh, 1, np.nan)
    if plot:
        fig, axs = plt.subplots(1,1)
        axs.plot(df.speed)
        axs.plot(df.stop)
    stops = df[df['roll'] < speed_thresh].groupby((df['roll'] >speed_thresh).cumsum())
    return stops, df

def return_to_edge(df):
    """
    determine if fly returns to the edge
    """
    xpos = df.ft_posx.to_numpy()
    if np.abs(xpos[-1]-xpos[0])<1:
        return True
    else:
        return False

def conv_cart_upwind(angle):
    """
    for an angle in cartesian coordinates, convert to upwind.
    0       <>      pi/2
    pi/2    <>      0
    pi/-pi  <>      -pi/2
    -pi/2   <>      pi/-pi
    """
    angle = wrap(-angle+np.pi/2)
    return angle

def fit_cos(y, offset=0, nglom=16):
    def cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline
    t=np.linspace(0,2*np.pi,nglom)

    params, _ = curve_fit(cosine_func, t, y, p0=[1, 1, 1], bounds=((0,-np.inf, -np.inf), (np.inf,np.inf, np.inf)))
    params[2] = params[2]+offset
    fit = cosine_func(t,*params)
    return fit, params

def correct_fit_cos_phase(phase):
    corrected_phase = wrap(np.array([phase+np.pi/2]))
    return corrected_phase

def ang_mag_to_cos(ang, mag):
    def cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline

def calculate_net_motion(df):
    """
    Calculate net motion of ball

    Using the three rotational axes, calculate the net motion of the ball

    Parameters
    ----------
    df: DataFrame
        experimental dataframe, must contain fields df_roll, df_pitch, df_yaw, seconds

    Returns
    -------
    df: Dataframe
        adds net motion to experimental dataframe, units in radians/s
    """
    del_t = np.mean(np.diff(df.seconds))
    effective_rate = 1/del_t
    netmotion = np.sqrt(df.ft_roll**2+df.ft_pitch**2+df.ft_yaw**2)*effective_rate
    df['net_motion'] = netmotion
    return df



def calculate_speeds(df):
    """
    Calculate speed of animal in a 2D plane, and x/y velocity

    Parameters
    ----------
    df: DataFrame
        experimental dataframe, must contain ft_posx, ft_posy

    Returns
    -------
    df: DataFrame
        adds speed, x velocity and y velocity, curvature
    """
    del_t = np.mean(np.diff(df.seconds))
    effective_rate = 1/del_t
    xv = np.gradient(df.ft_posx)*effective_rate
    yv = np.gradient(df.ft_posy)*effective_rate
    try:
        angvel = np.abs(df.ft_yaw)*effective_rate
    except:
        angvel = np.abs(df.df_yaw)*effective_rate
    speed = np.sqrt(xv**2+yv**2)
    df['abs_angvel'] = angvel
    df['speed'] = speed
    df['xv'] = xv
    df['yv'] = yv
    df['curvature'] = df['abs_angvel']/df['speed']
    return df

def calculate_trav_dir(df):
    """
    calculate the travelling direction
    """
    x = df.ft_posx
    y = df.ft_posy
    dir = np.arctan2(np.gradient(y), np.gradient(x))
    df['trav_dir'] = dir
    return df

def dff(f):
    """
    calculate the delta F/F0 where F0 is the bottom 10% of raw values

    Parameteres
    -----------
    f:
        raw fluoresence values

    Returns
    -------
    a:
        delta F/F0 values
    """
    f0 = f.quantile(0.1)
    a=(f-f0)/f0
    return a

def znorm_old(signal):
    """
    z score normalize, adapted from Cheng

    Parameters
    ----------
    signal: numpy array
        raw fluoresence values

    Returns
    -------
    newsignal: numpy array
        z score normalized fluoresence
    """
    newsignal = signal - signal.mean(axis=0)
    div = newsignal.std(axis=0)
    div[div == 0] = 1
    newsignal /= div
    return newsignal

def znorm(signal):
    import scipy.stats as stats
    newsignal = stats.zscore(signal)
    return newsignal

def max_min_norm(signal):
    max = np.max(signal)
    min = np.min(signal)
    newsignal = (signal-min)/(max-min)
    return newsignal


def lnorm(signal):
    """
    linearly normalized signal

    Parameters
    ----------
    signal: numpy array
        raw fluoresence values

    Returns
    -------
    newsignal: numpy array
        normalized fluoresence signal
    """
    signal_sorted = np.sort(signal, axis=0)
    f0 = signal_sorted[:int(len(signal)*.05)].mean(axis=0)
    fm = signal_sorted[int(len(signal)*.97):].mean(axis=0)
    if f0.ndim > 1:
        f0[f0 == 0] = np.nan
        fm[fm == 0] = np.nan
    newsignal = (signal - f0) / (fm - f0)
    return newsignal

def closest_argmin(A, B):
    """
    incomplete
    return the closest argument between A and B

    Parameters
    ----------

    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

def closest_val_ix(array, val):
    """
    find index of array element closest to val

    Parameters
    ----------
    array: np.array
        array of values
    val: float
        value of interest

    Returns
    -------
    idx: int
        index of value in array closes to val
    """
    if type(array) is not np.ndarray:
        array = np.array(array)
    idx = (np.abs(array - val)).argmin()
    return idx

def ft_bout_imaging(ix, bout, pv):
    """
    For an behavioral bout, return a df of corresponding imaging data

    Parameters
    ----------
    ix: list
        ix of ft dtaframe corresponding to each frame in imaging data
    bout: pandas.DataFrame
        behavioral df of bout
    pv: pandas.DataFrame
        imaging df

    Returns
    -------
    pv_bout: pandas.DataFrame
        dataframe
    """
    # bout start and stop indices
    start_ix = bout.index[0]
    end_ix = bout.index[-1]

    # pv start and stop indices
    start_pv_ix = closest_val_ix(ix, start_ix)
    end_pv_ix = closest_val_ix(ix, end_ix)

    # cropped pv dataframe
    pv_bout = pv.iloc[start_pv_ix:end_pv_ix]

    return pv_bout

def line_intersection(p1,p2,p3,p4):
    """
    Given endpoints coordinates p1 and p2 for one line, and p2 and p3 for a
    second line, calculate the x,y point where the lines intersect
    """

    # Line 1 passing through points p1 (x1,y1) and p2 (x2,y2)
    # Line 2 passing through points p3 (x3,y3) and p4 (x4,y4)
    # Line 1 dy, dx and determinant
    a11 = (p1[1] - p2[1])
    a12 = (p2[0] - p1[0])
    b1 = (p1[0]*p2[1] - p2[0]*p1[1])

    # Line 2 dy, dx and determinant
    a21 = (p3[1] - p4[1])
    a22 = (p4[0] - p3[0])
    b2 = (p3[0]*p4[1] - p4[0]*p3[1])

    # Construction of the linear system
    # coefficient matrix
    A = np.array([[a11, a12],
                  [a21, a22]])

    # right hand side vector
    b = -np.array([b1,
                   b2])
    # solve
    try:
        intersection_point = np.linalg.solve(A,b)
        # print(intersection_point)
        return intersection_point
    except np.linalg.LinAlgError:
        print('No single intersection point detected')

def find_borders(data, strip_width = 10, strip_spacing = 200):
    from scipy import signal as sg
    x = data.ft_posx
    y = data.ft_posy
    x_idx = data.index[data.mfc2_stpt>0.01].tolist()[0]
    x0 = data.ft_posx[x_idx] # First point where the odor turn on
    duty = strip_width/(strip_width+strip_spacing)
    freq = 1/(strip_width+strip_spacing)
    x_temp = np.linspace(min(x)-strip_width, max(x)+strip_width, 1000)
    mult = 0.5*sg.square(2*np.pi*freq*(x_temp+strip_width/2-x0), duty=duty)+0.5
    x_borders,_ = sg.find_peaks(np.abs(np.gradient(mult)))
    x_borders = x_temp[x_borders]
    y_borders = np.array([y.iloc[x_idx], max(y)])
    #t_borders = np.array([min(t), max(t)])
    all_x_borders = [np.array([x_borders[i], x_borders[i]]) for i, _ in enumerate(x_borders)]
    all_y_borders = [y_borders for i,_ in enumerate(x_borders)]
    #all_t_borders = [t_borders for i,_ in enumerate(x_borders)]
    return all_x_borders, all_y_borders

def coordinate_rotation(x, y, theta):
    """
    used for taking x,y trajectories and rotating them by an angle theta
    """
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)
    return xp,yp

def consolidate_out(data, do, t_cutoff=0.5):
    for out in do.keys():
        a=do[out]
        t = a.seconds.iloc[-1]-a.seconds.iloc[0]
        if t<t_cutoff:
            ix = a.index
            data.loc[ix, 'instrip']=True
    return data

def consolidate_in(data, di, t_cutoff=0.5):
    for i in di.keys():
        a=di[i]
        t = a.seconds.iloc[-1]-a.seconds.iloc[0]
        if t<t_cutoff:
            ix = a.index
            data.loc[ix, 'instrip']=False
    return data

def consolidate_in_out(data):
    d, di, do = inside_outside(data)
    data = consolidate_out(data, do)
    d, di, do = inside_outside(data)
    data = consolidate_in(data, di)
    return data

def select_et(data):
    d, di, do = inside_outside(data)
    inside_return = 0
    for key in list(di.keys()):
        df = di[key]
        if return_to_edge(df):
            inside_return+=1
        if inside_return>=3:
            return True
    return False

def center_x_y(bout_df):
    x = bout_df.ft_posx.to_numpy()
    y = bout_df.ft_posy.to_numpy()
    x = x-x[0]
    y = y-y[0]
    return x,y

def interp_append_x_y(x, y, avgx, avgy, pts = 10000):
    t = np.arange(len(x))
    t_common = np.linspace(t[0], t[-1], pts)
    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)
    #axs.plot(fx(t_common), fy(t_common))
    avgx.append(fx(t_common))
    avgy.append(fy(t_common))
    return avgx, avgy

def list_matrix_nan_fill(lst):
    """
    takes a single lists consisting of lists that thave differing lengths
    and returns a matrix whose rows are the individual lists with elemenets paded with nan as needed
    """
    pad = len(max(lst, key=len))
    m = np.array([i + [np.nan]*(pad-len(i)) for i in lst])
    return m

def interp_crop(t, y, t_common, pts = 100):
    """
    interpolate,
    crop from t1 to t2
    useful for making triggered averages
    """
    fy = interpolate.interp1d(t, y)
    y_crop = fy(t_common)
    return y_crop


def average_trajectory(dict, side='inside', pts=5000):
    """
    find the averge (x,y) trajectory for a dict of inside or outside trajectories
    excludes first and last trajectories, excludes trajectories that don't
    return to the edge. flips trajectories to align all inside and outside.
    """
    from scipy import interpolate
    if len(dict)<3:
        avg_x, avg_y=[],[]
    else:
        numel = len(dict)-2
        avg = np.zeros((numel, pts, 2))

        for i, key in enumerate(list(dict.keys())[1:-2]):
            df = dict[key]
            if len(df)>10:
                x = df.ft_posx.to_numpy()
                x = x-x[0]
                if np.abs(x[0]-x[-1]): # fly must make it back to edge
                    if side == 'outside':
                        x = -np.sign(np.mean(x))*x
                    elif side == 'inside':
                        x = np.sign(np.mean(x))*x
                    y = df.ft_posy.to_numpy()
                    y = y-y[0]
                    t = np.arange(len(x))
                    t_common = np.linspace(t[0], t[-1], pts)
                    fx = interpolate.interp1d(t, x)
                    fy = interpolate.interp1d(t, y)
                    avg[i,:,0] = fx(t_common)
                    avg[i,:,1] = fy(t_common)
        avg_x = np.mean(avg[:,:,0], axis=0)
        avg_y = np.mean(avg[:,:,1], axis=0)
    return avg_x, avg_y

def find_cutoff(temp):
    """
    for a trajectory, find the point where it fictrac loses tracking and take
    larger component of the trajectory
    """
    x = temp.ft_posx.to_numpy()
    y = temp.ft_posy.to_numpy()
    delta = np.sqrt(x**2+y**2)
    delta = np.abs(np.gradient(delta))
    ix,_=sg.find_peaks(delta, height = 5)
    if ix.any():
        ix = ix[0]
        temp = temp.iloc[0:ix-1]
    return temp

def exclude_lost_tracking(data, thresh=10):
    jumps = np.sqrt(np.gradient(data.ft_posy)**2+np.gradient(data.ft_posx)**2)
    resets, _ = sg.find_peaks(jumps, thresh)
    #resets = resets + 10
    l_mod = np.concatenate(([0], resets.tolist(), [len(data)-1]))
    l_mod = l_mod.astype(int)
    list_of_dfs = [data.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    if len(list_of_dfs)>1:
        data = max(list_of_dfs, key=len)
        data.reset_index()
        print('LOST TRACKING, SELECTION MADE',)
    return data


def savgolay_smooth(signal, window=11, order=5):
    import scipy
    smoothed = scipy.signal.savgol_filter(signal, window, order)
    return smoothed

def conv_signal(time, signal):
    cirf =-1*(pow(2,-time/.01)-pow(2,-time/0.1));
    conv_signal = np.convolve(signal, cirf)
    conv_signal = conv_signal[0:len(signal)]
    return conv_signal

def wrap(arr, cmin=-np.pi, cmax=np.pi):
    period = cmax - cmin
    arr = arr%period
    arr[arr>=cmax] = arr[arr>=cmax] - period
    arr[arr<cmin] = arr[arr<cmin] + period
    return arr

def unwrap(signal, period=2*np.pi):
    unwrapped = np.unwrap(signal*2*np.pi/period)*period/np.pi/2
    return unwrapped

def circ_moving_average(a, n=3, low=-np.pi, high=np.pi):
    """
    calculate a circular moving average, adapted from Cheng
    """
    assert len(a) > n # ensure that the array is long enough
    assert n%2 != 0 # make sure moving average is odd, or this screws up time points
    shoulder = int((n-1) / 2)
    ma = np.zeros(len(a))
    ind0 = np.arange(-shoulder, shoulder+1)
    inds = np.tile(ind0, (len(a)-2, 1)) + np.arange(1, len(a)-1)[:, None]
    ma[1:-1] = circmean(a[inds], low, high, axis=1)
    ma[[0, -1]] = a[[0, -1]]
    return ma

def circ_corr_coeff(x,y):
    import cmath
    deg = True
    x = np.rad2deg(x)
    y = np.rad2deg(y)
    def mean(angles, deg=True):
        '''Circular mean of angle data(default to degree)
        '''
        a = np.deg2rad(angles) if deg else np.array(angles)
        angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
        mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
        return round(np.rad2deg(mean) if deg else mean, 7)

    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x, deg)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
    return r

def circgrad(signal, method=np.gradient, **kwargs):
    signaluw = unwrap(signal, **kwargs)
    dsignal = method(signaluw)
    return dsignal

def rotate_heading(signal, angle):
    rotated_heading = wrap(unwrap(signal)+angle)
    return rotated_heading

def rdp2(M, epsilon=1):
    """
    faster RDP algorithm, doesn't use loop
    """
    def line_dists(points, start, end):
        if np.all(start == end):
            return np.linalg.norm(points - start, axis=1)

        vec = end - start
        cross = np.cross(vec, start - points)
        return np.divide(abs(cross), np.linalg.norm(vec))
    M = np.array(M)
    start, end = M[0], M[-1]
    dists = line_dists(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > epsilon:
        result1 = rdp2(M[:index + 1], epsilon)
        result2 = rdp2(M[index:], epsilon)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result

def rdp_simp(x,y, epsilon=1):
    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
    simplified = rdp(xy0, epsilon=1)
    # dx = np.diff(simplified[:,0])
    # dy = np.diff(simplified[:,1])
    # heading = np.arctan2(dy, dx)
    return simplified

def rdp_simp_heading_angles_len(x,y, epsilon=2):
    xy0 = np.concatenate((x[:,None],y[:,None]),axis=-1)
    simplified = rdp2(xy0, epsilon=1)
    dx = np.diff(simplified[:,0])
    dy = np.diff(simplified[:,1])
    L=[]
    angles = []
    for i in np.arange(len(dx)):
        l = np.sqrt(dx[i]**2+dy[i]**2)
        L.append(l)
    for i in np.arange(len(dx)-1):
        # angle between adjacent segments
        ang_between =-(np.arctan2(-dy[i],-dx[i])-np.arctan2(dy[i+1], dx[i+1]))

        # left vs right
        if ang_between>np.pi:
            ang_between -= 2*np.pi
        elif ang_between<-np.pi:
            ang_between += 2*np.pi

        # take outer angle
        if ang_between<0:
            ang_between = -np.pi-ang_between
        elif ang_between>0:
            ang_between = np.pi-ang_between

        angles.append(ang_between)

    heading = np.arctan2(dy, dx)
    return simplified, heading, angles, L

# def rdp_pts(xy0, points=4):
#     """
#     define a number of points to preserve
#     iterate to fine epsilon that give you the correct number of points
#     """
#     epsilon=1
#     simplified = rdp2(xy0, epsilon=epsilon)
#     while simplified.shape[0] !=points:
#         if simplified.shape[0]<points:
#             epsilon = epsilon-(epsilon/2+0.1*random.random())
#         else:
#             epsilon+=0.1*random.random()
#         simplified = rdp2(xy0, epsilon=epsilon)
#     return simplified

def rdp_pts(xy0,epsilon=1, nodes=4, run_lim=1000):
    run=0
    simplified = rdp2(xy0, epsilon=epsilon)
    while simplified.shape[0] != nodes:
        
        if simplified.shape[0] == (nodes-1): # save two run solution in case RDP can't converge on three run solution
            simplified_save = simplified
        elif simplified.shape[0]<nodes: # need to make epsilon smaller
            epsilon = epsilon-(epsilon/2+0.1*random.random())
        else: # need to make epsilon bigger
            epsilon+=0.1*random.random()
        simplified = rdp2(xy0, epsilon=epsilon)
        
        # abort if cannot find three run solution, return two run solution
        if run>run_lim:
            return simplified_save
        
        run+=1
    return simplified

def plot_mean_sem(data, axs, color='k'):
    from scipy.stats import sem
    mean = np.mean(data, axis=0)
    se = sem(data)
    x = np.arange(len(se))
    axs.fill_between(x,mean+se, mean-se, alpha=0.5)
    axs.plot(mean, color=color)
    return axs

def centroid_weightedring(data):
    if len(data.shape) != 2:
        print('Wrong Dimension of signals for calculating phase.')
        return nan
    phase = np.zeros(data.shape[0])
    amp = np.zeros(data.shape[0])
    pi = np.pi
    num = data.shape[1]
    # angle = np.arange(num)*2.0*pi/num - pi
    angle = wrap(np.arange(num) * 2.0 * pi / num - pi + pi/16., cmin=-pi, cmax=pi)
    # CD note: above does not get whole 360 space b.c. div by num not (num-1)
    # this leaves phase blank spot behind animal.
    for irow, row in enumerate(data):
        x = row * np.cos(angle)
        y = row * np.sin(angle)
        phase[irow] = np.arctan2(y.mean(), x.mean())
        amp[irow] = np.sqrt(y.mean()**2+x.mean()**2)
        
    return phase, amp

def get_fftphase(sig, n=100, axis=-1):
    from scipy import fftpack
    axlen = sig.shape[axis]*n
    epg_fft = fftpack.fft(sig, axlen, axis)
    power = np.abs(epg_fft)**2
    freq = fftpack.fftfreq(axlen, 1/n)/n
    phase = np.angle(epg_fft)
    midpoint = int(freq.size/2)
    freq = freq[1:midpoint]
    period = (1./freq)
    power = power[:, 1:midpoint]
    phase = phase[:, 1:midpoint]
    ix = np.where(period==8)
    phase_8 = phase[:,ix]
    return phase_8.flatten()

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def zero2nan(arr, make_copy=False):
    if make_copy:
        arr2 = copy.copy(arr)
        arr2[arr2 == 0] = np.nan
        return arr2
    else:
        arr[arr == 0] = np.nan

def circmean(arr, low=-np.pi, high=np.pi, axis=None):
    return stats.circmean(arr, high, low, axis, nan_policy='omit')


# %%
# class for connecting to drive and uploading files/folders
class drive_hookup:
    """
    Class for connecting to drive, downloading files, and uploading folders


    """
    def __init__(self):
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        gauth = GoogleAuth()
        if gauth.credentials is None:
            print('enter')
            gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
        self.drive = GoogleDrive(gauth)

    def list_files(self, *args):
        if not args:
            fileList = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
            for file in fileList:
                print('Title: %s, ID: %s' % (file['title'], file['id']))
        else:
            for arg in args:
                id = str(arg)
                st = "'id' in parents and trashed=false"
                st = st.replace('id', id)
                fileList = self.drive.ListFile({'q': st}).GetList()
                for file in fileList:
                    print('Title: %s, ID: %s' % (file['title'], file['id']))

    def download_file(self, *args):
        """
        Download files from drive

        Creates a folder temp_file_store which is the site of all downloads on
        local machine.

        Parameters
        ----------
        *args: str
            list all Drive file ids to download
        """
        if not os.path.exists('temp_file_store'):
            os.makedirs('temp_file_store')
        for arg in args:
            file = self.drive.CreateFile({'id': arg})
            file_name = file['originalFilename']
            if not os.path.isfile('temp_file_store/'+file_name):
                file.GetContentFile('temp_file_store/'+file_name)

    def delete_file_store(self):
        """
        Delete temp_file_store.  Should do every session
        """
        import shutil
        shutil.rmtree('temp_file_store')

    #def upload_folder(self):
