from behavior_analysis import *
from imaging_analysis import *

# Plot a trajectory
folder_path = '/Users/noelleeghbali/Desktop/exp/tethered_behavior/spring_2025_exp/MAIN/acv_light_disappearing'
select_file = '01192024-140219_PAM_Chr_light_crisscross_ctrl_oct_Fly33.log'
# trajectory_plotter_bw(folder_path, 50, 100000, -100, [-500, 500], [0,1500], led='red', hlines=[500], select_file= None, plot_type='odor', save=True)
disappearing_trajectory_plotter_bw(folder_path, led='red', select_file=None, save=True)
# pulse_xpos_distribution(folder_path, 'xpos_distribution', size=(5, 5), groups=3)
# pulse_uw_tracking(folder_path, 'no vs multiple', size=(3, 5), plotc='black', groups=2)
# uw_tracking_id(folder_path, 'distance tracked bw', size=(3,5), groups=2, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc='black')
uws_response_id(folder_path, 'uw speed at odor onset bw', size=(7,5), groups=2, sample_rate=20, buf_pts=4, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc='black')
# walking_speed_id(folder_path, 'walking speed in odor bw', size=(3,5), groups=2, keywords=['ctrl', 'light'], colors=['grey', '#0bdf51'], plotc='black')
#alt_pulse_return_efficiency_light(folder_path, 'return efficiency', size=(6,6), groups=2)
# block_return_efficiency(folder_path, 'returns per meter', size=(3,5), cutoff=350, labels = ['ctrl', 'crisscross'], colors=['#c1ffc1', '#6f00ff'])#trajectory_plotter_bw(folder_path, 50, 2000, 0, [-500, 500], [1000, 2000], led='green', hlines=None, select_file=None, plot_type='odor', save=True)
# cc_return_efficiency(folder_path, 'cc return eff', size=(3,5), groups=2, keywords = ['ctrl', 'crisscross_oct'], colors=['#7ed4e6', '#FF355E'])
#distance_df = create_bout_df(folder_path, data_type='x_distance_from_plume', plot_type='odor')
# pulse_cw_dist(folder_path, 'cw tracking bw', size=(5, 5), plotc='black', groups=3)
# octs_return_efficiency(folder_path, '2 octs return efficiency', size=(3, 4), groups=2, plotc='black', keywords=['plume', 'crisscross_oct'], colors=['#7ed4e6','#ff355e'])

# plot_histograms(folder_path,
#                 boutdf=distance_df,
#                 plot_variable='data',
#                 group_variable='condition',
#                 group_values=['in', 'out'],
#                 group_colors=['#FF355E', '#48bf91'],
#                 title='avg distance from plume in strip vs. out of strip (FB4R>GtACR)',
#                 x_label='distance from plume (mm)',
#                 y_label='number of bouts',
#                 x_limits=(-50, 200))

# plot_scatter(folder_path, 
#              boutdf=distance_df,
#              plot_variable='data',
#              group_variable='condition',
#              group_values=['in', 'out'],
#              group_colors=['#FF355E', '#48bf91'],
#              title='avg distance from plume in strip vs. out of strip (Orco>Chr)',
#              x_label=None,
#              y_label=None,
#              x_limits=(-50, 200))


# PLOT IMAGING DATA

# fig_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/fig'
# filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
# lobes = ['G2', 'G3', 'G4']
# colors = ['#00ccff', '#ff43a4', '#4cbb17', '#9f00ff']
# plot_FF_trajectory(fig_folder, filename, lobes, colors, strip_width=10, strip_length=1000, xlim=[-100,100], ylim=[0,500], hlines=None, save=True, keyword=26)
# plot_speed_trajectory(fig_folder, filename, strip_width=10, strip_length=1000, xlim=[-200,200], ylim=[0,1000], hlines=None, save=True, keyword=24)


#plot_trial_FF_odor(fig_folder, filename, lobes, colors, keyword=4)
#plot_triggered_norm_FF(fig_folder, filename, lobes, colors, window_size=5, event_type='entry')  # or event_type='exit'

