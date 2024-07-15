from behavior_analysis import trajectory_plotter, trajectory_plotter_bw, create_bout_df, plot_histograms, plot_scatter
from imaging_analysis import plot_FF_trajectory, plot_speed_trajectory, plot_triggered_norm_FF

# Plot a trajectory
folder_path = '/Users/noelleeghbali/Desktop/exp/tethered_behavior/summer_2024_exp/MAIN/EPG_GtACR'
select_file = '07022024-144551_KCs_Dop1R1shRNA_acv_plume_Fly8.log'

trajectory_plotter(folder_path, 50, 4000, -1000, [-500, 500], [0,1000], led='green', hlines=[500], select_file=None, plot_type='odor', save=True)
#trajectory_plotter_bw(folder_path, 50, 2000, 0, [-500, 500], [1000, 2000], led='green', hlines=None, select_file=None, plot_type='odor', save=True)

#distance_df = create_bout_df(folder_path, data_type='x_distance_from_plume', plot_type='odor')

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

