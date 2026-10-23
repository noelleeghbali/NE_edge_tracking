#%%
from imaging_analysis import *
from behavior_analysis import *
from scipy.stats import spearmanr, wilcoxon
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap


figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
lobes =['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff','#cc33cc', '#ff43a4']
# lobes =['G2', 'G3', 'G4', 'G5']
# colors = ['#00f5ff', '#cc33cc']

#%% df/f trace w/ net motion
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
print(len(flylist))
for fly in flylist:
    flydf = data[fly]['a1'] # dataframe for each fly
    print(list(flydf.columns))
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    fig, axs, markc = configure_white_plot(size=(8,6), xaxis=True, nrows=len(lobes)+1)
    axs = np.ravel(axs)         
    fig.suptitle(fly)
    odor1_et = flydf["mfc2_stpt"].to_numpy()
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe] 
        time = flydf['relative_time']
        axs[i].plot(time, FF, linewidth=0.5, color=colors[i])
        shade_on_intervals(axs[i], time, odor1_et, mode='span', alpha=0.65, color="#fbceb1")
        axs[i].set_ylabel(lobe, color=markc)
    
    # add net motion
    net_motion = flydf['net_motion']
    ax = axs[len(lobes)]
    ax.plot(flydf['relative_time'], net_motion, linewidth=0.5, color=markc)
    shade_on_intervals(ax, time, odor1_et, mode='span', alpha=0.65, color="#fbceb1")
    ax.set_ylabel('net motion', color=markc)
    ax.set_xlabel('time (s)', color=markc)

    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/traces'):
        os.makedirs(f'{figure_folder}/fig/traces')
    fig.savefig(f'{figure_folder}/fig/traces/{fly}_trace_white.pdf', bbox_inches='tight')

#%% trajectories

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle

for fly in flylist:
    flydf = data[fly]['a1'] # dataframe for each fly
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    df_odor = exp_df.where(exp_df.mfc2_stpt > 0)
    strip_width=10
    fig, axs, markc = configure_white_plot(size=(10,10), xaxis=True)
    xo = exp_df.iloc[0]['ft_posx']
    yo = exp_df.iloc[0]['ft_posy']
    yf = exp_df.iloc[-1]['ft_posy']
    plt.gca().add_patch(patches.Rectangle((-strip_width / 2, 0), 10, (yf-yo), facecolor='grey'))    
    plt.gca().add_patch(patches.Rectangle(((-strip_width / 2)-210, 0), 10, (yf-yo), facecolor='grey'))    
    plt.gca().add_patch(patches.Rectangle(((-strip_width / 2)+210, 0), 10, (yf-yo), facecolor='grey'))    
    plt.plot(exp_df['ft_posx'] - xo, exp_df['ft_posy'] - yo, color='lightgrey', label='clean air')
    plt.plot(df_odor['ft_posx'] - xo, df_odor['ft_posy'] - yo, color='red', label='odor only')
    # plt.gca().add_patch(patches.Rectangle((-strip_width / 2, plume_start), strip_width, strip_length, facecolor=plume_color, alpha=0.5))

    plt.title(fly, fontsize=14)
    axs.set_xlabel('x-position (mm)', fontsize=14, color='white')
    axs.set_ylabel('y-position (mm)', fontsize=14, color='white')
    # Further customization
    axs.tick_params(which='both', axis='both', labelsize=12, length=3, width=2, color='black', direction='out', left=True, bottom=True)
    for pos in ['right', 'top']:
        axs.spines[pos].set_visible(False)
    plt.tight_layout()
    sns.despine(offset=10)
    for _, spine in axs.spines.items():
        spine.set_linewidth(2)
    for spine in axs.spines.values():
        spine.set_edgecolor(markc)
    axs.set_aspect('equal', adjustable='datalim') 
    if not os.path.exists(f'{figure_folder}/fig/traj'):
        os.makedirs(f'{figure_folder}/fig/traj')
    fig.savefig(f'{figure_folder}/fig/traj/{fly}_traj.pdf', bbox_inches='tight')

#%% compare exp / bl auc (over time) in each lobe w/ return rate

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
all_return_rates = []
all_ratios = {f'tone_G{i+2}': [] for i in range(len(lobes))} # dict of lists of ratios for each lobe 
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1'] # dataframe for each fly
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    # exp_df = exp_df[~exp_df['instrip']] 
    bl_df = flydf.loc[:first_on_index]
    exp_dur = exp_df['relative_time'].iloc[-1] - exp_df['relative_time'].iloc[0]
    bl_dur = bl_df['relative_time'].iloc[-1] - bl_df['relative_time'].iloc[0]

    # calculate returns per meter
    returns = 0
    pl = get_a_bout_calc(exp_df, 'path_length') / 1000
    for key, df in do.items():
        if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 0.5 and return_to_edge(df):
           returns += 1
    return_rate = returns/pl
    print(f'return rate: {return_rate}')

    # Calculate DA tones first
    exp_aucs = []
    da_tones = []
    for i, (lobe, color) in enumerate(zip(lobes, colors)):
        baseline_auc = (np.trapz(bl_df[lobe], bl_df['relative_time'])) / bl_dur
        exp_auc = (np.trapz(exp_df[lobe], exp_df['relative_time'])) / exp_dur
        da_tone = exp_auc / baseline_auc
        da_tones.append(da_tone)
        print(f'lobe: {lobe}, auc ratio: {da_tone}')
    
    # If any da_tone is negative, skip this fly
    if any(tone < 0 for tone in da_tones):
        print("Negative DA tone detected. Skipping this fly.")
        continue

    # Otherwise, append results
    all_return_rates.append(return_rate)
    for i, tone in enumerate(da_tones):
        all_ratios[f'tone_G{i+2}'].append(tone)

fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for i in range(len(lobes)):
    x = all_ratios[f'tone_G{i+2}']
    y = all_return_rates
    axs[i].scatter(x, y, color=colors[i], alpha=0.8)
    axs[i].set_title(lobes[i])  # updated to show lobe name
    axs[i].set_xlabel('DA tone', color=markc, fontsize=14)
    axs[i].set_ylabel('return rate', color=markc, fontsize=14)

#%% compare relative activity (over time) in each lobe w/ return rate
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
all_return_rates = []
all_ratios = {f'tone_G{i+2}': [] for i in range(len(lobes))} # dict of lists of ratios for each lobe 
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    bl_df = flydf.loc[:first_on_index]
    exp_dur = exp_df['relative_time'].iloc[-1] - exp_df['relative_time'].iloc[0]
    bl_dur = bl_df['relative_time'].iloc[-1] - bl_df['relative_time'].iloc[0]
    # calculate returns per meter
    returns = 0
    pl = get_a_bout_calc(exp_df, 'path_length') / 1000
    for key, df in do.items():
        if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 0.5 and return_to_edge(df):
            returns += 1
    return_rate = returns / pl
    print(f'return rate: {return_rate}')
    # Calculate exp_auc for each lobe
    exp_aucs = []
    for lobe in lobes:
        exp_auc = np.trapz(exp_df[lobe], exp_df['relative_time']) / exp_dur
        baseline_auc = (np.trapz(bl_df[lobe], bl_df['relative_time'])) / bl_dur
        exp_aucs.append(exp_auc/baseline_auc)
    # Skip if any AUC is negative (optional — for quality control)
    if any(auc < 0 for auc in exp_aucs):
        print("Negative exp AUC detected. Skipping this fly.")
        continue
    total_exp_auc = sum(exp_aucs)
    if total_exp_auc == 0:
        print("Total exp AUC is zero. Skipping this fly.")
        continue
    # Compute relative tone per lobe (i.e., proportion of total)
    rel_tones = [auc / total_exp_auc for auc in exp_aucs]
    # Append results
    all_return_rates.append(return_rate)
    for i, rel_tone in enumerate(rel_tones):
        all_ratios[f'tone_G{i+2}'].append(rel_tone)
# Plotting
fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for i in range(len(lobes)):
    x = np.array(all_ratios[f'tone_G{i+2}'])
    y = np.array(all_return_rates)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    
    axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('relative activity', color=markc, fontsize=14)
    axs[0].set_ylabel('return rate', color=markc, fontsize=14)
    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            # Only apply linear model if R² is decent
            if r_squared >= 0.3:
                x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color='black')
                axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                            transform=axs[i].transAxes,
                            verticalalignment='top',
                            fontsize=12,
                            color='black')
            # else:
            #     axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
            #                 transform=axs[i].transAxes,
            #                 verticalalignment='top',
            #                 fontsize=10,
            #                 color='gray')

        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")

#%% compare relative activity (over time) in each lobe w/ return efficiency

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
all_return_effs = []
all_ratios = {f'tone_G{i+2}': [] for i in range(len(lobes))} # dict of lists of ratios for each lobe 
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    exp_df = exp_df[~exp_df['instrip']]
    bl_df = flydf.loc[:first_on_index]
    exp_dur = exp_df['relative_time'].iloc[-1] - exp_df['relative_time'].iloc[0]
    bl_dur = bl_df['relative_time'].iloc[-1] - bl_df['relative_time'].iloc[0]
    # calculate returns per meter
    fly_eff = []
    for key, df in do.items():
        if df['relative_time'].iloc[-1] - df['relative_time'].iloc[0] >= 0.5 and return_to_edge(df):
            pl = get_a_bout_calc(exp_df, 'path_length') 
            dist = get_a_bout_calc(exp_df, 'furthest_distance_from_plume')
            fly_eff.append(pl / dist)
    all_fly_effs = sum(fly_eff) / len(fly_eff)
    # Calculate exp_auc for each lobe
    exp_aucs = []
    for lobe in lobes:
        exp_auc = np.trapz(exp_df[lobe], exp_df['relative_time']) / exp_dur
        baseline_auc = (np.trapz(bl_df[lobe], bl_df['relative_time'])) / bl_dur
        exp_aucs.append(exp_auc/baseline_auc) #normalized over baseline
    # Skip if any AUC is negative (optional — for quality control)
    if any(auc < 0 for auc in exp_aucs):
        print("Negative exp AUC detected. Skipping this fly.")
        continue
    total_exp_auc = sum(exp_aucs)
    if total_exp_auc == 0:
        print("Total exp AUC is zero. Skipping this fly.")
        continue
    # Compute relative tone per lobe (i.e., proportion of total)
    rel_tones = [auc / total_exp_auc for auc in exp_aucs]
    # Append results
    all_return_effs.append(1/all_fly_effs)
    for i, rel_tone in enumerate(rel_tones):
        all_ratios[f'tone_G{i+2}'].append(rel_tone)
fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for i in range(len(lobes)):
    x = np.array(all_ratios[f'tone_G{i+2}'])
    y = np.array(all_return_effs)

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('relative activity', color=markc, fontsize=14)
    axs[0].set_ylabel('avg. return efficiency', color=markc, fontsize=14)

    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            # Only apply linear model if R² is decent
            if r_squared >= 0.3:
                x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color='black')
                axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                            transform=axs[i].transAxes,
                            verticalalignment='top',
                            fontsize=12,
                            color='black')
            # else:
            #     axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
            #                 transform=axs[i].transAxes,
            #                 verticalalignment='top',
            #                 fontsize=10,
            #                 color='gray')

        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")


# %% compare change in activity (auc/time) at entry w/ efficiency of subsequent return (PER FLY)

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=(12,3.5), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    entry_key_dict = {f'G{i+2}': [] for i in range(len(lobes))} # bout IDs
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        time = flydf['relative_time']
        aucs = []
        return_efficiencies = []

        for key, entry_df in di.items():
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            duration = time_off - time_on
            if duration <= 0 or np.isnan(auc / duration):
                continue
            auc_value = auc / duration
            print ('integrating normalized df/f')
            # Match to corresponding return bout using key + 1
            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    pl = get_a_bout_calc(return_df, 'path_length')
                    dist = get_a_bout_calc(return_df, 'furthest_distance_from_plume')
                    print('calculating return efficiency')
                    if pl > 0:
                        return_eff = dist / pl
                        aucs.append(auc_value)
                        return_efficiencies.append(return_eff)
        # axs[i].scatter(aucs, return_efficiencies, color=colors[i], alpha=0.8)
        # axs[i].set_title(lobes[i])
        
        x = np.array(aucs)
        y = np.array(return_efficiencies)

        # Remove NaNs/Infs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('integrated dF/F', color=markc, fontsize=14)
        axs[0].set_ylabel('return efficiency', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)

                # Only apply linear model if R² is decent
                if r_squared >= 0.3:
                    x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                    axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                    axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)
                # else:
                #     axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
                #                 transform=axs[i].transAxes,
                #                 verticalalignment='top',
                #                 fontsize=10,
                #                 color='gray')

            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.suptitle(fly, fontsize=14)


# %% compare peak in activity at entry w/ efficiency of subsequent return (PER FLY)
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=((3.5*len(lobes)),3.5), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    entry_key_dict = {f'G{i+2}': [] for i in range(len(lobes))} # bout IDs
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        time = flydf['relative_time']
        aucs = []
        return_efficiencies = []

        for key, entry_df in di.items():
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = FF_adjusted.max() #np.trapz(FF_adjusted, time[interval_mask])
            auc_value = auc 
            print ('calculating peak df/f')
            # Match to corresponding return bout using key + 1
            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if 0.5 <= bout_dur <= 30 and return_to_edge(return_df):
                    pl = get_a_bout_calc(return_df, 'path_length')
                    dist = get_a_bout_calc(return_df, 'furthest_distance_from_plume')
                    print('calculating return efficiency')
                    if pl > 0:
                        return_eff = dist / pl
                        aucs.append(auc_value)
                        return_efficiencies.append(return_eff)

        x = np.array(aucs)
        y = np.array(return_efficiencies)
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('peak dF/F', color=markc, fontsize=14)
        axs[0].set_ylabel('return efficiency', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)

                # Only apply linear model if R² is decent
                # if r_squared >= 0.3:
                x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                            transform=axs[i].transAxes,
                            verticalalignment='top',
                            fontsize=12,
                            color=markc)
                # else:
                    # axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
                    #             transform=axs[i].transAxes,
                    #             verticalalignment='top',
                    #             fontsize=10,
                    #             color='gray')

            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.suptitle(fly, fontsize=14)


#%%

data = open_pickle(filename)
flylist = np.array(list(data.keys()))  # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}

for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=(12, 3.5), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    entry_key_dict = {f'G{i+2}': [] for i in range(len(lobes))}  # bout IDs

    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        time = flydf['relative_time']
        aucs = []
        return_efficiencies = []

        for key, entry_df in di.items():
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = FF_adjusted.max()  # Using peak ΔF/F instead of integral
            # auc = np.trapz(FF_adjusted, time[interval_mask])
            auc_value = auc
            print('calculating peak df/f')

            # Match to corresponding return bout using key + 1
            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if 2 <= bout_dur <= 30 and return_to_edge(return_df):
                    pl = get_a_bout_calc(return_df, 'path_length')
                    dist = get_a_bout_calc(return_df, 'furthest_distance_from_plume')
                    print('calculating return efficiency')
                    if pl > 0:
                        return_eff = dist / pl
                        aucs.append(auc_value)
                        return_efficiencies.append(return_eff)

        x = np.array(aucs)
        y = np.array(return_efficiencies)

        # Remove NaNs/Infs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]

        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('peak dF/F', color=markc, fontsize=14)
        axs[0].set_ylabel('return efficiency', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)

                # Only apply linear model if R² is decent
                if r_squared >= 0.3:
                    x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                    axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                    axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)

            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")

        # Store data for summary
        summary_aucs[f'G{i+2}'].extend(x_clean)
        summary_return_eff[f'G{i+2}'].extend(y_clean)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.suptitle(fly, fontsize=14)

# ---- SUMMARY PLOT ACROSS ALL FLIES ----
fig_sum, axs_sum, markc = configure_bw_plot(size=(12, 3.5), xaxis=True, ncols=len(lobes))
for i, lobe in enumerate(lobes):
    x = np.array(summary_aucs[f'G{i+2}'])
    y = np.array(summary_return_eff[f'G{i+2}'])
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    axs_sum[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.6)
    axs_sum[i].set_title(f"{lobe} (all flies)")
    axs_sum[i].set_xlabel('peak dF/F', color=markc, fontsize=14)
    axs_sum[0].set_ylabel('return efficiency', color=markc, fontsize=14)

    # Optional linear fit
    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            print(lobe)
            print(r_squared)
            if r_squared >= 0.3:
                x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                axs_sum[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                axs_sum[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs_sum[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)
        except np.linalg.LinAlgError:
            print(f"Linear regression failed for summary lobe {lobes[i]}.")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Summary Across All Flies', fontsize=16)
plt.savefig(f'{figure_folder}/fig/peak_vs_re.pdf', bbox_inches='tight', dpi=300)
plt.show()




# %% compare change in activity (auc/time) at entry w/ time since last return (PER FLY)

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=(12,4), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']

    entry_key_dict = {f'G{i+2}': [] for i in range(len(lobes))} # bout IDs
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        time = flydf['relative_time']
        aucs = []
        return_efficiencies = []

        for key, entry_df in di.items():
            if key == 2: # skip first entry
                continue 
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            duration = time_off - time_on
            if duration <= 0 or np.isnan(auc / duration):
                continue
            auc_value = auc / duration
            # Match to corresponding return bout using key + 1
            return_key = key - 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if 30 >= bout_dur >= 1 and return_to_edge(return_df):
                    aucs.append(auc_value)
                    return_efficiencies.append(bout_dur)
        # axs[i].scatter(aucs, return_efficiencies, color=colors[i], alpha=0.8)
        # axs[i].set_title(lobes[i])
        axs[i].set_xlabel('entry AUC', color=markc, fontsize=14)
        axs[i].set_ylabel('time since last return (s)', color=markc, fontsize=14)

        x = np.array(aucs)
        y = np.array(return_efficiencies)

        # Remove NaNs/Infs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('entry AUC', color=markc, fontsize=14)
        axs[i].set_ylabel('time since last return (s)', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
                # Only apply linear model if R² is decent
                if r_squared >= 0.3:
                    x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                    axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                    axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)
                else:
                    axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=10,
                                color='gray')

            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.suptitle(fly, fontsize=14)




# %% compare change in activity (auc/time) at entry w/ time since last return

data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
fig, axs, markc = configure_bw_plot(size=(12,3.5), xaxis=True, ncols=len(lobes))
for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        aucs = []
        return_efficiencies = []
        for key, entry_df in di.items():
            if key == 2:  # skip first entry
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            duration = time_off - time_on
            if duration <= 0 or np.isnan(auc / duration):
                continue
            auc_value = auc / duration
            # Match to corresponding return bout using key - 1
            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    aucs.append(auc_value)
                    return_efficiencies.append(bout_dur)
        # Accumulate results across flies
        summary_aucs[f'G{i+2}'].extend(aucs)
        summary_return_eff[f'G{i+2}'].extend(return_efficiencies)
# Plot all data at the end
for i in range(len(lobes)):
    aucs = summary_aucs[f'G{i+2}']
    return_eff = summary_return_eff[f'G{i+2}']
    axs[i].scatter(return_eff, aucs, color=colors[i], alpha=0.8)
    x = np.array(return_eff)
    y = np.array(aucs)
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) > 1 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            # Only apply linear model if R² is decent
            x_fit = np.linspace(min(x_clean), max(x_clean), 100)
            axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
            axs[i].text(0.05, 0.95, f'R²={r_squared:.2f}',
                        transform=axs[i].transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        color=markc)
        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('next return length (s)', color=markc, fontsize=14)
    axs[0].set_ylabel('dF/F', color=markc, fontsize=14)
# fig.suptitle('All Flies Combined', fontsize=16)


# %% compare peak activity at entry w/ time since last return *****************
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        aucs = []
        return_efficiencies = []
        for key, entry_df in di.items():
            if key == 2:  # skip first entry
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc_value = FF_adjusted.max() # - FF_adjusted.min()
            # Match to previous return bout using key - 1
            return_key = key - 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    aucs.append(auc_value)
                    return_efficiencies.append(bout_dur)
        # Accumulate results across flies
        summary_aucs[f'G{i+2}'].extend(aucs)
        summary_return_eff[f'G{i+2}'].extend(return_efficiencies)
# Plot all data at the end
for i in range(len(lobes)):
    aucs = summary_aucs[f'G{i+2}']
    return_eff = summary_return_eff[f'G{i+2}']
    axs[i].scatter(return_eff, aucs, color=colors[i], alpha=0.8)
    x = np.array(return_eff)
    y = np.array(aucs)
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            x_fit = np.linspace(min(x_clean), max(x_clean), 100)
            axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
            axs[i].text(0.05, 0.95, f'R²={r_squared:.2f}',
                        transform=axs[i].transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        color=markc)
        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('time since last return (s)', color=markc, fontsize=14)
    axs[i].set_xscale('log')
    axs[0].set_ylabel('dF/F', color=markc, fontsize=14)
out_dir = f'{figure_folder}/fig/time_corr'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/peak_vs_time_since_last_entry.pdf', bbox_inches='tight')

    
# %% compare activity (auc/time) outside w/ efficiency of the return (PER FLY)
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
for fly in flylist:
    fig, axs, markc = configure_white_plot(size=(12,3), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    entry_key_dict = {f'G{i+2}': [] for i in range(len(lobes))} # bout IDs
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        time = flydf['relative_time']
        aucs = []
        return_efficiencies = []

        for key, entry_df in do.items(): # taking outside activity
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            duration = time_off - time_on
            if duration <= 0 or np.isnan(auc / duration):
                continue
            auc_value = auc / duration
            print ('integrating normalized df/f')
            # Match to corresponding return bout using key + 1
            return_key = key
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 1 and return_to_edge(return_df):
                    pl = get_a_bout_calc(return_df, 'path_length')
                    dist = get_a_bout_calc(return_df, 'furthest_distance_from_plume')
                    print('calculating return efficiency')
                    if pl > 0:
                        return_eff = dist / pl
                        aucs.append(auc_value)
                        return_efficiencies.append(return_eff)
        # axs[i].scatter(aucs, return_efficiencies, color=colors[i], alpha=0.8)
        # axs[i].set_title(lobes[i])
        axs[i].set_xlabel('entry AUC', color=markc, fontsize=14)
        axs[i].set_ylabel('return efficiency', color=markc, fontsize=14)
        x = np.array(aucs)
        y = np.array(return_efficiencies)

        # Remove NaNs/Infs
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('dF/F0 outside', color=markc, fontsize=14)
        axs[i].set_ylabel('return efficiency', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)

                # Only apply linear model if R² is decent
                if r_squared >= 0.3:
                    x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                    axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                    axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)
                else:
                    axs[i].text(0.05, 0.95, f'no good linear fit (R²={r_squared:.2f})',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=10,
                                color='gray')

            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")

    fig.suptitle(fly)

#%% per fly normalized max-min vs log of time
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    
    # Collect this fly's AUCs and return efficiencies per lobe first
    fly_aucs = {i: [] for i in range(len(lobes))}
    fly_return_eff = {i: [] for i in range(len(lobes))}
    
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        for key, entry_df in di.items():
            if key == 2:  # skip first entry
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc_value = FF_adjusted.max() #- FF_adjusted.min()
            # Match to previous return bout using key - 1
            return_key = key - 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    fly_aucs[i].append(auc_value)
                    fly_return_eff[i].append(bout_dur)
    
    # Normalize this fly's AUCs by the max across all lobes
    all_fly_aucs = [v for lobe_aucs in fly_aucs.values() for v in lobe_aucs]
    finite_aucs = [v for v in all_fly_aucs if np.isfinite(v)]
    if len(finite_aucs) == 0:
        continue
    fly_max = max(finite_aucs)
    if fly_max == 0 or not np.isfinite(fly_max):
        continue
    
    # Accumulate normalized results across flies
    for i in range(len(lobes)):
        normalized_aucs = [a / fly_max for a in fly_aucs[i]]
        summary_aucs[f'G{i+2}'].extend(normalized_aucs)
        summary_return_eff[f'G{i+2}'].extend(fly_return_eff[i])

# Plot all data at the end
for i in range(len(lobes)):
    aucs = summary_aucs[f'G{i+2}']
    return_eff = summary_return_eff[f'G{i+2}']
    axs[i].scatter(return_eff, aucs, color=colors[i], alpha=0.8)
    x = np.array(return_eff)
    y = np.array(aucs)
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            x_fit = np.linspace(min(x_clean), max(x_clean), 100)
            axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
            axs[i].text(0.05, 0.95, f'R²={r_squared:.2f}',
                        transform=axs[i].transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        color=markc)
        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('time since last return (s)', color=markc, fontsize=14)
    axs[i].set_xscale('log')
    axs[i].set_ylim(0, 1.05)  # enforce max y of 1
    axs[0].set_ylabel('normalized dF/F', color=markc, fontsize=14)    

#%% across flies normalized peak vs time
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
for fly in flylist:
    print(fly)
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        aucs = []
        return_efficiencies = []
        for key, entry_df in di.items():
            if key == 2:  # skip first entry
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)
            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue
            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc_value = FF_adjusted.max() - FF_adjusted.min()
            # Match to previous return bout using key - 1
            return_key = key - 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    aucs.append(auc_value)
                    return_efficiencies.append(bout_dur)
        # Accumulate results across flies
        summary_aucs[f'G{i+2}'].extend(aucs)
        summary_return_eff[f'G{i+2}'].extend(return_efficiencies)

# Find the global max across all lobes and all flies, then normalize
all_values = [v for lobe_key in summary_aucs for v in summary_aucs[lobe_key] if np.isfinite(v)]
global_max = max(all_values) if all_values else 1.0
if global_max == 0 or not np.isfinite(global_max):
    global_max = 1.0
for lobe_key in summary_aucs:
    summary_aucs[lobe_key] = [v / global_max for v in summary_aucs[lobe_key]]

# Plot all data at the end
for i in range(len(lobes)):
    aucs = summary_aucs[f'G{i+2}']
    return_eff = summary_return_eff[f'G{i+2}']
    axs[i].scatter(return_eff, aucs, color=colors[i], alpha=0.8)
    x = np.array(return_eff)
    y = np.array(aucs)
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
        try:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            fit_fn = np.poly1d(coeffs)
            y_pred = fit_fn(x_clean)
            r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            x_fit = np.linspace(min(x_clean), max(x_clean), 100)
            axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
            axs[i].text(0.05, 0.95, f'R²={r_squared:.2f}',
                        transform=axs[i].transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        color=markc)
        except np.linalg.LinAlgError:
            print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")
    axs[i].set_title(lobes[i])
    axs[i].set_xlabel('time since last return (s)', color=markc, fontsize=14)
    axs[i].set_xscale('log')
    axs[i].set_ylim(0, 1.05)
    axs[0].set_ylabel('normalized dF/F', color=markc, fontsize=14)

# %% histogram of max df/f per exit/entry bout over heading (all experiment)
data = open_pickle(filename)
flylist = np.array(list(data.keys()))  # list of flies
bins = np.arange(-180, 181, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    mean_entry_heading = compute_mean_entry_heading(do)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    bl_df = flydf.loc[:first_on_index]
    # exp_df = exp_df[exp_df['instrip']] # squiggle is outside. delete the squiggle and its inside
    exp_df = exp_df[exp_df['net_motion']>0]
    fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
    axs = np.ravel(axs)
    fig.suptitle(fly)
    for i, lobe in enumerate(lobes):
        FF = exp_df[lobe]
        heading_rad = ((exp_df['heading'] + np.pi) % (2 * np.pi)) - np.pi
        heading_deg = np.rad2deg(heading_rad)
        binned_FF = [[] for _ in range(len(bin_centers))]
        for angle, ff_val in zip(heading_deg, FF):
            bin_idx = np.digitize(angle, bins) - 1
            if 0 <= bin_idx < len(binned_FF):
                binned_FF[bin_idx].append(ff_val)
        mean_FF = [np.mean(vals) if len(vals) > 0 else np.nan for vals in binned_FF]
        # Plot in current subplot
        ax = axs[i]
        for j, vals in enumerate(binned_FF):
            if vals:
                jitter = np.random.uniform(-4, 4, size=len(vals))
                ax.scatter(np.full(len(vals), bin_centers[j]), vals, color=colors[i], alpha=0.6, s=10)
        ax.plot(bin_centers, mean_FF, color=markc, linestyle='-')
        ax.set_title(f'{lobe}')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.axvline(mean_entry_heading, color=markc, linestyle=':', linewidth=1)
        ax.axhline(0, color=markc, linestyle='--')
        ax.axvline(0, color=markc, linestyle='--')
        ax.set_xlabel('heading (deg)')
        ax.set_ylabel('dF/F')
    plt.tight_layout()
    plt.show()
    
# %% df/f heatmap on trajectories
data = open_pickle(filename)
flylist = np.array(list(data.keys()))  # list of flies in the pickle

for fly in flylist:
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), facecolor='black')
    axs = axs.flatten()
    for i, lobe in enumerate(lobes):
        flydf = data[fly]['a1']
        d, di, do = inside_outside(flydf)
        first_on_index = flydf[flydf['instrip']].index[0]
        exp_df = flydf.loc[first_on_index:]

        # Normalize fluorescence for color mapping
        colour = exp_df[lobe].to_numpy().flatten()
        cmin = np.round(np.nanpercentile(colour, 1), 1)
        cmax = np.round(np.nanpercentile(colour, 99), 1)
        cmap = cm.get_cmap('coolwarm')
        norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        color_map = cmap(norm(colour))

        xo = exp_df.iloc[0]['ft_posx']
        yo = exp_df.iloc[0]['ft_posy']
        x = exp_df['ft_posx'] - xo
        y = exp_df['ft_posy'] - yo
        strip_width = 10
        # Background and strip layout
        yf = exp_df.iloc[-1]['ft_posy'] - yo
        axs[i].add_patch(patches.Rectangle((-strip_width / 2, 0), 10, yf, facecolor='grey'))    
        axs[i].add_patch(patches.Rectangle(((-strip_width / 2) - 210, 0), 10, yf, facecolor='grey'))    
        axs[i].add_patch(patches.Rectangle(((-strip_width / 2) + 210, 0), 10, yf, facecolor='grey'))    

        # Plot trajectory with fluorescence-based color
        for j in range(len(x) - 1):
            axs[i].plot(x.iloc[j:j+2], y.iloc[j:j+2], color=color_map[j+1, :3])

        axs[i].set_title(f'{fly} {lobe}', color='white')
        axs[i].set_xlabel('x-position (mm)', fontsize=12, color='white')
        axs[i].set_ylabel('y-position (mm)', fontsize=12, color='white')
        axs[i].set_aspect('equal', adjustable='datalim')

        # Axis style
        axs[i].tick_params(which='both', axis='both', labelsize=10, length=3, width=2, color='white')
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        for spine in axs[i].spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('white')

    sns.despine(offset=10)
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/traj_ff'):
        os.makedirs(f'{figure_folder}/fig/traj_ff')
    fig.savefig(f'{figure_folder}/fig/traj/{fly}_traj_ff', bbox_inches='tight')

# %%
data = open_pickle(filename)
flylist = np.array(list(data.keys()))  # list of flies in the pickle

for fly in flylist:
    if fly == '20210528_KCdLight_Fly2_003':

        fig, axs, markc = configure_bw_plot(size=(12, 12), xaxis=True, nrows=2, ncols=2)

        for i, lobe in enumerate(lobes):
            flydf = data[fly]['a1']
            d, di, do = inside_outside(flydf)
            first_on_index = flydf[flydf['instrip']].index[0]
            exp_df = flydf.loc[first_on_index:]

            colour = exp_df[lobe].to_numpy().flatten()
            cmin = np.round(np.nanpercentile(colour, 5), 1)
            cmax = np.round(np.nanpercentile(colour, 95), 1)
            cmap = cm.get_cmap('coolwarm')
            norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            color_map = cmap(norm(colour))

            xo = exp_df.iloc[0]['ft_posx']
            yo = exp_df.iloc[0]['ft_posy']
            x = exp_df['ft_posx'] - xo
            y = exp_df['ft_posy'] - yo
            yf = exp_df.iloc[-1]['ft_posy'] - yo
            strip_width = 10
            axs[i].add_patch(patches.Rectangle((-strip_width / 2, 0), 10, yf, facecolor='grey'))
            axs[i].add_patch(patches.Rectangle(((-strip_width / 2) - 210, 0), 10, yf, facecolor='grey'))
            axs[i].add_patch(patches.Rectangle(((-strip_width / 2) + 210, 0), 10, yf, facecolor='grey'))

            for j in range(len(x) - 1):
                axs[i].plot(x.iloc[j:j+2], y.iloc[j:j+2], color=color_map[j+1, :3],linewidth=1)

            axs[i].set_title(f'{fly} {lobe}', color='white', fontsize=12)
            axs[i].set_xlabel('x-position (mm)', fontsize=10, color='white')
            axs[i].set_ylabel('y-position (mm)', fontsize=10, color='white')
            axs[i].set_aspect('equal', adjustable='datalim')

        sns.despine(offset=10)
        plt.tight_layout()
        plt.show()
        if not os.path.exists(f'{figure_folder}/fig/traj_ff'):
            os.makedirs(f'{figure_folder}/fig/traj_ff')
        fig.savefig(f'{figure_folder}/fig/traj_ff/{fly}_traj_ff.pdf', bbox_inches='tight')


#%% triggered Z scored entry or exit dff (SUMMARY ACROSS FLIES, baseline-relative)
tbef = 2
taf = 2
event_type = 'exit'
first = False
data = open_pickle(filename)
flylist = np.array(list(data.keys()))

fly_avgs = {lobe: [] for lobe in lobes}
common_len = None

for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    exp_df = exp_df[exp_df['net_motion'] > 0]

    for i, lobe in enumerate(lobes):
        FF = exp_df[lobe].to_numpy()
        
        time = exp_df['relative_time'].to_numpy()
        td = exp_df['instrip'].to_numpy()
        if event_type == 'entry':
            son = np.where((td[:-1] == False) & (td[1:] == True))[0]
        elif event_type == 'exit':
            son = np.where((td[:-1] == True) & (td[1:] == False))[0]
        if len(son) == 0:
            print(f"No {event_type} events for {fly}, {lobe}")
            continue

        # --- Exclude events where the opposite-direction transition ---
        good = []
        for s in son:
            state = td[s]  # state before the transition (True=inside, for exit)
            j = s
            while j - 1 >= 0 and td[j - 1] == state:
                j -= 1
            # j is now the index of the start of this bout (i.e. the entry, for exit events)
            bout_start_time = time[j]
            gap = time[s] - bout_start_time
            if gap >= tbef:
                good.append(s)
        son = np.array(good, dtype=int)

        if len(son) == 0:
            print(f"No {event_type} events with clean pre-windows for {fly}, {lobe}")
            continue

        if first:
            son = son[:1]

        tinc = np.mean(np.diff(time))
        idx_bef = int(np.round(tbef / tinc))
        idx_af = int(np.round(taf / tinc))
        seg_len = idx_bef + idx_af + 1
        if common_len is None:
            common_len = seg_len

        mn_mat = np.full((len(son), seg_len), np.nan)
        for j, s in enumerate(son):
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            mask = (idx_array >= 0) & (idx_array < len(FF))
            valid_positions = np.where(mask)[0]
            if len(valid_positions) == 0:
                continue
            mn_mat[j, valid_positions] = FF[idx_array[mask]]

        # --- Baseline-relative z-scoring ---
        # Baseline = the pre-event window (first idx_bef samples of each row)
        # Normalize each event to the mean/std of its own baseline.
        baseline = mn_mat[:, :idx_bef]                       # (n_events, idx_bef)
        b_mean = np.nanmean(baseline, axis=1, keepdims=True) # (n_events, 1)
        b_std = np.nanstd(baseline, axis=1, keepdims=True)   # (n_events, 1)
        # Guard against zero/NaN std (flat or empty baselines)
        b_std = np.where((b_std == 0) | np.isnan(b_std), np.nan, b_std)
        mn_mat = (mn_mat - b_mean) / b_std

        # Drop events whose baseline was unusable (all-NaN row after normalization)
        valid_rows = ~np.all(np.isnan(mn_mat), axis=1)
        mn_mat = mn_mat[valid_rows]
        if mn_mat.shape[0] == 0:
            print(f"No usable baselines for {fly}, {lobe}")
            continue

        # Average across events for this fly
        fly_avg = np.nanmean(mn_mat, axis=0)

        if len(fly_avg) != common_len:
            t_src = np.linspace(-tbef, taf, len(fly_avg))
            t_dst = np.linspace(-tbef, taf, common_len)
            fly_avg = np.interp(t_dst, t_src, fly_avg)

        fly_avgs[lobe].append(fly_avg)

# Build the summary figure
fig, axs, markc = configure_white_plot(size=(10, 4), xaxis=True, ncols=len(lobes))
axs = np.ravel(axs)
fig.suptitle(f'all flies summary ({event_type}, n={len(flylist)}, baseline z-score)',
             color=markc)

t = np.linspace(-tbef, taf, common_len)

for i, lobe in enumerate(lobes):
    traces = fly_avgs[lobe]
    if len(traces) == 0:
        axs[i].set_title(f'{lobe} (no data)', color=markc)
        continue

    mat = np.vstack(traces)
    n_flies = mat.shape[0]
    grand_avg = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(n_flies)

    color = colors[i]

    # Faint individual fly traces underneath
    for trace in mat:
        axs[i].plot(t, trace, color=color, alpha=0.25, linewidth=0.8)

    # SEM band and grand average on top
    axs[i].fill_between(t, grand_avg + sem, grand_avg - sem, color=color, alpha=0.4)
    axs[i].plot(t, grand_avg, color=color, linewidth=2)
    axs[i].axhline(0, color=markc, linestyle=':', linewidth=0.6, alpha=0.5)
    axs[i].axvline(0, color=markc, linestyle='--')
    axs[i].fill_between(t, grand_avg + sem, grand_avg - sem, color=color, alpha=0.3, edgecolor=color, linewidth=0)
    # axs[i].set_ylim(-2, 8)
    axs[i].set_xlabel('time (s)', color=markc)
    axs[i].set_title(f'{lobe} (n={n_flies})', color=markc)

axs[0].set_ylabel('Z-score', color=markc)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

if not os.path.exists(f'{figure_folder}/fig/triggered_ff'):
    os.makedirs(f'{figure_folder}/fig/triggered_ff')
fig.savefig(f'{figure_folder}/fig/triggered_ff/summary_{event_type}_baselineZ_bw.pdf',
            bbox_inches='tight')

        
#%% triggered Z scored entry or exit dff (PER FLY)
tbef = 2
taf = 2
event_type = 'entry'
first = False
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle

for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    exp_df = exp_df[exp_df['net_motion'] > 0]  # Filter for net motion
    fig, axs, markc = configure_white_plot(size=(10, 4), xaxis=True, ncols=len(lobes))
    axs = np.ravel(axs)
    fig.suptitle(fly)
    for i, lobe in enumerate(lobes):
        FF = exp_df[lobe].to_numpy()
        time = exp_df['relative_time'].to_numpy()
        td = exp_df['instrip'].to_numpy()
        if event_type == 'entry':
            son = np.where((td[:-1] == False) & (td[1:] == True))[0]
        elif event_type == 'exit':
            son = np.where((td[:-1] == True) & (td[1:] == False))[0]
        if len(son) == 0:
            print(f"No {event_type} events for {fly}, {lobe}")
            continue
        if first:
            son = son[:1]
        tinc = np.mean(np.diff(time))
        idx_bef = int(np.round(tbef / tinc))
        idx_af = int(np.round(taf / tinc))
        mn_mat = np.full((len(son), idx_bef + idx_af + 1), np.nan)
        for j, s in enumerate(son):
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            idx_array = idx_array[(idx_array >= 0) & (idx_array < len(FF))]
            segment = FF[idx_array]
            mn_mat[j, :len(segment)] = segment
        mn_mat = stats.zscore(mn_mat, axis=None, nan_policy='omit')
        avg = np.nanmean(mn_mat, axis=0)
        std = np.nanstd(mn_mat, axis=0)
        t = np.linspace(-tbef, taf, len(avg))
        color = colors[i]
        axs[i].fill_between(t, avg + std, avg - std, color=color, alpha=0.3)
        axs[i].plot(t, avg, color=color)
        axs[i].axvline(0, color='white', linestyle='--')
        axs[i].fill_between(t, grand_avg + sem, grand_avg - sem,
                    color=color, alpha=0.6, edgecolor=color, linewidth=0)
        # axs[i].set_ylim(-2, 10)
        axs[0].set_ylabel('Z-score', color='white')
        axs[i].set_xlabel('time (s)', color='white')
        axs[i].set_title(lobe, color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/triggered_ff'):
        os.makedirs(f'{figure_folder}/fig/triggered_ff')
    fig.savefig(f'{figure_folder}/fig/triggered_ff/{fly}_{event_type}.pdf', bbox_inches='tight')

#%% exp wide Z triggered entry/exit traces
tbef = 2
taf = 2
event_type = 'exit'
first = False
data = open_pickle(filename)
flylist = np.array(list(data.keys()))

fly_avgs = {lobe: [] for lobe in lobes}
common_len = None

for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    exp_df = exp_df[exp_df['net_motion'] > 0]

    for i, lobe in enumerate(lobes):
        FF = exp_df[lobe].to_numpy()
        time = exp_df['relative_time'].to_numpy()
        td = exp_df['instrip'].to_numpy()

        # --- Whole-experiment baseline stats (per fly, per lobe) ---
        exp_mean = np.nanmean(FF)
        exp_std = np.nanstd(FF)
        if exp_std == 0 or np.isnan(exp_std):
            print(f"Degenerate experiment-wide std for {fly}, {lobe}")
            continue

        if event_type == 'entry':
            son = np.where((td[:-1] == False) & (td[1:] == True))[0]
        elif event_type == 'exit':
            son = np.where((td[:-1] == True) & (td[1:] == False))[0]
        if len(son) == 0:
            print(f"No {event_type} events for {fly}, {lobe}")
            continue

        # --- Exclude events where the opposite-direction transition ---
        # --- (e.g. entry, for exit events) falls within tbef of the event ---
        good = []
        for s in son:
            state = td[s]  # state before the transition (True=inside, for exit)
            j = s
            while j - 1 >= 0 and td[j - 1] == state:
                j -= 1
            # j is now the index of the start of this bout
            bout_start_time = time[j]
            gap = time[s] - bout_start_time
            if gap >= tbef:
                good.append(s)
        son = np.array(good, dtype=int)

        if len(son) == 0:
            print(f"No {event_type} events with clean pre-windows for {fly}, {lobe}")
            continue

        if first:
            son = son[:1]

        tinc = np.mean(np.diff(time))
        idx_bef = int(np.round(tbef / tinc))
        idx_af = int(np.round(taf / tinc))
        seg_len = idx_bef + idx_af + 1
        if common_len is None:
            common_len = seg_len

        mn_mat = np.full((len(son), seg_len), np.nan)
        for j, s in enumerate(son):
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            mask = (idx_array >= 0) & (idx_array < len(FF))
            valid_positions = np.where(mask)[0]
            if len(valid_positions) == 0:
                continue
            mn_mat[j, valid_positions] = FF[idx_array[mask]]

        # --- Z-score relative to whole-experiment baseline ---
        mn_mat = (mn_mat - exp_mean) / exp_std

        # Drop events that are entirely NaN (e.g. ran off the edge of the data)
        valid_rows = ~np.all(np.isnan(mn_mat), axis=1)
        mn_mat = mn_mat[valid_rows]
        if mn_mat.shape[0] == 0:
            print(f"No usable events for {fly}, {lobe}")
            continue

        # Average across events for this fly
        fly_avg = np.nanmean(mn_mat, axis=0)

        if len(fly_avg) != common_len:
            t_src = np.linspace(-tbef, taf, len(fly_avg))
            t_dst = np.linspace(-tbef, taf, common_len)
            fly_avg = np.interp(t_dst, t_src, fly_avg)

        fly_avgs[lobe].append(fly_avg)

# Build the summary figure
fig, axs, markc = configure_bw_plot(size=(10, 4), xaxis=True, ncols=len(lobes))
axs = np.ravel(axs)
fig.suptitle(f'all flies summary ({event_type}, n={len(flylist)}, experiment-wide z-score)',
             color=markc)

t = np.linspace(-tbef, taf, common_len)

for i, lobe in enumerate(lobes):
    traces = fly_avgs[lobe]
    if len(traces) == 0:
        axs[i].set_title(f'{lobe} (no data)', color=markc)
        continue

    mat = np.vstack(traces)
    n_flies = mat.shape[0]
    grand_avg = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(n_flies)

    color = colors[i]

    # Faint individual fly traces underneath
    for trace in mat:
        axs[i].plot(t, trace, color=color, alpha=0.25, linewidth=0.8)

    # SEM band and grand average on top
    axs[i].fill_between(t, grand_avg + sem, grand_avg - sem, color=color, alpha=0.4)
    axs[i].plot(t, grand_avg, color=color, linewidth=2)
    axs[i].axhline(0, color=markc, linestyle=':', linewidth=0.6, alpha=0.5)
    axs[i].axvline(0, color=markc, linestyle='--')
    axs[i].fill_between(t, grand_avg + sem, grand_avg - sem, color=color, alpha=0.3, edgecolor=color, linewidth=0)
    # axs[i].set_ylim(-2, 8)
    axs[i].set_xlabel('time (s)', color=markc)
    axs[i].set_title(f'{lobe} (n={n_flies})', color=markc)

axs[0].set_ylabel('Z-score', color=markc)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

if not os.path.exists(f'{figure_folder}/fig/triggered_ff'):
    os.makedirs(f'{figure_folder}/fig/triggered_ff')
fig.savefig(f'{figure_folder}/fig/triggered_ff/summary_{event_type}_expwideZ_bw.pdf',
            bbox_inches='tight')



#%% triggered Z-scored entry/exit dF/F — group summary across flies
data = open_pickle(filename)
flylist = np.array(list(data.keys()))
tbef = 3
taf = 3
event_type = 'entry'        # 'entry' or 'exit'
first = False               # True = use only first event per fly
max_events = None           # cap events per fly (after `first`); None = no cap
zscore_mode = 'baseline'      # 'global', 'trial', 'baseline', or 'raw'
motion_filter = False       # filter events (not samples) where fly isn't moving
plot_per_fly = False        # also produce per-fly figures

n_lobes = len(lobes)

per_fly_traces = {lobe: [] for lobe in lobes}
common_t = None
for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    try:
        first_on_index = flydf[flydf['instrip']].index[0]
    except IndexError:
        print(f'{fly}: no instrip events, skipping')
        continue
    exp_df = flydf.loc[first_on_index:].reset_index(drop=True)

    time = exp_df['relative_time'].to_numpy()
    tinc = np.nanmean(np.diff(time))
    if not np.isfinite(tinc) or tinc <= 0:
        print(f'{fly}: bad timebase, skipping')
        continue

    idx_bef = int(np.round(tbef / tinc))
    idx_af  = int(np.round(taf / tinc))
    win_len = idx_bef + idx_af + 1
    t = np.linspace(-tbef, taf, win_len)

    td = exp_df['instrip'].to_numpy()
    if event_type == 'entry':
        son = np.where((td[:-1] == False) & (td[1:] == True))[0]
    elif event_type == 'exit':
        son = np.where((td[:-1] == True) & (td[1:] == False))[0]
    else:
        raise ValueError("event_type must be 'entry' or 'exit'")

    if motion_filter and 'net_motion' in exp_df.columns:
        moving = exp_df['net_motion'].to_numpy() > 0
        son = son[moving[son]]

    if len(son) == 0:
        print(f'{fly}: no {event_type} events')
        continue
    if first:
        son = son[:1]
    elif max_events is not None:
        son = son[:max_events]

    # per-fly figure (optional)
    if plot_per_fly:
        fig_f, axs_f, markc_f = configure_bw_plot(
            size=(2.5 * n_lobes, 3.5), xaxis=True, ncols=n_lobes)
        if n_lobes == 1:
            axs_f = np.array([axs_f])

    for r_idx, lobe in enumerate(lobes):
        if lobe not in exp_df.columns:
            print(f'{fly}: missing lobe column {lobe}')
            continue
        FF = exp_df[lobe].to_numpy()
        mn_mat = np.full((len(son), win_len), np.nan)

        for i, s in enumerate(son):
            idx_array = np.arange(s - idx_bef, s + idx_af + 1, dtype=int)
            valid = (idx_array >= 0) & (idx_array < len(FF))
            idx_array = idx_array[valid]
            seg = FF[idx_array]
            mn_mat[i, :len(seg)] = seg

        # z-score within this fly+lobe
        if zscore_mode in ('none', 'raw', None):
            pass
        elif zscore_mode == 'global':
            mn_mat = stats.zscore(mn_mat, axis=None, nan_policy='omit')
        elif zscore_mode == 'trial':
            mn_mat = stats.zscore(mn_mat, axis=1, nan_policy='omit')
        elif zscore_mode == 'baseline':
            baseline = mn_mat[:, :idx_bef]
            mu = np.nanmean(baseline, axis=1, keepdims=True)
            sd = np.nanstd(baseline, axis=1, keepdims=True)
            mn_mat = (mn_mat - mu) / sd
        else:
            raise ValueError("zscore_mode must be 'global', 'trial', 'baseline', or 'raw'")

        fly_mean = np.nanmean(mn_mat, axis=0)
        per_fly_traces[lobe].append(fly_mean)
        if common_t is None or len(common_t) != len(fly_mean):
            common_t = t

        # per-fly plotting
        if plot_per_fly:
            color = None if colors is None else colors[r_idx % len(colors)]
            fly_std = np.nanstd(mn_mat, axis=0)
            ax = axs_f[r_idx]
            if color is None:
                ax.fill_between(t, fly_mean - fly_std, fly_mean + fly_std, alpha=0.3)
                ax.plot(t, fly_mean, lw=1.5)
            else:
                ax.fill_between(t, fly_mean - fly_std, fly_mean + fly_std,
                                color=color, alpha=0.3)
                ax.plot(t, fly_mean, color=color, lw=1.5)
            ax.axvline(0, color=markc_f, ls='--', lw=1)
            ax.set_title(lobe, color=markc_f, size=14)
            ax.set_xlabel('time (s)', color=markc_f, size=12)

    if plot_per_fly:
        axs_f[0].set_ylabel('Z-score', color=markc_f, size=12)
        fig_f.suptitle(f'{savename} — {fly}')
        tag = 'first' if first else 'every'
        fname_f = f"{savename}_{fly}_{'_'.join(lobes)}_{tag}_{event_type}_{zscore_mode}.pdf"
        fig_f.savefig(os.path.join(out_folder, fname_f), bbox_inches='tight')
        plt.close(fig_f)

# ---- group summary plot for this file ----
fig, axs, markc = configure_bw_plot(
    size=(2.5 * n_lobes, 3.5), xaxis=True, ncols=n_lobes)
if n_lobes == 1:
    axs = np.array([axs])

any_plotted = False
for r_idx, lobe in enumerate(lobes):
    ax = axs[r_idx]
    traces = per_fly_traces[lobe]
    if not traces:
        ax.set_title(f'{lobe} (no data)', color=markc, size=14)
        continue

    L = min(len(tr) for tr in traces)
    arr = np.vstack([tr[:L] for tr in traces])
    t_use = common_t[:L]
    n = arr.shape[0]
    color = None if colors is None else colors[r_idx % len(colors)]

    # individual flies, faint
    for row in arr:
        if color is None:
            ax.plot(t_use, row, lw=0.6, alpha=0.3)
        else:
            ax.plot(t_use, row, color=color, lw=0.6, alpha=0.3)

    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)

    if color is None:
        ax.fill_between(t_use, mean - sem, mean + sem, alpha=0.3)
        ax.plot(t_use, mean, lw=2)
    else:
        ax.fill_between(t_use, mean - sem, mean + sem, color=color, alpha=0.3)
        ax.plot(t_use, mean, color=color, lw=2)

    ax.axvline(0, color=markc, ls='--', lw=1)
    ax.set_title(f'{lobe} (n={n})', color=markc, size=14)
    ax.set_xlabel('time (s)', color=markc, size=12)
    any_plotted = True

axs[0].set_ylabel('Z-score', color=markc, size=12)
plt.show()
if not os.path.exists(f'{figure_folder}/fig/triggered_ff'):
    os.makedirs(f'{figure_folder}/fig/triggered_ff')
fig.savefig(f'{figure_folder}/fig/triggered_ff/summary_{event_type}.pdf', bbox_inches='tight')

# %%
tbef = 5
taf = 5
event_type = 'entry'
first = False

summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}

for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=(12, 3.5), xaxis=True, ncols=len(lobes))
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe].to_numpy()
        time = flydf['relative_time'].to_numpy()
        aucs = []
        return_efficiencies = []

        for key, entry_df in di.items():
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]

            # Z-score based on [-5s, +5s] window around entry
            center_idx = np.searchsorted(time, time_on)
            tinc = np.mean(np.diff(time))
            idx_bef = int(np.round(tbef / tinc))
            idx_af = int(np.round(taf / tinc))
            segment_idx = np.arange(center_idx - idx_bef, center_idx + idx_af + 1)
            segment_idx = segment_idx[(segment_idx >= 0) & (segment_idx < len(FF))]

            if len(segment_idx) == 0:
                continue

            segment = FF[segment_idx]
            z_segment = stats.zscore(segment, nan_policy='omit')
            auc_value = np.nanmax(z_segment)  # Use peak Z-score

            # Find return bout
            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 0.5 and return_to_edge(return_df):
                    pl = get_a_bout_calc(return_df, 'path_length')
                    dist = get_a_bout_calc(return_df, 'furthest_distance_from_plume')
                    if pl > 0:
                        return_eff = dist / pl
                        aucs.append(auc_value)
                        return_efficiencies.append(return_eff)

        x = np.array(aucs)
        y = np.array(return_efficiencies)
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]

        axs[i].scatter(x_clean, y_clean, color=colors[i], alpha=0.8)
        axs[i].set_title(lobes[i])
        axs[i].set_xlabel('peak Z-scored dF/F', color=markc, fontsize=14)
        axs[0].set_ylabel('return efficiency', color=markc, fontsize=14)

        if len(x_clean) >= 3 and not np.all(x_clean == x_clean[0]):
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_fn = np.poly1d(coeffs)
                y_pred = fit_fn(x_clean)
                r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
                if r_squared >= 0.3:
                    x_fit = np.linspace(min(x_clean), max(x_clean), 100)
                    axs[i].plot(x_fit, fit_fn(x_fit), linestyle='--', color=markc)
                    axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$',
                                transform=axs[i].transAxes,
                                verticalalignment='top',
                                fontsize=12,
                                color=markc)
            except np.linalg.LinAlgError:
                print(f"Linear regression failed for lobe {lobes[i]} due to matrix error.")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(fly, fontsize=14)
    plt.show()


#%%  returned peak df/f vs not returned

data = open_pickle(filename)
flylist = np.array(list(data.keys()))
auc_by_lobe = {lobe: {'returned': [], 'not_returned': []} for lobe in lobes}
fig, axs, markc = configure_bw_plot(size=(10, 4), xaxis=True, ncols=len(lobes))

for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        for key, entry_df in di.items(): #outside df/f
            if key == 2:
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)

            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue

            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            duration = time_off - time_on
            auc_value = auc / duration # for integrated
            #auc_value = FF_adjusted.max() #- FF_adjusted.min() for peak or range

            return_key = key + 1
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 1:
                    if return_to_edge(return_df):
                        auc_by_lobe[lobe]['returned'].append(auc_value)
                    else:
                        auc_by_lobe[lobe]['not_returned'].append(auc_value)

# Plotting
for i, lobe in enumerate(lobes):
    data1 = np.array(auc_by_lobe[lobe]['returned'])
    data2 = np.array(auc_by_lobe[lobe]['not_returned'])
    # Clean data
    data1 = data1[np.isfinite(data1)]
    data2 = data2[np.isfinite(data2)]
    # Scatter with jitter
    x1 = 1 + 0.1 * (np.random.rand(len(data1)) - 0.5)
    x2 = 2 + 0.1 * (np.random.rand(len(data2)) - 0.5)
    axs[i].scatter(x1, data1, color=colors[i], alpha=0.5)
    axs[i].scatter(x2, data2, facecolors='none', edgecolors=colors[i], alpha=0.8, linewidth=1.5)
    
    # Mean and STD
    groups = 2
    all_data = [data1, data2]
    averages = [np.mean(g) if len(g) > 0 else np.nan for g in all_data]
    std_devs = [np.std(g) if len(g) > 1 else 0 for g in all_data]
    print(f"\nLobe: {lobe}")
    print(f"  Returned      : mean = {averages[0]:.3f}, std = {std_devs[0]:.3f}, n = {len(data1)}")
    print(f"  Did not return: mean = {averages[1]:.3f}, std = {std_devs[1]:.3f}, n = {len(data2)}")


    for j in range(groups):
        if len(all_data[j]) == 0:
            continue
        axs[i].hlines(averages[j], xmin=j+0.9, xmax=j+1.1, colors='white', linewidth=2)
        axs[i].errorbar(j+1, averages[j], yerr=std_devs[j], color='white', capsize=0)

    # Statistical test
    if len(data1) > 1 and len(data2) > 1 and np.std(data1) > 0 and np.std(data2) > 0:
        _, p_value = ttest_ind(data1, data2, equal_var=False)
        p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
    else:
        p_text = 'n/a'
    plt.tight_layout()

    ymax = max([np.max(g) if len(g) > 0 else 0 for g in all_data]) * 1.2
    axs[i].set_title(lobe, fontsize=12, color='white')
    axs[i].text(0.5, 1.02, p_text,
                transform=axs[i].transAxes,
                ha='center', va='top',
                fontsize=10, color=markc)
    axs[i].tick_params(which='both', axis='both', labelsize=12, length=3, width=2,
                       color='black', direction='out', left=True, bottom=True)
    for pos in ['right', 'top']:
        axs[i].spines[pos].set_visible(False)
    for _, spine in axs[i].spines.items():
        spine.set_linewidth(2)
        spine.set_edgecolor('white')
    axs[i].tick_params(axis='both', colors='white')
    axs[i].set_xticks([1, 2])
    axs[i].set_xticklabels(['return', 'no return'], fontsize=16, color=markc, rotation=30)
    axs[0].set_ylabel('integrated ΔF/F', fontsize=14, color='white')
    axs[i].set_xlim(0.5, 2.5)
    # axs[i].set_ylim(bottom=0)

sns.despine(offset=10)
plt.show()


#
# %% return vs no return per fly
data = open_pickle(filename)
flylist = np.array(list(data.keys()))

for fly in flylist:
    auc_by_lobe = {lobe: {'returned': [], 'not_returned': []} for lobe in lobes}
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    time = flydf['relative_time']

    for i, lobe in enumerate(lobes):
        FF = flydf[lobe]
        for key, entry_df in do.items():
            if key == 2:
                continue
            time_on = entry_df['relative_time'].iloc[0]
            time_off = entry_df['relative_time'].iloc[-1]
            baseline_mask = (time >= time_on - 0.5) & (time < time_on)
            interval_mask = (time >= time_on) & (time <= time_off)

            if baseline_mask.sum() == 0 or interval_mask.sum() == 0:
                continue

            baseline = FF[baseline_mask].mean()
            FF_adjusted = FF[interval_mask] - baseline
            auc = np.trapz(FF_adjusted, time[interval_mask])
            auc_value = FF_adjusted.max()

            return_key = key
            if return_key in do:
                return_df = do[return_key]
                bout_dur = return_df['relative_time'].iloc[-1] - return_df['relative_time'].iloc[0]
                if bout_dur >= 1:
                    if return_to_edge(return_df):
                        auc_by_lobe[lobe]['returned'].append(auc_value)
                    else:
                        auc_by_lobe[lobe]['not_returned'].append(auc_value)

    # === PLOTTING (per file) ===
    fig, axs, markc = configure_bw_plot(size=(10, 4), xaxis=True, ncols=len(lobes))
    for i, lobe in enumerate(lobes):
        data1 = np.array(auc_by_lobe[lobe]['returned'])
        data2 = np.array(auc_by_lobe[lobe]['not_returned'])

        data1 = data1[np.isfinite(data1)]
        data2 = data2[np.isfinite(data2)]

        x1 = 1 + 0.1 * (np.random.rand(len(data1)) - 0.5)
        x2 = 2 + 0.1 * (np.random.rand(len(data2)) - 0.5)
        axs[i].scatter(x1, data1, color=colors[i], alpha=0.5)
        axs[i].scatter(x2, data2, facecolors='none', edgecolors=colors[i], alpha=0.8, linewidth=1.5)

        all_data = [data1, data2]
        averages = [np.mean(g) if len(g) > 0 else np.nan for g in all_data]
        std_devs = [np.std(g) if len(g) > 1 else 0 for g in all_data]

        print(f"\nFly: {fly}, Lobe: {lobe}")
        print(f"  Returned      : mean = {averages[0]:.3f}, std = {std_devs[0]:.3f}, n = {len(data1)}")
        print(f"  Did not return: mean = {averages[1]:.3f}, std = {std_devs[1]:.3f}, n = {len(data2)}")

        for j in range(2):
            if len(all_data[j]) == 0:
                continue
            axs[i].hlines(averages[j], xmin=j+0.9, xmax=j+1.1, colors='white', linewidth=2)
            axs[i].errorbar(j+1, averages[j], yerr=std_devs[j], color='white', capsize=0)

        if len(data1) > 1 and len(data2) > 1 and np.std(data1) > 0 and np.std(data2) > 0:
            _, p_value = ttest_ind(data1, data2, equal_var=False)
            p_text = f'p = {p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'
        else:
            p_text = 'n/a'

        axs[i].set_title(lobe, fontsize=12, color='white')
        axs[i].text(0.5, 1.02, p_text,
                    transform=axs[i].transAxes,
                    ha='center', va='top',
                    fontsize=10, color=markc)
        axs[i].tick_params(which='both', axis='both', labelsize=12, length=3, width=2,
                           color='black', direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
            axs[i].spines[pos].set_visible(False)
        for _, spine in axs[i].spines.items():
            spine.set_linewidth(2)
            spine.set_edgecolor('white')
        axs[i].tick_params(axis='both', colors='white')
        axs[i].set_xticks([1, 2])
        axs[i].set_xticklabels(['return', 'no return'], fontsize=16, color=markc, rotation=30)
        axs[0].set_ylabel('peak ΔF/F', fontsize=14, color='white')
        axs[i].set_xlim(0.5, 2.5)

    sns.despine(offset=10)
    plt.suptitle(f"Fly: {fly}", fontsize=16, color='white')
    plt.tight_layout()
    plt.show()

# %%
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3']
colors = ['#00f5ff', '#cc33cc']
window_s = 30

data = open_pickle(filename)
flylist = np.array(list(data.keys()))
print(len(flylist))
r_in_all = []
r_out_all = []
fly_ids = []
for fly in flylist:
    flydf = data[fly]['a1']  # pv2-like

    time = flydf['relative_time'].to_numpy()
    FF_g0 = flydf[rois[0]].to_numpy()
    # FF_g0 = flydf['net_motion'].to_numpy()
    FF_g1 = flydf[rois[1]].to_numpy()

    odor1_on = flydf['mfc2_stpt'].to_numpy() if 'mfc2_stpt' in flydf.columns else None
    odor2_on = flydf['mfc3_stpt'].to_numpy() if 'mfc3_stpt' in flydf.columns else None
    if odor1_on is None and odor2_on is None:
        in_odor = np.zeros_like(FF_g0, dtype=bool)
    else:
        in_odor = np.zeros_like(FF_g0, dtype=bool)
        if odor1_on is not None:
            in_odor |= (odor1_on > 0)
        if odor2_on is not None:
            in_odor |= (odor2_on > 0)
    out_odor = ~in_odor
    finite = np.isfinite(FF_g0) & np.isfinite(FF_g1)

    fig, axs, markc = configure_bw_plot(size=(10, 5), xaxis=True, nrows=2)
    fig.suptitle(f"{filename} | {fly}", color=markc)

    ax_traces = axs[0]
    ax_corr   = axs[1]

    ax_traces.plot(time, FF_g0, linewidth=0.75, color=colors[0], label=f'{rois[0]}')
    ax_traces.plot(time, FF_g1, linewidth=0.75, color=colors[1], label=f'{rois[1]}')

    if odor1_on is not None:
        shade_on_intervals(ax_traces, time, odor1_on, alpha=0.3, color='#fbceb1')
        shade_on_intervals(ax_corr,   time, odor1_on, alpha=0.2, color='#fbceb1')
    if odor2_on is not None:
        shade_on_intervals(ax_traces, time, odor2_on, alpha=0.3, color='#7ed4e6')
        shade_on_intervals(ax_corr,   time, odor2_on, alpha=0.2, color='#7ed4e6')

    ax_traces.set_ylabel('dF/F', color=markc)
    ax_traces.legend(frameon=False)

    mask_in = in_odor & finite
    if mask_in.sum() >= 3:
        r_in, _ = pearsonr(zscore(FF_g0[mask_in], nan_policy='omit'),
                           zscore(FF_g1[mask_in], nan_policy='omit'))
    else:
        r_in = np.nan
    mask_out = out_odor & finite
    if mask_out.sum() >= 3:
        r_out, _ = pearsonr(zscore(FF_g0[mask_out], nan_policy='omit'),
                            zscore(FF_g1[mask_out], nan_policy='omit'))
    else:
        r_out = np.nan
    r_in_all.append(r_in)
    r_out_all.append(r_out)
    fly_ids.append(fly)

    t_corr, r_corr = sliding_pearson(FF_g0, FF_g1, time, window_s, step_s=1)
    ax_corr.plot(t_corr, r_corr, color=markc, linewidth=1)
    ax_corr.axhline(0, linestyle='--', color='gray', linewidth=1)

    ax_corr.set_ylabel('Pearson r')
    ax_corr.set_xlabel('time (s)')
    ax_corr.set_ylim(-1, 1)

    mask = np.isfinite(FF_g0) & np.isfinite(FF_g1)
    if mask.sum() >= 3:
        r, p = pearsonr(FF_g0[mask], FF_g1[mask])
    else:
        r, p = np.nan, np.nan
    ax_corr.set_title(f"r_in-odor={r_in:.3f} | r_out-odor={r_out:.3f} | r_all={r:.3f}",color=markc,fontsize=10)

    plt.tight_layout()
    plt.show()

    if not os.path.exists(f'{figure_folder}/fig/corrs/{rois[0]}_{rois[1]}'):
        os.makedirs(f'{figure_folder}/fig/corrs/{rois[0]}_{rois[1]}')

    outname = f"{fly}_{rois[0]}_{rois[1]}_{window_s}window.pdf"
    fig.savefig(f'{figure_folder}/fig/corrs/{rois[0]}_{rois[1]}/{outname}.pdf', bbox_inches='tight')
    plt.close(fig)

r_in_all = np.array(r_in_all)
r_out_all = np.array(r_out_all)

r_in_all = np.array(r_in_all)
r_out_all = np.array(r_out_all)

valid = np.isfinite(r_in_all) & np.isfinite(r_out_all)

r_in = r_in_all[valid]
r_out = r_out_all[valid]
n = len(r_in)
stat, p = wilcoxon(r_in, r_out, alternative='two-sided')
print(f'p-value: {p}')
fig, axs, markc = configure_bw_plot(size=(3,4), xaxis=True)

for rin, rout in zip(r_in, r_out):
    axs.plot([0, 1], [rout, rin], color='gray', alpha=0.5, linewidth=1)
# axs.scatter(np.zeros(n), r_out, color='b', s=25, zorder=3)
# axs.scatter(np.ones(n),  r_in,  color='r', s=25, zorder=3)

mean_out = np.nanmean(r_out)
mean_in  = np.nanmean(r_in)
sem_out = np.nanstd(r_out, ddof=1) / np.sqrt(n)
sem_in  = np.nanstd(r_in,  ddof=1) / np.sqrt(n)
axs.errorbar(0, mean_out, yerr=sem_out, color=markc, capsize=0, linewidth=3,zorder=4)
axs.errorbar(1, mean_in, yerr=sem_in, color=markc, capsize=0, linewidth=3,zorder=4)
axs.plot([0, 1], [mean_out, mean_in], color=markc, linewidth=3, zorder=4)

axs.axhline(0, linestyle='--', color='gray', linewidth=1)
axs.set_xlim(-0.3, 1.3)
axs.set_xticks([0, 1])
axs.set_xticklabels(['out of odor', 'in odor'], color=markc)
axs.set_ylabel('Pearson r')
axs.set_title(f'Wilcoxon signed-rank\np = {p:.4g}',fontsize=8)
fig.savefig(f'{figure_folder}/fig/corrs/{rois[0]}_{rois[1]}/{rois[0]}_{rois[1]}_outside_inside.pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()



# %% g2/g4 tone vs. return efficiency
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G4']
pre_sec = 1.0
data = open_pickle(filename)
flylist = np.array(list(data.keys()))
roiA, roiB = rois[0], rois[1]
for fly in flylist:
    flydf = data[fly]['a1']  # pv2-like

    time = flydf['relative_time'].to_numpy()
    if time.size < 3 or not np.all(np.isfinite(time)):
        print(f"Skipping {filename}: bad timebase")
        continue

    dt = np.nanmean(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        print(f"Skipping {filename}: bad dt")
        continue

    pre_n = int(np.round(float(pre_sec) / dt))
    pre_n = max(pre_n, 1)

    odor1 = flydf["mfc2_stpt"].to_numpy() 
    odor2 = flydf["mfc3_stpt"].to_numpy()

    use = 'mfc2'
    odor_on = odor1.astype(bool)
    
    in_bouts  = odor_bout_indices(odor_on)   # list of (s,e), e exclusive
    out_bouts = exit_bout_indices(odor_on)   # list of (s,e), e exclusive

    if len(in_bouts) == 0 or len(out_bouts) == 0:
        print(f"Skipping {filename}: no bouts found")
        continue

    out_by_start = {int(s): (int(s), int(e)) for s, e in out_bouts}

    FF_A = (flydf[roiA]/100).to_numpy()
    FF_B = (flydf[roiB]/100).to_numpy()

    x_eff = []
    y_ratio = []

    for in_s, in_e in in_bouts:
        in_s = int(in_s); in_e = int(in_e)

        if in_s - pre_n < 0:
            continue

        nxt_out = out_by_start.get(in_e, None)
        if nxt_out is None:
            continue
        out_s, out_e = nxt_out
        if out_e - out_s < 2:
            continue

        # --- compute return efficiency on OUT bout trajectory ---
        out_df = flydf.iloc[out_s:out_e].copy()

        furthest = get_a_bout_calc(out_df, "furthest_distance_from_plume")
        pathlen  = get_a_bout_calc(out_df, "path_length")

        if not (np.isfinite(furthest) and np.isfinite(pathlen)) or pathlen <= 1e-12:
            continue

        return_eff = float(furthest) / float(pathlen)

        # --- compute normalized integrated activity ratio on IN bout ---
        pre_idx  = np.arange(in_s - pre_n, in_s, dtype=int)
        bout_idx = np.arange(in_s, in_e, dtype=int)
        if bout_idx.size < 2:
            continue

        t_pre  = time[pre_idx]
        t_bout = time[bout_idx]

        Ipre_A  = np.trapz(FF_A[pre_idx],  t_pre)
        Ibout_A = np.trapz(FF_A[bout_idx], t_bout)
        Ipre_B  = np.trapz(FF_B[pre_idx],  t_pre)
        Ibout_B = np.trapz(FF_B[bout_idx], t_bout)

        if not all(np.isfinite(v) for v in [Ipre_A, Ibout_A, Ipre_B, Ibout_B]):
            continue

        if (Ipre_A <= 1e-12) or (Ipre_B <= 1e-12):
            continue
       

        normA = Ibout_A / Ipre_A
        normB = Ibout_B / Ipre_B

        if not (np.isfinite(normA) and np.isfinite(normB)) or abs(normB) < 1e-12:
            continue

        ratio = float(normB / normA)

        x_eff.append(return_eff)
        y_ratio.append(ratio)

    fig, ax, markc = configure_bw_plot(size=(5, 5), xaxis=True, ncols=1)

    ax.scatter(x_eff, y_ratio, s=20, alpha=0.8)
    ax.set_title(f"{roiB}/{roiA} normalized AUC ratio vs next OUT return efficiency",
                    color=markc, fontsize=10)
    ax.set_xlabel("return efficiency (furthest distance / path length)", fontsize=12)
    ax.set_ylabel(f"normalized integrated activity ratio: {roiA}/{roiB}", fontsize=12)

    # regression
    if len(x_eff) >= 2:
        x = np.asarray(x_eff, dtype=float)
        y = np.asarray(y_ratio, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        n_valid = np.count_nonzero(valid)
        if n_valid >= 2:
            xv = x[valid]
            yv = y[valid]
            if (np.nanstd(xv) > 0) and (np.nanstd(yv) > 0):
                r, p = pearsonr(xv, yv)
            else:
                r, p = np.nan, np.nan

            # line fit (optional; keep your existing polyfit)
            slope, intercept = np.polyfit(xv, yv, 1)
            x_fit = np.linspace(xv.min(), xv.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, linewidth=2)

            # R^2 from the same fit (your existing code)
            y_pred = slope * xv + intercept
            ss_res = np.sum((yv - y_pred) ** 2)
            ss_tot = np.sum((yv - np.mean(yv)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            ax.text(
                0.05, 0.95,
                f"slope = {slope:.3g}\nR² = {r2:.3f}\nPearson r = {r:.3f}\np = {p:.3g}\n(n={n_valid})",
                transform=ax.transAxes,
                va="top"
            )




# %% integrated relative activity 

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
pre_sec = 1.0
data = open_pickle(filename)
flylist = np.array(list(data.keys()))
fig, ax, markc = configure_bw_plot(size=(4, 3), xaxis=True, ncols=1)
n_rois = len(rois)
x = np.arange(len(rois))

for fly in flylist:
    flydf = data[fly]['a1']
    time = flydf['relative_time'].to_numpy()
    dt = np.nanmean(np.diff(time))
    pre_n = max(1, int(np.round(pre_sec / dt)))

    odor_on = (flydf["mfc2_stpt"].to_numpy() != 0).astype(bool)
    bouts = odor_bout_indices(odor_on)

    fractions_per_bout = []

    for (s, e) in bouts:
        s = int(s); e = int(e)
        if s - pre_n < 0:
            continue

        pre_idx  = np.arange(s - pre_n, s, dtype=int)
        bout_idx = np.arange(s, e, dtype=int)
        if bout_idx.size < 2:
            continue

        t_pre  = time[pre_idx]
        t_bout = time[bout_idx]

        norm_ints = np.full(n_rois, np.nan, float)
        for r_idx, roi in enumerate(rois):
            FF = flydf[roi].to_numpy()
            loc_reshaped = flydf['net_motion'].reshape(-1, 1)
            # Fit linear model: FF ~ loc
            reg = LinearRegression().fit(loc_reshaped, FF)
            # Predicted FF due to motion alone
            FF_motion_predicted = reg.predict(loc_reshaped)
            # Residuals = FF with motion contribution removed
            FF_regressed = FF - FF_motion_predicted
            baseline = np.nanmean(FF_regressed)
            pre_sig  = FF_regressed[pre_idx]
            bout_sig = FF_regressed[bout_idx]
            I_pre  = np.trapz(pre_sig,  t_pre)
            I_bout = np.trapz(bout_sig, t_bout)

            # I_pre  = np.trapz(FF[pre_idx],  t_pre)
            # I_bout = np.trapz(FF[bout_idx], t_bout)
            if not (np.isfinite(I_pre) and np.isfinite(I_bout)):
                continue
            if abs(I_pre) < 1e-12:
                continue
            norm_ints[r_idx] = I_bout / I_pre

        if np.all(~np.isfinite(norm_ints)):
            continue
        denom = np.nansum(norm_ints)
        if (not np.isfinite(denom)) or (abs(denom) < 1e-12):
            continue

        fractions_per_bout.append(norm_ints / denom)

    # ---- plot ONE line for this fly (mean across its bouts) ----
    if len(fractions_per_bout) == 0:
        continue

    frac_mat  = np.vstack(fractions_per_bout)   # (n_bouts_for_fly, n_rois)
    frac_mean = np.nanmean(frac_mat, axis=0)

    ax.plot(x, frac_mean, marker="o", linewidth=2.0, color=markc, alpha=0.35)

ax.set_xticks(x)
ax.set_xticklabels(rois, rotation=45, ha="right", color=markc)
ax.set_ylabel("relative activity", color=markc)
ax.set_ylim(0, 0.5)
plt.tight_layout()
plt.show()

        # if not os.path.exists(f'{figure_folder}/fig/tone'):
        #     os.makedirs(f'{figure_folder}/fig/tone')
        # fig.savefig(f'{figure_folder}/fig/tone/{filename}', bbox_inches='tight')


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
pre_sec = 1.0

data = open_pickle(filename)
flylist = np.array(list(data.keys()))

n_rois = len(rois)
x = np.arange(len(rois))

# ------------------ FIRST PASS: per-fly frac_mean (IN bouts) + avg return eff (OUT bouts) ------------------
fly_frac_mean = {}
fly_avg_eff   = {}

for fly in flylist:
    flydf = data[fly]['a1']
    time = flydf['relative_time'].to_numpy()
    dt = np.nanmean(np.diff(time))
    pre_n = max(1, int(np.round(pre_sec / dt)))
    pl = get_a_bout_calc(flydf, 'path_length') / 1000

    odor_on = (flydf["mfc2_stpt"].to_numpy() != 0).astype(bool)

    # ----- IN bouts for FF fractions -----
    in_bouts = odor_bout_indices(odor_on)
    fractions_per_bout = []

    for (s, e) in in_bouts:
        s = int(s); e = int(e)
        if s - pre_n < 0:
            continue

        pre_idx  = np.arange(s - pre_n, s, dtype=int)
        bout_idx = np.arange(s, e, dtype=int)
        if bout_idx.size < 2:
            continue

        t_pre  = time[pre_idx]
        t_bout = time[bout_idx]

        norm_ints = np.full(n_rois, np.nan, float)
        for r_idx, roi in enumerate(rois):
            FF = flydf[roi].to_numpy()
            I_pre  = np.trapz(FF[pre_idx],  t_pre)
            I_bout = np.trapz(FF[bout_idx], t_bout)
            if not (np.isfinite(I_pre) and np.isfinite(I_bout)):
                continue
            if abs(I_pre) < 1e-12:
                continue
            norm_ints[r_idx] = I_bout / I_pre

        if np.all(~np.isfinite(norm_ints)):
            continue
        denom = np.nansum(norm_ints)
        if (not np.isfinite(denom)) or (abs(denom) < 1e-12):
            continue

        fractions_per_bout.append(norm_ints / denom)

    if len(fractions_per_bout) == 0:
        continue

    frac_mat  = np.vstack(fractions_per_bout)

    frac_mean = np.nanmean(frac_mat, axis=0)
    eps = 1e-6
    finite = np.isfinite(frac_mean)
    if np.any(frac_mean[finite] < -eps) or np.any(frac_mean[finite] > 1.0 + eps):
        continue

    fly_frac_mean[fly] = frac_mean

    # ----- OUT bouts for return efficiency -----
    out_bouts = exit_bout_indices(odor_on)
    effs_per_bout = []
    for (out_s, out_e) in out_bouts:
        out_s = int(out_s); out_e = int(out_e)
        if out_e - out_s < 2:
            continue

        out_df = flydf.iloc[out_s:out_e].copy()
        furthest = get_a_bout_calc(out_df, "furthest_distance_from_plume")
        pathlen  = get_a_bout_calc(out_df, "path_length")

        effs_per_bout.append(furthest / pathlen)

    # effs_per_bout.append(len(out_bouts)-1/pl)

    fly_avg_eff[fly] = np.nanmean(effs_per_bout) if len(effs_per_bout) else np.nan
    


# ------------------ COLOR MAP: darkest = highest avg return efficiency ------------------
flies = list(fly_frac_mean.keys())
eff_vals = np.array([fly_avg_eff[f] for f in flies], dtype=float)
valid = np.isfinite(eff_vals)

# Percentile-based mapping (evenly spreads colors across flies)
pct = np.full_like(eff_vals, np.nan, dtype=float)
eff_valid = eff_vals[valid]
order = np.argsort(eff_valid)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(order.size)
pct_valid = ranks / (max(1, order.size - 1))  # 0..1
pct[valid] = pct_valid

light_hex = "white"   # lowest efficiency
dark_hex  = "blue"   # highest efficiency

cmap = mpl.colors.LinearSegmentedColormap.from_list("eff_cmap", [light_hex, dark_hex], N=256)

fig, ax, markc = configure_bw_plot(size=(5,3), xaxis=True, ncols=1)

for i, fly in enumerate(flies):
    frac_mean = fly_frac_mean[fly]
    if np.isfinite(pct[i]):
        color = cmap(pct[i])   # pct must be in [0,1]
    else:
        color = (0.7, 0.7, 0.7, 1.0)

    ax.plot(x, frac_mean, marker="o", linewidth=2.0, color=color)

ax.set_xticks(x)
ax.set_xticklabels(rois, rotation=45, ha="right", color=markc)
ax.set_ylabel("relative activity", color=markc)
ax.set_ylim(0, 1)

# optional colorbar for percentile
sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("avg return efficiency percentile", color=markc)

plt.tight_layout()
plt.show()
if not os.path.exists(f'{figure_folder}/fig/tone'):
    os.makedirs(f'{figure_folder}/fig/tone')
fig.savefig(f'{figure_folder}/fig/tone/gradient_entries_freq.pdf', bbox_inches='tight')


# %% duration sorted DA release profiles -- per fly plots
figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff','#cc33cc', '#ff43a4']
pre_s = 1.0
post_s = 15.0
min_post_after_trunc = 1.0
data = open_pickle(filename)
flylist = np.array(list(data.keys()))

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)  # global odor-on bouts

    if len(bouts) == 0:
        print(f"{fly}: no odor bouts")
        continue

    roi_traces = {roi: [] for roi in rois}
    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]

        if end_idx - 1 <= start_idx:
            odor_dur = 0.0
        else:
            odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s

        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset

        if t1 <= onset + min_post_after_trunc:
            continue
        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        tol = 0.25
        if win['t_rel'].min() > (-pre_s + tol):
            continue

        for roi in rois:
            t_rel = win['t_rel'].to_numpy()
            ff = win[roi].to_numpy()
            good = np.isfinite(t_rel) & np.isfinite(ff)
            if good.sum() < 2:
                continue
            roi_traces[roi].append((odor_dur, t_rel[good], ff[good], j))

    fig, axes, markc = configure_bw_plot(size=(3*len(rois),4), xaxis=True, ncols=len(rois))
    if len(rois) == 1:
        axes = [axes]

    for ax, roi, color in zip(axes, rois, colors):
        traces = roi_traces[roi]
        if len(traces) == 0:
            ax.set_title(f'{fly} - {roi} (no valid odor bouts)')
            continue

        # shortest odor bouts first
        traces_sorted = sorted(traces, key=lambda x: x[0])

        for y_idx, (odor_dur, t_rel, ff, bout_j) in enumerate(traces_sorted):
            y_base = y_idx

            # --- duration bar (from odor onset=0 to odor offset=odor_dur) ---
            # if odor_dur <= post_s:
            #     ax.hlines(y=y_base - 0.18,xmin=0,xmax=max(0, odor_dur),color='orange',linewidth=2,alpha=0.35)
            # else:
            #     ax.hlines(y=y_base - 0.18,xmin=0,xmax=post_s,color='orange',linewidth=2,alpha=0.35)

            # trace (normalized and vertically offset around the row)
            ff0 = ff - np.nanmean(ff)
            ff_scaled = ff0 / (np.nanstd(ff0) + 1e-9)
            y = y_base + 0.25 * ff_scaled
            ax.plot(t_rel, y, lw=1, color=color)
            ax.axvline(0, linestyle='--', color=markc, linewidth=0.75)
            ax.set_title(roi)
    plt.suptitle(fly)
    axes[0].set_ylabel('increasing bout duration')
    axes[0].set_xlabel('time from odor onset (s)')
    plt.tight_layout()
    plt.show()


# %% duration sorted DA release profiles MOTION REGRESSED OUT -- per fly plots

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff','#cc33cc', '#ff43a4']
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0
data = open_pickle(filename)
flylist = np.array(list(data.keys()))

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)  # DECIDE ENTRY/EXIT

    if len(bouts) == 0:
        print(f"{fly}: no odor bouts")
        continue
    
    # --- regress out motion from full session, per ROI ---
    loc_full = flydf['speed'].to_numpy()
    good_full = np.isfinite(loc_full)
    roi_corrected = {}
    for roi in rois:
        ff_full = flydf[roi].to_numpy() # raw dF/F
        good = good_full & np.isfinite(ff_full)
        if good.sum() < 2:
            print(f"{fly} - {roi}: not enough valid samples, skipping")
            continue
        reg = LinearRegression().fit(loc_full[good].reshape(-1,1), ff_full[good])
        print(f"{fly} - {roi}: R² = {reg.score(loc_full[good].reshape(-1,1), ff_full[good]):.3f}")
        ff_corrected = ff_full.copy()
        ff_corrected[good] = ff_full[good] - reg.predict(loc_full[good].reshape(-1,1)) + ff_full[good].mean()
        flydf[roi] = ff_full # decide to use full or corrected

    roi_traces = {roi: [] for roi in rois}
    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]

        if end_idx - 1 <= start_idx: # effective duration filter
            # odor_dur = 0.0
            continue
        else:
            odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s

        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset

        if t1 <= onset + min_post_after_trunc:
            continue
        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        tol = 0.25
        if win['t_rel'].min() > (-pre_s + tol):
            continue

        for roi in rois:
            t_rel = win['t_rel'].to_numpy()
            ff = win[roi].to_numpy()
            loc = win['speed'].to_numpy() # NET MOTION REGRESSOR
            good = np.isfinite(t_rel) & np.isfinite(ff)
            if good.sum() < 2:
                continue
            t_rel_g = t_rel[good]
            ff_g    = ff[good]
            loc_g   = loc[good]

            roi_traces[roi].append((odor_dur, t_rel_g, ff_g, j))

    fig, axes, markc = configure_bw_plot(size=(3*len(rois),(3*(len(bouts)/10))), xaxis=True, ncols=len(rois))
    if len(rois) == 1:
        axes = [axes]

    for ax, roi, color in zip(axes, rois, colors):
        traces = roi_traces[roi]
        if len(traces) == 0:
            ax.set_title(f'{fly} - {roi} (no valid odor bouts)')
            continue

        # shortest odor bouts first
        traces_sorted = sorted(traces, key=lambda x: x[0])

        for y_idx, (odor_dur, t_rel, ff, bout_j) in enumerate(traces_sorted):
            y_base = y_idx

            # --- duration bar (from odor onset=0 to odor offset=odor_dur) ---
            if odor_dur <= post_s:
                ax.hlines(y=y_base - 0.18,xmin=0,xmax=max(0, odor_dur),color=markc,linewidth=1,alpha=0.8)
            else:
                ax.hlines(y=y_base - 0.18,xmin=0,xmax=post_s,color='orange',linewidth=2,alpha=0.5)
            # trace (normalized and vertically offset around the row)
            ff0 = ff - np.nanmean(ff)
            ff_scaled = ff0 / (np.nanstd(ff0) + 1e-9)
            y = y_base + 0.25 * ff_scaled
            ax.plot(t_rel, y, lw=1, color=color)
            # ax.axvline(0, linestyle='--', color=markc, linewidth=0.75)
            ax.set_title(roi)
    plt.suptitle(fly)
    axes[0].set_ylabel('increasing bout duration')
    axes[0].set_xlabel('time from odor onset (s)')
    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/sorted_traces'):
        os.makedirs(f'{figure_folder}/fig/sorted_traces')
    fig.savefig(f'{figure_folder}/fig/sorted_traces/{fly}_sorted_raw_entries.pdf', bbox_inches='tight')


# %% duration-binned DA release heatmaps MOTION REGRESSED OUT -- per fly plots

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff', '#cc33cc', '#ff43a4']
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0

# --- binning parameters ---
n_duration_bins = 6           # number of duration bins
dt_common = 0.05              # common time grid resolution (s) for interpolation
# bin edges will be defined per-fly based on observed durations

data = open_pickle(filename)
flylist = np.array(list(data.keys()))

# common time axis used for interpolation/heatmap
t_common = np.arange(-pre_s, post_s + dt_common, dt_common)

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)

    if len(bouts) == 0:
        print(f"{fly}: no odor bouts")
        continue

    # --- regress out motion from full session, per ROI ---
    loc_full = flydf['speed'].to_numpy()
    good_full = np.isfinite(loc_full)
    for roi in rois:
        ff_full = flydf[roi].to_numpy()
        good = good_full & np.isfinite(ff_full)
        if good.sum() < 2:
            print(f"{fly} - {roi}: not enough valid samples, skipping")
            continue
        reg = LinearRegression().fit(loc_full[good].reshape(-1, 1), ff_full[good])
        print(f"{fly} - {roi}: R² = {reg.score(loc_full[good].reshape(-1, 1), ff_full[good]):.3f}")
        ff_corrected = ff_full.copy()
        ff_corrected[good] = ff_full[good] - reg.predict(loc_full[good].reshape(-1, 1)) + ff_full[good].mean()
        flydf[roi] = ff_corrected  # decide to use full or corrected

    # --- collect traces per ROI, interpolated onto common time grid ---
    roi_traces = {roi: [] for roi in rois}  # list of (odor_dur, ff_interp)
    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]

        if end_idx - 1 <= start_idx:
            continue
        odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s

        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset

        if t1 <= onset + min_post_after_trunc:
            continue

        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        tol = 0.25
        if win['t_rel'].min() > (-pre_s + tol):
            continue

        for roi in rois:
            t_rel = win['t_rel'].to_numpy()
            ff = win[roi].to_numpy()
            good = np.isfinite(t_rel) & np.isfinite(ff)
            if good.sum() < 2:
                continue
            t_rel_g = t_rel[good]
            ff_g = ff[good]

            # interpolate onto common time grid; mask times outside available window
            ff_interp = np.full_like(t_common, np.nan, dtype=float)
            valid_mask = (t_common >= t_rel_g.min()) & (t_common <= t_rel_g.max())
            if valid_mask.sum() >= 2:
                ff_interp[valid_mask] = np.interp(t_common[valid_mask], t_rel_g, ff_g)

            roi_traces[roi].append((odor_dur, ff_interp))

    # --- determine duration bin edges from this fly's data (pooled across ROIs) ---
    all_durs = [d for roi in rois for (d, _) in roi_traces[roi]]
    if len(all_durs) == 0:
        print(f"{fly}: no valid traces after filtering")
        continue

    dur_min = np.min(all_durs)
    dur_max = np.max(all_durs)
    if dur_max == dur_min:
        # single duration; just one bin
        bin_edges = np.array([dur_min - 1e-6, dur_max + 1e-6])
    else:
        bin_edges = np.linspace(dur_min, dur_max, n_duration_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # --- build heatmap matrix per ROI: rows = duration bins, cols = time ---
    fig, axes, markc = configure_white_plot(size=(12,4),xaxis=True, ncols=len(rois))
    if len(rois) == 1:
        axes = [axes]

    # shared color limits across ROIs for this fly (symmetric around 0)
    all_means = []
    roi_heatmaps = {}
    roi_counts = {}
    for roi in rois:
        traces = roi_traces[roi]
        heatmap = np.full((n_bins, len(t_common)), np.nan)
        counts = np.zeros(n_bins, dtype=int)
        if len(traces) > 0:
            durs = np.array([d for d, _ in traces])
            ff_stack = np.vstack([f for _, f in traces])  # (n_traces, n_time)

            # baseline subtract per trace using pre-odor period
            pre_mask = t_common < 0
            baseline = np.nanmean(ff_stack[:, pre_mask], axis=1, keepdims=True)
            ff_stack_bs = ff_stack - baseline

            # assign each trace to a bin
            bin_idx = np.digitize(durs, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            for b in range(n_bins):
                sel = bin_idx == b
                counts[b] = sel.sum()
                if sel.any():
                    heatmap[b, :] = np.nanmean(ff_stack_bs[sel], axis=0)

        roi_heatmaps[roi] = heatmap
        roi_counts[roi] = counts
        if np.any(np.isfinite(heatmap)):
            all_means.append(np.nanmax(np.abs(heatmap)))

    vmax = np.nanmax(all_means) if len(all_means) > 0 else 1.0
    vmin = -vmax

    extent = [t_common[0], t_common[-1], -0.5, n_bins - 0.5]

    for ax, roi in zip(axes, rois):
        heatmap = roi_heatmaps[roi]
        counts = roi_counts[roi]

        im = ax.imshow(
            heatmap,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='RdBu_r',
            vmin=vmin, vmax=vmax,
            interpolation='nearest'
        )
        ax.axvline(0, linestyle='--', color='k', linewidth=0.75)
        for b in range(n_bins):
            x_edge = bin_edges[b + 1]   # upper edge of this bin
            ax.plot([x_edge, x_edge], [b - 0.5, b + 0.5],
                    linestyle=':', color='k', linewidth=0.75)
        ax.set_title(f'{roi}')
        ax.set_yticks(np.arange(n_bins))
        ax.set_yticklabels([f'{bin_edges[b]:.1f}–{bin_edges[b+1]:.1f}s\n(n={counts[b]})'
                            for b in range(n_bins)], fontsize=7)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('ΔF/F (baseline-sub)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    axes[0].set_ylabel('odor duration bin')
    for ax in axes:
        ax.set_xlabel('time from odor onset (s)')
    plt.suptitle(fly)
    plt.tight_layout()
    plt.show()

    out_dir = f'{figure_folder}/fig/duration_binned_heatmaps'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{fly}_duration_binned_heatmap.pdf', bbox_inches='tight')


# %% duration-binned DA release heatmaps MOTION REGRESSED OUT -- pooled across flies

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff', '#cc33cc', '#ff43a4']
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0

bin_width = 2.0
n_duration_bins = 5
bin_edges = np.arange(0, bin_width * n_duration_bins + bin_width, bin_width)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
n_bins = len(bin_centers)
dt_common = 0.05    

data = open_pickle(filename)
flylist = np.array(list(data.keys()))

t_common = np.arange(-pre_s, post_s + dt_common, dt_common)

# --- collect traces from ALL flies, per ROI ---
roi_traces_all = {roi: [] for roi in rois}  # list of (odor_dur, ff_interp, fly)

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = exit_bout_indices(odor_on)

    if len(bouts) == 0:
        print(f"{fly}: no odor bouts")
        continue

    # --- regress out motion from full session, per ROI ---
    loc_full = flydf['speed'].to_numpy()
    good_full = np.isfinite(loc_full)
    for roi in rois:
        ff_full = flydf[roi].to_numpy()
        good = good_full & np.isfinite(ff_full)
        if good.sum() < 2:
            print(f"{fly} - {roi}: not enough valid samples, skipping")
            continue
        reg = LinearRegression().fit(loc_full[good].reshape(-1, 1), ff_full[good])
        print(f"{fly} - {roi}: R² = {reg.score(loc_full[good].reshape(-1, 1), ff_full[good]):.3f}")
        ff_corrected = ff_full.copy()
        ff_corrected[good] = ff_full[good] - reg.predict(loc_full[good].reshape(-1, 1)) + ff_full[good].mean()
        flydf[roi] = ff_corrected # decide to use full or corrected

    # --- collect traces interpolated onto common time grid ---
    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]

        if end_idx - 1 <= start_idx:
            continue
        odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s

        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset

        if t1 <= onset + min_post_after_trunc:
            continue

        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        tol = 0.25
        if win['t_rel'].min() > (-pre_s + tol):
            continue

        for roi in rois:
            t_rel = win['t_rel'].to_numpy()
            ff = win[roi].to_numpy()
            good = np.isfinite(t_rel) & np.isfinite(ff)
            if good.sum() < 2:
                continue
            t_rel_g = t_rel[good]
            ff_g = ff[good]

            ff_interp = np.full_like(t_common, np.nan, dtype=float)
            valid_mask = (t_common >= t_rel_g.min()) & (t_common <= t_rel_g.max())
            if valid_mask.sum() >= 2:
                ff_interp[valid_mask] = np.interp(t_common[valid_mask], t_rel_g, ff_g)

            roi_traces_all[roi].append((odor_dur, ff_interp, fly))

# --- determine duration bin edges from pooled data ---
all_durs = [d for roi in rois for (d, _, _) in roi_traces_all[roi]]
if len(all_durs) == 0:
    raise RuntimeError("No valid traces found across any fly")

# --- build heatmap matrix per ROI: rows = duration bins, cols = time ---
roi_heatmaps = {}
roi_counts = {}        # number of trials per bin
roi_fly_counts = {}    # number of unique flies contributing per bin
all_abs_max = []

for roi in rois:
    traces = roi_traces_all[roi]
    heatmap = np.full((n_bins, len(t_common)), np.nan)
    counts = np.zeros(n_bins, dtype=int)
    fly_counts = np.zeros(n_bins, dtype=int)

    if len(traces) > 0:
        durs = np.array([d for d, _, _ in traces])
        ff_stack = np.vstack([f for _, f, _ in traces])
        fly_arr = np.array([fl for _, _, fl in traces])

        # baseline subtract per trace
        pre_mask = t_common < 0
        baseline = np.nanmean(ff_stack[:, pre_mask], axis=1, keepdims=True)
        ff_stack_bs = ff_stack - baseline

        bin_idx = np.digitize(durs, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            sel = bin_idx == b
            counts[b] = sel.sum()
            fly_counts[b] = len(np.unique(fly_arr[sel])) if sel.any() else 0
            if sel.any():
                heatmap[b, :] = np.nanmean(ff_stack_bs[sel], axis=0)

    roi_heatmaps[roi] = heatmap
    roi_counts[roi] = counts

# --- plot ---
fig, axes, markc = configure_bw_plot(size=(12,3),xaxis=True, ncols=len(rois))
if len(rois) == 1:
    axes = [axes]

extent = [t_common[0], t_common[-1], -0.5, n_bins - 0.5]

for ax, roi in zip(axes, rois):
    heatmap = roi_heatmaps[roi]
    counts = roi_counts[roi]

    # per-compartment symmetric color scale
    if np.any(np.isfinite(heatmap)):
        roi_vmax = np.nanmax(np.abs(heatmap))
    else:
        roi_vmax = 1.0
    roi_vmin = -roi_vmax
    RdBkBu_r = LinearSegmentedColormap.from_list('RdBkBu_r', ['#89cff0', '#4575b4', 'black', '#960018', '#ff0800'])

    im = ax.imshow(heatmap,aspect='auto',origin='lower',extent=extent,cmap=RdBkBu_r,vmin=roi_vmin, vmax=roi_vmax,interpolation='nearest')
    ax.axvline(0, linestyle='--', color=markc, linewidth=0.75)
    # per-row dotted line at the upper edge of each duration bin
    for b in range(n_bins):
        x_edge = bin_edges[b + 1]   # upper edge of this bin
        print(x_edge)
        ax.plot([x_edge, x_edge], [b - 0.5, b + 0.5],
                linestyle=':', color=markc, linewidth=0.75)
    ax.set_title(f'{roi}')
    ax.set_yticks(np.arange(n_bins))
    ax.set_yticklabels(
        [f'{bin_edges[b]:.0f}–{bin_edges[b+1]:.0f}s' for b in range(n_bins)],
        fontsize=8
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('ΔF/F (baseline-sub)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

axes[0].set_ylabel('bout duration')
for ax in axes:
    ax.set_xlabel('time from exit (s)')

plt.suptitle(f'pooled across {len(flylist)} flies')
plt.tight_layout()
plt.show()

out_dir = f'{figure_folder}/fig/duration_binned_heatmaps'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/pooled_duration_binned_exit_reg_heatmap_bw.pdf', bbox_inches='tight')



# %% time-to-peak and peak width per compartment, pooled across flies

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
rois = ['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff', '#cc33cc', '#ff43a4']
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0
dt_common = 0.05

# --- peak detection parameters ---
peak_search_window = (0.0, post_s)   # search for peak within this window (s, relative to onset)
width_height_frac = 0.5              # peak width measured at this fraction of peak height (0.5 = FWHM)
min_peak_prominence = None           # set to a float (in dF/F units) to require minimum prominence; None = no filter

data = open_pickle(filename)
flylist = np.array(list(data.keys()))
t_common = np.arange(-pre_s, post_s + dt_common, dt_common)

# --- collect traces per ROI ---
roi_traces_all = {roi: [] for roi in rois}  # list of (odor_dur, ff_interp, fly)

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)

    if len(bouts) == 0:
        continue

    # motion regression (kept consistent with heatmap script)
    loc_full = flydf['speed'].to_numpy()
    good_full = np.isfinite(loc_full)
    for roi in rois:
        ff_full = flydf[roi].to_numpy()
        good = good_full & np.isfinite(ff_full)
        if good.sum() < 2:
            continue
        reg = LinearRegression().fit(loc_full[good].reshape(-1, 1), ff_full[good])
        ff_corrected = ff_full.copy()
        ff_corrected[good] = ff_full[good] - reg.predict(loc_full[good].reshape(-1, 1)) + ff_full[good].mean()
        flydf[roi] = ff_full # _corrected  # using raw, matching heatmap script

    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]
        if end_idx - 1 <= start_idx:
            continue
        odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue
        if odor_dur < 1:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s
        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset
        if t1 <= onset + min_post_after_trunc:
            continue

        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        if win['t_rel'].min() > (-pre_s + 0.25):
            continue

        for roi in rois:
            t_rel = win['t_rel'].to_numpy()
            ff = win[roi].to_numpy()
            good = np.isfinite(t_rel) & np.isfinite(ff)
            if good.sum() < 2:
                continue
            t_rel_g = t_rel[good]
            ff_g = ff[good]

            ff_interp = np.full_like(t_common, np.nan, dtype=float)
            valid_mask = (t_common >= t_rel_g.min()) & (t_common <= t_rel_g.max())
            if valid_mask.sum() >= 2:
                ff_interp[valid_mask] = np.interp(t_common[valid_mask], t_rel_g, ff_g)

            roi_traces_all[roi].append((odor_dur, ff_interp, fly))

# --- per-trial peak metrics ---
def compute_peak_metrics(t_axis, trace, odor_dur, height_frac, min_prom,baseline_sd_mult=2.0):
    """Return (time_to_peak, peak_amp, peak_width, time_above_thresh) for a single trace.
    Peak is searched within [0, odor_dur]. Time-above-threshold uses 2x SD of pre-odor
    baseline as the threshold and counts total time above it within [0, odor_dur]."""
    pre_mask = t_axis < 0
    if not np.any(np.isfinite(trace[pre_mask])):
        return np.nan, np.nan, np.nan, np.nan
    baseline_vals = trace[pre_mask]
    baseline = np.nanmean(baseline_vals)
    baseline_sd = np.nanstd(baseline_vals, ddof=1) if np.sum(np.isfinite(baseline_vals)) > 1 else np.nan
    trace_bs = trace - baseline

    search_mask = (t_axis >= 0) & (t_axis <= odor_dur)
    if not np.any(np.isfinite(trace_bs[search_mask])):
        return np.nan, np.nan, np.nan, np.nan

    seg = trace_bs[search_mask].copy()
    seg_t = t_axis[search_mask]

    if np.all(np.isnan(seg)):
        return np.nan, np.nan, np.nan, np.nan
    peak_local = int(np.nanargmax(seg))
    peak_amp = seg[peak_local]
    peak_time = seg_t[peak_local]

    if min_prom is not None:
        pre_peak = seg[:peak_local + 1]
        if pre_peak.size == 0 or np.all(np.isnan(pre_peak)):
            return np.nan, np.nan, np.nan, np.nan
        prom = peak_amp - np.nanmin(pre_peak)
        if prom < min_prom:
            return np.nan, np.nan, np.nan, np.nan

    # --- FWHM-style peak width (existing logic) ---
    if not np.isfinite(peak_amp) or peak_amp <= 0:
        width = np.nan
    else:
        half_level = peak_amp * height_frac
        left_idx = peak_local
        while left_idx > 0 and np.isfinite(seg[left_idx - 1]) and seg[left_idx - 1] >= half_level:
            left_idx -= 1
        right_idx = peak_local
        while right_idx < len(seg) - 1 and np.isfinite(seg[right_idx + 1]) and seg[right_idx + 1] >= half_level:
            right_idx += 1

        def interp_cross(i_inside, i_outside):
            if i_inside == i_outside:
                return seg_t[i_inside]
            y1, y2 = seg[i_inside], seg[i_outside]
            t1, t2 = seg_t[i_inside], seg_t[i_outside]
            if not np.isfinite(y1) or not np.isfinite(y2) or y1 == y2:
                return seg_t[i_inside]
            frac = (half_level - y1) / (y2 - y1)
            return t1 + frac * (t2 - t1)

        left_t = seg_t[left_idx] if left_idx == 0 else interp_cross(left_idx, left_idx - 1)
        right_t = seg_t[right_idx] if right_idx == len(seg) - 1 else interp_cross(right_idx, right_idx + 1)
        width = right_t - left_t

    # --- contiguous time above threshold around the initial response ---
    if not np.isfinite(baseline_sd) or baseline_sd <= 0:
        time_above = np.nan
    else:
        thresh = baseline_sd_mult * baseline_sd
        above = np.isfinite(seg) & (seg >= thresh)

        # find the first sample after onset that crosses threshold
        first_above_candidates = np.where(above)[0]
        if first_above_candidates.size == 0:
            time_above = 0.0
        else:
            # anchor on the peak if the peak itself is above threshold,
            # otherwise on the first threshold crossing
            if above[peak_local]:
                anchor = peak_local
            else:
                anchor = int(first_above_candidates[0])

            # walk left from anchor while still above threshold
            left_idx = anchor
            while left_idx > 0 and above[left_idx - 1]:
                left_idx -= 1
            # walk right from anchor while still above threshold
            right_idx = anchor
            while right_idx < len(seg) - 1 and above[right_idx + 1]:
                right_idx += 1

            # linear interpolation for sub-sample threshold crossings
            def interp_thresh_cross(i_inside, i_outside):
                if i_inside == i_outside:
                    return seg_t[i_inside]
                y1, y2 = seg[i_inside], seg[i_outside]
                t1, t2 = seg_t[i_inside], seg_t[i_outside]
                if not np.isfinite(y1) or not np.isfinite(y2) or y1 == y2:
                    return seg_t[i_inside]
                frac = (thresh - y1) / (y2 - y1)
                return t1 + frac * (t2 - t1)

            left_t = seg_t[left_idx] if left_idx == 0 else interp_thresh_cross(left_idx, left_idx - 1)
            right_t = seg_t[right_idx] if right_idx == len(seg) - 1 else interp_thresh_cross(right_idx, right_idx + 1)
            time_above = right_t - left_t

    return peak_time, peak_amp, width, time_above

records = []
for roi in rois:
    for odor_dur, ff_interp, fly in roi_traces_all[roi]:
        ttp, amp, width, t_above = compute_peak_metrics(
            t_common, ff_interp, odor_dur, width_height_frac, min_peak_prominence,
            baseline_sd_mult=2.0,
        )
        records.append({
            'fly': fly,
            'roi': roi,
            'odor_dur': odor_dur,
            'time_to_peak': ttp,
            'peak_amp': amp,
            'peak_width': width,
            'time_above_thresh': t_above,
        })

results = pd.DataFrame.from_records(records)

results = pd.DataFrame.from_records(records)

# --- summary stats per compartment ---
def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

summary = (
    results.groupby('roi')
    .agg(
        n_trials=('time_to_peak', lambda x: np.isfinite(x).sum()),
        n_flies=('fly', 'nunique'),
        ttp_mean=('time_to_peak', 'mean'),
        ttp_sem=('time_to_peak', sem),
        ttp_median=('time_to_peak', 'median'),
        width_mean=('peak_width', 'mean'),
        width_sem=('peak_width', sem),
        width_median=('peak_width', 'median'),
        amp_mean=('peak_amp', 'mean'),
        amp_sem=('peak_amp', sem),
    )
    .reindex(rois)
)

print(summary.round(3))

# also per-fly means, then averaged across flies (each fly weighted equally)
per_fly = (
    results.groupby(['fly', 'roi'])
    .agg(ttp=('time_to_peak', 'mean'), width=('peak_width', 'mean'))
    .reset_index()
)
fly_summary = (
    per_fly.groupby('roi')
    .agg(
        n_flies=('fly', 'nunique'),
        ttp_mean_of_fly_means=('ttp', 'mean'),
        ttp_sem_of_fly_means=('ttp', sem),
        width_mean_of_fly_means=('width', 'mean'),
        width_sem_of_fly_means=('width', sem),
    )
    .reindex(rois)
)
print('\nper-fly-then-across-flies:')
print(fly_summary.round(3))

# fig, axes, markc = configure_white_plot(size=(8,3), xaxis=True, ncols=3)

fig, axes, markc = configure_white_plot(size=(8,3), xaxis=True, ncols=3)

metrics = [
    ('time_to_peak', 'time to peak (s)'),
    ('peak_width', f'peak width at {int(width_height_frac*100)}% height (s)'),
    ('time_above_thresh', 'time above 2×SD baseline (s)'),
]

# y-limits per subplot, in the same order as `metrics`.
# Use a (low, high) tuple to set, or None to autoscale.
ylims = [
    (0, 7),      # time to peak
    (0, 1.5),      # peak width
    (0, 3),     # time above threshold
]

for ax, (metric, ylabel), ylim in zip(axes, metrics, ylims):
    # secondary: trial-level box plot
    trial_data = [results.loc[results['roi'] == r, metric].dropna().to_numpy() for r in rois]
    bp = ax.boxplot(
        trial_data,
        positions=np.arange(len(rois)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        zorder=1,
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor('gray')
        patch.set_alpha(0.25)
        patch.set_linewidth(0.6)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('gray')
            line.set_alpha(0.6)
            line.set_linewidth(0.6)
    for line in bp['medians']:
        line.set_color('gray')
        line.set_alpha(0.7)
        line.set_linewidth(1.0)

    # primary: per-fly means
    fly_means = (
        results.dropna(subset=[metric])
        .groupby(['fly', 'roi'])[metric].mean()
        .reset_index()
    )

    for xi, (r, c) in enumerate(zip(rois, colors)):
        vals = fly_means.loc[fly_means['roi'] == r, metric].to_numpy()
        if len(vals) == 0:
            continue
        rng = np.random.RandomState(0)
        jitter = (rng.rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                   s=32, color=c, edgecolor=None, linewidth=0.5,
                   alpha=0.9, zorder=3)

    for xi, r in enumerate(rois):
        vals = fly_means.loc[fly_means['roi'] == r, metric].dropna().to_numpy()
        if len(vals) < 1:
            continue
        m = np.mean(vals)
        if len(vals) > 1:
            s = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(xi, m, yerr=s, fmt='_', color=markc,
                        markersize=22, elinewidth=1.6, capsize=0, zorder=4)
        else:
            ax.plot(xi, m, '_', color='k', markersize=22, zorder=4)

    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

n_flies = results['fly'].nunique()
n_trials = len(results)
plt.suptitle(f'per-fly means (dots) — {n_flies} flies, {n_trials} trials', fontsize=10)
plt.tight_layout()
plt.show()

if not os.path.exists(f'{figure_folder}/fig/quant'):
    os.makedirs(f'{figure_folder}/fig/quant')
fig.savefig(f'{figure_folder}/fig/quant/exit.pdf', bbox_inches='tight')


# %% heading consistency around odor entries

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0

# --- binning ---
bin_width = 1.0
bin_edges = np.arange(-pre_s, post_s + bin_width, bin_width)   # [-2,-1,0,1,2,3,4,5]
bin_centers = np.arange(-pre_s, post_s + 1.0, 1.0)   # [-2, -1, 0, 1, 2, 3, 4, 5]
half_window = 0.5                                    # ±0.5 s around each center
n_bins = len(bin_centers)

data = open_pickle(filename)
flylist = np.array(list(data.keys()))

# --- collect heading samples per fly per bin ---
# structure: per_fly_headings[fly][bin_idx] = list of heading values (radians)
per_fly_headings = {fly: [[] for _ in range(n_bins)] for fly in flylist}
per_fly_entry_count = {fly: 0 for fly in flylist}

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)

    if len(bouts) == 0:
        continue

    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]
        if end_idx - 1 <= start_idx:
            continue
        odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s
        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset
        if t1 <= onset + min_post_after_trunc:
            continue

        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        if win['t_rel'].min() > (-pre_s + 0.25):
            continue

        t_rel = win['t_rel'].to_numpy()
        heading = win['heading'].to_numpy()
        good = np.isfinite(t_rel) & np.isfinite(heading)
        if good.sum() < 2:
            continue
        t_rel_g = t_rel[good]
        heading_g = heading[good]

        per_fly_entry_count[fly] += 1   # <-- add this line

        # assign each sample to every bin whose ±0.5 s window contains it
        for b, center in enumerate(bin_centers):
            sel = (t_rel_g >= center - half_window) & (t_rel_g < center + half_window)
            if sel.any():
                per_fly_headings[fly][b].extend(heading_g[sel].tolist())


# --- compute mean resultant vector R per fly per bin ---
def mean_resultant_length(angles_rad, min_n=3):
    """R = |mean(exp(i*theta))|; returns NaN if too few samples."""
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < min_n:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * a))))

min_entries = 5
included_flies = [fly for fly in flylist if per_fly_entry_count[fly] >= min_entries]
excluded_flies = [fly for fly in flylist if per_fly_entry_count[fly] < min_entries]

print(f'including {len(included_flies)} flies (≥{min_entries} entries)')
print(f'excluding {len(excluded_flies)} flies: '
      + ', '.join([f"{fly}(n={per_fly_entry_count[fly]})" for fly in excluded_flies]))

R_per_fly = np.full((len(included_flies), n_bins), np.nan)
for i, fly in enumerate(included_flies):
    for b in range(n_bins):
        R_per_fly[i, b] = mean_resultant_length(per_fly_headings[fly][b], min_n=3)

# --- summarize across flies ---
R_mean = np.nanmean(R_per_fly, axis=0)
R_sem  = np.nanstd(R_per_fly, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(R_per_fly), axis=0))
n_flies_per_bin = np.sum(np.isfinite(R_per_fly), axis=0)

# --- plot: line connecting per-bin means, with SEM band and per-fly dots ---
# fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
fig, ax, markc = configure_white_plot(size=(3,3), xaxis=True)

# SEM band
ax.fill_between(
    bin_centers,
    R_mean - R_sem,
    R_mean + R_sem,
    color='gray',
    alpha=0.25,
    linewidth=0,
    zorder=1,
)

# per-fly dots in the background
rng = np.random.RandomState(0)
for b, xc in enumerate(bin_centers):
    vals = R_per_fly[:, b]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        continue
    jitter = (rng.rand(len(vals)) - 0.5) * 0.18
    ax.scatter(np.full(len(vals), xc) + jitter, vals,
               s=12, color='k', alpha=0.25, linewidth=0, zorder=2)

# mean line + scatter on top
ax.plot(bin_centers, R_mean, color=markc, linewidth=1.5, zorder=3)
# ax.scatter(bin_centers, R_mean, s=40, color='k',
#            edgecolor='white', linewidth=0.8, zorder=4)

# odor onset marker
ax.axvline(0, linestyle='--', color=markc, linewidth=0.8, alpha=0.7, zorder=0)
ax.set_xlim(-pre_s - 0.3, post_s + 0.3)
# ax.set_ylim(0, 1.05)
ax.set_xlabel('time from entry (s)')
ax.set_ylabel('heading consistency')
# ax.set_xticks(bin_centers)
# ax.set_xticklabels([f'{c:.0f}' for c in bin_centers], fontsize=8)
ax.set_title(f'heading consistency')

plt.tight_layout()
plt.show()

out_dir = f'{figure_folder}/fig/heading_consistency'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fig.savefig(f'{out_dir}/heading_R_per_bin.pdf', bbox_inches='tight')


# %% crosswind component  around odor entries

figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
pre_s = 2.0
post_s = 10.0
min_post_after_trunc = 1.0

# --- binning ---
bin_width = 1.0
bin_edges = np.arange(-pre_s, post_s + bin_width, bin_width)   # [-2,-1,0,1,2,3,4,5]
bin_centers = np.arange(-pre_s, post_s + 1.0, 1.0)   # [-2, -1, 0, 1, 2, 3, 4, 5]
half_window = 0.5                                    # ±0.5 s around each center
n_bins = len(bin_centers)

data = open_pickle(filename)
flylist = np.array(list(data.keys()))

# --- collect heading samples per fly per bin ---
per_fly_headings = {fly: [[] for _ in range(n_bins)] for fly in flylist}
per_fly_entry_count = {fly: 0 for fly in flylist}

for fly in flylist:
    flydf = data[fly]['a1'].copy()
    t = flydf['relative_time'].to_numpy()
    odor_on = flydf['mfc2_stpt'].fillna(0).to_numpy() > 0
    bouts = odor_bout_indices(odor_on)

    if len(bouts) == 0:
        continue

    for j, (start_idx, end_idx) in enumerate(bouts):
        onset = t[start_idx]
        if end_idx - 1 <= start_idx:
            continue
        odor_dur = t[end_idx - 1] - t[start_idx]
        if odor_dur > post_s:
            continue

        t0 = onset - pre_s
        t1 = onset + post_s
        if j + 1 < len(bouts):
            next_start_idx, _ = bouts[j + 1]
            next_onset = t[next_start_idx]
            if next_onset < t1:
                t1 = next_onset
        if t1 <= onset + min_post_after_trunc:
            continue

        win = flydf[(flydf['relative_time'] >= t0) & (flydf['relative_time'] <= t1)].copy()
        if win.empty:
            continue
        win['t_rel'] = win['relative_time'] - onset
        if win['t_rel'].min() > (-pre_s + 0.25):
            continue

        t_rel = win['t_rel'].to_numpy()
        heading = win['heading'].to_numpy()
        good = np.isfinite(t_rel) & np.isfinite(heading)
        if good.sum() < 2:
            continue
        t_rel_g = t_rel[good]
        heading_g = heading[good]

        per_fly_entry_count[fly] += 1   # <-- add this line

        # assign each sample to every bin whose ±0.5 s window contains it
        for b, center in enumerate(bin_centers):
            sel = (t_rel_g >= center - half_window) & (t_rel_g < center + half_window)
            if sel.any():
                per_fly_headings[fly][b].extend(heading_g[sel].tolist())


# --- compute mean resultant vector R per fly per bin ---
def mean_crosswind_component(angles_rad, min_n=3):
    """Mean of |sin(heading)| — 0 = upwind/downwind, 1 = pure crosswind.
    Assumes ft_heading = 0 is upwind, in radians."""
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < min_n:
        return np.nan
    return float(np.mean(np.abs(np.sin(a))))

min_entries = 5
included_flies = [fly for fly in flylist if per_fly_entry_count[fly] >= min_entries]
excluded_flies = [fly for fly in flylist if per_fly_entry_count[fly] < min_entries]

print(f'including {len(included_flies)} flies (≥{min_entries} entries)')
print(f'excluding {len(excluded_flies)} flies: '
      + ', '.join([f"{fly}(n={per_fly_entry_count[fly]})" for fly in excluded_flies]))

crosswind_per_fly = np.full((len(included_flies), n_bins), np.nan)
for i, fly in enumerate(included_flies):
    for b in range(n_bins):
        crosswind_per_fly[i, b] = mean_crosswind_component(per_fly_headings[fly][b], min_n=3)

cw_mean = np.nanmean(crosswind_per_fly, axis=0)
cw_sem  = np.nanstd(crosswind_per_fly, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(crosswind_per_fly), axis=0))
n_flies_per_bin = np.sum(np.isfinite(crosswind_per_fly), axis=0)
# --- summarize across flies ---
R_mean = np.nanmean(R_per_fly, axis=0)
R_sem  = np.nanstd(R_per_fly, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(R_per_fly), axis=0))
n_flies_per_bin = np.sum(np.isfinite(R_per_fly), axis=0)

# --- plot: line connecting per-bin means, with SEM band and per-fly dots ---
# fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
fig, ax, markc = configure_white_plot(size=(3,3), xaxis=True)

# SEM band
ax.fill_between(
    bin_centers,
    cw_mean - cw_sem,
    cw_mean + cw_sem,
    color='gray',
    alpha=0.25,
    linewidth=0,
    zorder=1,
)

# per-fly dots in the background
rng = np.random.RandomState(0)
for b, xc in enumerate(bin_centers):
    vals = crosswind_per_fly[:, b]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        continue
    jitter = (rng.rand(len(vals)) - 0.5) * 0.18
    ax.scatter(np.full(len(vals), xc) + jitter, vals,
               s=12, color=markc, alpha=0.25, linewidth=0, zorder=2)

# mean line + scatter on top
ax.plot(bin_centers, cw_mean, color=markc, linewidth=1.5, zorder=3)

# odor onset marker
ax.axvline(0, linestyle='--', color=markc, linewidth=0.8, alpha=0.7, zorder=0)
ax.set_xlim(-pre_s - 0.3, post_s + 0.3)
ax.set_xlabel('time from entry')
ax.set_ylabel('crosswind component')
plt.tight_layout()
plt.show()

fig.savefig(f'{out_dir}/heading_crosswind_per_bin.pdf', bbox_inches='tight')
# %%
