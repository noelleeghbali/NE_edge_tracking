#%%
from imaging_analysis import *
from behavior_analysis import *
from scipy.stats import spearmanr
from collections import defaultdict


figure_folder = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/et'
filename = '/Users/noelleeghbali/Desktop/exp/imaging/as_imaging/all_data.pkl'
lobes =['G2', 'G3', 'G4', 'G5']
colors = ['#00f5ff', '#00ccff','#cc33cc', '#ff43a4']

#%% df/f trace w/ net motion
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
print(len(flylist))
for fly in flylist:
    flydf = data[fly]['a1'] # dataframe for each fly
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    fig, axs, markc = configure_bw_plot(size=(8,6), xaxis=True, nrows=len(lobes)+1)
    axs = np.ravel(axs)         
    fig.suptitle(fly)
    for i, lobe in enumerate(lobes):
        FF = flydf[lobe] 
        time = flydf['relative_time']
        axs[i].plot(time, FF, linewidth=0.5, color=colors[i])
        add_odor(axs[i], di, FF.min(), FF.max())
        axs[i].set_ylabel(lobe, color=markc)
    
    # add net motion
    net_motion = flydf['net_motion']
    ax = axs[len(lobes)]
    ax.plot(flydf['relative_time'], net_motion, linewidth=0.5, color=markc)
    add_odor(ax, di, net_motion.min(), net_motion.max())
    ax.set_ylabel('net motion', color=markc)
    ax.set_xlabel('time (s)', color=markc)

    plt.tight_layout()
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/traces'):
        os.makedirs(f'{figure_folder}/fig/traces')
    fig.savefig(f'{figure_folder}/fig/traces/{fly}_trace', bbox_inches='tight')

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
    fig, axs, markc = configure_bw_plot(size=(10,10), xaxis=True)
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
    fig.savefig(f'{figure_folder}/fig/traj/{fly}_traj', bbox_inches='tight')

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
            auc = FF_adjusted.max() #np.trapz(FF_adjusted, time[interval_mask])
            auc_value = auc 
            print ('calculating peak df/f')
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
                if bout_dur >= 0.5 and return_to_edge(return_df):
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

# %% compare change in activity (df/f max-min) at entry w/ time since last return

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

    
# %% compare activity (auc/time) outside w/ efficiency of the return (PER FLY)
data = open_pickle(filename)
flylist = np.array(list(data.keys())) # list of flies in the pickle
summary_aucs = {f'G{i+2}': [] for i in range(len(lobes))}
summary_return_eff = {f'G{i+2}': [] for i in range(len(lobes))}
for fly in flylist:
    fig, axs, markc = configure_bw_plot(size=(12,3), xaxis=True, ncols=len(lobes))
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
        
#%% triggered Z scored entry or exit dff (PER FLY)
tbef = 5
taf = 5
event_type = 'entry'
first = False

for fly in flylist:
    flydf = data[fly]['a1']
    d, di, do = inside_outside(flydf)
    first_on_index = flydf[flydf['instrip']].index[0]
    exp_df = flydf.loc[first_on_index:]
    exp_df = exp_df[exp_df['net_motion'] > 0]  # Filter for net motion
    fig, axs, markc = configure_bw_plot(size=(12, 4), xaxis=True, ncols=len(lobes))
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
        axs[i].set_ylim(-2, 8)
        axs[0].set_ylabel('Z-score', color='white')
        axs[i].set_xlabel('time (s)', color='white')
        axs[i].set_title(lobe, color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the suptitle
    plt.show()
    if not os.path.exists(f'{figure_folder}/fig/triggered_ff'):
        os.makedirs(f'{figure_folder}/fig/triggered_ff')
    fig.savefig(f'{figure_folder}/fig/triggered_ff/{fly}_{event_type}.pdf', bbox_inches='tight')


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
