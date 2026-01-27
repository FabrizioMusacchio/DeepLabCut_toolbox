""" 
A script to calculate the velocity of body parts from DeepLabCut output CSV files.
The script will loop over all CSV files in the data folder, calculate the velocity 
of each body part, and plot the x-y coordinates of the body parts and the velocity.
It also detects, based on a threshold, whether a body part is moving or not.

author: Fabrizio Musacchio
date:   Feb 20, 2025
        Jan 26, 2026 - added option to use either filtered or not-filtered data
"""
# %% IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATH AND PARAMETERS (ADJUST HERE)
# set your data and results paths here:
DATA_PATH = "/Users/husker/Workspace/Emine/DLC cFC/Data Test/"
RESULTS_PATH = "/Users/husker/Workspace/Emine/DLC cFC/results/"

# use filtered data?
use_filtered_data = True  # if True, use filtered data; if False, use raw data

# define frame rate and time step:
frame_rate = 30  # fps
time_step = 1 / frame_rate

read_frame_rate_from_DLC_pickled_file = False  # if True, read the frame rate from the DLC pickle file; if False, use the defined frame_rate above
""" 
NOTE: read_frame_rate_from_DLC_pickled_file is not implemented yet! We will
add Luk's' code to read the frame rate from the DLC pickle file in a future update.
"""

# define the size of a pixel (if available):
pixel_size = 1  # spatial_unit/px; leave as 1 if you don't know the size of a pixel
spatial_unit="cm"

# define likelihood threshold for valid points:
likelihood_threshold = 0.9 # this likelihood refers to the DLC assigned likelihood
                           # for each bodypart; it is a value between 0 and 1, where
                           # 1 means "very likely" and 0 means "not likely at all";
                           # "likely" roughly means "reliable"; by adjusting this 
                           # threshold, you can filter out low-confidence points.

# define a threshold for movement detection:
movement_threshold = 15  # px/frame; note, if you set pixel_size to 1, this is in px/s;
                         # if you set pixel_size to a value other than 1, this is in spatial_unit/s;
                         # this threshold is used to determine whether a body part is moving or not;
                         # if the velocity is above this threshold, the body part is considered to be moving;
                         # if the velocity is below this threshold, the body part is considered to be not moving;

# set (optional!) y-axis limit for velocity plot:
ylim = None # DON'T CHANGE THIS LINE
#
# uncomment if you want to set a fixed y-axis limit for the velocity plot:
#
#ylim = 200 # set to a value, e.g., 1000, for fixed scaling;
#
# note: this is useful if you want to compare the velocity plots of different files;
# if you set ylim to None, the y-axis limit will be automatically scaled to the data;

            
# define bodyparts to be excluded from the velocity plot:
bodypart_not_to_plot = None  # DO NOT CHANGE THIS LINE
#
# uncomment if you want to exclude some body parts from the velocity plot:
#
#bodypart_not_to_plot = ['center', 'tail', "ear_L", "ear_R", "neck"] # set to a list of body parts to be excluded from the velocity plot;


# define bodypart-groups:
# initialize bodypart_groups as None if not defined:
bodypart_groups = None # DON'T CHANGE THIS LINE
# 
# uncomment if you want to group body parts together:
#
# bodypart_groups = {
#     'head': ['nose', 'ear_L', 'ear_R', 'neck'],
#     'body': ['center']}
#
# grouping body parts together can be useful if you want to assess 
# moving/non moving only for a subset of body parts, e.g., for all 
# head related body parts (nose, neck, etc.) in, e.g., freezing behavior analysis

# define time intervals:
# initialize bodypart_groups as None if not defined:
time_intervals = None # DON'T CHANGE THIS LINE
# 
# uncomment if you want to separate the analysis into time intervals:
#
# time_intervals = {
#     'pre-shock': [0, 7200],  # in frames
#     'after_shock1': [7200, 11050], # in frames
#     'after_shock2': [11051, 14700],
#     'after_shock3': [14701, 17500]} # in frames
#
# note: if you define time intervals, the analysis will be additionally performed for each 
# interval separately; the results will be saved in separate CSV files for each interval;
# %% FUNCTIONS

# %% MAIN (LEAVE AS IT IS)
""" 
HERE, YOU DON'T NEED TO CHANGE ANYTHING – EXCEPT YOU WANT
TO CUSTOMIZE THE ANALYSIS ACCORDING TO YOUR NEEDS.
"""

# create the results folder if it does not exist:
RESULTS_EXCEL_PATH = RESULTS_PATH + "excel/"
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(RESULTS_EXCEL_PATH):
    os.makedirs(RESULTS_EXCEL_PATH)

# scan for all csv files in the data folder that do not start with a dot:
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and not f.startswith('.')]

"""
now check whether to use filtered or raw data; If true, only take those csv files that 
contain 'filtered' in their filename; if false, only take those csv files that do not 
contain 'filtered' in their filename;
"""
if use_filtered_data:
    csv_files = [f for f in csv_files if 'filtered' in f]
else:
    csv_files = [f for f in csv_files if 'filtered' not in f]

print(f"Found {len(csv_files)} CSV files in the data folder. 'use_filtered_data' is set to {use_filtered_data}.")

# prepare DataFrame in which we collect the averages per files:
all_velocity_df = pd.DataFrame()

# loop over all CSV files:
for curr_filename in csv_files:
    # curr_filename = csv_files[0]
    ## %%
    # load the DeepLabCut output CSV file:
    curr_file = DATA_PATH + curr_filename
    print(f"Processing {curr_filename}...")
    
    # read the CSV file with multi-index columns:
    df = pd.read_csv(curr_file, header=[1, 2])
    
    """ if read_frame_rate_from_DLC_pickled_file:
        # TODO: reade pickle file to get frame rate
        # frame_rate = ...
        # time_step = 1 / frame_rate """
    
    # clean the filename:
    curr_filename_clean = curr_filename.replace('.csv', '').replace(' ', '_')

    # drop the first column as it contains metadata:
    df_cleaned = df.iloc[:, 1:].copy()

    # Set proper multi-index column names:
    df_cleaned.columns = pd.MultiIndex.from_tuples(df_cleaned.columns)

    # convert data to numeric:
    df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors='coerce')


    # identify unique body parts:
    body_parts = df_cleaned.columns.get_level_values(0).unique()

    # define a list of colors for plotting, which is as long as the number of body parts:
    colors = plt.cm.tab10.colors[:len(body_parts)]

    # prepare a DataFrame for velocity results:
    velocity_df = pd.DataFrame()
    # add first columns with frame numbers and time corrspondence:
    velocity_df['frame'] = df_cleaned.index.values
    velocity_df['time'] = df_cleaned.index.values * time_step

    # initialize a plot with two subplots to plot the x-y coordinates of the body parts and the velocity:
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # compute velocity for each body part:
    for body_part_i, body_part in enumerate(body_parts):
        # body_part = body_parts[1]
        curr_sub_df = df_cleaned.loc[:, (df_cleaned.columns.get_level_values(0) == body_part)].copy()
        
        # remove the first header level:
        curr_sub_df.columns = curr_sub_df.columns.droplevel(0)
        
        # rename the columns by attaching body part name:
        curr_sub_df.columns = [body_part + '_' + col for col in curr_sub_df.columns]
        
        # extract coordinates and likelihood:
        x = curr_sub_df.loc[:, body_part + '_x'].values
        y = curr_sub_df.loc[:, body_part + '_y'].values
        likelihood = curr_sub_df.loc[:, body_part + '_likelihood'].values
        frames_array = curr_sub_df.index.values

        # compute velocity components:
        vx = np.diff(x) / time_step
        vy = np.diff(y) / time_step

        # compute overall velocity magnitude:
        velocity = np.sqrt(vx**2 + vy**2)
        
        # if pixel_size is not 1, convert velocity to spatial_unit/s:
        if pixel_size != 1:
            velocity = velocity * pixel_size

        # store results (aligning size with original DataFrame by padding with NaN at the start)
        velocity_df[body_part + '_velocity'] = np.insert(velocity, 0, np.nan)
        
        # check whether current velocity is above the threshold:
        velocity_df[body_part + '_moving'] = np.insert(velocity, 0, np.nan) > movement_threshold

        # filter low-confidence points:
        #velocity_df[body_part + '_likelihood'] = np.insert(likelihood[1:], 0, np.nan)
        velocity_df[body_part + '_likelihood'] = likelihood
        velocity_df[body_part + '_valid']      = likelihood > likelihood_threshold
        
        #fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(frames_array,x, label=body_part+' x', c=colors[body_part_i])
        ax[0].plot(frames_array,y, label=body_part+' y', c=colors[body_part_i], alpha=0.5)
        if bodypart_not_to_plot is not None:
            if body_part not in bodypart_not_to_plot:
                ax[1].plot(velocity, label=body_part, c=colors[body_part_i], lw=0.5)
        else:
            ax[1].plot(velocity, label=body_part, c=colors[body_part_i], lw=0.5)
        
        # indicate with a shaded area the frames where the body part is moving; to do so, filter for consecutive True values:
        moving_frames = velocity_df[body_part + '_moving']
        moving_frames_diff = np.diff(moving_frames.astype(int))
        moving_frames_start = np.where(moving_frames_diff == 1)[0] + 1
        moving_frames_end = np.where(moving_frames_diff == -1)[0] + 1
        for start, end in zip(moving_frames_start, moving_frames_end):
            ax[2].fill_between(np.arange(start, end), 
                            body_part_i * 2, (body_part_i + 1) * 2, 
                            color=colors[body_part_i], alpha=0.75, lw=0)        

    # in ax[2], indicate with a gray overlay the total consecutive frames where body parts are moving, i.e., all 
    # consecutive frames where at least one body part is moving:
    all_moving_frames = velocity_df.iloc[:,2:].iloc[:, 1::4].sum(axis=1)/velocity_df.iloc[:,2:].iloc[:, 1::4].sum(axis=1) # velocity_df.iloc[:, 1::4].any(axis=1)
    # set NaN to 0:
    if all_moving_frames.isnull().any():
        all_moving_frames = all_moving_frames.fillna(0)
    all_moving_frames_diff = np.diff(all_moving_frames.astype(int))
    all_moving_frames_start = np.where(all_moving_frames_diff == 1)[0] + 1
    all_moving_frames_end = np.where(all_moving_frames_diff == -1)[0] + 1
    for start, end in zip(all_moving_frames_start, all_moving_frames_end):
        ax[2].fill_between(np.arange(start, end), 
                        (body_part_i + 1) * 2, (body_part_i + 1) * 2 + 2, 
                        color='grey', alpha=0.85, lw=0)

    # calculate the number of frames with movement for each body part (as percentage of total valid frames):
    for body_part in body_parts:
        moving_frames = velocity_df[body_part + '_moving'].sum()
        valid_frames = velocity_df[body_part + '_valid'].sum()
        print(f"   {body_part}: {moving_frames} moving frames ({100 * moving_frames / valid_frames:.2f}%)")
        

    # finalize plot:
    ax[0].set_ylabel("position (px)")
    ax[0].set_title("mouse body parts positions")
    ax[0].legend(loc='upper right')
    if pixel_size == 1:
        ax[1].set_ylabel("velocity (px/s)")
    else:
        ax[1].set_ylabel(f"velocity ({spatial_unit}/s)")
    if pixel_size == 1:
        unit_snippet = "px"
    else:
        unit_snippet = spatial_unit
    ax[1].axhline(movement_threshold, color='r', linestyle='--', label=f'movement threshold\n({movement_threshold} {unit_snippet}/s)')
    ax[1].legend(loc='upper right')
    ax[1].set_xlim(0, len(velocity_df))
    if ylim is not None:
        ax[1].set_ylim(0, ylim)
    # change y-axis to log scale:
    #ax[1].set_yscale('log')
    ax[1].set_title(f"body parts movement velocity $v=\\sqrt{{v_x^2 + v_y^2}}$, with $v_{{x/y}}=\\frac{{\\Delta x/y}}{{\\Delta t}}$")
    ax[2].set_yticks(np.arange(0, 2 * len(body_parts)+1, 2) + 1)
    ax[2].set_yticklabels(body_parts.append(pd.Index(['any'])))
    ax[2].set_title("body parts in motion")
    ax[2].set_xlabel(f"frame (top) / time (bottom, [s]) ($\\Delta$t = {time_step:.3f} s)")
    # customize x-tick labels to show "frame/time" format:
    xticks = ax[2].get_xticks()
    ax[2].xaxis.set_major_locator(FixedLocator(xticks))
    xticklabels = [f"{int(tick)}\n{tick * time_step:.1f}" for tick in xticks]
    ax[2].set_xticklabels(xticklabels)
    # add a vertical line for each time interval (if defined):
    if time_intervals is not None:
        for interval_name, interval in time_intervals.items():
            ax[0].axvline(x=interval[0], color='gray', linestyle='--', lw=1)
            ax[0].axvline(x=interval[1], color='gray', linestyle='--', lw=1)
            # get the lower limit of ax[0] y-axis:
            lowest_position = ax[0].get_ylim()[0]
            ax[0].text((interval[0] + interval[1]) / 2, lowest_position, 
                       interval_name, va='bottom', ha='center',
                       fontsize=14, color='gray')
            ax[1].axvline(x=interval[0], color='gray', linestyle='--', lw=1)
            ax[1].axvline(x=interval[1], color='gray', linestyle='--', lw=1)
            ax[2].axvline(x=interval[0], color='gray', linestyle='--', lw=1)
            ax[2].axvline(x=interval[1], color='gray', linestyle='--', lw=1)
            # add a label for the interval:
            
        
    # annotate in ax[2] the calculated number of frames with movement for each body part:
    for body_part_i, body_part in enumerate(body_parts):
        moving_frames = velocity_df[body_part + '_moving'].sum()
        valid_frames = len(velocity_df)
        ax[2].text(len(velocity_df) * 0.95, (body_part_i*2 + 1), 
           f"{body_part}: {moving_frames} moving frames ({100 * moving_frames / valid_frames:.2f}%)",
           va='center', ha='right', transform=ax[2].transData, color='k', fontsize=14)
    # also annotate the total number of moving frames:
    moving_frames = int(all_moving_frames.sum())
    valid_frames = len(velocity_df)
    ax[2].text(len(velocity_df) * 0.95, (body_part_i*2 + 3), 
           f"any bodypart: {moving_frames} moving frames ({100 * moving_frames / valid_frames:.2f}%)",
           va='center', ha='right', transform=ax[2].transData, color='k', fontsize=14)
    
    # add a super title:
    fig.suptitle(curr_filename_clean)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, curr_filename+" velocity_plot.pdf"), dpi=300, transparent=True)
    plt.close(fig)


    # Prepare output DataFrames:
    # add a column containing "any body part moving" to the velocity_df:
    velocity_df['any_bodypart_moving'] = velocity_df.iloc[:, 3::4].sum(axis=1) > 0
    
    # collect the average velocity and percentage of moving frames for each body part,
    # add it to a tmp dataframe and concatenate it to the all_velocity_df:
    curr_line_df = pd.DataFrame()
    curr_line_df['filename'] = [curr_filename_clean]
    for bp in body_parts:
        if pixel_size == 1:
            unit_snippet = "px"
        else:
            unit_snippet = spatial_unit
        # calculate the mean velocity and percentage of moving frames for each body part:
        curr_line_df[bp + ' mean velocity ' + f' ({unit_snippet}/s)'] = [velocity_df[bp + '_velocity'].mean()]
        curr_line_df[bp + ' moving frames'] = [velocity_df[bp + '_moving'].sum()]
        curr_line_df[bp + ' moving frames %'] = [100 * velocity_df[bp + '_moving'].sum() / len(velocity_df)]
        
    # add the number of any_bodypart_moving=True frames and percentage:
    any_bodypart_moving_True = velocity_df['any_bodypart_moving'].sum()
    curr_line_df['any_bodypart_moving frames'] = [velocity_df['any_bodypart_moving'].sum()]
    curr_line_df['any_bodypart_moving frames %'] = [100 * velocity_df['any_bodypart_moving'].sum() / len(velocity_df)]
    # add the number of any_bodypart_moving=True frames and percentage:
    curr_line_df['any_bodypart_moving frames'] = [velocity_df['any_bodypart_moving'].sum()]
    curr_line_df['any_bodypart_moving frames %'] = [100 * velocity_df['any_bodypart_moving'].sum() / len(velocity_df)]
        
    
    # if bodypart groups are defined, calculate the mean velocity for each group:
    if bodypart_groups is not None:
        for group_name, group_bodyparts in bodypart_groups.items():
            # check if all body parts in the group are present in the DataFrame:
            if all([bp in body_parts for bp in group_bodyparts]):
                # calculate the velocity for the group as the average of the individual body parts within the group:
                group_velocity = velocity_df[[bp + '_velocity' for bp in group_bodyparts]].mean(axis=1)
                # add the group velocity to the DataFrame:
                velocity_df[group_name + '_mean_velocity'] = group_velocity
                # check whether the group is moving:
                velocity_df[group_name + '_moving'] = group_velocity > movement_threshold
                
                # calculate the mean velocity and percentage of moving frames for each group:
                curr_line_df[group_name + ' mean velocity ' + f' ({unit_snippet}/s)'] = [group_velocity.mean()]
                curr_line_df[group_name + ' moving frames'] = [velocity_df[group_name + '_moving'].sum()]
                curr_line_df[group_name + ' moving frames %'] = [100 * velocity_df[group_name + '_moving'].sum() / len(velocity_df)]
            else:
                print(f"   WARNING: Group '{group_name}' contains body parts not found in the DataFrame.")
                print(f"   Missing body parts: {[bp for bp in group_bodyparts if bp not in body_parts]}")
                print(f"   Skipping group '{group_name}'.")
                continue
    else:
        print("   No bodypart groups defined. Skipping group analysis.")
            
    # if time_intervals is defined: 
    if time_intervals is not None:
        # first just add another column indicating the time interval:
        velocity_df['time_interval'] = pd.Series(dtype='object')
        for interval_name, interval in time_intervals.items():
            # check if the interval is valid:
            if len(interval) == 2 and interval[0] < interval[1]:
                # assign the interval name to the corresponding frames:
                velocity_df.loc[interval[0]:interval[1], 'time_interval'] = interval_name
            else:
                print(f"   WARNING: Invalid time interval '{interval_name}': {interval}. Skipping.")
                continue
        # where velocity_df['time_interval'] has empty values, assign 'not defined':
        velocity_df['time_interval'] = velocity_df['time_interval'].fillna('not defined')
        
        # second, for curr_line_df, calculate the mean velocity and percentage of moving frames for each time interval
        # and bodypart:
        for interval_name, interval in time_intervals.items():
            # check if the interval is valid:
            if len(interval) == 2 and interval[0] < interval[1]:
                # filter the DataFrame for the current interval:
                curr_interval_df = velocity_df.loc[interval[0]:interval[1], :].copy()
                # calculate the mean velocity and percentage of moving frames for each body part:
                for bp in body_parts:
                    curr_line_df[interval_name + ' ' + bp + ' mean velocity ' + f' ({unit_snippet}/s)'] = [curr_interval_df[bp + '_velocity'].mean()]
                    curr_line_df[interval_name + ' ' + bp + ' moving frames'] = [curr_interval_df[bp + '_moving'].sum()]
                    curr_line_df[interval_name + ' ' + bp + ' moving frames %'] = [100 * curr_interval_df[bp + '_moving'].sum() / len(curr_interval_df)]
                # add the number of any_bodypart_moving=True frames and percentage:
                any_bodypart_moving_True = curr_interval_df['any_bodypart_moving'].sum()
                curr_line_df[interval_name + ' ' + 'any_bodypart_moving frames'] = [curr_interval_df['any_bodypart_moving'].sum()]
                curr_line_df[interval_name + ' ' + 'any_bodypart_moving frames %'] = [100 * curr_interval_df['any_bodypart_moving'].sum() / len(curr_interval_df)]
                
                # if bodypart groups are defined, calculate the mean velocity for each group:
                if bodypart_groups is not None:
                    for group_name, group_bodyparts in bodypart_groups.items():
                        # check if all body parts in the group are present in the DataFrame:
                        if all([bp in body_parts for bp in group_bodyparts]):
                            # calculate the velocity for the group as the average of the individual body parts within the group:
                            group_velocity = curr_interval_df[[bp + '_velocity' for bp in group_bodyparts]].mean(axis=1)
                            # add the group velocity to the DataFrame:
                            curr_line_df[interval_name + ' ' + group_name + '_mean_velocity'] = group_velocity
                            # if curr_line_df[interval_name + ' ' + group_name + '_mean_velocity'] is NaN (=no movement), set it to 0:
                            curr_line_df[interval_name + ' ' + group_name + '_mean_velocity'] = curr_line_df[interval_name + ' ' + group_name + '_mean_velocity'].fillna(0)
                            # check whether the group is moving:
                            curr_line_df[interval_name + ' ' + group_name + '_moving'] = group_velocity > movement_threshold
                            
                            # calculate the mean velocity and percentage of moving frames for each group:
                            curr_line_df[interval_name + ' ' + group_name + ' mean velocity ' + f' ({unit_snippet}/s)'] = [group_velocity.mean()]
                            curr_line_df[interval_name + ' ' + group_name + ' moving frames'] = [curr_interval_df[group_name + '_moving'].sum()]
                            curr_line_df[interval_name + ' ' + group_name + ' moving frames %'] = [100 * curr_interval_df[group_name + '_moving'].sum() / len(curr_interval_df)]
    
    # update the all_velocity_df with the current line:
    all_velocity_df = pd.concat([all_velocity_df, curr_line_df], ignore_index=True)

    # save or display the results:
    velocity_df.to_csv(RESULTS_EXCEL_PATH + curr_filename_clean + ' velocity_results.csv', index=False)
    print(f"   Saved results to {RESULTS_EXCEL_PATH + curr_filename_clean + ' velocity_results.csv'}")

# save the all_velocity_df to a CSV file:
all_velocity_df.to_csv(RESULTS_EXCEL_PATH + "all_velocity_results.csv", index=False)
print(f"Saved all results to {RESULTS_EXCEL_PATH + 'all_velocity_results.csv'}")
    
# %% END
print("Done!")