""" 
A script to calculate the velocity of body parts from DeepLabCut output CSV files.
The script will loop over all CSV files in the data folder, calculate the velocity 
of each body part, and plot the x-y coordinates of the body parts and the velocity.
It also detects, based on a threshold, whether a body part is moving or not.

author: Fabrizio Musacchio
date:   Feb 20, 2025
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
# %% DEFINE PATHS
DATA_PATH = "/Users/husker/Workspace/Denise/DLC project Test/Data/"
RESULTS_PATH = "/Users/husker/Workspace/Denise/DLC project Test/Analysis/"
RESULTS_EXCEL_PATH = RESULTS_PATH + "excel/"
# check if the path exists:
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(RESULTS_EXCEL_PATH):
    os.makedirs(RESULTS_EXCEL_PATH)
# %% DEFINE PARAMETERS
# define frame rate and time step:
frame_rate = 30  # fps
time_step = 1 / frame_rate

# define likelihood threshold for valid points:
likelihood_threshold = 0.9

# define a threshold for movement detection:
movement_threshold = 50  # px/frame
# %% FUNCTIONS

# %% MAIN (LEAVE AS IT IS)
""" 
HERE, YOU DON'T NEED TO CHANGE ANYTHING – EXCEPT YOU WANT
TO CUSTOMIZE THE ANALYSIS ACCORDING TO YOUR NEEDS.
"""


# scan for all csv files in the data folder that do not start with a dot:
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and not f.startswith('.')]
print(f"Found {len(csv_files)} CSV files in the data folder.")

# loop over all CSV files:
for curr_filename in csv_files:
    # curr_filename = csv_files[0]
    ## %%
    # load the DeepLabCut output CSV file:
    curr_file = DATA_PATH + curr_filename
    df = pd.read_csv(curr_file, header=[1, 2])
    
    print(f"Processing {curr_filename}...")

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
        ax[1].plot(velocity, label=body_part, c=colors[body_part_i])
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
    all_moving_frames = velocity_df.iloc[:, 1::4].sum(axis=1)/velocity_df.iloc[:, 1::4].sum(axis=1) # velocity_df.iloc[:, 1::4].any(axis=1)
    # set NaN to 0:
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
    ax[1].set_ylabel("velocity (px/s)")
    ax[1].axhline(movement_threshold, color='r', linestyle='--', label=f'movement threshold\n({movement_threshold} px/s)')
    ax[1].legend(loc='upper right')
    ax[1].set_xlim(0, len(velocity_df))
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

    # add a column containing "any body part moving" to the velocity_df:
    velocity_df['any_bodypart_moving'] = velocity_df.iloc[:, 2::4].sum(axis=1) > 0

    # save or display the results:
    velocity_df.to_csv(RESULTS_EXCEL_PATH + curr_filename_clean + ' velocity_results.csv', index=False)
    print(f"   Saved results to {RESULTS_EXCEL_PATH + curr_filename_clean + ' velocity_results.csv'}")

    
# %% END
print("Done!")