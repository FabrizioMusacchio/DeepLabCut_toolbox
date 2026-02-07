# %% IMPORTS
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', 7)
pd.set_option('display.max_columns', None)
from scipy import stats
import pingouin as pg
import seaborn as sns
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova, permdisp

from skbio.stats.ordination import pcoa, rda
from skbio.stats.distance import DistanceMatrix

import umap

# remove spines right and top for better aesthetics
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
# set global font site zu 12:
plt.rcParams.update({'font.size': 12})
# %% DATA PATHS
# define data paths:
#EthoVision_File_PATH = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/cFC EV/Statistics-cFC EthoVision.xlsx'
Unblinding_Key_File  = '/Users/husker/Workspace/Plastic Project/Revisions 2025/revision_mice.xlsx'

# create a result folder:
RESULTS_PATH = os.path.join(os.path.dirname(Unblinding_Key_File), 'Collective Results')
os.makedirs(RESULTS_PATH, exist_ok=True)

# define group order for plotting:
group_order = ['Control', 'Nano-Plastic', 'Micro-Plastic']

# assign sns colors to groups:
sns_palette = "Set2"
sns.set_palette(sns_palette, n_colors=len(group_order))
sns_palette_use = sns.color_palette(sns_palette, n_colors=len(group_order))

# do not change the following lines:
MICE_df = pd.read_excel(Unblinding_Key_File)
MICE_COLLECTOR_df = MICE_df.copy()
# remove entire column "Excluded cFC" from MICE_COLLECTOR_df:
MICE_COLLECTOR_df = MICE_COLLECTOR_df.drop(columns=['Exclude cFC', 'Behavior Batch', 'ET', 'S'])
MICE_COLLECTOR_df["ID_MICE"] = MICE_COLLECTOR_df["ID"]
# %% GLOBAL PARAMETERS
# set outlier removal parameters:
exclude_outlier = False  # True or False
outlier_method="iqr"     # 'iqr' or 'mad'; only takes effect if exclude_outlier=True
iqr_k=1.0
mad_z=3.5

# use pchance_use_tost in analyze_and_plot function?:
pchance_use_tost = False
""" 
This controls whether, when testing against chance level, we use a regular one-sample t-test
or a TOST equivalence test. This alters the interpretation of the results:
- regular one-sample t-test: tests whether the group mean is significantly different from chance level
- TOST equivalence test: tests whether the group mean is statistically equivalent to chance level
I.e., 
* ttest p-vals <-0.05 means "significantly different from chance level"
* TOST p-vals <-0.05 means "significantly not different from chance level, i.e., significantly equivalent to chance level!"
"""
# %% FUNCTIONS

def merge_cFC_and_MICE_data(cFC_DATA_df, 
                            MICE_df, 
                            id_col_cFC='ID', 
                            id_col_MICE='Alt ID (tail stripes)',
                            id_col_MICE_exclude='Exclude cFC',
                            exclude_marked=True):
    """ 
    cFC_DATA_df contains a column "ID" which is identical 
    to the IDs stored MICE_df's "Alt ID (tail stripes)" column. We need
    a function that merges the two dataframes based on this ID: 
    """
    # MICE_df could have columns, that have identical names as in cFC_DATA_df.
    # To avoid confusion, we can add a suffix to all columns in MICE_df except
    # the ID column:
    MICE_df = MICE_df.add_suffix('_MICE')
    MICE_df = MICE_df.rename(columns={f'{id_col_MICE}_MICE': id_col_MICE})
    
    merged_df = pd.merge(cFC_DATA_df, MICE_df, left_on=id_col_cFC, right_on=id_col_MICE, how='left')
    
    # MICE_df contains a column called "Exclude cFC" that indicates, whether
    # a mouse should be excluded from the analysis (indicate by 'yes'). If exclude_marked is True,
    # we filter the merged_df accordingly:
    if exclude_marked:
        merged_df = merged_df[merged_df[f'{id_col_MICE_exclude}_MICE'] != 'yes']
    return merged_df

def remove_outliers(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    method: str = "iqr",     # "iqr" (Tukey) oder "mad" (robust z)
    iqr_k: float = 1.5,
    mad_z: float = 3.5,
    drop_na: bool = True,
    return_mask: bool = False):
    """
    Entfernt Outlier aus df basierend auf value_col.
    Wenn group_col gesetzt ist, werden Outlier *pro Gruppe* bestimmt (empfohlen).

    Returns:
        df_filtered (und optional outlier_mask als pd.Series[bool])
    """
    df_use = df.copy()
    s = pd.to_numeric(df_use[value_col], errors="coerce")

    if drop_na:
        df_use = df_use.loc[s.notna()].copy()
        s = s.loc[s.notna()]

    def _mask_iqr(x: pd.Series) -> pd.Series:
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(False, index=x.index)
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        return (x < lo) | (x > hi)

    def _mask_mad(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = np.median(np.abs(x - med))
        if pd.isna(mad) or mad == 0:
            return pd.Series(False, index=x.index)
        modified_z = 0.6745 * (x - med) / mad
        return np.abs(modified_z) > mad_z

    if method not in {"iqr", "mad"}:
        raise ValueError("method must be 'iqr' or 'mad'")

    if group_col is not None:
        if method == "iqr":
            out_mask = df_use.groupby(group_col, observed=False)[value_col].transform(lambda x: _mask_iqr(pd.to_numeric(x, errors="coerce")))
        else:
            out_mask = df_use.groupby(group_col, observed=False)[value_col].transform(lambda x: _mask_mad(pd.to_numeric(x, errors="coerce")))
    else:
        out_mask = _mask_iqr(s) if method == "iqr" else _mask_mad(s)

    out_mask = out_mask.fillna(False)
    df_filt = df_use.loc[~out_mask].copy()
    
    print(f"Outlier removal using method '{method}' with iqr_k={iqr_k} mad_z={mad_z}:")
    
    # print per group, which values were removed as outliers:
    if group_col is not None:
        groups = df_use[group_col].unique()
        for group in groups:
            group_mask = df_use[group_col] == group
            outliers_in_group = df_use.loc[group_mask & out_mask, value_col]
            if not outliers_in_group.empty:
                print(f"Removed outliers in group '{group}':")
                print(outliers_in_group)
    else:
        outliers = df_use.loc[out_mask, value_col]
        if not outliers.empty:
            print("Removed outliers:")
            print(outliers)

    if return_mask:
        return df_filt, out_mask
    return df_filt

def append_n(group_name, group_sizes):
    return f"{group_name}\n(N={group_sizes.get(group_name, 0)})"

def add_pairwise_sig_bars_axescoords(ax, x_order, posthoc_results,
                                    y_start=1.02, y_step=0.04, h=0.015,
                                    text_offset=0.005,
                                    plot_pvalues=True, fontsize=10, lw=1.5):
    """
    Draw pairwise significance bars in axes y-coordinates (stable across plots).
    x is in data coordinates (category index), y is in axes fraction.

    Parameters
    ----------
    y_start : float
        Start height in axes fraction. >1 places bars above the axes area.
    y_step : float
        Vertical spacing per comparison in axes fraction.
    h : float
        Height of the bracket in axes fraction.
    """

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # We want comparisons in the same group order as the x-axis
    groups = [g.split("N=")[0] if "N=" in g else g for g in x_order]
    groups = [g.split("\n")[0] if "\n" in g else g for g in x_order]

    curr_y = y_start

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]

            row = posthoc_results[
                ((posthoc_results["A"] == g1) & (posthoc_results["B"] == g2)) |
                ((posthoc_results["A"] == g2) & (posthoc_results["B"] == g1))
            ]
            if row.empty:
                continue

            p = float(row["p-corr"].values[0])

            if not plot_pvalues:
                if p < 0.001:
                    p_text = "***"
                elif p < 0.01:
                    p_text = "**"
                elif p < 0.05:
                    p_text = "*"
                else:
                    p_text = "n.s."
            else:
                p_text = f"$p$ = {p:.3f}"

            x1 = i
            x2 = j

            # bracket: (x1,curr_y) -> (x1,curr_y+h) -> (x2,curr_y+h) -> (x2,curr_y)
            ax.plot([x1, x1, x2, x2],
                    [curr_y, curr_y + h, curr_y + h, curr_y],
                    transform=trans, lw=lw, color="k", clip_on=False)

            ax.text((x1 + x2) * 0.5, curr_y + h + text_offset,
                    p_text, transform=trans, ha="center", va="bottom",
                    fontsize=fontsize, color="k", clip_on=False)

            curr_y += y_step

    return curr_y

def add_chance_sig_markers_axescoords(ax, x_order, pchance_vals,
                                      y_start, h=0.015, text_offset=0.005,
                                      plot_pvalues=True, fontsize=10, lw=1.5):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    groups = [g.split("N=")[0] if "N=" in g else g for g in x_order]
    groups = [g.split("\n")[0] if "\n" in g else g for g in x_order]

    curr_y = y_start
    for i, g in enumerate(groups):
        if g not in pchance_vals:
            continue
        p = float(pchance_vals[g])

        if not plot_pvalues:
            if p < 0.001:
                p_text = "***"
            elif p < 0.01:
                p_text = "**"
            elif p < 0.05:
                p_text = "*"
            else:
                p_text = "n.s."
        else:
            p_text = f"$p_c$ = {p:.3f}"

        ax.plot([i, i], [curr_y, curr_y + h],
                transform=trans, lw=lw, color="k", clip_on=False)
        ax.text(i, curr_y + h + text_offset,
                p_text, transform=trans, ha="center", va="bottom",
                fontsize=fontsize, color="k", clip_on=False)

    return curr_y + h

def analyze_and_plot(data_df, value_col, group_col, result_path, 
                     plot_pvalues=True, figsize=(4,6),
                     stats_y_start=0.95, stats_y_step=0.07,stats_h=0.02,
                     ylim=None, max_title_length=35, pchance=None, pchance_use_tost=False,
                     exclude_outlier=False, outlier_method="iqr", iqr_k=1.0, mad_z=3.5):
    """ 
    Perform statistical analysis and plotting for the given value column
    grouped by the group column.
    
    DEBUG:
    data_df = cFC_RECALL_df
    value_col = 'Freezing relative Cumulative Duration %'
    group_col = 'Group_MICE'
    result_path = Resultspath
    """
    ## %%
    data_df_use = data_df.copy()
        
    # define marker edge color for outliers:
    if exclude_outlier:
        markeredgecolor = "red"
    else:
        markeredgecolor = "black"
        
    # in data_df's column group_col, append "N =xx" to each group name, but only once!
    data_df_use_for_plotting_only = data_df_use.copy()
    group_sizes = data_df_use[group_col].value_counts().to_dict()
    data_df_use_for_plotting_only[group_col] = data_df_use_for_plotting_only[group_col].apply(append_n, args=(group_sizes,))
        
    # main plotting:
    fig, ax = plt.subplots(figsize=figsize)
    x_order = list(data_df_use_for_plotting_only[group_col].unique())
    sns.boxplot(x=group_col, y=value_col, data=data_df_use_for_plotting_only, palette=sns_palette_use,
                width=0.6, hue=group_col, hue_order=x_order,
                flierprops=dict(marker='o',
                                markersize=8,
                                markerfacecolor='white',
                                markeredgecolor=markeredgecolor,
                                alpha=0.9), ax=ax)
    sns.swarmplot(x=group_col, y=value_col, data=data_df_use_for_plotting_only, color=".25", ax=ax)
    
    # remove outliers from subsequent statistics:
    if exclude_outlier:
        data_df_use = remove_outliers(
            data_df_use,
            value_col=value_col,
            group_col=group_col,
            method=outlier_method,   # alternativ: "mad"
            iqr_k=iqr_k, mad_z=mad_z)
        
    # statistical analysis:
    # 1st check whether data is normally distributed:
    normality_results = pg.normality(data=data_df_use, dv=value_col, group=group_col)
    print(f"Normality results for {value_col}:\n", normality_results)
    # if there is any group that is not normally distributed, we use Kruskal-Wallis test:
    if (normality_results['normal'].values == False).any():
        print(f"At least one group in {group_col} is not normally distributed for {value_col}. Using Kruskal-Wallis test.")
        kw_results = pg.kruskal(data=data_df_use, dv=value_col, between=group_col)
        print(f"Kruskal-Wallis results for {value_col}:\n", kw_results)
        
        # Post-hoc test:
        posthoc_results = pg.pairwise_tests(data=data_df_use, dv=value_col, between=group_col, padjust='fdr_bh', 
                                            parametric=False)
        print(f"Post-hoc results for {value_col}:\n", posthoc_results)   
    else:
        print(f"All groups in {group_col} are normally distributed for {value_col}. Using ANOVA.")
        anova_results = pg.anova(data=data_df_use, dv=value_col, between=group_col, detailed=True)
        print(f"ANOVA results for {value_col}:\n", anova_results)

        # Post-hoc test:
        posthoc_results = pg.pairwise_tests(data=data_df_use, dv=value_col, between=group_col, padjust='fdr_bh')
        print(f"Post-hoc results for {value_col}:\n", posthoc_results)
    # also, if pchance is not None: calculate whether each group is significantly different from chance level:
    pchance_vals = {}
    effect_sizes = {}
    if pchance is not None:
        print(f"Performing one-sample t-tests against chance level {pchance} for each group in {group_col}.")
        unique_groups = data_df_use[group_col].unique()
        for group in unique_groups:
            # group = unique_groups[0]
            # we now use pg.ttest() and test each group's data against pchance:
            group_data = data_df_use[data_df_use[group_col] == group][value_col]
            # is group_data normally distributed?
            res = group_data - pchance
            curr_normal = pg.normality(res)['normal'].values[0]
            #curr_normal = pg.normality(data=group_data)['normal'].values[0]
            if not pchance_use_tost:
                if not curr_normal:
                    #pchance_vals[group] = sp.stats.ks_2samp(group_data, pchance)[1]
                    x = group_data.dropna().to_numpy()
                    res = x - pchance
                    pchance_vals[group] = pg.wilcoxon(res)["p-val"].iat[0]
                else:
                    pchance_vals[group] = pg.ttest(group_data, pchance)["p-val"].values[0]
            else:
                # ignore normality, use TOST equivalence test:
                # NOTE: this alters the interpretation of the p-values! I.e., p-val <0.05 means "statistically equivalent to chance level"
                pchance_vals[group] = pg.tost(group_data, y=pchance, paired=False,bound=2.5, correction="auto")["pval"].values[0]
            
            # also calculate the effect size (Cohen's d) for each group against chance level:
            #effect_size = pg.compute_effsize(group_data, pchance, eftype='cohen')
            x = group_data.dropna().to_numpy()
            sd = x.std(ddof=1)
            effect_sizes[group] = np.nan if sd == 0 or not np.isfinite(sd) else (x.mean() - pchance) / sd
        print(f"One-sample test p-values against chance level {pchance}:")
        for group, p_val in pchance_vals.items():
            print(f"   Group {group}: p = {p_val:.3f}")
        # do we need to correct for multiple comparisons here? 
        multicomp_results = pg.multicomp(list(pchance_vals.values()), method='fdr_bh')
        print(f"Multiple comparison corrected p-values against chance level {pchance} ({multicomp_results[0]}):")
        for i, group in enumerate(unique_groups):
            corrected_p = multicomp_results[1][i]
            print(f"   Group {group}: corrected p = {corrected_p:.3f}")
            # overwrite pchance_vals with corrected p-values:
            pchance_vals[group] = corrected_p
        
        # print effect sizes:
        print(f"Effect sizes (Cohen's d) against chance level {pchance}:")
        for group, eff_size in effect_sizes.items():
            print(f"Group {group}: d = {eff_size:.3f}")
   
        
    # plot pairwise significance bars:
    y_end = add_pairwise_sig_bars_axescoords(
        ax=ax,
        x_order=x_order,
        posthoc_results=posthoc_results,
        y_start=stats_y_start,
        y_step=stats_y_step,
        h=stats_h,
        plot_pvalues=plot_pvalues)
    # plot chance level significance markers:
    if pchance is not None:
        # chance level line:
        ax.axhline(y=pchance, linestyle='--', label=f'Chance Level {pchance}', c='grey')
        # annotate chance level line:
        ax.text(0.00-0.5*0.6, pchance + (data_df_use[value_col].max() - data_df_use[value_col].min()) * 0.01, 
                 f'$p_c$', color='grey', ha='left', va='bottom')

        add_chance_sig_markers_axescoords(
            ax=ax,
            x_order=x_order,
            pchance_vals=pchance_vals,
            y_start=y_end + stats_h,   # small gap above pairwise bars; alt: + 0.02
            h=stats_h,
            plot_pvalues=plot_pvalues
        )
    # make room above axes for annotations that are >1 in axes coords
    ax.margins(y=0.10)
    fig.subplots_adjust(top=0.80)  # tune; gives extra top space
    ## %%
    
    # indicate by horizontal bars + annotation the p-values of all comparisons:
    # get unique groups:
    """ unique_groups = data_df[group_col].unique()
    y_max = data_df_use[value_col].max()
    y_min = data_df_use[value_col].min()
    y_range = y_max - y_min
    y_offset = y_range * 0.1  # 10% of the range
    bar_height = y_max + y_offset
    bar_spacing = y_range * bar_spacing_rel # * 0.05  # 5% of the range
    current_bar_height = bar_height
    for i in range(len(x_order)):
        for j in range(i + 1, len(unique_groups)):
            group1 = unique_groups[i]
            group2 = unique_groups[j]
            # get p-value from posthoc_results:
            p_value_row = posthoc_results[
                ((posthoc_results['A'] == group1) & (posthoc_results['B'] == group2)) |
                ((posthoc_results['A'] == group2) & (posthoc_results['B'] == group1))]
            if not p_value_row.empty:
                p_value = p_value_row['p-corr'].values[0]
                # plot horizontal bar:
                x1 = i
                x2 = j
                plt.plot([x1, x1, x2, x2], 
                         [current_bar_height, current_bar_height + y_offset * 0.2, 
                          current_bar_height + y_offset * 0.2, current_bar_height], 
                         lw=1.5, c='k')
                # annotate p-value:
                if not plot_pvalues:
                    if p_value < 0.001:
                        p_text = '***'
                    elif p_value < 0.01:
                        p_text = '**'
                    elif p_value < 0.05:
                        p_text = '*'
                    else:
                        p_text = 'n.s.'
                else:
                    p_text = f"$p$ = {p_value:.3f}"
                plt.text((x1 + x2) * 0.5, current_bar_height + y_offset * 0.25, p_text, 
                         ha='center', va='bottom', color='k')
                current_bar_height += bar_spacing
    
    # also, if pchance is not None: calculate whether each group is significantly different from chance level:
    if pchance is not None:
        for i in range(len(unique_groups)):
            # i = 0
            group = unique_groups[i]
            # now plot a vertical short dashed other each group and above that the corresponding p-value
            # from pchance_vals[group]; use current_bar_height + some offset for the height:
            x1 = i
            x2 = i
            plt.plot([x1, x1, x2, x2], 
                     [current_bar_height, current_bar_height + y_offset * 0.2, 
                      current_bar_height + y_offset * 0.2, current_bar_height], 
                     lw=1.5, c='k')
            p_value = pchance_vals[group]
            if not plot_pvalues:
                if p_value < 0.001:
                    p_text = '***'
                elif p_value < 0.01:
                    p_text = '**'
                elif p_value < 0.05:
                    p_text = '*'
                else:
                    p_text = 'n.s.'
            else:
                p_text = f"$p_c$ = {p_value:.3f}"
            plt.text(i, current_bar_height + y_offset * 0.25, p_text, 
                     ha='center', va='bottom', color='k')
            #current_bar_height += bar_spacing """
    ## %%
    
    # set y limit to make space for bars:
    if ylim is not None:
        ax.set_ylim(ylim)
    """ else:
        plt.ylim(y_min, current_bar_height + y_offset) """
    
    # turn off x label:
    plt.xlabel('')
    title = f'{value_col}'
    # in the title bar, only N=35 characters have space; thus, we need to
    # split title 1st by spaces, then re-join with newlines if necessary:
    if len(title) > max_title_length:
        words = title.split(' ')
        new_title = ''
        current_line = ''
        for word in words:
            if len(current_line) + len(word) + 1 <= max_title_length:
                current_line += ' ' + word if current_line else word
            else:
                new_title += current_line + '\n'
                current_line = word
        new_title += current_line
        title = new_title
    
    plt.title(title)
    fig.tight_layout()
    plot_name = os.path.join(result_path, f'{value_col}.png'.replace(' ', '_').replace('/', '_'))
    plt.savefig(plot_name)
    plt.close()
    ## %%
    # create a sub-folder for storing csv results:
    csv_result_path = os.path.join(result_path, 'CSV_Results')
    os.makedirs(csv_result_path, exist_ok=True)
    # save statistical results to csv:
    normality_results.to_csv(os.path.join(csv_result_path, f'Normality_{value_col}.csv'.replace(' ', '_').replace('/', '_')), 
                             index=False)
    posthoc_results.to_csv(os.path.join(csv_result_path, f'Posthoc_{value_col}.csv'.replace(' ', '_').replace('/', '_')), 
                           index=False)
    data_df.to_csv(os.path.join(csv_result_path, f'Data_{value_col}.csv'.replace(' ', '_').replace('/', '_')), 
                   index=False)
    # also save a sub-dataframe with only the data used for plotting:
    sub_data_df = data_df_use[[group_col, value_col]]
    sub_data_df.to_csv(os.path.join(csv_result_path, f'Data_subset_{value_col}.csv'.replace(' ', '_').replace('/', '_')),
                     index=False)
    
    # also save the effect sizes against chance level if calculated:
    if effect_sizes:
        eff_size_df = pd.DataFrame(list(effect_sizes.items()), columns=['Group', 'Cohen_d_vs_Chance'])
        eff_size_df.to_csv(os.path.join(csv_result_path, f'Effect_Sizes_vs_Chance_{value_col}_by_{group_col}.csv'.replace(' ', '_').replace('/', '_')),
                           index=False)

def final_label(key: str) -> str:
    # Einheit nur für bestimmte Variablen:
    if key in ['avg_speed_moving (cut)', 'avg_speed_overall (cut)', 'max_speed (cut)']:
        return key.replace(' (cut)', '').replace('_', ' ') + ' (cm/s)'
    if key == 'total_distance_moved_in_spatial_unit (cut)':
        return key.replace(' (cut)', '').replace('_', ' ') + ' (cm)'
    return key.replace(' (cut)', '').replace('_', ' ')

def plot_conditioning_cFC_trajectories(
    df_long: pd.DataFrame,
    value_col: str,
    subject_col='ID',
    group_col='Group_MICE',
    time_col='Time',
    result_path='.',
    figsize=(6.2, 4.6),
    ylim=None,
    alpha_lines=0.55,
    marker_size=4.0):
    """
    Plot one line per subject across time levels, color-coded by group.
    Also overlays group mean ± SEM.
    """
    os.makedirs(result_path, exist_ok=True)

    plt.figure(figsize=figsize)
    
    df_long_use = df_long.copy()
    # append n_subjects to the group_col for legend clarity:
    group_sizes = df_long_use.groupby(group_col)[subject_col].nunique().to_dict()
    def append_n(group_name):
        return f"{group_name} (N={group_sizes[group_name]})"
    df_long_use[group_col] = df_long_use[group_col].apply(append_n)
    
    # subject trajectories
    sns.lineplot(
        data=df_long_use,
        x=time_col,
        y=value_col,
        hue=group_col,
        units=subject_col,
        estimator=None,
        lw=1.0,
        alpha=alpha_lines,
        marker='o',
        markersize=marker_size,
        legend=False)

    # group mean ± SEM overlay (thicker line, no units)
    sns.pointplot(
        data=df_long_use,
        x=time_col,
        y=value_col,
        hue=group_col,
        dodge=0.15,
        #join=True,
        #scale=0.9,
        markersize=6,     # <- instead of scale=0.9
        linewidth=1.8,     # <- instead of scale=0.9
        errorbar=('se', 1),
        capsize=0.08,
        markers='D',
        linestyles='-',
        estimator='mean',
        legend=True)

    plt.legend(loc='upper left')
    plt.xlabel('')
    plt.ylabel(value_col)
    title = f'{value_col} during conditioning'
    # in the title bar, only N=35 characters have space; thus, we need
    # to split title 1st by spaces, then re-join with newlines if necessary:
    N_CHAR = 30
    if len(title) > N_CHAR:
        words = title.split(' ')
        new_title = ''
        current_line = ''
        for word in words:
            if len(current_line) + len(word) + 1 <= N_CHAR:
                current_line += ' ' + word if current_line else word
            else:
                new_title += current_line + '\n'
                current_line = word
        new_title += current_line
        title = new_title
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    out = os.path.join(result_path, f'conditioning_{value_col}.png')
    plt.savefig(out, dpi=300)
    plt.close()
    #return out

def update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, DATA_df, keys_for_MICE_COLLECTOR, curr_key):
    if curr_key in keys_for_MICE_COLLECTOR:
        # add a new column to MICE_COLLECTOR_df with the current key's values:
        measure_values = DATA_df[[curr_key, 'ID_MICE']].set_index('ID_MICE')[curr_key]
        MICE_COLLECTOR_df = MICE_COLLECTOR_df.set_index('ID')
        MICE_COLLECTOR_df[curr_key] = measure_values
        MICE_COLLECTOR_df = MICE_COLLECTOR_df.reset_index()
        print(f"  Added column '{curr_key}' to MICE_COLLECTOR_df.")
    return MICE_COLLECTOR_df
# %% NOR HAB ANALYSIS
DATA_PATH_NOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/NOR DLC/Hab/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_NOR_HAB = os.path.join(RESULTS_PATH, "NOR Hab")
os.makedirs(PLOT_PATH_NOR_HAB, exist_ok=True)

NOR_DATA_df = pd.read_csv(DATA_PATH_NOR_HAB)
""" 
NOR_DATA_df contains a column called filename with contents like "NOR Hab 1 ID135_3 G5 B6".
We need to extract "NOR" = experiment, "Hab XY" = trial, "IDXXX_XX" = Alt ID (tail stripes) 
from these strings and store it in a new columns:
"""
# extract experiment, trial, Alt ID (tail stripes) from filename:
NOR_DATA_df['Experiment'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
NOR_DATA_df['Trial'] = NOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

NOR_DATA_df.keys()
MICE_df.keys()

NOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=NOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

unique_bodyparts = NOR_DATA_df['bodypart_label'].unique()
unique_trials = NOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[2]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_NOR_HAB_BP = os.path.join(PLOT_PATH_NOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_NOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = NOR_DATA_df[NOR_DATA_df['bodypart_label'] == curr_bp]
    # average the following list of keys over all unique_trials:
    keys_to_average = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_A time_in_zone_s (cut)',
       'object_A zone_crossings (cut)',
       'object_B time_in_zone_s (cut)',
       'object_B zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)']
    averaged_df = curr_bp_df.groupby(['ID_MICE', 'Group_MICE'])[keys_to_average].mean().reset_index()
    
    # for another test, overwrite the averages in averaged_df with Hab 3 and Hab 2 values:
    for key in keys_to_average:
        hab3_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 3'][['ID_MICE', key]]
        hab2_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 2'][['ID_MICE', key]]
        # average Hab 2 and Hab 3 values:
        hab2_3_avg = pd.merge(hab2_values, hab3_values, on='ID_MICE', how='inner', suffixes=('_hab2', '_hab3'))
        hab2_3_avg[key] = hab2_3_avg[[f"{key}_hab2", f"{key}_hab3"]].mean(axis=1)
        hab2_3_avg = hab2_3_avg[['ID_MICE', key]]
        averaged_df = pd.merge(averaged_df, hab2_3_avg, on='ID_MICE', how='left', suffixes=('', '_hab2_3_avg'))
        averaged_df[key] = averaged_df[f"{key}_hab2_3_avg"]
        averaged_df = averaged_df.drop(columns=[f"{key}_hab2_3_avg"])
    # ⟶ I will use the average of Hab 2 and Hab 3 values as these two trials represent stabilized behavior.
    
    
    """ # for a test, overwrite the averages in averaged_df with Hab 3 values:
    for key in keys_to_average:
        hab3_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 3'][['ID_MICE', key]]
        hab3_values = hab3_values.rename(columns={key: f"{key}_hab3"})
        averaged_df = pd.merge(averaged_df, hab3_values, on='ID_MICE', how='left')
        averaged_df[key] = averaged_df[f"{key}_hab3"]
        averaged_df = averaged_df.drop(columns=[f"{key}_hab3"])
    # ⟶ I will use Hab 3 values for Hab-related analysis and plots as this is the last Habituation trial! """
    
    # calculate some relative values and add them to averaged_df:
    averaged_df["time_in_center_in_s (cut) (%)"] = (averaged_df["time_in_center_in_s (cut)"] / averaged_df["total_time_in_arena_s (cut)"]) * 100
    averaged_df["time_in_border_in_s (cut) (%)"] = (averaged_df["time_in_border_in_s (cut)"] / averaged_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    averaged_df["DI (%)"] = ((averaged_df["object_B time_in_zone_s (cut)"]) /
                               (averaged_df["object_A time_in_zone_s (cut)"] + averaged_df["object_B time_in_zone_s (cut)"])) * 100
    current_keys = keys_to_average + [
        "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 320), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 320), #(200, 320)
        'avg_speed_moving (cut)': (0,25),
        'avg_speed_overall (cut)': (0,25),
        'time_in_center_in_s (cut) (%)': (0, 109),
        'time_in_border_in_s (cut) (%)': (0, 109),
        'DI (%)': (0, 119),
        'object_A time_in_zone_s (cut)': (0, 40),
        'object_B time_in_zone_s (cut)': (0, 40),
        'object_A zone_crossings (cut)': (0, 130),
        'object_B zone_crossings (cut)': (0, 130),
        'num_center_border_crossings (cut)': (0, 150),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.27,
        'total_moving_time_in_s (cut)': 0.27,
        'DI (%)': 0.25,
        'time_in_border_in_s (cut) (%)': 0.14,
        'time_in_center_in_s (cut) (%)': 0.14,}
    pchance_dict = {
        'DI (%)': 50.0,
        }
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0][:-2] 
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    averaged_df = averaged_df.rename(columns=new_column_names)
    
    # sort averaged_df by Group_MICE:
    averaged_df['Group_MICE'] = pd.Categorical(averaged_df['Group_MICE'], categories=group_order, ordered=True)
    averaged_df = averaged_df.sort_values('Group_MICE')
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-1]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        pchance = None
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=averaged_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_NOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["NOR Hab headcenter DI (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, averaged_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% NOR TRIAL 1 ANALYSIS
DATA_PATH_NOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/NOR DLC/Trial 1/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_NOR_HAB = os.path.join(RESULTS_PATH, "NOR Trial 1")
os.makedirs(PLOT_PATH_NOR_HAB, exist_ok=True)

NOR_DATA_df = pd.read_csv(DATA_PATH_NOR_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
NOR_DATA_df['Experiment'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
NOR_DATA_df['Trial'] = NOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

NOR_DATA_df.keys()
MICE_df.keys()

NOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=NOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort the dataframe by group_order:
NOR_DATA_df['Group_MICE'] = pd.Categorical(NOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
NOR_DATA_df = NOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = NOR_DATA_df['bodypart_label'].unique()
unique_trials = NOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[0]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_NOR_HAB_BP = os.path.join(PLOT_PATH_NOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_NOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = NOR_DATA_df[NOR_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["time_in_center_in_s (cut) (%)"] = (curr_bp_df["time_in_center_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["time_in_border_in_s (cut) (%)"] = (curr_bp_df["time_in_border_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    curr_bp_df["DI (%)"] = ((curr_bp_df["object_C time_in_zone_s (cut)"]) /
                               (curr_bp_df["object_A time_in_zone_s (cut)"] + curr_bp_df["object_C time_in_zone_s (cut)"])) * 100
    curr_bp_df["DI by zone crossings (%)"] = ((curr_bp_df["object_C zone_crossings (cut)"]) /
                               (curr_bp_df["object_A zone_crossings (cut)"] + curr_bp_df["object_C zone_crossings (cut)"])) * 100
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_A time_in_zone_s (cut)',
       'object_A zone_crossings (cut)',
       'object_C time_in_zone_s (cut)',
       'object_C zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)',
       "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI by zone crossings (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 660), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 660), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'avg_speed_overall (cut)': (0,20),
        'time_in_center_in_s (cut) (%)': (0, 119),
        'time_in_border_in_s (cut) (%)': (0, 119),
        'DI by zone crossings (%)': (0, 119),
        'DI (%)': (0, 119),
        'object_A time_in_zone_s (cut)': (0, 100),
        'object_C time_in_zone_s (cut)': (0, 100),
        'object_A zone_crossings (cut)': (0, 200),
        'object_C zone_crossings (cut)': (0, 200),
        'num_center_border_crossings (cut)': (0, 250),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'DI (%)': 0.15,
        'time_in_border_in_s (cut) (%)': 0.13,
        'time_in_center_in_s (cut) (%)': 0.13,
        'num_center_border_crossings (cut)': 0.09,}
    pchance_dict = {
        'DI by zone crossings (%)': 50.0,
        'DI (%)': 50.0,}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        # key = current_keys[0]
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-1]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_NOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["NOR Trail 1 headcenter DI (%)", "NOR Trail 1 headcenter DI by zone crossings (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% NOR TRIAL 2 ANALYSIS
DATA_PATH_NOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/NOR DLC/Trial 2/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_NOR_HAB = os.path.join(RESULTS_PATH, "NOR Trial 2")
os.makedirs(PLOT_PATH_NOR_HAB, exist_ok=True)

NOR_DATA_df = pd.read_csv(DATA_PATH_NOR_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
NOR_DATA_df['Experiment'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
NOR_DATA_df['Trial'] = NOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
NOR_DATA_df['Alt ID (tail stripes)'] = NOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

NOR_DATA_df.keys()
MICE_df.keys()

NOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=NOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort the dataframe by group_order:
NOR_DATA_df['Group_MICE'] = pd.Categorical(NOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
NOR_DATA_df = NOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = NOR_DATA_df['bodypart_label'].unique()
unique_trials = NOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[-1]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_NOR_HAB_BP = os.path.join(PLOT_PATH_NOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_NOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = NOR_DATA_df[NOR_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["time_in_center_in_s (cut) (%)"] = (curr_bp_df["time_in_center_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["time_in_border_in_s (cut) (%)"] = (curr_bp_df["time_in_border_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    curr_bp_df["DI (%)"] = ((curr_bp_df["object_D time_in_zone_s (cut)"]) /
                               (curr_bp_df["object_A time_in_zone_s (cut)"] + curr_bp_df["object_D time_in_zone_s (cut)"])) * 100
    curr_bp_df["DI by zone crossings (%)"] = ((curr_bp_df["object_D zone_crossings (cut)"]) /
                               (curr_bp_df["object_A zone_crossings (cut)"] + curr_bp_df["object_D zone_crossings (cut)"])) * 100
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_A time_in_zone_s (cut)',
       'object_A zone_crossings (cut)',
       'object_D time_in_zone_s (cut)',
       'object_D zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)',
       "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI by zone crossings (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 660), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 660), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'avg_speed_overall (cut)': (0,20),
        'time_in_center_in_s (cut) (%)': (0, 119),
        'time_in_border_in_s (cut) (%)': (0, 119),
        'DI by zone crossings (%)': (0, 119),
        'DI (%)': (0, 119),
        'object_A time_in_zone_s (cut)': (0, 100),
        'object_D time_in_zone_s (cut)': (0, 100),
        'object_A zone_crossings (cut)': (0, 200),
        'object_D zone_crossings (cut)': (0, 200),
        'num_center_border_crossings (cut)': (0, 250),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'DI (%)': 0.15,
        'time_in_border_in_s (cut) (%)': 0.13,
        'time_in_center_in_s (cut) (%)': 0.13,
        'num_center_border_crossings (cut)': 0.09,}
    pchance_dict = {
        'DI by zone crossings (%)': 50.0,
        'DI (%)': 50.0,}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-1]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_NOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)

        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["NOR Trail 2 headcenter DI (%)", "NOR Trail 2 headcenter DI by zone crossings (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% SOR HAB ANALYSIS
DATA_PATH_SOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/SOR DLC/Hab/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_SOR_HAB = os.path.join(RESULTS_PATH, "SOR Hab")
os.makedirs(PLOT_PATH_SOR_HAB, exist_ok=True)

SOR_DATA_df = pd.read_csv(DATA_PATH_SOR_HAB)
""" 
SOR_DATA_df contains a column called filename with contents like "SOR Hab 1 ID135_3 G5 B6".
We need to extract "SOR" = experiment, "Hab XY" = trial, "IDXXX_XX" = Alt ID (tail stripes) 
from these strings and store it in a new columns:
"""
# extract experiment, trial, Alt ID (tail stripes) from filename:
SOR_DATA_df['Experiment'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
SOR_DATA_df['Trial'] = SOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

SOR_DATA_df.keys()
MICE_df.keys()

SOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=SOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort SOR_DATA_df according to group_order:
# SOR_DATA_df['Group_MICE'] = pd.Categorical(SOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
# SOR_DATA_df = SOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = SOR_DATA_df['bodypart_label'].unique()
unique_trials = SOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[0]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_SOR_HAB_BP = os.path.join(PLOT_PATH_SOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_SOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = SOR_DATA_df[SOR_DATA_df['bodypart_label'] == curr_bp]
    # average the following list of keys over all unique_trials:
    keys_to_average = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_A time_in_zone_s (cut)',
       'object_A zone_crossings (cut)',
       'object_B time_in_zone_s (cut)',
       'object_B zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)']
    averaged_df = curr_bp_df.groupby(['ID_MICE', 'Group_MICE'])[keys_to_average].mean().reset_index()
    
    """ # for a test, overwrite the averages in averaged_df with Hab 3 values:
    for key in keys_to_average:
        hab3_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 2'][['ID_MICE', key]]
        hab3_values = hab3_values.rename(columns={key: f"{key}_hab3"})
        averaged_df = pd.merge(averaged_df, hab3_values, on='ID_MICE', how='left')
        averaged_df[key] = averaged_df[f"{key}_hab3"]
        averaged_df = averaged_df.drop(columns=[f"{key}_hab3"])
    # ⟶ I will use Hab 3 values for Hab-related analysis and plots as this is the last Habituation trial! """
    
    # for another test, overwrite the averages in averaged_df with Hab 2 and Hab 3 values:
    for key in keys_to_average:
        hab3_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 2'][['ID_MICE', key]]
        hab2_values = curr_bp_df[curr_bp_df['Trial'] == 'Hab 3'][['ID_MICE', key]]
        # average Hab 2 and Hab 3 values:
        hab2_3_avg = pd.merge(hab2_values, hab3_values, on='ID_MICE', how='inner', suffixes=('_hab2', '_hab3'))
        hab2_3_avg[key] = hab2_3_avg[[f"{key}_hab2", f"{key}_hab3"]].mean(axis=1)
        hab2_3_avg = hab2_3_avg[['ID_MICE', key]]
        averaged_df = pd.merge(averaged_df, hab2_3_avg, on='ID_MICE', how='left', suffixes=('', '_hab2_3_avg'))
        averaged_df[key] = averaged_df[f"{key}_hab2_3_avg"]
        averaged_df = averaged_df.drop(columns=[f"{key}_hab2_3_avg"])
    # ⟶ I will use the average of Hab 2 and Hab 3 values as these two trials represent stabilized behavior.
    
    # calculate some relative values and add them to averaged_df:
    averaged_df["time_in_center_in_s (cut) (%)"] = (averaged_df["time_in_center_in_s (cut)"] / averaged_df["total_time_in_arena_s (cut)"]) * 100
    averaged_df["time_in_border_in_s (cut) (%)"] = (averaged_df["time_in_border_in_s (cut)"] / averaged_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    averaged_df["DI (%)"] = ((averaged_df["object_B time_in_zone_s (cut)"]) /
                               (averaged_df["object_A time_in_zone_s (cut)"] + averaged_df["object_B time_in_zone_s (cut)"])) * 100
    current_keys = keys_to_average + [
        "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 320), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 320), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'time_in_center_in_s (cut) (%)': (0, 109),
        'time_in_border_in_s (cut) (%)': (0, 109),
        'DI (%)': (0, 119),
        'object_A time_in_zone_s (cut)': (0, 40),
        'object_B time_in_zone_s (cut)': (0, 40),
        'object_A zone_crossings (cut)': (0, 130),
        'object_B zone_crossings (cut)': (0, 130),
        'num_center_border_crossings (cut)': (0, 150),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.27,
        'total_moving_time_in_s (cut)': 0.27,
        'DI (%)': 0.25,
        'time_in_border_in_s (cut) (%)': 0.14,
        'time_in_center_in_s (cut) (%)': 0.14,}
    pchance_dict = {
        'DI (%)': 50.0,
        }
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0][:-2]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    averaged_df = averaged_df.rename(columns=new_column_names)
    
    # sort averaged_df by Group_MICE:
    averaged_df['Group_MICE'] = pd.Categorical(averaged_df['Group_MICE'], categories=group_order, ordered=True)
    averaged_df = averaged_df.sort_values('Group_MICE')
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[0]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        pchance = None
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=averaged_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_SOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["SOR Hab headcenter DI (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, averaged_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% SOR TRIAL 1 ANALYSIS
DATA_PATH_SOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/SOR DLC/Trial 1/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_SOR_HAB = os.path.join(RESULTS_PATH, "SOR Trial 1")
os.makedirs(PLOT_PATH_SOR_HAB, exist_ok=True)

SOR_DATA_df = pd.read_csv(DATA_PATH_SOR_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
SOR_DATA_df['Experiment'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
SOR_DATA_df['Trial'] = SOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

SOR_DATA_df.keys()
MICE_df.keys()

SOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=SOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort the dataframe by group_order:
SOR_DATA_df['Group_MICE'] = pd.Categorical(SOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
SOR_DATA_df = SOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = SOR_DATA_df['bodypart_label'].unique()
unique_trials = SOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[-1]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_SOR_HAB_BP = os.path.join(PLOT_PATH_SOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_SOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = SOR_DATA_df[SOR_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["time_in_center_in_s (cut) (%)"] = (curr_bp_df["time_in_center_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["time_in_border_in_s (cut) (%)"] = (curr_bp_df["time_in_border_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    curr_bp_df["DI (%)"] = ((curr_bp_df["object_C time_in_zone_s (cut)"]) /
                               (curr_bp_df["object_A time_in_zone_s (cut)"] + curr_bp_df["object_C time_in_zone_s (cut)"])) * 100
    curr_bp_df["DI by zone crossings (%)"] = ((curr_bp_df["object_C zone_crossings (cut)"]) /
                               (curr_bp_df["object_A zone_crossings (cut)"] + curr_bp_df["object_C zone_crossings (cut)"])) * 100
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_A time_in_zone_s (cut)',
       'object_A zone_crossings (cut)',
       'object_C time_in_zone_s (cut)',
       'object_C zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)',
       "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI by zone crossings (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 660), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 660), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'avg_speed_overall (cut)': (0,20),
        'time_in_center_in_s (cut) (%)': (0, 119),
        'time_in_border_in_s (cut) (%)': (0, 119),
        'DI by zone crossings (%)': (0, 119),
        'DI (%)': (0, 119),
        'object_A time_in_zone_s (cut)': (0, 100),
        'object_C time_in_zone_s (cut)': (0, 100),
        'object_A zone_crossings (cut)': (0, 200),
        'object_C zone_crossings (cut)': (0, 200),
        'num_center_border_crossings (cut)': (0, 250),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'DI (%)': 0.15,
        'time_in_border_in_s (cut) (%)': 0.13,
        'time_in_center_in_s (cut) (%)': 0.13,
        'num_center_border_crossings (cut)': 0.09,}
    pchance_dict = {
        'DI by zone crossings (%)': 50.0,
        'DI (%)': 50.0,}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key

        # Wichtig: ylims_dict/bar_spacing/pchance auf denselben finalen new_key schreiben
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)

    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-1]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_SOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["SOR Trial 1 headcenter DI (%)", "SOR Trial 1 headcenter DI by zone crossings (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% SOR TRIAL 2 ANALYSIS
DATA_PATH_SOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/SOR DLC/Trial 2/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_SOR_HAB = os.path.join(RESULTS_PATH, "SOR Trial 2")
os.makedirs(PLOT_PATH_SOR_HAB, exist_ok=True)

SOR_DATA_df = pd.read_csv(DATA_PATH_SOR_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
SOR_DATA_df['Experiment'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
SOR_DATA_df['Trial'] = SOR_DATA_df['filename'].apply(lambda x: ' '.join(x.split(' ')[1:3]))
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[3])
# remove "ID" from leading of Alt ID (tail stripes):
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

SOR_DATA_df.keys()
MICE_df.keys()

SOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=SOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort the dataframe by group_order:
SOR_DATA_df['Group_MICE'] = pd.Categorical(SOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
SOR_DATA_df = SOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = SOR_DATA_df['bodypart_label'].unique()
unique_trials = SOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[0]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_SOR_HAB_BP = os.path.join(PLOT_PATH_SOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_SOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = SOR_DATA_df[SOR_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["time_in_center_in_s (cut) (%)"] = (curr_bp_df["time_in_center_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["time_in_border_in_s (cut) (%)"] = (curr_bp_df["time_in_border_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    # calculate discrimination index for object A and B:
    curr_bp_df["DI (%)"] = ((curr_bp_df["object_D time_in_zone_s (cut)"]) /
                               (curr_bp_df["object_C time_in_zone_s (cut)"] + curr_bp_df["object_D time_in_zone_s (cut)"])) * 100
    curr_bp_df["DI by zone crossings (%)"] = ((curr_bp_df["object_D zone_crossings (cut)"]) /
                               (curr_bp_df["object_C zone_crossings (cut)"] + curr_bp_df["object_D zone_crossings (cut)"])) * 100
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',  
       'object_C time_in_zone_s (cut)',
       'object_C zone_crossings (cut)',
       'object_D time_in_zone_s (cut)',
       'object_D zone_crossings (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)',
       "time_in_center_in_s (cut) (%)",
        "time_in_border_in_s (cut) (%)",
        "DI by zone crossings (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 660), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 660), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'avg_speed_overall (cut)': (0,20),
        'time_in_center_in_s (cut) (%)': (0, 119),
        'time_in_border_in_s (cut) (%)': (0, 119),
        'DI by zone crossings (%)': (0, 119),
        'DI (%)': (0, 119),
        'object_C time_in_zone_s (cut)': (0, 100),
        'object_D time_in_zone_s (cut)': (0, 100),
        'object_C zone_crossings (cut)': (0, 200),
        'object_D zone_crossings (cut)': (0, 200),
        'num_center_border_crossings (cut)': (0, 250),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'DI (%)': 0.15,
        'time_in_border_in_s (cut) (%)': 0.13,
        'time_in_center_in_s (cut) (%)': 0.13,
        'num_center_border_crossings (cut)': 0.09,}
    pchance_dict = {
        'DI by zone crossings (%)': 50.0,
        'DI (%)': 50.0,}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[0]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_SOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["SOR Trial 2 headcenter DI (%)", "SOR Trial 2 headcenter DI by zone crossings (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% EMP ANALYSIS
DATA_PATH_SOR_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/EMP DLC/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_SOR_HAB = os.path.join(RESULTS_PATH, "EMP")
os.makedirs(PLOT_PATH_SOR_HAB, exist_ok=True)

SOR_DATA_df = pd.read_csv(DATA_PATH_SOR_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
SOR_DATA_df['Experiment'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
SOR_DATA_df['Trial'] = "Trial 1"
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['filename'].apply(lambda x: x.split(' ')[1])
# remove "ID" from leading of Alt ID (tail stripes):
SOR_DATA_df['Alt ID (tail stripes)'] = SOR_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

SOR_DATA_df.keys()
MICE_df.keys()

SOR_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=SOR_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# sort the dataframe by group_order:
SOR_DATA_df['Group_MICE'] = pd.Categorical(SOR_DATA_df['Group_MICE'], categories=group_order, ordered=True)
SOR_DATA_df = SOR_DATA_df.sort_values('Group_MICE')

unique_bodyparts = SOR_DATA_df['bodypart_label'].unique()
unique_trials = SOR_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[-1]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_SOR_HAB_BP = os.path.join(PLOT_PATH_SOR_HAB, curr_bp)
    os.makedirs(PLOT_PATH_SOR_HAB_BP, exist_ok=True)
    
    curr_bp_df = SOR_DATA_df[SOR_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["center time_in_zone_s (cut) (%)"] = (curr_bp_df["center time_in_zone_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["arm_up time_in_zone_s (cut) (%)"] = (curr_bp_df["arm_up time_in_zone_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["arm_down time_in_zone_s (cut) (%)"] = (curr_bp_df["arm_down time_in_zone_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["arm_right time_in_zone_s (cut) (%)"] = (curr_bp_df["arm_right time_in_zone_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["arm_left time_in_zone_s (cut) (%)"] = (curr_bp_df["arm_left time_in_zone_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["open_arm time_in_zone_s (cut) (%)"] = ((curr_bp_df["arm_up time_in_zone_s (cut)"] + curr_bp_df["arm_down time_in_zone_s (cut)"]) / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["closed_arm time_in_zone_s (cut) (%)"] = ((curr_bp_df["arm_right time_in_zone_s (cut)"] + curr_bp_df["arm_left time_in_zone_s (cut)"]) / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    curr_bp_df["open_arm entries (cut)"] = (curr_bp_df["arm_up zone_entries (cut)"] + curr_bp_df["arm_down zone_entries (cut)"])
    curr_bp_df["closed_arm entries (cut)"] = (curr_bp_df["arm_right zone_entries (cut)"] + curr_bp_df["arm_left zone_entries (cut)"])
    
    # calculate discrimination index for object A and B:
    open_arms_time_total = (curr_bp_df["arm_up time_in_zone_s (cut)"] + curr_bp_df["arm_down time_in_zone_s (cut)"])
    closed_arms_time_total = (curr_bp_df["arm_right time_in_zone_s (cut)"] + curr_bp_df["arm_left time_in_zone_s (cut)"])
    curr_bp_df["DI (%)"] = ((open_arms_time_total) / (open_arms_time_total + closed_arms_time_total)) * 100
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'open_arm entries (cut)',
       'closed_arm entries (cut)',
       'center zone_crossings (cut)',
        "open_arm time_in_zone_s (cut) (%)",
        "closed_arm time_in_zone_s (cut) (%)",
        "center time_in_zone_s (cut) (%)",
        "DI (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 300),
        'total_moving_time_in_s (cut)': (0, 300),
        'avg_speed_moving (cut)': (0,10),
        'avg_speed_overall (cut)': (0,10),
        'center time_in_zone_s (cut) (%)': (0, 119),
        'open_arm time_in_zone_s (cut) (%)': (0, 119),
        'closed_arm time_in_zone_s (cut) (%)': (0, 119),
        'open_arm entries (cut)': (0, 80),
        'closed_arm entries (cut)': (0, 80),
        'DI (%)': (0, 119),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'center time_in_zone_s (cut) (%)': 0.16,
        'DI (%)': 0.15,
        # 'time_in_border_in_s (cut) (%)': 0.13,
        # 'time_in_center_in_s (cut) (%)': 0.13,
        # 'num_center_border_crossings (cut)': 0.09,
        }
    pchance_dict = {
        'DI (%)': 50.0,}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-1]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df,
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_SOR_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ["EPM Trial 1 headcenter DI (%)"]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% OF ANALYSIS
DATA_PATH_OF_HAB = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/OF DLC/DLC_analysis/all_mice_OF_measurements_all_bodyparts.csv'
PLOT_PATH_OF_HAB = os.path.join(RESULTS_PATH, "OF")
os.makedirs(PLOT_PATH_OF_HAB, exist_ok=True)

OF_DATA_df = pd.read_csv(DATA_PATH_OF_HAB)

# extract experiment, trial, Alt ID (tail stripes) from filename:
OF_DATA_df['Experiment'] = OF_DATA_df['filename'].apply(lambda x: x.split(' ')[0])
OF_DATA_df['Trial'] = "Trial 1"
OF_DATA_df['Alt ID (tail stripes)'] = OF_DATA_df['filename'].apply(lambda x: x.split(' ')[1])
# remove "ID" from leading of Alt ID (tail stripes):
OF_DATA_df['Alt ID (tail stripes)'] = OF_DATA_df['Alt ID (tail stripes)'].str.replace('ID', '')

OF_DATA_df.keys()
MICE_df.keys()
MICE_df['Alt ID (tail stripes)']

OF_DATA_df = merge_cFC_and_MICE_data(
    cFC_DATA_df=OF_DATA_df,
    MICE_df=MICE_df,
    id_col_cFC='Alt ID (tail stripes)',
    id_col_MICE='Alt ID (tail stripes)',
    exclude_marked=False)

# OFt the dataframe by group_order:
OF_DATA_df['Group_MICE'] = pd.Categorical(OF_DATA_df['Group_MICE'], categories=group_order, ordered=True)
OF_DATA_df = OF_DATA_df.sort_values('Group_MICE')

unique_bodyparts = OF_DATA_df['bodypart_label'].unique()
unique_trials = OF_DATA_df['Trial'].unique()

for curr_bp in unique_bodyparts:
    # curr_bp = unique_bodyparts[-1]
    ## %%
    print(f"Processing bodypart: {curr_bp}")
    
    # create a bodypart sub-folder for results:
    PLOT_PATH_OF_HAB_BP = os.path.join(PLOT_PATH_OF_HAB, curr_bp)
    os.makedirs(PLOT_PATH_OF_HAB_BP, exist_ok=True)
    
    curr_bp_df = OF_DATA_df[OF_DATA_df['bodypart_label'] == curr_bp].copy()
    
    
    # calculate some relative values and add them to curr_bp_df:
    curr_bp_df["time_in_center (cut) (%)"] = (curr_bp_df["time_in_center_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    curr_bp_df["time_in_border (cut) (%)"] = (curr_bp_df["time_in_border_in_s (cut)"] / curr_bp_df["total_time_in_arena_s (cut)"]) * 100
    
    
    # we now consider the same keys as for Hab:
    current_keys = [
       'avg_speed_moving (cut)', 
       'avg_speed_overall (cut)', 
       'max_speed (cut)',
       'total_distance_moved_in_spatial_unit (cut)',
       'total_time_in_arena_s (cut)',
       'total_moving_time_in_s (cut)', 
       'total_nonmoving_time_in_s (cut)',
       'time_in_center_in_s (cut)',
       'time_in_border_in_s (cut)',
       'num_center_border_crossings (cut)',
       "time_in_center (cut) (%)",
        "time_in_border (cut) (%)"]
    
    # define ylims for some keys:
    ylims_dict = {
        'total_nonmoving_time_in_s (cut)': (0, 660), #(0, 120)
        'total_moving_time_in_s (cut)': (0, 660), #(200, 320)
        'avg_speed_moving (cut)': (0,20),
        'avg_speed_overall (cut)': (0,20),
        'time_in_center (cut) (%)': (0, 119),
        'time_in_border (cut) (%)': (0, 119),
        'num_center_border_crossings (cut)': (0, 250),
    }
    bar_spacing_rel_dict = {
        'total_nonmoving_time_in_s (cut)': 0.25,
        'total_moving_time_in_s (cut)': 0.13,
        'time_in_border (cut) (%)': 0.13,
        'time_in_center (cut) (%)': 0.13,
        'num_center_border_crossings (cut)': 0.09,}
    pchance_dict = {}
    
    # rename averaged_df's columns like Experiment Trial Bodypart Key, all w/o "_" and w/p "(cut)":
    new_column_names = {}
    exp = curr_bp_df['Experiment'].unique()[0]
    trial = curr_bp_df['Trial'].unique()[0]
    for key in current_keys:
        new_key = f"{exp} {trial} {curr_bp} {final_label(key)}"
        new_column_names[key] = new_key
        ylims_dict[new_key] = ylims_dict.get(key, None)
        bar_spacing_rel_dict[new_key] = bar_spacing_rel_dict.get(key, 0.12)
        pchance_dict[new_key] = pchance_dict.get(key, None)
    curr_bp_df = curr_bp_df.rename(columns=new_column_names)
    
    
    # plot each of the averaged keys by Group_MICE:
    for key in new_column_names.values():
        # key = list(new_column_names.values())[-2]
        curr_ylim = None
        curr_bar_spacing_rel = 0.12
        pchance = None
        if key in ylims_dict:
            curr_ylim = ylims_dict[key]
        if key in bar_spacing_rel_dict:
            curr_bar_spacing_rel = bar_spacing_rel_dict[key]
        if key in pchance_dict:
            pchance = pchance_dict[key]
        if pchance is not None:
            stats_y_start= 0.74
        else:
            stats_y_start= 0.84
        analyze_and_plot(data_df=curr_bp_df, 
                         value_col=key,
                         group_col='Group_MICE',
                         result_path=PLOT_PATH_OF_HAB_BP,
                         figsize=(4,6),
                         ylim=curr_ylim,
                         stats_y_start=stats_y_start,
                         stats_y_step=0.06,
                         stats_h=0.01,
                         max_title_length=33,
                         pchance=pchance,
                         pchance_use_tost=pchance_use_tost,
                         exclude_outlier=exclude_outlier,
                         outlier_method=outlier_method,
                         iqr_k=iqr_k)
        # update MICE_COLLECTOR_df if applicable:
        keys_for_MICE_COLLECTOR = ['OF Trial 1 center point time in center (%)',
                                   'OF Trial 1 center point num center border crossings',
                                   'OF Trial 1 center point avg speed moving (cm/s)',
                                   'OF Trial 1 center point total moving time in s'
                                   ]
        MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, curr_bp_df, keys_for_MICE_COLLECTOR, curr_key=key)
# %% CFC DLC CONDITIONING ANALYSIS
# TODO
# %% CFC RELOAD DATASET ANALYSIS
DATA_PATH_CFC_RELOAD = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/MICE_collector_recall_freezing_data.csv'
PLOT_PATH_CFC_RELOAD = os.path.join(RESULTS_PATH, "CFC Reload")
os.makedirs(PLOT_PATH_CFC_RELOAD, exist_ok=True)

CFC_RELOAD_DATA_df = pd.read_csv(DATA_PATH_CFC_RELOAD)
""" 
CFC_RELOAD_DATA_df is already prepared and analyzed. It has columns:
We can directly plot the relevant columns:
"""

# sort the dataframe by group_order:
CFC_RELOAD_DATA_df['Group_MICE'] = pd.Categorical(CFC_RELOAD_DATA_df['Group_MICE'], categories=group_order, ordered=True)
CFC_RELOAD_DATA_df = CFC_RELOAD_DATA_df.sort_values('Group_MICE')

relevant_columns = [
    'cFC recall Freezing Freezing Frequency',
    'cFC recall Freezing relative Cumulative Duration %']

for key in relevant_columns:
    # key = relevant_columns[0]
    ylims_dict = {
        'cFC recall Freezing Freezing Frequency': (0, 110),
        'cFC recall Freezing relative Cumulative Duration %': (0, 100),
    }
    curr_ylim = None
    pchance_dict = {}
    if key in ylims_dict:
        curr_ylim = ylims_dict[key]
    pchance = None
    if key in pchance_dict:
        pchance = pchance_dict[key]
    if pchance is not None:
        stats_y_start= 0.74
    else:
        stats_y_start= 0.84 
    analyze_and_plot(data_df=CFC_RELOAD_DATA_df, 
                     value_col=key,
                     group_col='Group_MICE',
                     result_path=PLOT_PATH_CFC_RELOAD,
                     figsize=(4,6),
                     ylim=curr_ylim,
                     stats_y_start=0.84,
                     stats_y_step=0.06,
                     stats_h=0.01,
                     max_title_length=33,
                     pchance=None,
                     pchance_use_tost=pchance_use_tost,
                     exclude_outlier=exclude_outlier,
                     outlier_method=outlier_method,
                     iqr_k=iqr_k)

    # update MICE_COLLECTOR_df; here, directly add a new column to MICE_COLLECTOR_df with 
    # the current key's values, while matching by 'ID':
    MICE_COLLECTOR_df = update_MICE_COLLECTOR_df(MICE_COLLECTOR_df, CFC_RELOAD_DATA_df, [key], curr_key=key)

# we can also replot the conditioning data, , i.e., using plot_conditioning_cFC_trajectories:

# convert CFC_RELOAD_DATA_df into a long table as required by plot_conditioning_cFC_trajectories:
CFC_RELOAD_DATA_long_list = []
for _, row in CFC_RELOAD_DATA_df.iterrows():
    for stage in ['After S1', 'After S2', 'After S3']:
        new_row = {
            'ID': row['ID'],
            'Group_MICE': row['Group_MICE'],
            'Time': stage,
            'cFC conditioning Freezing Freezing Frequency': row[f'cFC conditioning Freezing Freezing Frequency_{stage}'],
            'cFC conditioning Freezing relative Cumulative Duration %': row[f'cFC conditioning Freezing relative Cumulative Duration %_{stage}'],
        }
        CFC_RELOAD_DATA_long_list.append(new_row)
CFC_RELOAD_DATA_long_df = pd.DataFrame(CFC_RELOAD_DATA_long_list)

relevant_columns = [
    'cFC conditioning Freezing Freezing Frequency',
    'cFC conditioning Freezing relative Cumulative Duration %']

for key in relevant_columns:
    # key = relevant_columns[0]
    ylims_cond = {
        'cFC conditioning Freezing Freezing Frequency': (0, 35),
        'cFC conditioning Freezing relative Cumulative Duration %': (0, 55)}
    
    plot_conditioning_cFC_trajectories(
            figsize=(3.8, 4.8),
            df_long=CFC_RELOAD_DATA_long_df,
            value_col=key,
            subject_col='ID',
            group_col='Group_MICE',
            time_col='Time',
            result_path=PLOT_PATH_CFC_RELOAD,
            ylim=ylims_cond.get(key),
            alpha_lines=0.25)
    
# %% PCA ANALYSIS

PCA_RESULT_PATH = os.path.join(RESULTS_PATH, "PCA")
os.makedirs(PCA_RESULT_PATH, exist_ok=True)

""" 
MICE_COLLECTOR_df's columns/keys:

Index(['ID', 'Weight Jul 22, 2025 [g]', 'Weight Nov 14, 2025 [g]',
       'Alt ID (tail stripes)', 'Group', 'ID_MICE',
       'NOR Hab headcenter DI (%)',
       'NOR Trail 1 headcenter DI by zone crossings (%)',
       'NOR Trail 1 headcenter DI (%)',
       'NOR Trail 2 headcenter DI by zone crossings (%)',
       'NOR Trail 2 headcenter DI (%)', 'SOR Hab headcenter DI (%)',
       'SOR Trial 1 headcenter DI by zone crossings (%)',
       'SOR Trial 1 headcenter DI (%)',
       'SOR Trial 2 headcenter DI by zone crossings (%)',
       'SOR Trial 2 headcenter DI (%)', 'EPM Trial 1 headcenter DI (%)',
       'OF Trial 1 center point avg speed moving (cm/s)',
       'OF Trial 1 center point total moving time in s',
       'OF Trial 1 center point num center border crossings',
       'OF Trial 1 center point time in center (%)',
       'cFC recall Freezing Freezing Frequency',
       'cFC recall Freezing relative Cumulative Duration %'],
      dtype='object')
"""

# PCA preparation:
meta_cols = ['ID', 'Alt ID (tail stripes)', 'Group']
measurement_cols = [
    c for c in MICE_COLLECTOR_df.columns
    if c not in meta_cols
    and c not in [
                  'Weight Jul 22, 2025 [g]',
                  'Weight Nov 14, 2025 [g]',
                  'ID_MICE']]
measurement_cols_to_remove = [
        'NOR Trail 1 headcenter DI by zone crossings (%)',
        'NOR Trail 2 headcenter DI by zone crossings (%)',
        'SOR Trial 1 headcenter DI by zone crossings (%)',
        'SOR Trial 2 headcenter DI by zone crossings (%)'
        ]
# remove the above from measurement_cols:
measurement_cols = [c for c in measurement_cols if c not in measurement_cols_to_remove]
X = MICE_COLLECTOR_df[measurement_cols].copy()
groups = MICE_COLLECTOR_df['Group'].copy()
ids = MICE_COLLECTOR_df['ID'].copy()
# drop rows with missing values
X = X.dropna()
groups = groups.loc[X.index]
ids = ids.loc[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
PCA_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'], index=X.index)
PCA_df['Group'] = groups.values
PCA_df['ID'] = ids.values

explained_var = pca.explained_variance_ratio_
for i, v in enumerate(explained_var, start=1):
    print(f"PC{i}: {v*100:.2f}% variance explained")
print(f"Total (PC1–PC3): {explained_var.sum()*100:.2f}%")

# save PCA loadings:
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=measurement_cols)
loadings.to_csv(os.path.join(PCA_RESULT_PATH, "PCA_loadings.csv"))


# 3D plot:
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
# plot each group separately for correct coloring + legend
for i, g in enumerate(group_order):
    sub = PCA_df[PCA_df["Group"] == g]
    ax.scatter(
        sub["PC1"].to_numpy(dtype=float),
        sub["PC2"].to_numpy(dtype=float),
        sub["PC3"].to_numpy(dtype=float),
        s=90,
        alpha=0.9,
        label=str(g),
        c=sns_palette_use[i],
        depthshade=True)
ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({explained_var[2]*100:.1f}%)")
ax.set_title(f"PCA of MICE behavioral data\n(variance explained: {explained_var.sum()*100:.1f}%)")
ax.view_init(elev=20, azim=35)
ax.legend(title="Group", loc="upper right", frameon=True,
          bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig(os.path.join(PCA_RESULT_PATH, "PCA_3D_scatter.png"), dpi=300)
plt.close(fig)

# plot 2D projections:
for pc_x, pc_y in [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]:
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, g in enumerate(group_order):
        sub = PCA_df[PCA_df["Group"] == g]
        ax.scatter(
            sub[pc_x].to_numpy(dtype=float),
            sub[pc_y].to_numpy(dtype=float),
            s=90,
            alpha=0.9,
            label=str(g),
            c=sns_palette_use[i])
    ax.set_xlabel(f"{pc_x} ({explained_var[int(pc_x[-1])-1]*100:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({explained_var[int(pc_y[-1])-1]*100:.1f}%)")
    ax.set_title(f"PCA of MICE behavioral data: {pc_x} vs {pc_y}")
    ax.legend(title="Group", loc="upper right", frameon=True,
              bbox_to_anchor=(1.3, 1))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    # add grid:
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PCA_RESULT_PATH, f"PCA_2D_scatter_{pc_x}_vs_{pc_y}.png"), dpi=300)
    plt.close(fig)

""" 
“Unsupervised principal component analysis of combined behavioral measures did not reveal a clear separation 
between control, micro-plastic, and nano-plastic groups. All groups largely overlapped in the space spanned by 
the first three principal components (≈44 % cumulative variance explained), indicating that inter-individual 
variability dominates the multivariate structure of the data.”
"""
# %% MULTI-VARIATE ANALYSIS

# PERMANOVA for your PCA input space (X_scaled) with Group labels
# Requires: scikit-bio
#   pip install scikit-bio


def run_permanova(
    X_scaled: np.ndarray,
    groups: pd.Series | np.ndarray,
    permutations: int = 9999,
    metric: str = "euclidean",
    seed: int = 0):
    """
    PERMANOVA on a distance matrix built from X_scaled.

    Parameters
    ----------
    X_scaled : (n, p) array
        Standardized features. Your X_scaled from StandardScaler is correct.
    groups : array-like length n
        Group labels aligned to rows of X_scaled.
    permutations : int
        Number of permutations.
    metric : str
        Distance metric for pdist. Use "euclidean" for standard PERMANOVA.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    perm : pandas.Series-like (scikit-bio result object)
        Contains test statistic and p-value.
    disp : pandas.Series-like (scikit-bio result object)
        PERMDISP (homogeneity of multivariate dispersions) with p-value.
    dm : skbio DistanceMatrix
        The distance matrix used.
    """
    X = np.asarray(X_scaled, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_scaled must be a 2D array of shape (n_samples, n_features).")

    g = np.asarray(groups)
    if g.shape[0] != X.shape[0]:
        raise ValueError("groups must have the same length as number of rows in X_scaled.")

    # Drop rows with non-finite values
    finite_rows = np.all(np.isfinite(X), axis=1) & pd.notna(g)
    X = X[finite_rows]
    g = g[finite_rows]

    # Build distance matrix
    D = squareform(pdist(X, metric=metric))
    ids = [str(i) for i in range(X.shape[0])]
    dm = DistanceMatrix(D, ids=ids)

    # PERMANOVA: tests group centroid differences in distance space
    perm = permanova(
        dm,
        grouping=g,
        permutations=int(permutations),
        seed=int(seed),
    )

    # PERMDISP: checks if groups differ in dispersion (variance) in distance space
    # If PERMDISP is significant, a significant PERMANOVA can be driven by dispersion, not centroid shift.
    disp = permdisp(
        dm,
        grouping=g,
        permutations=int(permutations),
        seed=int(seed),
    )

    return perm, disp, dm


# Example usage with your variables:
# X_scaled is your StandardScaler output (numpy array)
# groups is your groups Series already aligned to X rows (you did groups = groups.loc[X.index])

perm, disp, dm = run_permanova(
    X_scaled=X_scaled,
    groups=groups,
    permutations=9999,
    metric="euclidean",
    seed=1,
)

print("\nPERMANOVA")
print(perm)  # look for pseudo-F and p-value

print("\nPERMDISP (homogeneity of dispersions)")
print(disp)  # look for p-value

# %% dbRDA ANALYSIS



# X_scaled: (n_samples, n_features) standardized features
# X: the dataframe used to generate X_scaled (after dropna), same row order
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=measurement_cols)

# Ensure consistent string indices everywhere
X_scaled_df = X_scaled_df.copy()
X_scaled_df.index = X_scaled_df.index.astype(str)

PCA_df = PCA_df.copy()
PCA_df.index = PCA_df.index.astype(str)

# Distance matrix with matching IDs
D = squareform(pdist(X_scaled_df.to_numpy(dtype=float), metric="euclidean"))
dm = DistanceMatrix(D, ids=X_scaled_df.index.tolist())

# PCoA
pcoa_res = pcoa(dm)

# Keep only positive-eigenvalue axes, robustly by count
eigvals = np.asarray(pcoa_res.eigvals, dtype=float)
n_pos = int(np.sum(eigvals > 0))
Y = pcoa_res.samples.iloc[:, :n_pos].copy()

# Constraints: Group as dummies
X_design = pd.get_dummies(PCA_df.loc[Y.index, "Group"], drop_first=False)
X_design = X_design.astype(float)

# dbRDA = RDA on PCoA coordinates
ordination = rda(Y, X_design)

# ---- Plot dbRDA1 vs dbRDA2 ----
scores = ordination.samples.copy()
scores["Group"] = PCA_df.loc[scores.index, "Group"].values

# pick the first two ordination axes robustly
ord_axes = [c for c in scores.columns if c != "Group"]
ax1, ax2 = ord_axes[0], ord_axes[1]

fig, ax = plt.subplots(figsize=(5, 5))
for g in group_order:
    sub = scores[scores["Group"] == g]
    ax.scatter(sub[ax1], sub[ax2], s=80, alpha=0.85, label=str(g))

ax.axhline(0, color="grey", lw=0.8)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel(ax1)
ax.set_ylabel(ax2)
ax.set_title("dbRDA of behavioral feature space")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PCA_RESULT_PATH, "dbRDA_scatter.png"), dpi=300)
plt.close(fig)


""" 
Micro-Plastic-Tiere zeigen ein konsistentes, multivariates Abweichen vom Kontrollverhalten
	•	Dieses Abweichen ist:
	•	nicht dominiert von einem einzelnen Test
	•	nicht stark genug für visuelle Cluster
	•	aber über viele Maße hinweg kohärent
	•	Nano-Plastic liegt intermediär bzw. teilweise orthogonal, was auf qualitativ andere Effekte hindeuten kann
"""
# %% UMAP


reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.5,
    n_components=2,
    random_state=10,)

X_umap = reducer.fit_transform(X_scaled)

UMAP_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"], index=PCA_df.index)
UMAP_df["Group"] = PCA_df["Group"].values

# sort UMAP_df according to group_order:
UMAP_df["Group"] = pd.Categorical(UMAP_df["Group"], categories=group_order, ordered=True)
UMAP_df = UMAP_df.sort_values("Group")

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(
    data=UMAP_df,
    x="UMAP1",
    y="UMAP2",
    hue="Group",
    palette=sns_palette_use,
    s=90,
    alpha=0.9,
    ax=ax)
ax.set_title("UMAP of behavioral feature space")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PCA_RESULT_PATH, "UMAP_scatter.png"), dpi=300)
plt.close()


""" 
Sachlich korrekt:
	•	„Es gibt keine diskreten Verhaltenscluster, die einzelnen Gruppen zugeordnet werden können.“
	•	„Ähnliche Verhaltensprofile treten in allen Gruppen auf.“
	•	„Gruppenunterschiede manifestieren sich nicht als lokales Clustering, sondern als subtile Verschiebungen im globalen Merkmalsraum.“

Nicht korrekt wäre:
	•	„UMAP zeigt keine Unterschiede, also gibt es keine.“
	•	„dbRDA ist falsch, weil UMAP nichts zeigt.“
"""
# %% FEATURE CONTRIBUTIONS TO dbRDA1

# RDA-Scores der Samples
scores = ordination.samples[["RDA1"]]

# Feature-Beiträge (Korrelation)
feature_contrib = {}

for col in X_scaled_df.columns:
    feature_contrib[col] = np.corrcoef(
        X_scaled_df[col],
        scores["RDA1"]
    )[0, 1]

feature_contrib_df = (
    pd.Series(feature_contrib)
    .sort_values(key=np.abs, ascending=False)
    .to_frame("Correlation_with_dbRDA1")
)

feature_contrib_df.to_csv(
    os.path.join(PCA_RESULT_PATH, "dbRDA_feature_contributions.csv")
)
# %% END