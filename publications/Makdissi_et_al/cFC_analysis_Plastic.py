# %% IMPORTS
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', 7)
pd.set_option('display.max_columns', None)
from scipy import stats
import pingouin as pg
import seaborn as sns

# remove spines right and top for better aesthetics
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
# set global font site zu 12:
plt.rcParams.update({'font.size': 12})
# %% DATA PATHS
# define data paths:
EthoVision_File_PATH = '/Users/husker/Workspace/Plastic Project/Revisions 2025/Data/cFC EV/Statistics-cFC EthoVision.xlsx'
Unblinding_Key_File  = '/Users/husker/Workspace/Plastic Project/Revisions 2025/revision_mice.xlsx'

# define group order for plotting:
group_order = ['Control', 'Micro-Plastic', 'Nano-Plastic']

# assigne sns colors to groups:
sns_palette = "Set2"
sns.set_palette(sns_palette, n_colors=len(group_order))
sns_palette_use = sns.color_palette(sns_palette, n_colors=len(group_order))

# do not change lines below (!):
# load data:
cFC_DATA_df = pd.read_excel(EthoVision_File_PATH)
MICE_df = pd.read_excel(Unblinding_Key_File)

# create a result folder:
Resultspath = os.path.join(os.path.dirname(EthoVision_File_PATH), 'Results')
os.makedirs(Resultspath, exist_ok=True)
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

cFC_DATA_df = merge_cFC_and_MICE_data(cFC_DATA_df, MICE_df, 
                                      id_col_cFC='ID', 
                                      id_col_MICE='Alt ID (tail stripes)', 
                                      id_col_MICE_exclude='Exclude cFC',
                                      exclude_marked=True)
cFC_DATA_df[["ID", "Alt ID (tail stripes)"]]

# %% RECALL ANALYSIS

""" 
cFC_DATA_df contains a column "Selection Result". Extract a subset of the data
where "Selection Result" == "Recall":
"""
cFC_RECALL_df = cFC_DATA_df[cFC_DATA_df['Selection Result'] == 'Recall']
# sort the dataframe by group_order:
cFC_RECALL_df['Group_MICE'] = pd.Categorical(cFC_RECALL_df['Group_MICE'], categories=group_order, ordered=True)
cFC_RECALL_df = cFC_RECALL_df.sort_values('Group_MICE')

""" 
relevant data columns are 
* Freezing Freezing Frequency
* Freezing Freezing Cumulative Duration s
* Arena Arena / Center-point Cumulative Duration s
"""

cFC_RECALL_df = cFC_RECALL_df[['ID', "ID_MICE", 'Group_MICE',
                               'Freezing Freezing Frequency',
                               'Freezing Freezing Cumulative Duration s',
                               'Arena Arena / Center-point Cumulative Duration s']]
cFC_RECALL_df["Freezing relative Cumulative Duration %"] = (
    cFC_RECALL_df['Freezing Freezing Cumulative Duration s'] /
    (cFC_RECALL_df['Freezing Freezing Cumulative Duration s'] +
     cFC_RECALL_df['Arena Arena / Center-point Cumulative Duration s']) * 100)

# we need a function, that receives the dataframe, the column name and the group column name
# and performs statistical analysis (using pingouin) and plotting:
def analyze_and_plot(data_df, value_col, group_col, result_path, 
                     plot_pvalues=True, bar_spacing_rel=0.05, figsize=(4,6),
                     ylim=None):
    """ 
    Perform statistical analysis and plotting for the given value column
    grouped by the group column.
    
    DEBUG:
    data_df = cFC_RECALL_df
    value_col = 'Freezing relative Cumulative Duration %'
    group_col = 'Group_MICE'
    result_path = Resultspath
    """
    data_df_use = data_df.copy()
    
    
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
    
    # in data_df's column group_col, append "N =xx" to each group name, but only once!
    group_sizes = data_df_use[group_col].value_counts().to_dict()
    def append_n(group_name):
        return f"{group_name}\n(N={group_sizes[group_name]})"
    data_df_use[group_col] = data_df_use[group_col].apply(append_n)
    
    # plotting:
    plt.figure(figsize=figsize)
    sns.boxplot(x=group_col, y=value_col, data=data_df_use, palette=sns_palette_use,
                width=0.6)
    sns.swarmplot(x=group_col, y=value_col, data=data_df_use, color=".25")
    # indicate by horizontal bars + annotation the p-values of all comparisons:
    # get unique groups:
    unique_groups = data_df[group_col].unique()
    y_max = data_df_use[value_col].max()
    y_min = data_df_use[value_col].min()
    y_range = y_max - y_min
    y_offset = y_range * 0.1  # 10% of the range
    bar_height = y_max + y_offset
    bar_spacing = y_range * bar_spacing_rel # * 0.05  # 5% of the range
    current_bar_height = bar_height
    for i in range(len(unique_groups)):
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
                    p_text = f"p = {p_value:.4f}"
                plt.text((x1 + x2) * 0.5, current_bar_height + y_offset * 0.25, p_text, 
                         ha='center', va='bottom', color='k')
                current_bar_height += bar_spacing
    # set y limit to make space for bars:
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(y_min, current_bar_height + y_offset)
    
    # turn off x label:
    plt.xlabel('')
    title = f'{value_col} during recall'
    # in the title bar, only N=35 characters have space; thus, we need to
    # split title 1st by spaces, then re-join with newlines if necessary:
    if len(title) > 35:
        words = title.split(' ')
        new_title = ''
        current_line = ''
        for word in words:
            if len(current_line) + len(word) + 1 <= 35:
                current_line += ' ' + word if current_line else word
            else:
                new_title += current_line + '\n'
                current_line = word
        new_title += current_line
        title = new_title
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'recall_{value_col}_by_{group_col}.png'))
    plt.close()
    
    # create a sub-folder for storing csv results:
    csv_result_path = os.path.join(result_path, 'CSV_Results')
    os.makedirs(csv_result_path, exist_ok=True)
    # save statistical results to csv:
    normality_results.to_csv(os.path.join(csv_result_path, f'recall_Normality_{value_col}_by_{group_col}.csv'), index=False)
    posthoc_results.to_csv(os.path.join(csv_result_path, f'recall_Posthoc_{value_col}_by_{group_col}.csv'), index=False)
    data_df.to_csv(os.path.join(csv_result_path, f'Data_recall_{value_col}_by_{group_col}.csv'), index=False)
    
# analyze and plot for each relevant column:
relevant_columns = [
    'Freezing Freezing Frequency',
    'Freezing Freezing Cumulative Duration s',
    'Freezing relative Cumulative Duration %']
ylims = {
    'Freezing Freezing Frequency': (0,115),
    'Freezing Freezing Cumulative Duration s': (0, 306),
    'Freezing relative Cumulative Duration %': (0, 55)}
for col in relevant_columns:
    analyze_and_plot(cFC_RECALL_df, col, 'Group_MICE', Resultspath,
                     plot_pvalues=True, bar_spacing_rel=0.08, 
                     figsize=(4.0,4.8), ylim=ylims.get(col))

""" WHY DO ABSOLUTE AND RELATIVE FREEZING DURATION SHOW DIFFERENT STATISTICAL RESULTS?
## Why absolute freezing duration shows significant group differences, but relative freezing duration does not

The observed discrepancy between absolute and relative freezing measures is statistically and conceptually consistent. It does not indicate a flaw in the analysis pipeline, but follows directly from the mathematical structure of the variables and from basic properties of ratio-based measures.

### Absolute and relative freezing quantify different variables

The two quantities compared are fundamentally different random variables:

The absolute freezing duration is given by
$$
T_f
$$

The relative freezing duration is defined as
$$
R_f = \frac{T_f}{T_f + T_m}
$$
where $T_m$ denotes the cumulative duration of non-freezing behavior, here operationalized as center-point time.

A statistically significant difference in $T_f$ does not imply a significant difference in $R_f$. The relative measure is a nonlinear transformation of two stochastic variables, not a rescaling of the absolute measure.

### Variance inflation caused by ratio measures

The relative freezing duration is a quotient of two random variables. Standard error propagation shows that its variance depends on both numerator and denominator:
$$
\mathrm{Var}(R_f) \approx
\left(\frac{\partial R_f}{\partial T_f}\right)^2 \mathrm{Var}(T_f)
+
\left(\frac{\partial R_f}{\partial T_m}\right)^2 \mathrm{Var}(T_m)
+
2\,\mathrm{Cov}(T_f,T_m)
$$

As a consequence:

* The variance of the relative measure is typically larger than that of the absolute freezing duration alone.
* Group differences in the numerator can be partially compensated by correlated differences in the denominator.
* Effect sizes are systematically reduced.

This behavior is evident in the empirical distributions, where relative freezing percentages show reduced separation between group medians despite visible trends.

### Correlated scaling of freezing and non-freezing time

If experimental groups differ primarily in overall activity or arousal, freezing and non-freezing times tend to change in a correlated manner. In such cases, both $T_f$ and $T_m$ shift together, leaving the ratio approximately invariant.

The data suggest exactly this pattern:

* Micro-plastic exposure reduces absolute freezing duration.
* Non-freezing behavior changes in parallel.
* The relative allocation of time between freezing and non-freezing remains comparable across groups.

Thus, the relative measure masks differences that are clearly present in absolute time.

### Loss of statistical power after normalization

Normalizing by total time compresses the dynamic range of the data. Relative freezing values are bounded between 0 and 100 percent, which reduces sensitivity to group differences. With sample sizes on the order of $N \approx 17–19$ per group, this loss of effect size is sufficient to eliminate statistical significance even when absolute measures differ robustly.

### Interpretation in the context of contextual fear conditioning

From a behavioral perspective, the two measures capture different aspects of fear expression:

* Absolute freezing duration reflects the overall strength or intensity of the conditioned fear response.
* Relative freezing duration reflects how behavioral time is distributed between freezing and non-freezing states.

The finding that only absolute freezing duration differs between groups indicates that the manipulation affects the magnitude of fear expression rather than altering behavioral strategy or time allocation.

### Conclusion

The presence of significant group differences in absolute freezing duration but not in relative freezing duration is an expected and interpretable outcome. It reflects genuine differences in overall fear expression, not a statistical artifact or analytical inconsistency. Reporting both measures is informative, provided their conceptual distinction is made explicit.
"""
# %% CONDITIONING ANALYSIS

def prepare_conditioning_long_df(
    cFC_DATA_df: pd.DataFrame,
    group_order=('Control', 'Micro-Plastic', 'Nano-Plastic'),
    subject_col='ID',
    true_ID='ID_MICE',
    group_col='Group_MICE',
    selection_col='Selection Result',
    time_levels=('After S1', 'After S2', 'After S3'),
    value_cols=(
        'Freezing Freezing Frequency',
        'Freezing Freezing Cumulative Duration s',
        'Arena Arena / Center-point Cumulative Duration s',
    ),
):
    """
    Returns a long-format dataframe for conditioning phase.

    Output columns:
      subject_col, group_col, 'Time', plus derived 'Freezing relative Cumulative Duration %'
      and the two absolute freezing columns.
    """
    df = cFC_DATA_df.copy()

    # keep only conditioning phase rows
    df = df[df[selection_col].isin(time_levels)].copy()

    # enforce group ordering
    df[group_col] = pd.Categorical(df[group_col], categories=list(group_order), ordered=True)

    # keep required cols
    keep_cols = [subject_col, true_ID, group_col, selection_col] + list(value_cols)
    df = df[keep_cols].copy()

    # compute relative freezing (%)
    df["Freezing relative Cumulative Duration %"] = (
        df['Freezing Freezing Cumulative Duration s']
        / (df['Freezing Freezing Cumulative Duration s'] + df['Arena Arena / Center-point Cumulative Duration s'])
        * 100
    )

    # rename time factor
    df = df.rename(columns={selection_col: 'Time'})
    df['Time'] = pd.Categorical(df['Time'], categories=list(time_levels), ordered=True)

    # some animals might have missing rows in a time level
    # keep only animals with complete data for all time points
    n_levels = len(time_levels)
    complete_subjects = (
        df.groupby(subject_col)['Time']
        .nunique()
        .reset_index()
        .query('Time == @n_levels')[subject_col]
        .tolist()
    )
    df = df[df[subject_col].isin(complete_subjects)].copy()

    return df


def plot_conditioning_trajectories(
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
        join=True,
        errorbar=('se', 1),
        capsize=0.08,
        markers='D',
        scale=0.9,
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
    return out


def conditioning_stats_mixed(
    df_long: pd.DataFrame,
    value_col: str,
    subject_col='ID',
    group_col='Group_MICE',
    time_col='Time',
    padjust='fdr_bh',
):
    """
    Preferred approach:
      Mixed design with within-subject factor Time and between-subject factor Group.

    Uses:
      * Mixed ANOVA if residual assumptions are reasonably met (practically, often acceptable)
      * plus optional pairwise post-hoc:
          - within each group: Time comparisons (paired)
          - at each time: Group comparisons (independent)

    Notes:
      Strict normality is hard to guarantee; for small N, mixed ANOVA is commonly used,
      but you may want to additionally report robust/nonparametric checks.
    """
    res = {}

    # Mixed ANOVA (between: group, within: time)
    aov = pg.mixed_anova(
        data=df_long,
        dv=value_col,
        within=time_col,
        between=group_col,
        subject=subject_col,
        effsize='np2',
    )
    res['mixed_anova'] = aov

    # Post-hoc A: within each group, compare time levels (paired)
    within_time = pg.pairwise_tests(
        data=df_long,
        dv=value_col,
        within=time_col,
        subject=subject_col,
        between=group_col,
        padjust=padjust,
        parametric=True,   # paired t by default; set False for Wilcoxon
    )
    res['posthoc_within_time_by_group'] = within_time

    # Post-hoc B: at each time, compare groups (independent)
    between_groups = pg.pairwise_tests(
        data=df_long,
        dv=value_col,
        between=group_col,
        within=time_col,
        subject=subject_col,
        padjust=padjust,
        parametric=True,   # t-test by default; set False for Mann-Whitney
    )
    res['posthoc_between_groups_by_time'] = between_groups

    return res


def conditioning_stats_nonparametric_checks(
    df_long: pd.DataFrame,
    value_col: str,
    subject_col='ID',
    group_col='Group_MICE',
    time_col='Time',
    padjust='fdr_bh',
):
    """
    Practical nonparametric sanity checks:
      * Within each group: Friedman test across time (paired, k=3)
      * At each time: Kruskal-Wallis across groups
      * Post-hoc:
          - within-group: paired Wilcoxon between time points, FDR
          - between-group: pairwise Mann-Whitney at each time, FDR

    This does not test the full Group×Time interaction in one unified nonparametric model,
    but provides robust complementary evidence.
    """
    res = {}

    # Friedman within each group (paired across time)
    friedman_rows = []
    for g in df_long[group_col].cat.categories:
        dfg = df_long[df_long[group_col] == g].copy()
        if dfg.empty:
            continue
        wide = dfg.pivot(index=subject_col, columns=time_col, values=value_col)
        if wide.shape[0] < 3:
            continue
        # scipy friedman needs arrays; pingouin has friedman for long
        fr = pg.friedman(data=dfg, dv=value_col, within=time_col, subject=subject_col)
        fr['Group'] = g
        friedman_rows.append(fr)
    res['friedman_by_group'] = pd.concat(friedman_rows, ignore_index=True) if friedman_rows else pd.DataFrame()

    # Kruskal at each time (between groups)
    kw_rows = []
    for t in df_long[time_col].cat.categories:
        dft = df_long[df_long[time_col] == t].copy()
        if dft.empty:
            continue
        kw = pg.kruskal(data=dft, dv=value_col, between=group_col)
        kw['Time'] = t
        kw_rows.append(kw)
    res['kruskal_by_time'] = pd.concat(kw_rows, ignore_index=True) if kw_rows else pd.DataFrame()

    # Post-hoc within each group: Wilcoxon paired across time
    within_wilcoxon = pg.pairwise_tests(
        data=df_long,
        dv=value_col,
        within=time_col,
        subject=subject_col,
        between=group_col,
        padjust=padjust,
        parametric=False,  # Wilcoxon
    )
    res['posthoc_within_time_by_group_wilcoxon'] = within_wilcoxon

    # Post-hoc between groups at each time: Mann-Whitney
    between_mwu = pg.pairwise_tests(
        data=df_long,
        dv=value_col,
        between=group_col,
        within=time_col,
        subject=subject_col,
        padjust=padjust,
        parametric=False,  # Mann-Whitney
    )
    res['posthoc_between_groups_by_time_mwu'] = between_mwu

    return res


conditioning_time_levels = ('After S1', 'After S2', 'After S3')

cFC_COND_long = prepare_conditioning_long_df(
    cFC_DATA_df=cFC_DATA_df,
    group_order=group_order,
    subject_col='ID',
    true_ID='ID_MICE',
    group_col='Group_MICE',
    selection_col='Selection Result',
    time_levels=conditioning_time_levels)

relevant_columns = [
    'Freezing Freezing Frequency',
    'Freezing Freezing Cumulative Duration s',
    'Freezing relative Cumulative Duration %',
]

ylims_cond = {
    'Freezing Freezing Frequency': (0, 35),
    'Freezing Freezing Cumulative Duration s': (0, 60),
    'Freezing relative Cumulative Duration %': (0, 55),
}

# create plots + stats for each metric:
for col in relevant_columns:
    
    # plot:
    plot_conditioning_trajectories(
        figsize=(3.8, 4.8),
        df_long=cFC_COND_long,
        value_col=col,
        subject_col='ID',
        group_col='Group_MICE',
        time_col='Time',
        result_path=Resultspath,
        ylim=ylims_cond.get(col),
        alpha_lines=0.25)

    # stats: mixed ANOVA + posthoc:
    stats_out = conditioning_stats_mixed(
        df_long=cFC_COND_long,
        value_col=col,
        subject_col='ID',
        group_col='Group_MICE',
        time_col='Time',
        padjust='fdr_bh')

    print("\n" + "=" * 90)
    print(f"CONDITIONING: mixed ANOVA for {col}")
    print(stats_out['mixed_anova'])
    
    def print_pairwise_table(df, preferred_cols):
        """Print only columns that exist in df, in a sensible order."""
        cols = [c for c in preferred_cols if c in df.columns]
        missing = [c for c in preferred_cols if c not in df.columns]
        print("Available columns:", list(df.columns))
        if missing:
            print("Missing columns (version dependent):", missing)
        print(df[cols] if cols else df)

    print("\nPost-hoc: time comparisons within each group (paired, FDR)")
    print_pairwise_table(
        stats_out['posthoc_within_time_by_group'],
        preferred_cols=['Contrast', 'A', 'B', 'Paired', 'parametric', 'p-unc', 'p-corr', 'p-adjust', 'hedges', 'cohen'])

    print("\nPost-hoc: group comparisons at each time point (independent, FDR)")
    print_pairwise_table(
        stats_out['posthoc_between_groups_by_time'],
        preferred_cols=['Contrast', 'A', 'B', 'Paired', 'parametric', 'p-unc', 'p-corr', 'p-adjust', 'hedges', 'cohen'])

    # optional robust checks:
    robust = conditioning_stats_nonparametric_checks(
        df_long=cFC_COND_long,
        value_col=col,
        subject_col='ID',
        group_col='Group_MICE',
        time_col='Time',
        padjust='fdr_bh')

    print("\nRobust checks: Friedman within each group")
    print(robust['friedman_by_group'])

    print("\nRobust checks: Kruskal-Wallis across groups at each time")
    print(robust['kruskal_by_time'])
    
    
    # save csv results:
    csv_folder = os.path.join(Resultspath, 'CSV_Results')
    os.makedirs(csv_folder, exist_ok=True)
    stats_out['mixed_anova'].to_csv(
        os.path.join(csv_folder, f'conditioning_mixed_anova_{col}.csv'), index=False)
    stats_out['posthoc_within_time_by_group'].to_csv(
        os.path.join(csv_folder, f'conditioning_posthoc_within_time_by_group_{col}.csv'), index=False)
    stats_out['posthoc_between_groups_by_time'].to_csv(
        os.path.join(csv_folder, f'conditioning_posthoc_between_groups_by_time_{col}.csv'), index=False)
    robust['friedman_by_group'].to_csv(
        os.path.join(csv_folder, f'conditioning_robust_friedman_by_group_{col}.csv'), index=False)
    robust['kruskal_by_time'].to_csv(
        os.path.join(csv_folder, f'conditioning_robust_kruskal_by_time_{col}.csv'), index=False)
    cFC_COND_long.to_csv(
        os.path.join(csv_folder, f'Data_conditioning_long_data_{col}.csv'), index=False)
    


#cFC_COND_long[cFC_COND_long["Group_MICE"]=="Nano-Plastic"]
# %% CONCATENATE RESULTS
""" 
We now take MICE_df and append to it the recall freezing results
"""
MICE_collector_df = cFC_RECALL_df.copy()
# drop Freezing Freezing Cumulative Duration s and Arena Arena / Center-point Cumulative Duration s
MICE_collector_df = MICE_collector_df.drop(columns=[
    'Freezing Freezing Cumulative Duration s',
    'Arena Arena / Center-point Cumulative Duration s'])
# append columns ET	Weight Jul 22, 2025 [g]	Weight Nov 14, 2025 [g]	Behavior Batch	Alt ID (tail stripes)	S from MICE_df to MICE_collector_df
MICE_cols_to_add = ['ET',
                     'Weight Jul 22, 2025 [g]',
                     'Weight Nov 14, 2025 [g]',
                     'Behavior Batch', "S"]
for col in MICE_cols_to_add:
    # col = MICE_cols_to_add[0]
    MICE_collector_df[col] = MICE_df.set_index('ID')[col].reindex(MICE_collector_df['ID_MICE']).values
#MICE_collector_df = MICE_collector_df.drop(columns=['ID_MICE'])
# re-arrange columns to have ID first, then ET, Weight Jul, Weight Nov, Behavior Batch, S, Group_MICE, then the rest
cols_order = ['ID_MICE','ID', 'ET', 'S', 'Group_MICE', 'Weight Jul 22, 2025 [g]', 'Weight Nov 14, 2025 [g]', 'Behavior Batch', 
              'Freezing Freezing Frequency',
              'Freezing relative Cumulative Duration %']
MICE_collector_df = MICE_collector_df[cols_order]

# prepend "cFC recall" to the relevant columns:
cols_to_rename = [
    'Freezing Freezing Frequency',
    'Freezing relative Cumulative Duration %']
MICE_collector_df = MICE_collector_df.rename(columns={col: f'cFC recall {col}' for col in cols_to_rename})


# now, also append the conditioning freezing data:
# pivot cFC_COND_long to wide format for easier merging:
cFC_COND_wide = cFC_COND_long.pivot_table(
    index=['ID', 'ID_MICE', 'Group_MICE'],
    columns='Time',
    observed=False,
    values=[
        'Freezing Freezing Frequency',
        'Freezing Freezing Cumulative Duration s',
        'Arena Arena / Center-point Cumulative Duration s',
        'Freezing relative Cumulative Duration %']
    ).reset_index()
# flatten multiindex columns:
cFC_COND_wide.columns = [f'{val}_{time}' if time else val for val, time in cFC_COND_wide.columns]
# drop Arena Arena / Center-point Cumulative Duration s_After S... and Freezing Freezing Cumulative Duration s_After S...:
cols_to_drop = [
    col for col in cFC_COND_wide.columns 
    if col.startswith('Arena Arena / Center-point Cumulative Duration s_') or
       col.startswith('Freezing Freezing Cumulative Duration s_')]
cFC_COND_wide = cFC_COND_wide.drop(columns=cols_to_drop)
# rename all columns that start with "Freezing Freezing..." to "cFC conditioning Freezing...":
cols_to_rename_cond = [
    col for col in cFC_COND_wide.columns 
    if col.startswith('Freezing Freezing Frequency_') or
       col.startswith('Freezing relative Cumulative Duration %_')]
cFC_COND_wide = cFC_COND_wide.rename(columns={col: f'cFC conditioning {col}' for col in cols_to_rename_cond})
# merge into MICE_collector_df:
MICE_collector_df = pd.merge(
    MICE_collector_df,
    cFC_COND_wide,
    on=['ID', 'ID_MICE', 'Group_MICE'],
    how='left')


# save MICE_collector_df to csv:
# Collector_df_Path  should be two level above Resultspath, named MICE
Collector_df_Path = os.path.dirname(os.path.dirname(Resultspath))
MICE_collector_df.to_csv(os.path.join(Collector_df_Path, 'MICE_collector_recall_freezing_data.csv'), index=False)


# %% END