"""
Analysis for Morris Water Maze EthoVision data.

author:  Fabrizio Musacchio (fabriziomusacchio.com)
date:    Nov 27, 2022
"""
# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import os as os
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
# turn off warnings:
import warnings
warnings.filterwarnings("ignore")
# %% PATHS
ROOT_path = "/Users/husker/Workspace/Henrike DLC/MWM/"
DATA_path = ROOT_path
RESULTS_path = "/Users/husker/Workspace/Henrike DLC/Collective Results/MWM/"
RESULTS_excel_path = RESULTS_path+"excel/"
# check whether RESULTS folder exists, if not create:
if not os.path.exists(RESULTS_excel_path):
    os.makedirs(RESULTS_excel_path)
if not os.path.exists(RESULTS_path):
    os.makedirs(RESULTS_path)
# check whether exists, if not create:
if not os.path.exists(RESULTS_path):
    os.makedirs(RESULTS_path)
mice_infotable = "/Users/husker/Workspace/Henrike DLC/Behavior_Groups_with_IDs.xlsx"
# %% SET PARAMETERS AND VARIABLES
# define colors:
colors = ["#71b5a0", "#e99675"]
Group_Styles2 = pd.DataFrame(columns=["Adrb2-wt", "Adrb2-flox"],
                            index=['Color', 'Linestyle', 'Markerstyle', 'Name'])
Group_Styles2['Adrb2-wt']['Color']  = "#71b5a0"
Group_Styles2['Adrb2-flox']['Color'] = "#e99675" # ff1c00 57b4e9
Group_Styles2['Adrb2-wt']['Linestyle']  = '-'
Group_Styles2['Adrb2-flox']['Linestyle'] = '-'
Group_Styles2['Adrb2-wt']['Markerstyle']  = 'o'
Group_Styles2['Adrb2-flox']['Markerstyle'] = 'o'
Group_Styles2['Adrb2-wt']['Name']  = "Adrb2-wt"
Group_Styles2['Adrb2-flox']['Name'] = "Adrb2-flox"


Group_Styles = pd.DataFrame(columns=[0, 1],
                            index=['Color', 'Linestyles'])
Group_Styles[0]['Color']  = Group_Styles2['Adrb2-wt']['Color']
Group_Styles[1]['Color']  = Group_Styles2['Adrb2-flox']['Color']
Group_Styles[0]['Linestyles']  = "-"
Group_Styles[1]['Linestyles']  = "-"
colors = ["#56B4E9", "#E69F00"]

Group_Pairs_Markers = ["o", "o", "d", "s", "h", "v", "^", "2", "1", "4", "*"]
Group_Pairs_Linestyles = ["-", "-", "--", "s", "h", "v", "^", "2", "1", "4", "*"]
group_order = ["Adrb2-wt", "Adrb2-flox"]
# %% FUNCTIONS
palette_N2= sns.color_palette('Set2', 2)  # husl  deep, muted, pastel, bright, dark,  colorblind

def check_pval(pval):
    if pval <  0.0001: pval_string = "$\,\\ast\!\!\!\\ast\!\!\!\\ast\!\!\!\!\\ast$" #****"
    elif pval <  0.001: pval_string = "$\,\\ast\!\!\!\\ast\!\!\!\\ast$" #***
    elif pval <  0.01: pval_string = "$\,\\ast\!\!\!\\ast$" #**
    elif pval <  0.05: pval_string = "$\\ast$" #*
    else: pval_string = "n.s."
    return pval_string

def independent_two_sample_test(A, B):
    if pg.normality(A)['normal'].bool() and pg.normality(B)[ 'normal'].bool():
        normality=True
    else:
        normality = False

    if normality:
        pval = pg.ttest(A, B, paired=False)['p-val'][0]
        test = 'unpaired ttest'
    else:
        pval = stats.ks_2samp(A, B)[1]
        #pval = pg.mwu(A, B, tail='two-sided')['p-val'][0]
        test = 'kolmogorov-smirnov test'
    if pval<0.05:
        sigDifference = True
    else:
        sigDifference = False
    cohen = pg.compute_effsize(A, B, paired=False, eftype='cohen')
    power = pg.power_ttest2n(nx=len(A), ny=len(B), d=cohen, alpha=0.05)
    cohen_estimate_for_power80 = pg.power_ttest2n(nx=len(A), ny=len(B), power=0.8, alpha=0.05)

    significance_results = pd.DataFrame(columns=['pval', 'difference', 'normal', 'cohen d', 'min. d',
                                                 'power','test type'], index=['entry'])


    significance_results['pval'] = pval
    significance_results['difference'] = sigDifference
    significance_results['normal'] = normality
    significance_results['cohen d'] = cohen
    significance_results['min. d'] = cohen_estimate_for_power80
    significance_results['power'] = power
    significance_results['test type'] = test
    # return pval, sigDifference, normality, cohen, test
    return significance_results

def plot_2_samples(A, B, fignum=1, title='', Alabel='A', Blabel='B', ylim='', yticks=np.arange(0),
                   ylabel='', figaspect=(4,4), plotsavepath='', significance=False, statsOffset=0,
                   boxplot=True,barOffset=1.0, A_significance=False, B_significance=False,
                   clevel=False):
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    ## %%
    fig.clf()
    #sns.set(style="ticks")
    plt.grid(color='gainsboro', linestyle='-', linewidth=0.25, zorder=1)

    # Arange Input-Data into pandas construct:
    data_DF_dump_A = pd.DataFrame(data=[A]).T
    data_DF_dump_A.columns = ['data']
    data_DF_dump_A['Group'] = 1  # Alabel
    data_DF_dump_B = pd.DataFrame(data=[B]).T
    data_DF_dump_B.columns = ['data']
    data_DF_dump_B['Group'] = 2  # Blabel
    data_DF_dump = data_DF_dump_A.append(data_DF_dump_B, ignore_index=True)

    if not type(clevel) == bool:
        plt.plot([-0.5,2], [clevel, clevel], '--', c="silver", lw=0.75)

    # Plot Violins:
    pv = plt.violinplot(A, positions=[0], showmeans=False, showmedians=True, bw_method='scott') #0.450
    palette_iterator=0
    for pc in pv['bodies']:
        pc.set_facecolor(palette_N2[palette_iterator])
        pc.set_edgecolor(palette_N2[palette_iterator])
        pc.set_linewidth(0)
        pc.set_alpha(0.20)
    for partname in ('cbars', 'cmins', 'cmaxes',  'cmedians'):
        v = pv[partname]
        # v.set_edgecolor(palette(iterator))
        v.set_edgecolor(palette_N2[palette_iterator])
        v.set_linewidth(1.75)
        v.set_alpha(0)
    pv['cmedians'].set_alpha(1)
    pv['cmedians'].set_linewidth(2)

    pv = plt.violinplot(B, positions=[1], showmeans=False, showmedians=True, bw_method='scott')  # 0.450
    palette_iterator = 1
    for pc in pv['bodies']:
        pc.set_facecolor(palette_N2[palette_iterator])
        pc.set_edgecolor(palette_N2[palette_iterator])
        pc.set_linewidth(0)
        pc.set_alpha(0.20)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        v = pv[partname]
        # v.set_edgecolor(palette(iterator))
        v.set_edgecolor(palette_N2[palette_iterator])
        v.set_linewidth(1.75)
        v.set_alpha(0)
    pv['cmedians'].set_alpha(1)
    pv['cmedians'].set_linewidth(2)

    # Plot Boxplots:
    if boxplot:
        sns.boxplot(x='Group', y='data', data=data_DF_dump, width=0.095, color='white',
                boxprops={'color': 'silver'}, showmeans=False, medianprops=dict(color="w", linewidth=2, alpha=0 ))

    # Plot data points:
    ax2 = sns.swarmplot(x='Group',y='data', data=data_DF_dump, size=8, alpha=0.5,c="grey")
    if ylabel:
        ax2.set(ylabel=ylabel, xlabel='')
    else:
        ax2.set(xlabel='')

    # Mark medians:
    plt.plot(0, np.median(A), '.', markersize=13, linewidth=30, markeredgewidth=1.5, color=palette_N2[palette_iterator-1], markerfacecolor='white', zorder=3)
    plt.plot(1, np.median(B), '.', markersize=13, linewidth=30, markeredgewidth=1.5, color=palette_N2[palette_iterator], markerfacecolor='white', zorder=3)


    if yticks.any():
        plt.yticks(yticks)
    if ylim:
        plt.ylim(ylim)
    # Significance statistics:
    if not type(significance)==bool:
        y = ylim[1] - statsOffset
        x1 = 0
        x2 = 1
        if significance['difference'][0]:
            plt.plot([x1, x2], [y, y], lw=1.5, color="k")
            plt.text(x1 + (x2 - x1) / 2, y, '$d$=' + str(significance['cohen d'][0].round(2)),
                     ha='center', va='bottom', fontsize=9, fontweight="normal")
            # plt.text(-0.45,  ylim[1]-statsOffset,
            #          'significant difference\n'+
            #          'cohen\'s d= ' + str(significance['cohen d'][0].round(2)) +
            #          ', p= ' + str(significance['pval'][0].round(4)) + '\n'+
            #          significance['test type'][0],
            #          ha='left', va='top', fontsize=10, color='dimgrey')
        else:
            plt.plot([x1, x2], [y, y], lw=1.5, color="k")
            plt.text(x1 + (x2 - x1) / 2, y, 'n.s.',
                     ha='center', va='bottom', fontsize=9,fontweight="normal")
            # plt.text(-0.45, ylim[1]-statsOffset,
            #          'no significant difference\n'+
            #          'cohen\'s d= ' + str(significance['cohen d'][0].round(2)) +
            #          ', p= ' + str(significance['pval'][0].round(4)) + '\n'+
            #          significance['test type'][0],
            #          ha='left', va='top', fontsize=10, color='dimgrey')

    if not type(A_significance) == bool:
        y = ylim[1] - statsOffset-barOffset
        if A_significance['p-val'][0]<0.05:
            plt.text(0, y, '$d$=' + str(A_significance['cohen-d'][0].round(2)),
                     ha='center', va='bottom', fontsize=9, fontweight="normal")
        else:
            plt.text(0, y, 'n.s.',
                     ha='center', va='bottom', fontsize=9, fontweight="normal")
    if not type(B_significance) == bool:
        y = ylim[1] - statsOffset-barOffset
        if B_significance['p-val'][0]<0.05:
            plt.text(1, y, '$d$=' + str(B_significance['cohen-d'][0].round(2)),
                     ha='center', va='bottom', fontsize=9, fontweight="normal")
        else:
            plt.text(1, y, 'n.s.',
                     ha='center', va='bottom', fontsize=9, fontweight="normal")


    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.xlim(-0.5, 1.5)
    plt.xticks((0,1), [Alabel, Blabel])
    plt.title(title, fontweight='bold')

    plt.tight_layout()
    #aa = plt.get_current_fig_manager()
    #aa.window.wm_geometry('+1000+0')
    #plt.show()
    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
            # plt.savefig(os.path.join(plotsavepath, title + '.png'), dpi=200)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
            # plt.savefig(os.path.join(plotsavepath, 'plot' + '.png'), dpi=200)

def plot_2_samples_pub(A, B, fignum=1, title='', plotname='', Alabel='A', Blabel='B', ylim='',
                       yticks=np.arange(0), ylabel='', figaspect=(4,4), plotsavepath='',
                       significance=(), statsOffset=0, boxplot=False, barOffset=16.0,
                       palette_N2=palette_N2, fontsize_axes=10, fontsize_labels=10,
                       boxborder_width=2, colors=["k","k"], plot_median=False,
                       show_title=False, A_significance=False, B_significance=False,
                       clevel=False, AB_statsOffset=3):
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    ## %%
    fig.clf()
    sns.reset_orig()
    
    # change global matplotliv font to arial:
    plt.rcParams['font.family'] = 'Arial'

    # Arange Input-Data into pandas construct:
    data_DF_dump_A = pd.DataFrame(data=[A]).T
    data_DF_dump_A.columns = ['data']
    data_DF_dump_A['Group'] = 1  # Alabel
    data_DF_dump_B = pd.DataFrame(data=[B]).T
    data_DF_dump_B.columns = ['data']
    data_DF_dump_B['Group'] = 2  # Blabel
    data_DF_dump = data_DF_dump_A.append(data_DF_dump_B, ignore_index=True)

    if not type(clevel) == bool:
        plt.plot([-0.5,2], [clevel, clevel], '--', c="silver", lw=0.75)

    # Plot Violins:
    pv = plt.violinplot(A, positions=[0], showmeans=False, showmedians=False, bw_method='scott') #0.450
    palette_iterator=0
    for pc in pv['bodies']:
        pc.set_facecolor(colors[0]) #"black"
        pc.set_edgecolor(colors[0]) #"black"
        pc.set_linewidth(0)
        pc.set_alpha(0.1) #0.1
    for partname in ('cbars', 'cmins', 'cmaxes'): #,  'cmedians'
        v = pv[partname]
        # v.set_edgecolor(palette(iterator))
        v.set_edgecolor(palette_N2[palette_iterator])
        v.set_linewidth(1.75)
        v.set_alpha(0)
    # pv['cmedians'].set_alpha(1)
    # pv['cmedians'].set_linewidth(2)

    pv = plt.violinplot(B, positions=[1], showmeans=False, showmedians=False, bw_method='scott')  # 0.450
    palette_iterator = 1
    for pc in pv['bodies']:
        pc.set_facecolor(colors[1])  # "black"
        pc.set_edgecolor(colors[1])  # "black"
        pc.set_linewidth(0)
        pc.set_alpha(0.1)  # 0.1
    for partname in ('cbars', 'cmins', 'cmaxes'):  #, 'cmedians'
        v = pv[partname]
        # v.set_edgecolor(palette(iterator))
        v.set_edgecolor(palette_N2[palette_iterator])
        v.set_linewidth(1.75)
        v.set_alpha(0)
    # pv['cmedians'].set_alpha(1)
    # pv['cmedians'].set_linewidth(2)

    # Plot Boxplots:
    if boxplot:
        sns.boxplot(x='Group', y='data', data=data_DF_dump, width=0.095, color='white',
                boxprops={'color': 'silver'}, showmeans=False, medianprops=dict(color="w", linewidth=2, alpha=0 ))

    # Create an array with the colors you want to use
    #colors_dots = ["black", "#d9d9d9"]
    colors_dots = ["black", "white"]
    # Set your custom color palette
    palette =  sns.color_palette(colors_dots) # sns.set_palette(sns.color_palette(colors))

    # colors_N = ["k"]*9
    # tmp = ["m"]*9
    # colors_N.append(tmp[:])
    # colors_N = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm']
    # Plot data points:
    ax2 = sns.swarmplot(x='Group', y='data', data=data_DF_dump, size=7,
                        linewidth=2, alpha=1, hue="Group", edgecolor="black",
                        palette=palette)
    ## %%
    if ylabel:
        ax2.set(ylabel=ylabel, xlabel='')
        plt.ylabel(ylabel=ylabel, fontweight="normal", fontsize=fontsize_labels)
        #plt.ylabel(ylabel=ylabel)
    else:
        ax2.set(xlabel='')
    ax2.legend_.remove()

    if not plot_median:
        calc_avrg = np.mean
    else:
        calc_avrg = np.median

    # Mark medians:
    # plt.plot([0.8-1, 1.2-1], [np.median(A), np.median(A)], '-',
    #          linewidth=2, color="white", zorder=3, alpha=0.95)
    plt.errorbar(0, calc_avrg(A), yerr=np.std(A), linewidth=2, color="black", zorder=3, alpha=1,
                 capsize=8, capthick=2.0)
    # plt.plot(0, np.median(A), 'o', markersize=15, linewidth=30, markeredgewidth=2.5, color="black",
    #          markerfacecolor='black', zorder=3)
    plt.plot([0 - 0.2, 0 + 0.2], [calc_avrg(A), calc_avrg(A)], '-', linewidth=2.5, color="black",
             zorder=3)
    # plt.plot([0.8, 1.2], [np.median(B), np.median(B)], '-',
    #          linewidth=2, color="white", zorder=3, alpha=0.95)
    plt.errorbar(1, calc_avrg(B), yerr=np.std(B), linewidth=2, color="black", zorder=3, alpha=1,
                 capsize=8, capthick=2.0)
    # plt.plot(1, np.median(B), 'o', markersize=15, linewidth=30, markeredgewidth=2.5, color="black",
    #          markerfacecolor='white', zorder=3)
    plt.plot([1-0.2, 1+0.2], [calc_avrg(B), calc_avrg(B)], '-', linewidth=2.5, color="black",
             zorder=3)


    if yticks.any():
        # plt.yticks(yticks, fontweight="bold")
        plt.yticks(yticks)
    if ylim:
        plt.ylim(ylim)
    # Significance statistics:
    if not significance.empty:
        #print('hallo')
        if significance['difference'][0]:

            if significance['pval'][0] <  0.0001:
                pval_string = "****"
            elif significance['pval'][0] <  0.001:
                pval_string = "***"
            elif significance['pval'][0] <  0.01:
                pval_string = "**"
            elif significance['pval'][0] <  0.05:
                pval_string = "*"

            x1 = 0
            x2 = 1
            y, h = ylim[1] - statsOffset , barOffset / 2
            plt.plot([x1, x2], [y + h, y + h], lw=2, color="k")
            plt.text(0.5, y + h + barOffset, pval_string,
                     ha='center', va='bottom', fontsize=fontsize_labels,
                     fontweight="normal")
            # plt.text(0.5, y + h - barOffset, "d= " + str(significance['cohen d'][0].round(2)),
            #          ha='center', va='top', fontsize=10,
            #          fontweight="bold")
        else:
            x1 = 0
            x2 = 1

            y, h = ylim[1] - statsOffset, barOffset / 2
            plt.plot([x1, x2], [y + h, y + h], lw=2, color="k")
            plt.text(0.5, y + h + barOffset, "n.s.", ha='center', va='bottom', 
                     fontsize=fontsize_labels, fontweight="normal")

    if not type(A_significance) == bool:
        y = ylim[1] - AB_statsOffset
        if A_significance['p-val'][0]<0.05:
            if A_significance['p-val'][0] <  0.0001:
                pval_string = "****"
            elif A_significance['p-val'][0] <  0.001:
                pval_string = "***"
            elif A_significance['p-val'][0] <  0.01:
                pval_string = "**"
            elif A_significance['p-val'][0] <  0.05:
                pval_string = "*"
            
            plt.text(0, y, pval_string,
                     ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
        else:
            plt.text(0, y, 'n.s.',
                     ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
    if not type(B_significance) == bool:
        y = ylim[1] - AB_statsOffset
        if B_significance['p-val'][0]<0.05:
            if B_significance['p-val'][0] <  0.0001:
                pval_string = "****"
            elif B_significance['p-val'][0] <  0.001:
                pval_string = "***"
            elif B_significance['p-val'][0] <  0.01:
                pval_string = "**"
            elif B_significance['p-val'][0] <  0.05:
                pval_string = "*"

            plt.text(1, y, pval_string,
                     ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
        else:
            plt.text(1, y, 'n.s.',
                     ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")

    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_linewidth(boxborder_width)
    axis.spines['left'].set_linewidth(boxborder_width)
    #axis.spines['bottom'].set_visible(False)
    #axis.spines['left'].set_visible(False)

    plt.xlim(-0.5, 1.5)
    plt.xticks((0,1), [Alabel, Blabel], fontweight="normal", fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_axes, fontweight="normal")
    if show_title:
        plt.title(title, fontweight='bold', fontsize=fontsize_axes)
    ax = plt.gca()
    ax.xaxis.set_tick_params(width=boxborder_width, length=8)
    ax.yaxis.set_tick_params(width=boxborder_width, length=8)

    plt.tight_layout()
    #aa = plt.get_current_fig_manager()
    #aa.window.wm_geometry('+1000+0')
    #plt.show()
    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
        elif plotname:
            plt.savefig(os.path.join(plotsavepath, plotname+'.pdf'), dpi=120)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)

# new plot_2_samples_pub:
def plot_N_samples_pub(df_in, variable, title='', plotname='', groups=[], ylim='',
                       yticks=np.arange(0), ylabel='', figaspect=(4,4), plotsavepath='', excelpath='',
                       show_stats=(), show_stats_ns=True, statsOffset=0, stats_text_correct=1, barOffset=16.0,
                       multicomp=True,  Group_Styles=Group_Styles,
                       fontsize_axes=10, fontsize_labels=10, fontweight="normal",
                       boxborder_width=1, plot_median=False, fignum=1, 
                       show_title=False, xlabel_rot = 0, violin_width=0.5, swarm_dots_alpha=1,
                       swarm_dots_size=35, clevel=False, clevel_stats_offset=3, plot_clevel_stats=False,
                       swarm_tol=0.04, swarm_offset = 0.05, detect_outliers=False):
    
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    
    fig.clf()
    
    # change matplotlib font to arial:
    plt.rcParams['font.family'] = 'Arial'

    if not type(clevel) == bool:
        plt.plot([-0.5,len(groups)+0.5], [clevel, clevel], '--', c="gray", lw=1)
    if not plot_median:
        calc_avrg = np.mean
    else:
        calc_avrg = np.median
    
    # plot Violins:
    markers = [Group_Styles[group]["Markerstyle"] for group in groups]
    colors_dots = [Group_Styles[group]["Color"] for group in groups]
    group_names = [Group_Styles[group]["Name"] for group in groups]
    palette =  sns.color_palette(colors_dots) # sns.set_palette(sns.color_palette(colors))
    #df_melted = df_in.dropna().melt(var_name="Group")
    # df_in is already melted; we need a wide format for the violinplot using pivot:
    df_wide = df_in.pivot(columns="Group", values=variable)
    
    ax = plt.gca()
    # violin plot:
    for group_i, group in enumerate(groups):
        pv = plt.violinplot(df_wide[group].dropna(), positions=[group_i], 
                            showmeans=False, showmedians=False, bw_method='scott', widths=violin_width)
        for pc in pv['bodies']:
            pc.set_facecolor(Group_Styles[group]["Color"]) #"black"
            pc.set_edgecolor(Group_Styles[group]["Color"]) #"black"
            pc.set_linewidth(0)
            pc.set_alpha(0.3) #0.1
        for partname in ('cbars', 'cmins', 'cmaxes'): #,  'cmedians'
            v = pv[partname]
            # v.set_edgecolor(palette(iterator))
            v.set_edgecolor("black")
            v.set_linewidth(1.75)
            v.set_alpha(0)
    # swarmplot (customized):
    for group_i, group in enumerate(groups):
        group_data = df_in[df_in["Group"] == group][variable]
        # sort group_data ascending:
        group_data = group_data.sort_values()
        #reset index:
        group_data.reset_index(drop=True, inplace=True)
        
        # create x-values for swarmplot (shift x-value if the corresponding y-value are slightly identical)):
        x_values = np.zeros(len(group_data))+group_i
        sign = 1
        for i, value in enumerate(group_data):
            if i>0:
                if value >= group_data.values[i-1]-swarm_tol*group_data.values[i-1]*np.sign(group_data.values[i-1]) and value <= group_data.values[i-1]+swarm_tol*group_data.values[i-1]*np.sign(group_data.values[i-1]):
                    x_values[i] = x_values[i-1]+swarm_offset*sign
                    # check whether previous x-value was already shifted:
                    if x_values[i] == x_values[i-1]:
                        x_values[i] = x_values[i-1]+2*swarm_offset*sign
                    
                    # change sign for next iteration:
                    sign *=-1
        plt.scatter(x_values, group_data, marker=markers[group_i], color=palette[group_i], 
                    edgecolor=Group_Styles[group]["Color"], s=swarm_dots_size, alpha=swarm_dots_alpha, linewidths=0)
        
        # plot medians
        plt.errorbar(group_i, calc_avrg(group_data), yerr=np.std(group_data), 
                     linewidth=2, color=Group_Styles[group]["Color"], zorder=3, alpha=1,
                 capsize=8, capthick=2.0)
        plt.plot([group_i - 0.2, group_i + 0.2], [calc_avrg(group_data), calc_avrg(group_data)], 
                 '-', linewidth=2.5, color=Group_Styles[group]["Color"],zorder=3)

        # detect outliers:
        if detect_outliers:
            # detect outliers:
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = group_data[(group_data < (Q1 - 1.5 * IQR)) | (group_data > (Q3 + 1.5 * IQR))]
            #outliers = group_data[group_data > 3 * pg.mad(group_data)]
            for outlier in outliers:
                # outlier = outliers.iloc[0]
                # find the index of the outlier in the group_data:
                outlier_i = group_data[group_data == outlier].index[0]
                
                # mark outliers with the same color as the group as edgecolor, but white facecolor:
                plt.scatter(x_values[outlier_i], group_data[outlier_i], marker=markers[group_i],
                            color=Group_Styles[group]["Color"], facecolor="white", s=swarm_dots_size, 
                            alpha=1, linewidths=1)
        

    if ylabel:
        ax.set(ylabel=ylabel, xlabel='')
        plt.ylabel(ylabel=ylabel, fontweight=fontweight, fontsize=fontsize_labels)
        #plt.ylabel(ylabel=ylabel)
    else:
        ax.set(xlabel='')
    #ax.legend_.remove()

    if yticks.any():
        # plt.yticks(yticks, fontweight="bold")
        plt.yticks(yticks)
    if ylim:
        plt.ylim(ylim)
       
    # statistics: 
    if show_stats:
        # perform normality test and calculate and indicate whether group's 
        # average is above or below chance level:
        y = ylim[1] - clevel_stats_offset
        normality = []
        for group_i, group in enumerate(groups):
            group_data = df_in[df_in["Group"] == group][variable]
            # test for normality:
            parametric = pg.normality(group_data)['normal'][0]
            normality.append(parametric)
            if not type(clevel) == bool and plot_clevel_stats:
                if parametric:
                    # test for significant difference from chance level:
                    pval = pg.ttest(group_data, clevel, paired=False)['p-val'][0]
                else:
                    #pval = pg.mwu(group_data, clevel)['p-val'][0]
                    x = group_data.dropna().to_numpy()
                    res = x - clevel
                    pval = pg.wilcoxon(res)["p-val"].iat[0]
                pval_string = check_pval(pval)
                plt.text(group_i, y, pval_string,
                        ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
            
        # perform pairwise tests:
        if False in normality:
            #parametric = False
            if len(groups)==2:
                pval = pg.mwu(df_in[df_in["Group"] == groups[0]][variable],
                              df_in[df_in["Group"] == groups[1]][variable])['p-val'][0]
                pval_string = check_pval(pval)
                pttest = {}
                pttest["N Groups"] = 2
                pttest["Parametric"] = parametric
                pttest["Test type"] = "Mann-Whitney U"
                pttest["pval"] = pval
                pttest = pd.DataFrame(pttest, index=[0])
                y = ylim[1] - statsOffset
                plt.plot([0, 1], [y, y], lw=1, color="k")
                plt.text(0.5, y-stats_text_correct, pval_string,
                        ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
            else:
                if multicomp:
                    # omnibus test:
                    pval = pg.kruskal(df_in, dv=variable, between="Group")['p-unc'][0]
                    # if pval>0.05, don't plot any pairwise comparisons, i.e. don't indicate n.s.
                    
                    if pval<=0.05 or show_stats_ns:
                        # perform pairwise comparisons:
                        pttest = pg.pairwise_ttests(data=df_in, dv=variable, between="Group", 
                                                    parametric=parametric, padjust="fdr_bh", 
                                                    effsize="cohen")
                        pttest["Omnibus test"] = "Kruskal-Wallis"
                        pttest["Omnibus pval"] = pval
                        pttest_use = pttest[['A', 'B', 'p-corr']].copy()
                        
                        # in pttest, replace in column "A" the entries of the group names with the integer of the order of the respective group in the list "groups":
                        for group_i, group in enumerate(groups):
                            pttest_use["A"].replace(group, group_i, inplace=True)
                            pttest_use["B"].replace(group, group_i, inplace=True)
                        # iterate over all pairwise comparisons; connect the corresponding groups with a horizontal bar in the plots; add an annotation with the pval_string above the bar; increase the offset for the next bar:
                        #stats_text_correct =1
                        for i, row in pttest_use.iterrows():
                            if row['p-corr']<0.05 or show_stats_ns:
                                x1 = row['A']
                                x2 = row['B']
                                running_offset = barOffset * i
                                y = ylim[1] - statsOffset-running_offset
                                plt.plot([x1, x2], [y , y ], lw=2, color="k")
                                plt.text(x1 + (x2 - x1) / 2, y-stats_text_correct  , check_pval(row['p-corr']),
                                            ha='center', va='bottom', fontsize=fontsize_labels,
                                            fontweight="normal")
                    else:
                        pttest = {}
                        pttest["N Groups"] = len(groups)
                        pttest["Parametric"] = parametric
                        pttest["Test type"] = "Kruskal-Wallis"
                        pttest["pval"] = pval
                        pttest = pd.DataFrame(pttest, index=[0])
                else:
                    # perform t-tests of all groups against first group (=control group):
                    pttest = pd.DataFrame()
                    for group in groups[1:]:
                        A = df_in[df_in["Group"] == groups[0]][variable]
                        B = df_in[df_in["Group"] == group][variable]
                        # exclude outliers from A and B:
                        if detect_outliers:
                            print(f"excluding outliers from in non-parametric test group-wise ttests")
                            Q1 = A.quantile(0.25)
                            Q3 = A.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = A[(A < (Q1 - 1.5 * IQR)) | (A > (Q3 + 1.5 * IQR))]
                            A = A[~A.isin(outliers)]
                            Q1 = B.quantile(0.25)
                            Q3 = B.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = B[(B < (Q1 - 1.5 * IQR)) | (B > (Q3 + 1.5 * IQR))]
                            B = B[~B.isin(outliers)]
                        pval = pg.mwu(A, B)['p-val'][0]
                        pttest_curr = pd.DataFrame(index=[0])
                        pttest_curr["A"] = groups[0]
                        pttest_curr["B"] = group
                        pttest_curr["Paired"] = False
                        pttest_curr["Parametric"] = parametric
                        pttest_curr["p-val"] = pval
                        pttest_curr["p-corr"] = pval
                        pttest_curr["test"] = "Mann-Whitney U, just against control"
                        pttest = pd.concat([pttest, pttest_curr], ignore_index=True)
                    # in pttest, replace in column "A" the entries of the group names with the integer of the order of the respective group in the list "groups":
                    pttest_use = pttest[['A', 'B', 'p-corr']].copy()
                    for group_i, group in enumerate(groups):
                        pttest_use["A"].replace(group, group_i, inplace=True)
                        pttest_use["B"].replace(group, group_i, inplace=True)
                    # iterate over all pairwise comparisons; connect the corresponding groups with a horizontal bar in the plots; add an annotation with the pval_string above the bar; increase the offset for the next bar:
                    #stats_text_correct =1
                    for i, row in pttest_use.iterrows():
                        if row['p-corr']<0.05 or show_stats_ns:
                            x1 = row['A']
                            x2 = row['B']
                            running_offset = barOffset * i
                            y = ylim[1] - statsOffset-running_offset
                            plt.plot([x1, x2], [y , y ], lw=2, color="k")
                            plt.text(x1 + (x2 - x1) / 2, y-stats_text_correct  , check_pval(row['p-corr']),
                                        ha='center', va='bottom', fontsize=fontsize_labels,
                                        fontweight="normal")
        else:
            parametric = True
            if len(groups)==2:
                pval = pg.ttest(df_in[df_in["Group"] == groups[0]][variable],
                                df_in[df_in["Group"] == groups[1]][variable],
                                paired=False)['p-val'][0]
                pval_string = check_pval(pval)
                pttest = {}
                pttest["N Groups"] = 2
                pttest["Parametric"] = parametric
                pttest["Test type"] = "t-test"
                pttest["pval"] = pval
                pttest = pd.DataFrame(pttest, index=[0])
                y = ylim[1] - statsOffset
                plt.plot([0, 1], [y, y], lw=1, color="k")
                plt.text(0.5, y-stats_text_correct, pval_string,
                        ha='center', va='bottom', fontsize=fontsize_labels, fontweight="normal")
            else:
                if multicomp:
                    # omnibus test:
                    pval = pg.anova(data=df_in, dv=variable, between="Group")['p-unc'][0]
                    # if pval>0.05, don't plot any pairwise comparisons, i.e. don't indicate n.s.
                    
                    if pval<=0.05 or show_stats_ns:
                        # perform pairwise comparisons:
                        pttest = pg.pairwise_ttests(data=df_in, dv=variable, between="Group", 
                                                    parametric=parametric, padjust="fdr_bh", 
                                                    effsize="cohen")
                        pttest["Omnibus test"] = "ANOVA"
                        pttest["Omnibus pval"] = pval
                        pttest_use = pttest[['A', 'B', 'p-corr']].copy()
                        
                        # in pttest, replace in column "A" the entries of the group names with the integer of the order of the respective group in the list "groups":
                        for group_i, group in enumerate(groups):
                            pttest_use["A"].replace(group, group_i, inplace=True)
                            pttest_use["B"].replace(group, group_i, inplace=True)
                        # iterate over all pairwise comparisons; connect the corresponding groups with a horizontal bar in the plots; add an annotation with the pval_string above the bar; increase the offset for the next bar:
                        #stats_text_correct =1
                        for i, row in pttest_use.iterrows():
                            if row['p-corr']<0.05 or show_stats_ns:
                                x1 = row['A']
                                x2 = row['B']
                                running_offset = barOffset * i
                                y = ylim[1] - statsOffset-running_offset
                                plt.plot([x1, x2], [y , y ], lw=2, color="k")
                                plt.text(x1 + (x2 - x1) / 2, y-stats_text_correct  , check_pval(row['p-corr']),
                                            ha='center', va='bottom', fontsize=fontsize_labels,
                                            fontweight="normal")
                    else:
                        pttest = {}
                        pttest["N Groups"] = len(groups)
                        pttest["Parametric"] = parametric
                        pttest["Test type"] = "ANOVA"
                        pttest["pval"] = pval
                        pttest = pd.DataFrame(pttest, index=[0])
                else:
                    # perform t-tests of all groups against first group (=control group):
                    pttest = pd.DataFrame()
                    for group in groups[1:]:
                        A = df_in[df_in["Group"] == groups[0]][variable]
                        B = df_in[df_in["Group"] == group][variable]
                        # exclude outliers from A and B:
                        if detect_outliers:
                            print(f"excluding outliers from in non-parametric test group-wise ttests")
                            Q1 = A.quantile(0.25)
                            Q3 = A.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = A[(A < (Q1 - 1.5 * IQR)) | (A > (Q3 + 1.5 * IQR))]
                            A = A[~A.isin(outliers)]
                            Q1 = B.quantile(0.25)
                            Q3 = B.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = B[(B < (Q1 - 1.5 * IQR)) | (B > (Q3 + 1.5 * IQR))]
                            B = B[~B.isin(outliers)]
                        pval = pg.ttest(A, B, paired=False)['p-val'][0]
                        pttest_curr = pd.DataFrame(index=[0])
                        pttest_curr["A"] = groups[0]
                        pttest_curr["B"] = group
                        pttest_curr["Paired"] = False
                        pttest_curr["Parametric"] = parametric
                        pttest_curr["p-val"] = pval
                        pttest_curr["p-corr"] = pval
                        pttest_curr["test"] = "t-test, just against control"
                        pttest = pd.concat([pttest, pttest_curr], ignore_index=True)
                    # in pttest, replace in column "A" the entries of the group names with the integer of the order of the respective group in the list "groups":
                    pttest_use = pttest[['A', 'B', 'p-corr']].copy()
                    for group_i, group in enumerate(groups):
                        pttest_use["A"].replace(group, group_i, inplace=True)
                        pttest_use["B"].replace(group, group_i, inplace=True)
                    # iterate over all pairwise comparisons; connect the corresponding groups with a horizontal bar in the plots; add an annotation with the pval_string above the bar; increase the offset for the next bar:
                    #stats_text_correct =1
                    for i, row in pttest_use.iterrows():
                        if row['p-corr']<0.05 or show_stats_ns:
                            x1 = row['A']
                            x2 = row['B']
                            running_offset = barOffset * i
                            y = ylim[1] - statsOffset-running_offset
                            plt.plot([x1, x2], [y , y ], lw=2, color="k")
                            plt.text(x1 + (x2 - x1) / 2, y-stats_text_correct  , check_pval(row['p-corr']),
                                        ha='center', va='bottom', fontsize=fontsize_labels,
                                        fontweight="normal")
    
    #ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(boxborder_width)
    ax.spines['left'].set_linewidth(boxborder_width)

    plt.xlim(-0.5, len(groups)-0.5)
    plt.xticks(np.arange(len(groups)), group_names, fontweight=fontweight, fontsize=fontsize_axes,
               rotation=xlabel_rot)
    plt.yticks(fontsize=fontsize_axes, fontweight=fontweight)
    if show_title:
        plt.title(title, fontweight=fontweight, fontsize=fontsize_axes)
    
    ax.xaxis.set_tick_params(width=boxborder_width, length=8)
    ax.yaxis.set_tick_params(width=boxborder_width, length=8)

    plt.tight_layout()
    
    # prepare df_wide for saving: remove NaN in each row and restack the values to the removed NaNs:
    col_vals = []
    for col in df_wide.columns:
        curr_col_val = df_wide[col].dropna().values
        col_vals.append(curr_col_val)
    # find max length of all col_vals:
    max_len = max([len(col) for col in col_vals])
    df_wide_clean = pd.DataFrame(index=np.arange(max_len), columns=groups)
    for i, col in enumerate(df_wide.columns):
        df_wide_clean[col][:col_vals[i].shape[0]] = col_vals[i]

    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
        elif plotname:
            plt.savefig(os.path.join(plotsavepath, plotname+'.pdf'), dpi=120)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
    if excelpath:
        if title:
            pttest.to_excel(os.path.join(excelpath, title+'_stats.xlsx'))
            df_wide_clean.to_excel(os.path.join(excelpath, title+'_data.xlsx'))
        elif plotname:
            pttest.to_excel(os.path.join(excelpath, plotname+'_stats.xlsx'))
            df_wide_clean.to_excel(os.path.join(excelpath, plotname+'_data.xlsx'))
        else:
            pttest.to_excel(os.path.join(excelpath, 'plot_stats.xlsx'))
            df_wide_clean.to_excel(os.path.join(excelpath, 'plot_data.xlsx'))
    return

def plot_2_samples_timeseries(df_in, fignum=1, title='', Alabel='A', Blabel='B', figaspect=(6,4),
                              plotsavepath='', xlim=(0.75, 5.25), show_stats=True,
                              fontsize_axes=12, fontsize_labels=12, fontsize_stats=12,
                              plotname = "", boxborder_width=2,
                              ylabel="average time until\nreaching the platform [s]",
                              dv=" training time mean s", dv2=" training time std s",
                              ylim=(0, 64), yticks=np.arange(0, 61, 10),
                              statsOffset=61, SEM=False, legend_loc="upper right"):
    # df_in = mwm_training_df
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    fig.clf()

    # change global matplotliv font to arial:
    plt.rcParams['font.family'] = 'Arial'
    
    if SEM:
        error_A = df_in[Alabel + dv2].values / np.sqrt(df_in[Alabel + " N_samples"][0])
        error_B = df_in[Blabel + dv2].values / np.sqrt(df_in[Blabel + " N_samples"][0])
    else:
        error_A = df_in[Alabel + dv2].values
        error_B = df_in[Blabel + dv2].values

    plt.plot(df_in["day"].values, df_in[Alabel + dv].values,
             color=Group_Styles[0]['Color'],lw=2, label=Alabel, ls="-")
    # sem_divider = np.sqrt(curr_df_sub["N_samples"].values)
    plt.fill_between(df_in["day"].values.astype("int"),
                     df_in[Alabel + dv].values - error_A,
                     df_in[Alabel + dv].values + error_A,
                     edgecolor=Group_Styles[0]['Color'],
                     facecolor=Group_Styles[0]['Color'],
                     alpha=0.25, linewidth=0.0)
    plt.plot(df_in["day"].values, df_in[Blabel + dv].values,
             color=Group_Styles[1]['Color'],lw=2, label=Blabel)
    # sem_divider = np.sqrt(curr_df_sub["N_samples"].values)
    plt.fill_between(df_in["day"].values.astype("int"),
                     df_in[Blabel + dv].values - error_B,
                     df_in[Blabel + dv].values + error_B,
                     edgecolor=Group_Styles[1]['Color'],
                     facecolor=Group_Styles[1]['Color'],
                     alpha=0.25, linewidth=0.0)
    
    # legend dummy plot:
    plt.plot([0, 0], [0, 0], color="w", lw=2, label="$\mu \pm$ SEM")
    
    if show_stats:
        for day in range(df_in.shape[0]):
            if df_in["pval"][day]<0.05:
                if df_in["pval"][day] <  0.0001:
                    pval_string = "****"
                elif df_in["pval"][day] <  0.001:
                    pval_string = "***"
                elif df_in["pval"][day] <  0.01:
                    pval_string = "**"
                elif df_in["pval"][day] <  0.05:
                    pval_string = "*"

                
                plt.text(day+1, statsOffset, pval_string,
                         ha='center', va='bottom', fontsize=fontsize_stats, fontweight="normal")
            else:
                plt.text(day+1, statsOffset, 'n.s.',
                         ha='center', va='bottom', fontsize=fontsize_stats, fontweight="normal")

    plt.legend(loc=legend_loc, fontsize=fontsize_labels, frameon=False)
    plt.xlim(xlim)
    plt.xticks(np.arange(1, xlim[1], 1), fontsize=fontsize_axes, fontweight="normal")
    plt.ylim(ylim)
    plt.yticks(yticks, fontsize=fontsize_axes, fontweight="normal")
    plt.xlabel("training day", fontsize=fontsize_labels, fontweight="normal")
    plt.ylabel(ylabel, fontsize=fontsize_labels, fontweight="normal")
    plt.title(title, fontsize=fontsize_labels, fontweight="normal")

    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(boxborder_width)
    ax.spines["left"].set_linewidth(boxborder_width)
    ax.tick_params(width=boxborder_width, length=10, bottom=True, left=True)

    plt.tight_layout()

    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, title + '.png'), dpi=200)
        elif plotname:
            plt.savefig(os.path.join(plotsavepath, plotname+'.pdf'), dpi=120)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, 'plot' + '.png'), dpi=200)

def plot_N_samples_timeseries(df_in, fignum=1, title='', labels=[], figaspect=(6,4),
                              plotsavepath='', xlim=(0.75, 5.25), 
                              fontsize_axes=12, fontsize_labels=12,
                              plotname = "", boxborder_width=2,
                              ylabel="average time until\nreaching the platform [s]",
                              dv=" training time mean s", dv2=" training time std s",
                              ylim=(0, 64), yticks=np.arange(0, 61, 10),
                              SEM=False, legend_loc="upper right"):
    # df_in = mwm_training_eval_df
    # labels = groups
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    fig.clf()

    # change global matplotliv font to arial:
    plt.rcParams['font.family'] = 'Arial'
    
    # rewrite the existing code to run over N groups instead of 2:
    
    if SEM:
        errors_df = df_in[labels + dv2] / np.sqrt(df_in[labels + " N_samples"]).values
    else:
        errors_df = df_in[labels + dv2]

    for group_i, group in enumerate(labels):
        # get the index of the current group in the labels np-array:
        group_index = np.where(labels == group)[0][0]
        plt.plot(df_in["day"].values, df_in[group + dv].values,
                 color=Group_Styles[group_index]['Color'],lw=2, label=group, 
                 ls=Group_Styles[group_i]['Linestyles'])
        plt.fill_between(df_in["day"].values.astype("int"),
                         df_in[group + dv].values - errors_df[group + dv2],
                         df_in[group + dv].values + errors_df[group + dv2],
                         edgecolor=Group_Styles[group_index]['Color'],
                         facecolor=Group_Styles[group_index]['Color'],
                         alpha=0.25, linewidth=0.0)
    
    # legend dummy plot:
    plt.plot([0, 0], [0, 0], color="w", lw=2, label="$\mu \pm$ SEM")
    
    
    plt.legend(loc=legend_loc, fontsize=fontsize_labels, frameon=False)
    plt.xlim(xlim)
    plt.xticks(np.arange(1, xlim[1], 1), fontsize=fontsize_axes, fontweight="normal")
    plt.ylim(ylim)
    plt.yticks(yticks, fontsize=fontsize_axes, fontweight="normal")
    plt.xlabel("training day", fontsize=fontsize_labels, fontweight="normal")
    plt.ylabel(ylabel, fontsize=fontsize_labels, fontweight="normal")
    plt.title(title, fontsize=fontsize_labels, fontweight="normal")

    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(boxborder_width)
    ax.spines["left"].set_linewidth(boxborder_width)
    ax.tick_params(width=boxborder_width, length=10, bottom=True, left=True)

    plt.tight_layout()

    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, title + '.png'), dpi=200)
        elif plotname:
            plt.savefig(os.path.join(plotsavepath, plotname+'.pdf'), dpi=120)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, 'plot' + '.png'), dpi=200)

def fit_and_plot_2_samples_timeseries(df_in, fignum=1, title='', Alabel='A', Blabel='B',
                                      figaspect=(6,4), order=3, colors=["k","k"],
                                      dv="Latency to platform Platform / Center-point Latency to Last s",
                                      plotsavepath='', xlim=(0.75, 5.25), show_stats=True,
                                      fontsize_axes=12, fontsize_labels=12, fontsize_stats=12,
                                      ylabel="average time until\nreaching the platform [s]",
                                      ylim=(0, 64), yticks=np.arange(0, 61, 10),
                                      statsOffset=61):
    # df_in = mwm_training_df
    sns.reset_orig()
    # plt.close(fignum)
    # fig=plt.figure(fignum, figsize=figaspect)
    # fig.clf()

    # prepare the input dataframe:
    # mwm_training_eval_df = pd.DataFrame(columns=columns)
    # for group in groups:
    #     # day = 1
    #     # group = groups[0]
    #     N_mice = df_in[df_in["Group"]==group]['ID'].unique().shape[0]
    #     curr_mice_trace_1D_x = np.repeat(np.arange(1,6,1), [N_mice]*5)
    #     curr_mice_trace_1D_y = np.zeros(N_mice * 5)
    #     iter_start=0
    #     for day in range(1, 6):
    #         curr_mice_trace_1D_y[iter_start:iter_start+N_mice] = df_in[df_in["Day"]==day][df_in[mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].values
    #         iter_start+=N_mice
    #     poly = np.polyfit(curr_mice_trace_1D_x, curr_mice_trace_1D_y, deg=3)
    #     plt.plot(curr_mice_trace_1D_x, curr_mice_trace_1D_y, '.')
    #     plt.plot(np.arange(0,6,0.1), np.polyval(poly, np.arange(0,6,0.1)))
    #     init_vals = [1, 0, 1]  # for [amp, cen, wid]
    #     best_vals, covar = curve_fit(polynomial_function(curr_mice_trace_1D_x,5),
    #                                  curr_mice_trace_1D_x, curr_mice_trace_1D_y, p0=init_vals)
    groups = df_in["Group"].unique()
    df_tmp = pd.DataFrame(columns=["Day", "Values", "Group"])
    df_tmp["Day"] = df_in["Day"].values.astype("float32")
    df_tmp["Values"] = df_in[dv].values.astype("float32")
    df_tmp["Group"] = df_in["Group"]
    palette = sns.color_palette(colors)
    sns.set(rc={"figure.figsize": (10, 4)}, style="white", palette=palette)
    #plt.rcParams["figure.figsize"] = (10,4)
    #sns.reset_orig()
    #order=3
    ax = sns.lmplot(x="Day", y="Values", data=df_tmp, hue="Group",
                    x_jitter=0.00, legend=False, hue_order=[groups[1],groups[0]],
                    order=order, ci=95, scatter_kws={"s": 80, 'alpha': 0.2},
                    line_kws={"ls": "-"})
    ax = plt.gca()
    ax.set_xlabel("test")
    ax.plot(-1,-1, '-', lw=2, c="k", label=f"polynomial fit (deg={order}) ± CI95")
    #ax.plot(np.arange(1,5.1,0.1), np.polyval(poly, np.arange(1,5.1,0.1)))

    # plt.plot(df_in["day"].values, df_in[Blabel + dv].values,
    #          color=Group_Styles[1]['Color'],lw=2, label=Blabel)
    # # sem_divider = np.sqrt(curr_df_sub["N_samples"].values)
    # plt.fill_between(df_in["day"].values.astype("int"),
    #                  df_in[Blabel + dv].values - df_in[Blabel + dv2].values,
    #                  df_in[Blabel + dv].values + df_in[Blabel + dv2].values,
    #                  edgecolor=Group_Styles[1]['Color'],
    #                  facecolor=Group_Styles[1]['Color'],
    #                  alpha=0.25, linewidth=0.0)
    # plt.plot(df_in["day"].values, df_in[Alabel + dv].values,
    #          color=Group_Styles[0]['Color'],lw=2, label=Alabel, ls="--")
    # # sem_divider = np.sqrt(curr_df_sub["N_samples"].values)
    # plt.fill_between(df_in["day"].values.astype("int"),
    #                  df_in[Alabel + dv].values - df_in[Alabel + dv2].values,
    #                  df_in[Alabel + dv].values + df_in[Alabel + dv2].values,
    #                  edgecolor=Group_Styles[0]['Color'],
    #                  facecolor=Group_Styles[0]['Color'],
    #                  alpha=0.25, linewidth=0.0)

    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim(xlim)
    #ax.set_xticks(np.arange(1, xlim[1], 1, dtype="int"), fontsize=fontsize_axes, fontweight="bold")
    ax.set_xticks(np.arange(1, xlim[1], 1, dtype="int"))
    ax.set_xticklabels(np.arange(1, xlim[1], 1, dtype="int"), fontsize=fontsize_axes, fontweight="bold")
    ax.set_ylim(ylim)
    #ax.set_yticks(yticks, fontsize=fontsize_axes, fontweight="bold")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize_axes, fontweight="bold")
    ax.set_xlabel("training day", fontsize=fontsize_labels, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fontsize_labels, fontweight="bold")
    ax.set_title(title, fontsize=fontsize_labels, fontweight="bold")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(width=2, length=10, bottom=True, left=True)

    plt.tight_layout()

    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, title + '.png'), dpi=200)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
            #plt.savefig(os.path.join(plotsavepath, 'plot' + '.png'), dpi=200)
    sns.reset_orig()

def plot_1_samples_timeseries(df_in, fignum=1, title='', figaspect=(6,6), bar_offset=4, xlim=(0.75, 5.25),
                              plotsavepath='', group_to_eval=0, groups=["A", "B"], significance=False,
                              statsoffset=10, ylim=(0, 100), yticks = np.arange(0, 61, 10),
                              ylabel="average time until\nreaching the platform [s]",
                              dv=" training time mean s", dv2=" training time std s",):
    # df_in = mwm_training_df
    plt.close(fignum)
    fig=plt.figure(fignum, figsize=figaspect)
    fig.clf()

    #plt.grid(color='gainsboro', linestyle='-', linewidth=0.25, zorder=1)

    plt.plot(df_in["day"].values, df_in[groups[group_to_eval] + dv].values,
             color=Group_Styles[group_to_eval]['Color'],lw=2, label=groups[group_to_eval],
             ls=Group_Styles[group_to_eval]['Linestyles'])
    # sem_divider = np.sqrt(curr_df_sub["N_samples"].values)
    plt.fill_between(df_in["day"].values.astype("int"),
                     df_in[groups[group_to_eval] + dv].values - df_in[groups[group_to_eval] + dv2].values,
                     df_in[groups[group_to_eval] + dv].values + df_in[groups[group_to_eval] + dv2].values,
                     edgecolor=Group_Styles[group_to_eval]['Color'],
                     facecolor=Group_Styles[group_to_eval]['Color'],
                     alpha=0.25, linewidth=0.0)

    if not type(significance)==bool:
        for test_i in range(significance.shape[0]):
            #test_i = 0
            x1 = significance["A"][test_i]
            x2 = significance["B"][test_i]
            y = (ylim[1]-statsoffset) - (bar_offset*test_i)

            if significance["p-corr"][test_i] < 0.05:
                plt.plot([x1, x2], [y, y], "-", lw=2, c="k")
                plt.text(x1+(x2-x1)/2, y, '$d$=' + str(significance["cohen"][test_i].round(2)),
                         ha='center', va='bottom', fontsize=14, fontweight="normal")
            else:
                plt.plot([x1, x2], [y, y], "-", lw=2, c="k")
                plt.text(x1 + (x2 - x1) / 2, y, 'n.s.',
                         ha='center', va='bottom', fontsize=14, fontweight="normal")

    plt.legend(loc="lower left", fontsize=14)
    plt.xlim(xlim)
    plt.xticks(np.arange(1, xlim[1], 1), fontsize=14)
    plt.ylim(ylim)
    plt.yticks(yticks, fontsize=14)

    plt.xlabel("training day", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14, fontweight="bold")

    ax = plt.gca()  # get current axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(width=2, length=10)

    plt.tight_layout()

    if plotsavepath:
        if title:
            plt.savefig(os.path.join(plotsavepath, title+'.pdf'), dpi=120)
            # plt.savefig(os.path.join(plotsavepath, title + '.png'), dpi=200)
        else:
            plt.savefig(os.path.join(plotsavepath, 'plot'+'.pdf'), dpi=120)
            # plt.savefig(os.path.join(plotsavepath, 'plot' + '.png'), dpi=200)
# %% READ THE DATA
mice_info_df = pd.read_excel(mice_infotable)
# drop mice where "Exclude from NOR" == "yes":
mice_info_df = mice_info_df[mice_info_df["Exclude from NOR"]!="yes"]
# drop all rows, where all entries are NaN:
mice_info_df = mice_info_df.dropna(axis=0, how="all")

# read all Excel files, that are in DATA_path; files should not start with "~$":
excel_files = []
for file in os.listdir(DATA_path):
    if file.endswith(".xlsx") and not file.startswith("~$"):
        excel_files.append(os.path.join(DATA_path, file))

mwm_all_df = pd.DataFrame()

# read and unblind the data:
for file in excel_files:
    # file = excel_files[0]
    mwm_datatable_name = file
    mwm_datatable = os.path.join(DATA_path, mwm_datatable_name)

    mwm_df = pd.read_excel(mwm_datatable)

    mwm_df["Velocity [cm/s]"] = 0.0
    mwm_df["ID"] = 0
    mwm_df["Group"] = "n/a"
    mwm_df["Sex"] = "unknown"
    for mouse_id in mice_info_df["Pseudo_ID"]:
        # mouse_id = mice_info_df["Pseudo_ID"][0]
        # find mouse_id in mice_info_df and add "ID" and "Group" columns: 
        curr_mouse_idx = mwm_df[mwm_df["Animal Pseudo-ID"]==mouse_id].index
        mwm_df.loc[curr_mouse_idx, "ID"] = mice_info_df[mice_info_df["Pseudo_ID"]==mouse_id]["ID"].values[0]
        mwm_df.loc[curr_mouse_idx, "Group"] = mice_info_df[mice_info_df["Pseudo_ID"]==mouse_id]["Group"].values[0]
        mwm_df.loc[curr_mouse_idx, "Sex"] = mice_info_df[mice_info_df["Pseudo_ID"]==mouse_id]["Sex"].values[0]
        
        # calculate the velocity:
        mwm_df["Velocity [cm/s]"] = mwm_df["Total Distance moved Center-point Total cm"] / mwm_df["Total Duration Arena / Center-point Cumulative Duration s"]
        
        # replace ♀ and ♂ with "female" and "male":
        mwm_df["Sex"] = mwm_df["Sex"].replace({"♀": "female", "♂": "male"})
        
    # update the mwm_all_df:
    mwm_all_df = pd.concat([mwm_all_df, mwm_df], ignore_index=True)
    
# drop all rows, where mwm_df["Group"] = "n/a":
mwm_all_df = mwm_all_df[mwm_all_df["Group"]!="n/a"]
# replace all '-' entries with 60.0:
mwm_all_df = mwm_all_df.replace("-", 60.0)

# overwrite the mwm_all_df with the mwm_df to stay consistent with the remaining script:
mwm_df = mwm_all_df.copy()
mwm_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_all.xlsx"))

# swap order of groups:
groups_tmp = mwm_df["Group"].unique()
groups = group_order.copy()
# ensure, that groups is ordered group_order; also check, whether groups contains all groups; 
# if not, skip it in group_order:
for group in group_order:
    if group not in groups_tmp:
        groups.remove(group)
groups = np.array(groups).astype("object")
mice   = mwm_df["ID"].unique()

column_max=26

# average the 4 trials per training day (day1-5) per mouse:
mwm_training_df = pd.DataFrame(columns=mwm_df.columns)
for day in range(1,6):
    # day = 1
    curr_sub_mwm_df = mwm_df[mwm_df["Day"]==day]
    for mouse in mice:
        # mouse = mice[0]
        curr_sub_mwm_mouse_df = curr_sub_mwm_df[curr_sub_mwm_df["ID"] == mouse]
        # treat "-" entries as 60s:
        curr_sub_mwm_mouse_df = curr_sub_mwm_mouse_df.replace("-", 60.0)
        # find the number of column with name "Latency to platform Platform / Center-point Latency to First s":
        start_idx = curr_sub_mwm_mouse_df.columns.get_loc("Latency to platform Platform / Center-point Latency to First s")
        #curr_averages_ds = curr_sub_mwm_mouse_df.iloc[:,start_idx:26].mean(axis=0, numeric_only=False)
        curr_averages_ds = curr_sub_mwm_mouse_df.iloc[:,start_idx:column_max+start_idx].mean(axis=0, numeric_only=False)
        curr_averages_ds["Group"] = curr_sub_mwm_mouse_df["Group"].values[0]
        curr_averages_ds["ID"] = curr_sub_mwm_mouse_df["ID"].values[0]
        curr_averages_ds["Sex"]= curr_sub_mwm_mouse_df["Sex"].values[0]
        curr_sub_mwm_mouse_df_tmp = pd.DataFrame(columns=mwm_df.columns, index=[0])
        for key in curr_averages_ds.keys():
            curr_sub_mwm_mouse_df_tmp[key] = curr_averages_ds[key]
        for key_i in range(0,6):
            curr_sub_mwm_mouse_df_tmp.iloc[0,key_i] = curr_sub_mwm_mouse_df.iloc[0,key_i]
        curr_sub_mwm_mouse_df_tmp["Release quadrant"] = "S-W-NW-SE-SW"
        mwm_training_df = pd.concat([mwm_training_df, curr_sub_mwm_mouse_df_tmp], ignore_index=True)
mwm_training_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training.xlsx"))

# normalize to day 1 values:
mwm_training_df_day1norm = pd.DataFrame(columns=mwm_training_df.columns)
for mouse in mice:
    # mouse = mice[0]
    curr_sub_mwm_mouse_df = mwm_training_df[mwm_training_df["ID"]==mouse]
    # update values in columns 6 to 25 by dividing them by the value in the first row:
    curr_sub_mwm_mouse_df.iloc[:,6:26] = curr_sub_mwm_mouse_df.iloc[:,6:column_max].div(curr_sub_mwm_mouse_df.iloc[0,6:column_max].values, axis=1).astype("float32")
    mwm_training_df_day1norm = pd.concat([mwm_training_df_day1norm, curr_sub_mwm_mouse_df], ignore_index=True)
mwm_training_df_day1norm.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_day1norm.xlsx"))

# normalize the "training time mean s" in mwm_training_df via z-scoring per mouse:
mwm_training_df_zscored = pd.DataFrame(columns=mwm_training_df.columns)
for mouse in mice:
    # mouse = mice[0]
    curr_sub_mwm_mouse_df = mwm_training_df[mwm_training_df["ID"]==mouse]
    # find the number of column with name "Latency to platform Platform / Center-point Latency to First s":
    start_idx = curr_sub_mwm_mouse_df.columns.get_loc("Latency to platform Platform / Center-point Latency to First s")
    mean_trial = curr_sub_mwm_mouse_df.iloc[:,start_idx:column_max].mean(axis=0, numeric_only=False)
    std_trial = curr_sub_mwm_mouse_df.iloc[:,start_idx:column_max].std(axis=0, numeric_only=False)
    curr_sub_mwm_mouse_df.iloc[:,start_idx:column_max] = (curr_sub_mwm_mouse_df.iloc[:,start_idx:column_max]-mean_trial)/std_trial
    mwm_training_df_zscored = pd.concat([mwm_training_df_zscored, curr_sub_mwm_mouse_df], ignore_index=True)
mwm_training_df_zscored.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_zscored.xlsx"))

mwm_training_and_probe_df = pd.DataFrame(columns=mwm_df.columns)
mwm_training_and_probe_df = pd.concat([mwm_training_df, mwm_df[mwm_df["Day"]==6], mwm_df[mwm_df["Day"]==7]])
mwm_training_and_probe_df = mwm_training_and_probe_df.reset_index()
mwm_training_and_probe_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_and_probe.xlsx"))
# %% PLOT GROUP SIZES
# plot group-sizes from of_df as barplot:
fig=plt.figure(1, figsize=(4,4))
plt.rcParams['font.family'] = 'Arial'
plt.clf()
for group_i, group in enumerate(groups):
    plt.bar(group_i, len(mwm_training_df[mwm_training_df["Group"]==group]["ID"].unique()), color=Group_Styles2[group]["Color"], 
            edgecolor=Group_Styles2[group]["Color"])
plt.xticks(np.arange(len(groups)), [Group_Styles2[group]["Name"] for group in groups],
              rotation=55, fontsize=20)
plt.yticks(np.arange(0,21,2),fontsize=20)
plt.ylabel("group size", fontsize=20)
plt.tight_layout()
# remove top and right spines:
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig(os.path.join(RESULTS_path, "MWM_group_sizes.pdf"), dpi=120)

# plot it now sex-separated:
for curr_sex in ["female", "male"]:
    fig=plt.figure(1, figsize=(4,4))
    plt.rcParams['font.family'] = 'Arial'
    plt.clf()
    for group_i, group in enumerate(groups):
        plt.bar(group_i, len(mwm_training_df[(mwm_training_df["Group"]==group) & (mwm_training_df["Sex"]==curr_sex)]["ID"].unique()), 
                color=Group_Styles2[group]["Color"], edgecolor=Group_Styles2[group]["Color"])
    plt.xticks(np.arange(len(groups)), [Group_Styles2[group]["Name"] for group in groups],
                    rotation=55, fontsize=20)
    plt.yticks(np.arange(0,21,2),fontsize=20)
    plt.ylabel("group size", fontsize=20)
    plt.tight_layout()
    # remove top and right spines:
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.savefig(os.path.join(RESULTS_path, f"MWM_group_sizes {curr_sex}.pdf"), dpi=120)
 
# %% EVALUATE TRAINING DAY (LATENCY)
curr_mwm_training_df = mwm_training_df.copy()
columns = ["day"]
for group in groups:
    columns.append(group + " training time mean s")
    columns.append(group + " training time std s")

mwm_training_eval_df = pd.DataFrame(columns=columns)
for day in range(1,6):
    # day = 1
    mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
    group_Ns = []
    for group in groups:
        # Latency to platform Platform / Center-point Latency to Last s
        mwm_training_df_tmp[group+ " training time mean s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].mean()
        mwm_training_df_tmp[group+ " training time std s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].std()
        mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0]
        group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0])
    mwm_training_df_tmp["day"] = day
    
    if len(groups)==2:
        A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
        B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
        Significance = independent_two_sample_test(A, B)
        mwm_training_df_tmp["pval"] = Significance["pval"].values
        mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
        
    mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_eval.xlsx"))

if len(groups)>2:
    plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days",
                            ylabel="average time to\nreach the platform [s]",
                            ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.75, 5.25),
                            labels=groups, figaspect=(4.9,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17,
                            SEM=True,boxborder_width=1.5)
    
    # test, if the groups are normally distributed:
    norm_pvals = []
    for group in groups:
        # group = groups[0]
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
        pval = pg.normality(curr_df, dv="Latency to platform Platform / Center-point Latency to Last s", 
                            group="Day")["normal"].values
        norm_pvals.append(list(pval))
    # flatten norm_pvals:
    norm_pvals = [item for sublist in norm_pvals for item in sublist]
    if False in norm_pvals:
        parametric = False
        # perform a kruskal-wallis test:
        kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                 dv="Latency to platform Platform / Center-point Latency to Last s", 
                                 between="Group")
        # correct p-unc for multiple comparisons:
        kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test =kruskal_test["p-corr"].values
                                 
    else:
        parametric = True
        rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                     dv="Latency to platform Platform / Center-point Latency to Last s", 
                                     within="Day", subject="ID", between="Group")
        rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test = rm_anova_test["p-corr"][2]
        

    pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                dv="Latency to platform Platform / Center-point Latency to Last s",
                                within="Day", subject="ID", parametric = parametric,
                                between="Group", padjust='fdr_bh',
                                effsize='cohen', correction=True, within_first=True)
    
    pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
    pttest["p-corr"]
    pttest["Normality"] = False if parametric==False in norm_pvals else True
    pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
    pttest["Omnibus-test p-value"] = prehoc_test[0]
    pttest.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_pttest.xlsx"))
elif len(groups)==2:
    plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days",
                            ylabel="average time to\nreach the platform [s]",
                            ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                            Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                            SEM=True,boxborder_width=1.5)

# plot individual groups with their intra-group statistics:
for group_i, group in enumerate(groups):
    curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
    pval = pg.rm_anova(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                within="Day", subject="ID", correction=True)["p-GG-corr"].values
    """ if pval<0.05:
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)
    else:
        sig_test=False """
    sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)

    plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                            plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                            significance=sig_test,
                            ylabel="average time to\nreach the platform (pooled) [s]",
                            title="MWM training days "+group.replace("/", " "))

# plot heatmap:
plt.figure(figsize=(6, len(groups)))
plt.rcParams.update({'font.size': 12})
eval_columns = ["day"]
for group in groups:
    eval_columns.append(group + " training time mean s")
curr_heatmap_df = mwm_training_eval_df[eval_columns].set_index('day')
curr_heatmap_df.columns = [col.replace(" training time mean s", "") for col in curr_heatmap_df.columns]
curr_heatmap_df = curr_heatmap_df.T
ax = sns.heatmap(curr_heatmap_df, cmap='coolwarm', annot=True, vmin=0, vmax=60,
            cbar_kws={'label': 'average time to reach\nthe platform (lower=better) [s]'})
ax.set_yticklabels(ax.get_yticklabels(), rotation=00)
plt.title('Group training performance')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_path, 'MWM heatmap'+'.pdf'), dpi=120)
# %% EVALUATE TRAINING DAY (LATENCY, NORM TO DAY 1)
curr_mwm_training_df = mwm_training_df_day1norm.copy()
columns = ["day"]
for group in groups:
    columns.append(group + " training time mean s")
    columns.append(group + " training time std s")

mwm_training_eval_df = pd.DataFrame(columns=columns)
for day in range(1,6):
    # day = 1
    mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
    group_Ns = []
    for group in groups:
        # Latency to platform Platform / Center-point Latency to Last s
        mwm_training_df_tmp[group+ " training time mean s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].mean()
        mwm_training_df_tmp[group+ " training time std s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].std()
        mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0]
        group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0])
    mwm_training_df_tmp["day"] = day
    
    if len(groups)==2:
        A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
        B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
        Significance = independent_two_sample_test(A, B)
        mwm_training_df_tmp["pval"] = Significance["pval"].values
        mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
        
    mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_eval_day1norm.xlsx"))

if len(groups)>2:
    plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days, norm to day 1",
                            ylabel="average time to\nreach the platform [s]",
                            ylim=(0, 1.4), yticks=np.arange(0, 1.5, 0.2), xlim=(0.75, 5.25),
                            labels=groups, figaspect=(4.9,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17,
                            SEM=True,boxborder_width=1.5)
    
    # test, if the groups are normally distributed:
    norm_pvals = []
    for group in groups:
        # group = groups[0]
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
        # exclude all day 1 data (in the normed data, day 1 is 1.0 for all mice):
        curr_df = curr_df[curr_df["Day"]>1]
        #curr_df["Latency to platform Platform / Center-point Latency to Last s"] = curr_df["Latency to platform Platform / Center-point Latency to Last s"].astype("float32")
        pval = pg.normality(curr_df, dv="Latency to platform Platform / Center-point Latency to Last s", 
                            group="Day")["normal"].values
        norm_pvals.append(list(pval))
    # flatten norm_pvals:
    norm_pvals = [item for sublist in norm_pvals for item in sublist]
    if False in norm_pvals:
        parametric = False
        # perform a kruskal-wallis test:
        kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                 dv="Latency to platform Platform / Center-point Latency to Last s", 
                                 between="Group")
        # correct p-unc for multiple comparisons:
        kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test =kruskal_test["p-corr"].values
                                 
    else:
        parametric = True
        rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                     dv="Latency to platform Platform / Center-point Latency to Last s", 
                                     within="Day", subject="ID", between="Group")
        rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test = rm_anova_test["p-corr"][2]
        

    pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                dv="Latency to platform Platform / Center-point Latency to Last s",
                                within="Day", subject="ID", parametric = parametric,
                                between="Group", padjust='fdr_bh',
                                effsize='cohen', correction=True, within_first=True)
    
    #pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
    #pttest["p-corr"]
    pttest["Normality"] = False if parametric==False in norm_pvals else True
    pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
    pttest["Omnibus-test p-value"] = prehoc_test[0]
    pttest.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_pttest norm day 1.xlsx"))
elif len(groups)==2:
    plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days norm to day 1",
                            ylabel="average time to\nreach the platform [s]",
                            ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                            Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                            SEM=True,boxborder_width=1.5)

# plot individual groups with their intra-group statistics:
for group_i, group in enumerate(groups):
    curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
    pval = pg.rm_anova(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                within="Day", subject="ID", correction=True)["p-GG-corr"].values
    """ if pval<0.05:
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)
    else:
        sig_test=False """
    sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)

    plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                            plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                            significance=sig_test,
                            ylabel="average time to\nreach the platform [s]",
                            title="MWM training days norm to day 1 "+group.replace("/", " "),
                            ylim=(0,1.4), yticks=np.arange(0, 1.5, 0.2),
                            statsoffset=0.1, bar_offset=0.06)

# plot heatmap:
plt.figure(figsize=(6, 3))
sns.heatmap(mwm_training_eval_df[["day", "Adrb2-wt training time mean s","Adrb2-flox training time mean s", "50 mg/kg training time mean s" ]].set_index('day').T, 
            cmap='coolwarm', annot=True,
            cbar_kws={'label': 'normalized time to reach\nthe platform (lower=better)'})
plt.title('Group training performance (norm to day 1)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_path, 'MWM heatmap norm to day 1'+'.pdf'), dpi=120)
# %% EVALUATE TRAINING DAY (LATENCY, Z-SCORE NORMALIZED)
curr_mwm_training_df = mwm_training_df_zscored.copy()
columns = ["day"]
for group in groups:
    columns.append(group + " training time mean s")
    columns.append(group + " training time std s")

mwm_training_eval_df = pd.DataFrame(columns=columns)
for day in range(1,6):
    # day = 1
    mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
    for group in groups:
        # Latency to platform Platform / Center-point Latency to Last s
        mwm_training_df_tmp[group+ " training time mean s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].mean()
        mwm_training_df_tmp[group+ " training time std s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].std()
        mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0]
    mwm_training_df_tmp["day"] = day
    
    if len(groups)==2:
        A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
        B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
        Significance = independent_two_sample_test(A, B)
        mwm_training_df_tmp["pval"] = Significance["pval"].values
        mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
        
    mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_eval_zscored.xlsx"))

if len(groups)>2:
    plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days, zscored",
                            ylabel="average time to reach\nthe platform [z-scored]",
                            ylim=(-1, 1.5), yticks=np.arange(-1, 1.6, 0.3), xlim=(0.75, 5.25),
                            labels=groups, figaspect=(4.9,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17,
                            SEM=True,boxborder_width=1.5)
    
    # test, if the groups are normally distributed:
    norm_pvals = []
    for group in groups:
        # group = groups[0]
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
        # exclude all day 1 data (in the normed data, day 1 is 1.0 for all mice):
        curr_df = curr_df[curr_df["Day"]>1]
        #curr_df["Latency to platform Platform / Center-point Latency to Last s"] = curr_df["Latency to platform Platform / Center-point Latency to Last s"].astype("float32")
        pval = pg.normality(curr_df, dv="Latency to platform Platform / Center-point Latency to Last s", 
                            group="Day")["normal"].values
        norm_pvals.append(list(pval))
    # flatten norm_pvals:
    norm_pvals = [item for sublist in norm_pvals for item in sublist]
    if False in norm_pvals:
        parametric = False
        # perform a kruskal-wallis test:
        kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                 dv="Latency to platform Platform / Center-point Latency to Last s", 
                                 between="Group")
        # correct p-unc for multiple comparisons:
        kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test =kruskal_test["p-corr"].values
                                 
    else:
        parametric = True
        rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                     dv="Latency to platform Platform / Center-point Latency to Last s", 
                                     within="Day", subject="ID", between="Group")
        rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test = rm_anova_test["p-corr"][2]
        

    pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                dv="Latency to platform Platform / Center-point Latency to Last s",
                                within="Day", subject="ID", parametric = parametric,
                                between="Group", padjust='fdr_bh',
                                effsize='cohen', correction=True, within_first=True)
    
    #pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
    #pttest["p-corr"]
    pttest["Normality"] = False if parametric==False in norm_pvals else True
    pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
    pttest["Omnibus-test p-value"] = prehoc_test[0]
    pttest.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_pttest zscored.xlsx"))
elif len(groups)==2:
    plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days zscored",
                            ylabel="average time to\nreach the platform [z-score]",
                            ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                            Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                            SEM=True,boxborder_width=1.5)

# plot individual groups with their intra-group statistics:
for group_i, group in enumerate(groups):
    # group_i = 0
    # group = groups[0]
    curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
    pval = pg.rm_anova(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                within="Day", subject="ID", correction=True)["p-GG-corr"].values
    """ if pval<0.05:
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)
    else:
        sig_test=False """
    sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)

    plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                            plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                            significance=sig_test,
                            ylabel="average time to reach\nthe platform [z-scored]",
                            title="MWM training days z-scored "+group.replace("/", " "),
                            ylim=(-1.2, 1.9), yticks=np.arange(-1, 1.6, 0.3),
                            statsoffset=0.1, bar_offset=0.15)

# plot heatmap:
plt.figure(figsize=(6, 3))
plt.rcParams.update({'font.size': 12})
curr_heatmap_df = mwm_training_eval_df[["day", "Adrb2-wt training time mean s","Adrb2-flox training time mean s", "50 mg/kg training time mean s" ]].set_index('day')
# from the column names, remove "mean s":
curr_heatmap_df.columns = [col.replace(" training time mean s", "") for col in curr_heatmap_df.columns]
curr_heatmap_df = curr_heatmap_df.T
ax = sns.heatmap(curr_heatmap_df, cmap='coolwarm', annot=True, vmin=-1, vmax=1,
                cbar_kws={'label': 'z-scored time to reach\nthe platform (lower=better)'})
ax.set_yticklabels(ax.get_yticklabels(), rotation=00)
plt.title('Group training performance (z-scored)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_path, 'MWM heatmap zscored'+'.pdf'), dpi=120)
# %% EVALUATE TRAINING DAY (LATENCY) (M/F)
columns = ["day", "pval", "cohen d"]
for group in groups:
    columns.append(group + " training time mean s")
    columns.append(group + " training time std s")

# take only a subset of of_df with "Sex"=m/f:
for curr_sex in ["male", "female"]:
    # curr_sex = "m"
    curr_mwm_training_df = mwm_training_df[mwm_training_df["Sex"]==curr_sex]

    mwm_training_eval_df = pd.DataFrame(columns=columns)
    for day in range(1,6):
        # day = 1
        mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
        group_Ns = []
        for group in groups:
            # Latency to platform Platform / Center-point Latency to Last s:
            mwm_training_df_tmp[group+ " training time mean s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].mean()
            mwm_training_df_tmp[group+ " training time std s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].std()
            mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0]
            group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0])
        mwm_training_df_tmp["day"] = day
        
        if len(groups)==2:
            A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
            B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
            Significance = independent_two_sample_test(A, B)
            mwm_training_df_tmp["pval"] = Significance["pval"].values
            mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
            
        mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
    mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, f"MWM_training_eval {curr_sex}.xlsx"))

    if len(groups)>2:
        plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname=f"MWM training days ({curr_sex})",
                                ylabel="average time to\nreach the platform ("+curr_sex+") [s]",
                                ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.75, 5.25),
                                labels=groups, figaspect=(4.9,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17,
                                SEM=True,boxborder_width=1.5)
        
        # test, if the groups are normally distributed:
        norm_pvals = []
        for group in groups:
            # group = groups[0]
            curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
            pval = pg.normality(curr_df, dv="Latency to platform Platform / Center-point Latency to Last s", 
                                group="Day")["normal"].values
            norm_pvals.append(list(pval))
        # flatten norm_pvals:
        norm_pvals = [item for sublist in norm_pvals for item in sublist]
        if False in norm_pvals:
            parametric = False
            # perform a kruskal-wallis test:
            kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                    dv="Latency to platform Platform / Center-point Latency to Last s", 
                                    between="Group")
            # correct p-unc for multiple comparisons:
            kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test =kruskal_test["p-corr"].values
                                    
        else:
            parametric = True
            rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                        dv="Latency to platform Platform / Center-point Latency to Last s", 
                                        within="Day", subject="ID", between="Group")
            rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test = rm_anova_test["p-corr"][2]
            
        pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                    dv="Latency to platform Platform / Center-point Latency to Last s",
                                    within="Day", subject="ID", parametric = parametric,
                                    between="Group", padjust='fdr_bh',
                                    effsize='cohen', correction=True, within_first=True)
        
        pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
        pttest["p-corr"]
        pttest["Normality"] = False if parametric==False in norm_pvals else True
        pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
        pttest["Omnibus-test p-value"] = prehoc_test[0]
        pttest.to_excel(os.path.join(RESULTS_excel_path, f"MWM_training_pttest {curr_sex}.xlsx"))
    elif len(groups)==2:
        plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname=f"MWM training days ({curr_sex})",
                                ylabel="average time to\nreach the platform ("+curr_sex+") [s]",
                                ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                                Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                                SEM=True,boxborder_width=1.5)


    # plot individual groups with their intra-group statistics:
    for group_i, group in enumerate(groups):
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
        pval = pg.rm_anova(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                    within="Day", subject="ID", correction=True)["p-GG-corr"].values
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                        between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                        effsize='cohen', correction=True)
        """ if pval<0.05:
            sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                        between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                        effsize='cohen', correction=True)
        else:
            sig_test=False """
        plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                                plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                                significance=sig_test,
                                ylabel=f"average time to\nreach the platform ({curr_sex}) [s]",
                                title="MWM training days "+group.replace("/", " ") + " (" + curr_sex +")")

    # plot heatmap:
    plt.figure(figsize=(6, 3))
    plt.rcParams.update({'font.size': 12})
    eval_columns = ["day"]
    for group in groups:
        eval_columns.append(group + " training time mean s")
    curr_heatmap_df = mwm_training_eval_df[eval_columns].set_index('day')
    curr_heatmap_df.columns = [col.replace(" training time mean s", "") for col in curr_heatmap_df.columns]
    curr_heatmap_df = curr_heatmap_df.T
    ax = sns.heatmap(curr_heatmap_df, cmap='coolwarm', annot=True, vmin=0, vmax=60,
                cbar_kws={'label': 'average time to reach\nthe platform (lower=better) [s]'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=00)
    plt.title(f'Group training performance ({curr_sex})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_path, f'MWM heatmap ({curr_sex}).pdf'), dpi=120)
# %% EVALUATE TRAINING DAY (LATENCY) (M/F) (Z-SCORED)
columns = ["day", "pval", "cohen d"]
for group in groups:
    columns.append(group + " training time mean s")
    columns.append(group + " training time std s")

# take only a subset of of_df with "Sex"=m/f:
for curr_sex in ["male", "female"]:
    # curr_sex = "m"
    curr_mwm_training_df = mwm_training_df_zscored[mwm_training_df_zscored["Sex"]==curr_sex]

    mwm_training_eval_df = pd.DataFrame(columns=columns)
    for day in range(1,6):
        # day = 1
        mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
        group_Ns = []
        for group in groups:
            # Latency to platform Platform / Center-point Latency to Last s:
            mwm_training_df_tmp[group+ " training time mean s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].mean()
            mwm_training_df_tmp[group+ " training time std s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].std()
            mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0]
            group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"].shape[0])
        mwm_training_df_tmp["day"] = day
        
        if len(groups)==2:
            A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
            B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
            Significance = independent_two_sample_test(A, B)
            mwm_training_df_tmp["pval"] = Significance["pval"].values
            mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
            
        mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
    mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, f"MWM_training_eval {curr_sex} zscored.xlsx"))

    if len(groups)>2:
        plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname=f"MWM training days ({curr_sex}) zscored",
                                ylabel="average time to reach\nthe platform ("+curr_sex+") [z-scored]",
                                ylim=(-1, 1.5), yticks=np.arange(-1, 1.6, 0.3), xlim=(0.75, 5.25),
                                labels=groups, figaspect=(4.9,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17,
                                SEM=True,boxborder_width=1.5)
        
        # test, if the groups are normally distributed:
        norm_pvals = []
        for group in groups:
            # group = groups[0]
            curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
            pval = pg.normality(curr_df, dv="Latency to platform Platform / Center-point Latency to Last s", 
                                group="Day")["normal"].values
            norm_pvals.append(list(pval))
        # flatten norm_pvals:
        norm_pvals = [item for sublist in norm_pvals for item in sublist]
        if False in norm_pvals:
            parametric = False
            # perform a kruskal-wallis test:
            kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                    dv="Latency to platform Platform / Center-point Latency to Last s", 
                                    between="Group")
            # correct p-unc for multiple comparisons:
            kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test =kruskal_test["p-corr"].values
                                    
        else:
            parametric = True
            rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                        dv="Latency to platform Platform / Center-point Latency to Last s", 
                                        within="Day", subject="ID", between="Group")
            rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test = rm_anova_test["p-corr"][2]
            
        pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                    dv="Latency to platform Platform / Center-point Latency to Last s",
                                    within="Day", subject="ID", parametric = parametric,
                                    between="Group", padjust='fdr_bh',
                                    effsize='cohen', correction=True, within_first=True)
        
        pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
        pttest["p-corr"]
        pttest["Normality"] = False if parametric==False in norm_pvals else True
        pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
        pttest["Omnibus-test p-value"] = prehoc_test[0]
        pttest.to_excel(os.path.join(RESULTS_excel_path, f"MWM_training_pttest {curr_sex} zscored.xlsx"))
    elif len(groups)==2:
        plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname=f"MWM training days ({curr_sex}) z-scored",
                                ylabel="average time to\nreach the platform ("+curr_sex+") [z-scored]",
                                ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                                Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                                SEM=True,boxborder_width=1.5)


    # plot individual groups with their intra-group statistics:
    for group_i, group in enumerate(groups):
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Latency to platform Platform / Center-point Latency to Last s"]]
        pval = pg.rm_anova(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                    within="Day", subject="ID", correction=True)["p-GG-corr"].values
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Latency to platform Platform / Center-point Latency to Last s",
                                        between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                        effsize='cohen', correction=True)
        plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                                plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                                significance=sig_test,
                                ylabel=f"average time to reach\nthe platform ({curr_sex}) [z-scored]",
                                title="MWM training days "+group.replace("/", " ") + " (" + curr_sex +") zscored",
                                ylim=(-1.2, 1.9), yticks=np.arange(-1, 1.6, 0.3),
                                statsoffset=0.1, bar_offset=0.15)

    # plot heatmap:
    plt.figure(figsize=(6, 0+len(groups)))
    plt.rcParams.update({'font.size': 12})
    eval_columns = ["day"]
    for group in groups:
        eval_columns.append(group + " training time mean s")
    curr_heatmap_df = mwm_training_eval_df[eval_columns].set_index('day')
    curr_heatmap_df.columns = [col.replace(" training time mean s", "") for col in curr_heatmap_df.columns]
    curr_heatmap_df = curr_heatmap_df.T
    ax = sns.heatmap(curr_heatmap_df, cmap='coolwarm', annot=True, vmin=-1, vmax=1,
                cbar_kws={'label': 'average time to reach\nthe platform (lower=better)\n[z-scored]'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=00)
    plt.title(f'Group training performance ({curr_sex}) z-scored')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_path, f'MWM heatmap ({curr_sex}) zscored.pdf'), dpi=120)
# %% EVALUATE TRAINING DAY (DISTANCE) CURRENTLY NOT USED
""" columns = ["day", "pval", "cohen d"]
for group in groups:
    columns.append(group + " traveled distance mean cm")
    columns.append(group + " traveled distance std cm")

mwm_training_eval_df = pd.DataFrame(columns=columns)
for day in range(1,6):
    # day = 1
    mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
    for group in groups:
        mwm_training_df_tmp[group+ " traveled distance mean cm"] = mwm_training_df[mwm_training_df["Day"]==day][mwm_training_df[mwm_training_df["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"].mean()
        mwm_training_df_tmp[group+ " traveled distance std cm"]  = mwm_training_df[mwm_training_df["Day"]==day][mwm_training_df[mwm_training_df["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"].std()
        mwm_training_df_tmp[group+ " N_samples"]  = mwm_training_df[mwm_training_df["Day"]==day][mwm_training_df[mwm_training_df["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"].shape[0]
    A = mwm_training_df[mwm_training_df["Day"]==day][mwm_training_df[mwm_training_df["Day"]==day]["Group"]==groups[0]]["Total Distance moved center-point Total cm"]
    B = mwm_training_df[mwm_training_df["Day"]==day][mwm_training_df[mwm_training_df["Day"]==day]["Group"]==groups[1]]["Total Distance moved center-point Total cm"]
    Significance = independent_two_sample_test(A, B)
    mwm_training_df_tmp["day"] = day
    mwm_training_df_tmp["pval"] = Significance["pval"].values
    mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
    mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)

plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                          plotname="MWM training days – distance",
                          ylabel="average traveled distance [cm]",
                          dv=" traveled distance mean cm",
                          dv2=" traveled distance std cm",
                          ylim=(0, 1000), yticks=np.arange(0, 1001, 200),
                          xlim=(0.55, 5.45), figaspect=(4.5,4),
                          Alabel=groups[0], Blabel=groups[1], 
                          plotsavepath=RESULTS_path, 
                          fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                          SEM=True,boxborder_width=1.5, statsOffset=950)
fit_and_plot_2_samples_timeseries(df_in=mwm_training_df, fignum=1,
                                  title="MWM training days – distance\n(fitted)",
                                  dv="Total Distance moved center-point Total cm",
                                  ylabel="average traveled distance [cm]",
                                  Alabel=groups[0], Blabel=groups[1], figaspect=(5,4),
                                  ylim=(0, 1000), yticks=np.arange(0, 1001, 100),
                                  plotsavepath=RESULTS_path, colors=colors,
                                  fontsize_axes=16,fontsize_labels=16)
fit_and_plot_2_samples_timeseries(df_in=mwm_training_df, fignum=1,
                                  title="MWM training days – distance\n(fitted, 1D)",
                                  dv="Total Distance moved center-point Total cm",
                                  ylabel="average traveled distance [cm]",
                                  Alabel=groups[0], Blabel=groups[1], figaspect=(5,4), order=1,
                                  ylim=(0, 1000), yticks=np.arange(0, 1001, 100),
                                  plotsavepath=RESULTS_path, colors=colors,
                                  fontsize_axes=16, fontsize_labels=16) """
# %% EVALUATE TRAINING DAY (VELOCITY)
curr_mwm_training_df = mwm_training_df.copy()
columns = ["day"]
for group in groups:
    columns.append(group + " velocity mean cm/s")
    columns.append(group + " velocity std cm/s")

mwm_training_eval_df = pd.DataFrame(columns=columns)
for day in range(1,6):
    # day = 1
    mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
    group_Ns = []
    for group in groups:
        # Latency to platform Platform / Center-point Latency to Last s
        mwm_training_df_tmp[group+ " velocity mean cm/s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].mean()
        mwm_training_df_tmp[group+ " velocity std cm/s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].std()
        mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].shape[0]
        group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].shape[0])
    mwm_training_df_tmp["day"] = day
    
    if len(groups)==2:
        A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Velocity [cm/s]"]
        B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Velocity [cm/s]"]
        Significance = independent_two_sample_test(A, B)
        mwm_training_df_tmp["pval"] = Significance["pval"].values
        mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
        
    mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_eval_velocity.xlsx"))

if len(groups)>2:
    plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days velocity",
                            ylabel="average velocity (pooled) [cm/s]",
                            dv=" velocity mean cm/s",
                            dv2=" velocity std cm/s",
                            ylim=(0, 28), yticks=np.arange(0, 28, 3), xlim=(0.75, 5.25),
                            labels=groups, figaspect=(4.9,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17,
                            SEM=True,boxborder_width=1.5)
    
    # test, if the groups are normally distributed:
    norm_pvals = []
    for group in groups:
        # group = groups[0]
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Velocity [cm/s]"]]
        pval = pg.normality(curr_df, dv="Velocity [cm/s]", 
                            group="Day")["normal"].values
        norm_pvals.append(list(pval))
    # flatten norm_pvals:
    norm_pvals = [item for sublist in norm_pvals for item in sublist]
    if False in norm_pvals:
        parametric = False
        # perform a kruskal-wallis test:
        kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                 dv="Velocity [cm/s]", 
                                 between="Group")
        # correct p-unc for multiple comparisons:
        kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test =kruskal_test["p-corr"].values[0]
                                 
    else:
        parametric = True
        rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                     dv="Velocity [cm/s]", 
                                     within="Day", subject="ID", between="Group")
        rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
        prehoc_test = rm_anova_test["p-corr"][2]
        

    pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                dv="Velocity [cm/s]",
                                within="Day", subject="ID", parametric = parametric,
                                between="Group", padjust='fdr_bh',
                                effsize='cohen', correction=True, within_first=True)
    
    pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
    pttest["p-corr"]
    pttest["Normality"] = False if parametric==False in norm_pvals else True
    pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
    pttest["Omnibus-test p-value"] = prehoc_test#[0]
    pttest.to_excel(os.path.join(RESULTS_excel_path, "MWM_training_velocity_pttest.xlsx"))
elif len(groups)==2:
    plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days velocity",
                            ylabel="average velocity [cm/s]",
                            ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                            Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                            SEM=True,boxborder_width=1.5)

# plot individual groups with their intra-group statistics:
for group_i, group in enumerate(groups):
    curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Velocity [cm/s]"]]
    pval = pg.rm_anova(data=curr_df, dv="Velocity [cm/s]",
                within="Day", subject="ID", correction=True)["p-GG-corr"].values
    sig_test = pg.pairwise_ttests(data=curr_df, dv="Velocity [cm/s]",
                                    between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                    effsize='cohen', correction=True)

    plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                              dv=" velocity mean cm/s", dv2=" velocity std cm/s",
                            plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                            significance=sig_test,
                            ylim=(0, 68), yticks=np.arange(0, 68, 10),
                            ylabel="average velocity (pooled) [cm/s]",
                            title="MWM training days velocity "+group.replace("/", " "))

# %% EVALUATE TRAINING DAY (VELOCITY) (M/F)
curr_mwm_training_df = mwm_training_df.copy()
columns = ["day"]
for group in groups:
    columns.append(group + " velocity mean cm/s")
    columns.append(group + " velocity std cm/s")

# take only a subset of of_df with "Sex"=m/f:
for curr_sex in ["male", "female"]:
    #curr_sex="m"
    curr_mwm_training_df = mwm_training_df[mwm_training_df["Sex"]==curr_sex]

    mwm_training_eval_df = pd.DataFrame(columns=columns)

    for day in range(1,6):
        # day = 1
        mwm_training_df_tmp = pd.DataFrame(columns=columns, index=[0])
        group_Ns = []
        for group in groups:
            # Latency to platform Platform / Center-point Latency to Last s
            mwm_training_df_tmp[group+ " velocity mean cm/s"] = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].mean()
            mwm_training_df_tmp[group+ " velocity std cm/s"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].std()
            mwm_training_df_tmp[group+ " N_samples"]  = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].shape[0]
            group_Ns.append(curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==group]["Velocity [cm/s]"].shape[0])
        mwm_training_df_tmp["day"] = day
        
        if len(groups)==2:
            A = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[0]]["Velocity [cm/s]"]
            B = curr_mwm_training_df[curr_mwm_training_df["Day"]==day][curr_mwm_training_df[curr_mwm_training_df["Day"]==day]["Group"]==groups[1]]["Velocity [cm/s]"]
            Significance = independent_two_sample_test(A, B)
            mwm_training_df_tmp["pval"] = Significance["pval"].values
            mwm_training_df_tmp["cohen d"] = Significance["cohen d"].values
            
        mwm_training_eval_df = pd.concat([mwm_training_eval_df, mwm_training_df_tmp], ignore_index=True)
    mwm_training_eval_df.to_excel(os.path.join(RESULTS_excel_path, f"MWM_training_eval_velocity_{curr_sex}.xlsx"))

    if len(groups)>2:
        plot_N_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname=f"MWM training days velocity {curr_sex}",
                                ylabel=f"average velocity ({curr_sex}) [cm/s]",
                                dv=" velocity mean cm/s",
                                dv2=" velocity std cm/s",
                                ylim=(0, 28), yticks=np.arange(0, 28, 3), xlim=(0.75, 5.25),
                                labels=groups, figaspect=(4.9,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17,
                                SEM=True,boxborder_width=1.5)
        
        # test, if the groups are normally distributed:
        norm_pvals = []
        for group in groups:
            # group = groups[0]
            curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Velocity [cm/s]"]]
            pval = pg.normality(curr_df, dv="Velocity [cm/s]", 
                                group="Day")["normal"].values
            norm_pvals.append(list(pval))
        # flatten norm_pvals:
        norm_pvals = [item for sublist in norm_pvals for item in sublist]
        if False in norm_pvals:
            parametric = False
            # perform a kruskal-wallis test:
            kruskal_test = pg.kruskal(data=curr_mwm_training_df, 
                                    dv="Velocity [cm/s]", 
                                    between="Group")
            # correct p-unc for multiple comparisons:
            kruskal_test["p-corr"] = pg.multicomp(kruskal_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test =kruskal_test["p-corr"].values[0]
                                    
        else:
            parametric = True
            rm_anova_test = pg.mixed_anova(data=curr_mwm_training_df, 
                                        dv="Velocity [cm/s]", 
                                        within="Day", subject="ID", between="Group")
            rm_anova_test["p-corr"] = pg.multicomp(rm_anova_test["p-unc"].values, method="fdr_bh")[1]
            prehoc_test = rm_anova_test["p-corr"][2]
            

        pttest = pg.pairwise_ttests(data=curr_mwm_training_df, 
                                    dv="Velocity [cm/s]",
                                    within="Day", subject="ID", parametric = parametric,
                                    between="Group", padjust='fdr_bh',
                                    effsize='cohen', correction=True, within_first=True)
        
        pg.multicomp(pttest["p-unc"][13:28].values, method="fdr_bh")[1]
        pttest["p-corr"]
        pttest["Normality"] = False if parametric==False in norm_pvals else True
        pttest["Omnibus-test type"] = "Kruskal-Wallis" if parametric==False else "ANOVA"
        pttest["Omnibus-test p-value"] = prehoc_test#[0]
        pttest.to_excel(os.path.join(RESULTS_excel_path,f"MWM_training_velocity_pttest_{curr_sex}.xlsx"))
    elif len(groups)==2:
        plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                                plotname="MWM training days velocity",
                                ylabel="average velocity [cm/s]",
                                ylim=(0, 64), yticks=np.arange(0, 61, 10), xlim=(0.55, 5.45),
                                Alabel=groups[0], Blabel=groups[1], figaspect=(4.5,4),
                                plotsavepath=RESULTS_path, 
                                fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                                SEM=True,boxborder_width=1.5)

    # plot individual groups with their intra-group statistics:
    for group_i, group in enumerate(groups):
        curr_df = curr_mwm_training_df[curr_mwm_training_df["Group"]==group][["ID", "Day", "Velocity [cm/s]"]]
        pval = pg.rm_anova(data=curr_df, dv="Velocity [cm/s]",
                    within="Day", subject="ID", correction=True)["p-GG-corr"].values
        sig_test = pg.pairwise_ttests(data=curr_df, dv="Velocity [cm/s]",
                                        between=None, within="Day", subject="ID", parametric=True, padjust='fdr_bh',
                                        effsize='cohen', correction=True)

        plot_1_samples_timeseries(df_in=mwm_training_eval_df, fignum=1, figaspect=(6,6),
                                dv=" velocity mean cm/s", dv2=" velocity std cm/s",
                                plotsavepath=RESULTS_path, group_to_eval=group_i, groups=groups,
                                significance=sig_test,
                                ylim=(0, 68), yticks=np.arange(0, 68, 10),
                                ylabel=f"average velocity ({curr_sex}) [cm/s]",
                                title="MWM training days velocity "+group.replace("/", " ") + f" {curr_sex}")

# %% EVALUATE TRAINING DAY (VELOCITY) (M/F) CURRENTLY NOT USED
""" columns = ["day", "pval", "cohen d"]
for group in groups:
    columns.append(group + " velocity mean cm/s")
    columns.append(group + " velocity std cm/s")

# take only a subset of of_df with "Sex"=m/f:
for curr_sex in ["male", "female"]:
    curr_of_df_sub = mwm_training_df[mwm_training_df["Sex"]==curr_sex]

    mwm_training_eval_df = pd.DataFrame(columns=columns)
    for day in range(1,6):
        # day = 1
        curr_of_df_sub_tmp = pd.DataFrame(columns=columns, index=[0])
        for group in groups:
            curr_of_df_sub_tmp[group+ " velocity mean cm/s"] = (curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"]/ \
                                                                curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"]).mean()
            curr_of_df_sub_tmp[group+ " velocity std cm/s"]  = (curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"]/ \
                                                                curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==group]["Latency to platform Platform / Center-point Latency to Last s"]).std()
            curr_of_df_sub_tmp[group+ " N_samples"]  = curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==group]["Total Distance moved center-point Total cm"].shape[0]
        A = curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==groups[0]]["Total Distance moved center-point Total cm"]/ \
            curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==groups[0]]["Latency to platform Platform / Center-point Latency to Last s"]
        B = curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==groups[1]]["Total Distance moved center-point Total cm"]/ \
            curr_of_df_sub[curr_of_df_sub["Day"]==day][curr_of_df_sub[curr_of_df_sub["Day"]==day]["Group"]==groups[1]]["Latency to platform Platform / Center-point Latency to Last s"]
        Significance = independent_two_sample_test(A, B)
        curr_of_df_sub_tmp["day"] = day
        curr_of_df_sub_tmp["pval"] = Significance["pval"].values
        curr_of_df_sub_tmp["cohen d"] = Significance["cohen d"].values
        mwm_training_eval_df = pd.concat([mwm_training_eval_df, curr_of_df_sub_tmp], ignore_index=True)

    plot_2_samples_timeseries(df_in=mwm_training_eval_df, fignum=1,
                            plotname="MWM training days – velocity "+curr_sex,
                            ylabel="average velocity ("+curr_sex+") [cm/s]",
                            dv=" velocity mean cm/s",
                            dv2=" velocity std cm/s",
                            ylim=(0, 20), yticks=np.arange(0, 21, 2),statsOffset=18.75,
                            xlim=(0.55, 5.45), figaspect=(4.5,4),
                            Alabel=groups[0], Blabel=groups[1], 
                            plotsavepath=RESULTS_path, 
                            fontsize_axes=17, fontsize_labels=17, fontsize_stats=17, 
                            SEM=True,boxborder_width=1.5, legend_loc="lower right")
"""
""" fit_and_plot_2_samples_timeseries(df_in=curr_of_df_sub, fignum=1,
                                    title="MWM training days – velocity\n(fitted)",
                                    dv=" velocity mean cm/s",
                                    ylabel="average velocity [s]",
                                    Alabel=groups[0], Blabel=groups[1], figaspect=(5,4),
                                    ylim=(0, 20), yticks=np.arange(0, 21, 2),
                                    plotsavepath=RESULTS_path, colors=colors,
                                    fontsize_axes=16,fontsize_labels=16)
    fit_and_plot_2_samples_timeseries(df_in=curr_of_df_sub, fignum=1,
                                    title="MWM training days – velocity\n(fitted, 1D)",
                                    dv=" velocity mean cm/s",
                                    ylabel="average velocity [s]",
                                    Alabel=groups[0], Blabel=groups[1], figaspect=(5,4), order=1,
                                    ylim=(0, 20), yticks=np.arange(0, 21, 2),
                                    plotsavepath=RESULTS_path, colors=colors,
                                    fontsize_axes=16, fontsize_labels=16) """
# %% EVALUATE PROBE DAYS
quadrants = ["In quadrants Northwest / Center-point Cumulative Duration s",
             "In quadrants Northeast / Center-point Cumulative Duration s",
             "In quadrants Southwest / Center-point Cumulative Duration s",
             "In quadrants Southeast / Center-point Cumulative Duration s"]
quadrants_clean = ["NW quadrant", "NE quadrant", "SW quadrant", "SE quadrant"]

A_arena = np.pi*84**2
A_platform = np.pi*10**2
A_platformzone = np.pi*15**2
pchance_A_platform = 100*A_platform/A_arena
pchance_A_platformzone = 100*A_platformzone/A_arena

probe_day = 6

# extract the time spent in each quadrant for each group and plot it:
figaspect = (3.8, 5)
ylim = (0,70)
yticks = np.arange(0, 61, 10)
statsOffset = 2.5
barOffset = 5
clevel = 25
plot_clevel_stats=True
clevel_stats_offset=15
violin_width = 0.70
detect_outliers = True
# the plots:
for quadrant in range(4):
    # quadrant = 0
    # prepare the dataframe for plotting:
    mwm_probe_quadrant_times_df = pd.DataFrame()
    group_Ns = []
    groups_rename = []
    for group in groups:
        # group = groups[0]
        mwm_probe_df_tmp = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][quadrants[quadrant]].values /\
            mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values * 100
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", f"{quadrants_clean[quadrant]}"])
        temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[f"{quadrants_clean[quadrant]}"] = mwm_probe_df_tmp
        mwm_probe_quadrant_times_df = pd.concat([mwm_probe_quadrant_times_df, temp_df], ignore_index=True)
        groups_rename.append(group + "\nN="+str(len(mwm_probe_df_tmp)))
    # plot the data:
    plot_N_samples_pub(df_in=mwm_probe_quadrant_times_df,
                    variable=quadrants_clean[quadrant],
                    plotname="MWM "+quadrants_clean[quadrant] + " probe day",
                    groups=groups,
                    Group_Styles=Group_Styles2,
                    fignum=1,
                    figaspect=figaspect,
                    ylim=ylim,
                    yticks=yticks,
                    ylabel='average cumulative duration\n'+quadrants_clean[quadrant]+' [%]',
                    xlabel_rot=55,
                    plotsavepath=RESULTS_path,
                    excelpath=RESULTS_excel_path,
                    violin_width=violin_width,
                    show_stats=True,
                    multicomp=False,
                    show_stats_ns=True,
                    stats_text_correct=0,
                    statsOffset=statsOffset,
                    barOffset=barOffset,
                    clevel=clevel,
                    clevel_stats_offset=clevel_stats_offset,
                    plot_clevel_stats=plot_clevel_stats,
                    swarm_tol=0.04,
                    swarm_offset = 0.10,
                    swarm_dots_size=45,
                    swarm_dots_alpha=0.75,
                    detect_outliers=detect_outliers,
                    fontsize_axes=17,
                    fontsize_labels=17,
                    boxborder_width=1,
                    plot_median=False,
                    show_title=False,
                    fontweight="normal")

    
    # pttest.to_excel(os.path.join(RESULTS_excel_path, "MWM "+quadrants_clean[quadrant] + " probe day stats.xlsx"))
    # mwm_probe_quadrant_times_df.to_excel(os.path.join(RESULTS_excel_path, "MWM "+quadrants_clean[quadrant] + " probe day data.xlsx"))

# plot further probe day data:
mwm_probe_platform_times_df = pd.DataFrame()
mwm_probe_platform_zone_times_df = pd.DataFrame()
mwm_probe_platform_freq_df = pd.DataFrame()
mwm_probe_platform_zone_freq_df = pd.DataFrame()
mwm_probe_platform_latency_to_first = pd.DataFrame()
mwm_probe_platform_zone_latency_to_first = pd.DataFrame()
# first, collect data for each group:
for group in groups:
    # group = groups[0]
    variable = "Platform Platform / Center-point Cumulative Duration s"
    mwm_probe_df_tmp = 100 * mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values /\
                       mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_times_df = pd.concat([mwm_probe_platform_times_df, temp_df], ignore_index=True)
    
    
    variable = "Platform Platform Zone / Center-point Cumulative Duration s"
    mwm_probe_df_tmp = 100 * mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values /\
                       mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_zone_times_df = pd.concat([mwm_probe_platform_zone_times_df, temp_df], ignore_index=True)
    
    
    variable = "Platform Platform / Center-point Frequency"
    mwm_probe_df_tmp = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_freq_df = pd.concat([mwm_probe_platform_freq_df, temp_df], ignore_index=True)
    
    variable = "Platform Platform Zone / Center-point Frequency"
    mwm_probe_df_tmp = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_zone_freq_df = pd.concat([mwm_probe_platform_zone_freq_df, temp_df], ignore_index=True)
    
    variable = "Latency to platform Platform / Center-point Latency to First s"
    mwm_probe_df_tmp = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_latency_to_first = pd.concat([mwm_probe_platform_latency_to_first, temp_df], ignore_index=True)
    
    variable = "Latency to platform Platform Zone / Center-point Latency to First s"
    mwm_probe_df_tmp = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group][variable].values
    # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
    temp_df = pd.DataFrame(columns=["ID", "Group", variable])
    temp_df["ID"] = mwm_df[mwm_df["Day"]==probe_day][mwm_df[mwm_df["Day"]==probe_day]["Group"]==group]["ID"].values
    temp_df["Group"] = group
    temp_df[variable] = mwm_probe_df_tmp
    mwm_probe_platform_zone_latency_to_first = pd.concat([mwm_probe_platform_zone_latency_to_first, temp_df], ignore_index=True)
    
# plot the data:
variable = "Platform Platform / Center-point Cumulative Duration s"
plotname="MWM probe day platform times"
ylabel = 'average cumulative duration\nspent at former platform location [%]'
ylim=(0,10)
yticks=np.arange(0, 11, 1)
statsOffset=0.1
barOffset=0.75
clevel=pchance_A_platform
clevel_stats_offset=2.2
curr_df = mwm_probe_platform_times_df
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")


variable = "Platform Platform Zone / Center-point Cumulative Duration s"
plotname="MWM probe day platform zone times"
ylabel = 'average cumulative duration\nspent in platform zone [%]'
ylim=(0,10)
yticks=np.arange(0, 18, 1)
statsOffset=0.1
barOffset=0.75
clevel=pchance_A_platformzone
clevel_stats_offset=9.5
curr_df = mwm_probe_platform_zone_times_df
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

variable = "Platform Platform / Center-point Frequency"
plotname="MWM probe day platform frequency"
ylabel = 'platform crossing frequency'
ylim=(0,10)
yticks=np.arange(0, 16, 1)
statsOffset=0.1
barOffset=0.7
clevel=False
curr_df = mwm_probe_platform_freq_df
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

variable = "Platform Platform Zone / Center-point Frequency"
plotname="MWM probe day platform zone frequency"
ylabel = 'platform zone crossing frequency'
ylim=(0,22)
yticks=np.arange(0, 22, 2)
statsOffset=0.1
barOffset=1.5
clevel=False
curr_df = mwm_probe_platform_zone_freq_df
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

variable = "Latency to platform Platform / Center-point Latency to First s"
plotname="MWM probe day platform latency to first"
ylabel = 'latency to first platform crossing [s]'
ylim=(0,70)
yticks=np.arange(0, 61, 5)
statsOffset=0.5
barOffset=5.5
clevel=False
curr_df = mwm_probe_platform_latency_to_first
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

variable = "Latency to platform Platform Zone / Center-point Latency to First s"
plotname="MWM probe day platform zone latency to first"
ylabel = 'latency to first\nplatform zone crossing [s]'
ylim=(0,70)
yticks=np.arange(0, 61, 5)
statsOffset=0.5
barOffset=5.5
clevel=False
curr_df = mwm_probe_platform_zone_latency_to_first
plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")
# %% EVALUATE PROBE DAYS (M/F)
probe_day = 6

# extract the time spent in each quadrant for each group and plot it:
# the sex-specific plots:
for curr_sex in ["male", "female"]:
    # curr_sex="m"
    figaspect = (3.8, 5)
    ylim = (0,70)
    yticks = np.arange(0, 61, 10)
    statsOffset = 2.5
    barOffset = 5
    clevel = 25
    plot_clevel_stats=True
    clevel_stats_offset=15
    violin_width = 0.70
    detect_outliers = True
    
    mwm_df_sub = mwm_df[mwm_df["Sex"]==curr_sex]
    # extract the time spent in each quadrant for each group:
    for quadrant in range(4):
        # quadrant = 0
        # prepare the dataframe for plotting:
        mwm_probe_quadrant_times_df = pd.DataFrame()
        group_Ns = []
        groups_rename = []
        for group in groups:
            # group = groups[0]
            mwm_probe_df_tmp = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][quadrants[quadrant]].values /\
                               mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values * 100
            # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
            temp_df = pd.DataFrame(columns=["ID", "Group", f"{quadrants_clean[quadrant]}"])
            temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
            temp_df["Group"] = group
            temp_df[f"{quadrants_clean[quadrant]}"] = mwm_probe_df_tmp
            mwm_probe_quadrant_times_df = pd.concat([mwm_probe_quadrant_times_df, temp_df], ignore_index=True)
            groups_rename.append(group + "\nN="+str(len(mwm_probe_df_tmp)))
        # plot the data:
        plot_N_samples_pub(df_in=mwm_probe_quadrant_times_df,
                            variable=quadrants_clean[quadrant],
                            plotname="MWM "+quadrants_clean[quadrant] + f" probe day ({curr_sex})",
                            groups=groups,
                            Group_Styles=Group_Styles2,
                            fignum=1,
                            figaspect=figaspect,
                            ylim=ylim,
                            yticks=yticks,
                            ylabel='average cumulative duration\n'+quadrants_clean[quadrant]+f' ({curr_sex}) [%]',
                            xlabel_rot=55,
                            plotsavepath=RESULTS_path,
                            excelpath=RESULTS_excel_path,
                            violin_width=violin_width,
                            show_stats=True,
                            multicomp=False,
                            show_stats_ns=True,
                            stats_text_correct=0,
                            statsOffset=statsOffset,
                            barOffset=barOffset,
                            clevel=clevel,
                            clevel_stats_offset=clevel_stats_offset,
                            plot_clevel_stats=plot_clevel_stats,
                            swarm_tol=0.04,
                            swarm_offset = 0.10,
                            swarm_dots_size=45,
                            swarm_dots_alpha=0.75,
                            detect_outliers=detect_outliers,
                            fontsize_axes=17,
                            fontsize_labels=17,
                            boxborder_width=1,
                            plot_median=False,
                            show_title=False,
                            fontweight="normal")

    # plot further probe day data:
    mwm_probe_platform_times_df = pd.DataFrame()
    mwm_probe_platform_zone_times_df = pd.DataFrame()
    mwm_probe_platform_freq_df = pd.DataFrame()
    mwm_probe_platform_zone_freq_df = pd.DataFrame()
    mwm_probe_platform_latency_to_first = pd.DataFrame()
    mwm_probe_platform_zone_latency_to_first = pd.DataFrame()
    # first, collect data for each group:
    for group in groups:
        # group = groups[0]
        variable = "Platform Platform / Center-point Cumulative Duration s"
        mwm_probe_df_tmp = 100 * mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values /\
                           mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_times_df = pd.concat([mwm_probe_platform_times_df, temp_df], ignore_index=True)
        
        
        variable = "Platform Platform Zone / Center-point Cumulative Duration s"
        mwm_probe_df_tmp = 100 * mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values /\
                           mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["Total Duration Arena / Center-point Cumulative Duration s"].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_zone_times_df = pd.concat([mwm_probe_platform_zone_times_df, temp_df], ignore_index=True)
        
        variable = "Platform Platform / Center-point Frequency"
        mwm_probe_df_tmp = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_freq_df = pd.concat([mwm_probe_platform_freq_df, temp_df], ignore_index=True)
        
        variable = "Platform Platform Zone / Center-point Frequency"
        mwm_probe_df_tmp = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_zone_freq_df = pd.concat([mwm_probe_platform_zone_freq_df, temp_df], ignore_index=True)
        
        variable = "Latency to platform Platform / Center-point Latency to First s"
        mwm_probe_df_tmp = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_latency_to_first = pd.concat([mwm_probe_platform_latency_to_first, temp_df], ignore_index=True)
        
        variable = "Latency to platform Platform Zone / Center-point Latency to First s"
        mwm_probe_df_tmp = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group][variable].values
        # construct a long-format dataframe with columns "ID", "Group" and f"{quadrants[quadrant]}" (=mwm_probe_df_tmp):
        temp_df = pd.DataFrame(columns=["ID", "Group", variable])
        temp_df["ID"] = mwm_df_sub[mwm_df_sub["Day"]==probe_day][mwm_df_sub[mwm_df_sub["Day"]==probe_day]["Group"]==group]["ID"].values
        temp_df["Group"] = group
        temp_df[variable] = mwm_probe_df_tmp
        mwm_probe_platform_zone_latency_to_first = pd.concat([mwm_probe_platform_zone_latency_to_first, temp_df], ignore_index=True)
        
    # plot the data:
    variable = "Platform Platform / Center-point Cumulative Duration s"
    plotname=f"MWM probe day platform times ({curr_sex})"
    ylabel=f'average cumulative duration spent\nat former platform location ({curr_sex}) [%]'
    ylim=(0,10)
    yticks=np.arange(0, 11, 1)
    statsOffset=0.1
    barOffset=0.75
    clevel=pchance_A_platform
    clevel_stats_offset=2.2
    curr_df = mwm_probe_platform_times_df
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

    variable = "Platform Platform Zone / Center-point Cumulative Duration s"
    plotname=f"MWM probe day platform zone times ({curr_sex})"
    ylabel=f'average cumulative duration\nspent in platform zone ({curr_sex}) [%]'
    ylim=(0,10)
    yticks=np.arange(0, 18, 1)
    statsOffset=0.1
    barOffset=0.75
    clevel=pchance_A_platformzone
    clevel_stats_offset=9.5
    curr_df = mwm_probe_platform_zone_times_df
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

    variable = "Platform Platform / Center-point Frequency"
    plotname=f"MWM probe day platform frequency ({curr_sex})"
    ylabel=f'platform crossing frequency ({curr_sex})'
    ylim=(0,10)
    yticks=np.arange(0, 16, 1)
    statsOffset=0.1
    barOffset=0.7
    clevel=False
    curr_df = mwm_probe_platform_freq_df
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

    variable = "Platform Platform Zone / Center-point Frequency"
    plotname=f"MWM probe day platform zone frequency ({curr_sex})"
    ylabel=f'platform zone crossing frequency ({curr_sex})'
    ylim=(0,22)
    yticks=np.arange(0, 22, 2)
    statsOffset=0.1
    barOffset=1.5
    clevel=False
    curr_df = mwm_probe_platform_zone_freq_df
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

    variable = "Latency to platform Platform / Center-point Latency to First s"
    plotname=f"MWM probe day platform latency to first ({curr_sex})"
    ylabel=f'latency to first\nplatform crossing ({curr_sex}) [s]'
    ylim=(0,70)
    yticks=np.arange(0, 61, 5)
    statsOffset=0.5
    barOffset=5.5
    clevel=False
    curr_df = mwm_probe_platform_latency_to_first
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")

    variable = "Latency to platform Platform Zone / Center-point Latency to First s"
    plotname=f"MWM probe day platform zone latency to first ({curr_sex})"
    ylabel=f'latency to first\nplatform zone crossing ({curr_sex}) [s]'
    ylim=(0,70)
    yticks=np.arange(0, 61, 5)
    statsOffset=0.5
    barOffset=5.5
    clevel=False
    curr_df = mwm_probe_platform_zone_latency_to_first
    plot_N_samples_pub(df_in=curr_df,
                variable=variable,
                plotname=plotname,
                groups=groups,
                Group_Styles=Group_Styles2,
                fignum=1,
                figaspect=figaspect,
                ylim=ylim,
                yticks=yticks,
                ylabel=ylabel,
                xlabel_rot=55,
                plotsavepath=RESULTS_path,
                excelpath=RESULTS_excel_path,
                violin_width=violin_width,
                show_stats=True,
                multicomp=False,
                show_stats_ns=True,
                stats_text_correct=0,
                statsOffset=statsOffset,
                barOffset=barOffset,
                clevel=clevel,
                clevel_stats_offset=clevel_stats_offset,
                plot_clevel_stats=plot_clevel_stats,
                swarm_tol=0.04,
                swarm_offset = 0.10,
                swarm_dots_size=45,
                swarm_dots_alpha=0.75,
                detect_outliers=detect_outliers,
                fontsize_axes=17,
                fontsize_labels=17,
                boxborder_width=1,
                plot_median=False,
                show_title=False,
                fontweight="normal")
# %% END