import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
plt.rcParams['axes.axisbelow'] = True

colors = ['darkorange', 'green', 'darkcyan', 'yellowgreen', 'deepskyblue', 'fuchsia', 'darkkhaki']
markers = ['o', 's', '^', 'D', 'h', 'X', '*', 'v']

def gdf_stats(df):
    print(f'Total number of responses: {len(df)}')
    print(f'Total number of finished responses: {df.loc[df.Finished==1].shape[0]}')
    print(f'Range of dates is from {df.StartDate.min()} to {df.StartDate.max()}')
    print(f'Number of unique country or regions: {df.country_agg.nunique()}')
    pass

def gwcrosstab(df, col1, col2):
    num = pd.DataFrame(df[[col1, col2, 'weight']].groupby([col1, col2]).sum()['weight'])
    den = df[[col1, 'weight']].groupby([col1]).sum().reset_index()
    num = num.reset_index().rename(columns={'weight': 'value'})
    val = num.merge(den, on=col1, how='inner')
    val['Prop'] = val['value'] / val['weight']
    q = df.groupby([col1, col2]).count()
    val['Freq'] = pd.Series(q['StartDate'].values)
    val.drop(['value', 'weight'], axis=1, inplace=True)
    val = val[[col1, col2, 'Freq', 'Prop']]
    return val

def cross_plot_3(d, g, a, col, suptitle):
    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    for i, resp in enumerate(d[col].unique()):
        ax[0].scatter(d.loc[d[col]==resp, 'E2'], d.loc[d[col]==resp, 'Prop'], label=resp, marker=markers[i], c=colors[i], s=30)
    ax[0].legend()
    ax[0].grid(which='major', axis='y', color='#DDD', linestyle='--')
    ax[0].set_yticks(np.arange(0,1.1,0.2))
    ax[0].set_title('By area of residence')
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    for i, resp in enumerate(g[col].unique()):
        ax[1].scatter(g.loc[g[col]==resp, 'E3'], g.loc[g[col]==resp, 'Prop'], label=resp, marker=markers[i], c=colors[i], s=30)
    ax[1].legend()
    ax[1].grid(which='major', axis='y', color='#DDD', linestyle='--')
    ax[1].set_yticks(np.arange(0,1.1,0.2))
    ax[1].set_title('By gender')
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    for i, resp in enumerate(a[col].unique()):
        ax[2].scatter(a.loc[a[col]==resp, 'E4'], a.loc[a[col]==resp, 'Prop'], label=resp, marker=markers[i], c=colors[i], s=30)
    ax[2].legend()
    ax[2].grid(which='major', axis='y', color='#DDD', linestyle='--')
    ax[2].set_yticks(np.arange(0,1.1,0.2))
    ax[2].set_title('By age group')
    for tick in ax[2].get_xticklabels():
        tick.set_rotation(90)
    plt.suptitle(suptitle, fontsize=14)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Proportion (weighted)')
    plt.show()
    pass

def worried_label(df, cols):
    for col in cols:
        relab = {col: {1: "Very worried", 2: "Somewhat worried", 3: "Not too worried", 4: "Not worried at all"}}
        df = df.replace(relab)
    return df

grelabel_variables = {"E3": {1: "Male", 2: "Female", 3: "Prefer to self describe", 4: "Prefer not to answer", -99: "NA"},
                     "E4": {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65-74", 7:"75+", -99: "NA" },
                     "E2": {1: "City", 2: "Town", 3: "Rural"},
                     "C4": {1: "All of the time", 2: "Most of the time", 3: "About half the time", 4: "Some of the time", 5: "None of the time", 6: "Have not been in public", -99: "NA"},
                     "C5": {1: "All of the time", 2: "Most of the time", 3: "About half the time", 4: "Some of the time", 5: "None of the time", 6: "Have not been in public", -99: "NA"},
                     "D1": {1: "All the time", 2: "Most of the time", 3: "Some of the time", 4: "A little of the time", 5: "None of the time"},
                     "D2": {1: "All the time", 2: "Most of the time", 3: "Some of the time", 4: "A little of the time", 5: "None of the time"},
                     "B8": {1: "Yes", 2: "No", 3: "I don't know"},
                     "C2": {1: "1-4 people", 2: "5-9 people", 3: "10-19 people", 4: "20 or more people"},
                     "C6": {1: "0 days", 2: "1 day", 3: "2-4 days", 4: "5-7 days"}}
