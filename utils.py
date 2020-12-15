import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
plt.rcParams['axes.axisbelow'] = True

colors = ['darkorange', 'green', 'darkcyan', 'yellowgreen', 'deepskyblue', 'fuchsia', 'darkkhaki']
markers = ['o', 's', '^', 'D', 'h', 'X', '*', 'v']

relabel_variables = {"D1": {1: "Male", 2: "Female", 3: "Non-binary", 4: "Prefer to self describe", 5: "Prefer not to answer"},
                     "D2": {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65-74", 7:"75+" },
                     "B4": {1: "Yes, I had a lot of mucus", 2: "Yes, I had a little mucus", 3: "No, I had a dry cough"},
                     "B5": {1: "Yes, tested positive for COVID-19", 2: "Yes, tested negative for COVID-19",
                            3: "Yes, tested but not received result", 4: "No, tried but not tested",
                            5: "No, not tried to get tested"},
                     "B6": {1: "Yes", 2: "No", 3: "Tried, not received"},
                     "C7": {1: "All of the time", 2: "Most of the time", 3: "Some of the time", 4: "None of the time"},
                     "C8_1": {1: "All of the time", 2: "Most of the time", 3: "Some of the time", 4: "None of the time"},
                     "C8_2": {1: "All of the time", 2: "Most of the time", 3: "Some of the time", 4: "None of the time"},
                     "C9": {1: "Very worried", 2: "Somewhat worried", 3: "Not too worried", 4: "Not worried at all"},
                     "C11": {1: "Yes", 2: "Not to my knowledge"},
                     "D1b": {1: "Yes", 2: "No", 3: "Prefer not to answer", 4: "Not applicable"},
                     "Q36": {1: "A substantial threat", 2: "A moderate threat", 3: "Not much of a threat", 4: "Not a threat at all"}}

def append_files(path_to_files, list_of_files):
    df = pd.read_csv(os.path.join(path_to_files, list_of_files[0]), low_memory=False)
    for file in list_of_files[1:]:
        temp = pd.read_csv(os.path.join(path_to_files, file), low_memory=False, parse_dates=['StartDatetime', 'EndDatetime'])
        df = df.append(temp)
    return df

def yesno_label(df, cols):
    for col in cols:
        relab = {col: {1: "Yes", 2: "No"}}
        df = df.replace(relab)
    return df

def tab(df, col):
    a = pd.Series(df[col].value_counts().index)
    b = pd.Series(df[col].value_counts().values)
    c = df[col].isna().sum()
    den = b.sum()
    if c > 0:
        a = a.append(pd.Series(['NA']))
        b = b.append(pd.Series([c]))
        den += c
    p = b/den
    x = pd.DataFrame({'Cat': a, 'Freq': b, 'Prop': p})
    x['Cat'] = x['Cat'].astype(str)
    x['Prop'] = x['Prop'].round(decimals=4)
    x = x.sort_values(by = 'Cat', na_position='last')
    x = x.reset_index().drop('index', axis=1)
    return x

def wtab(df, x, y='weight'):
    a = pd.Series(df[[x,y]].groupby(x).sum()[y])/df[y].sum()
    b = pd.Series(a.index)
    c = pd.Series(a.values)
    d = pd.Series(df[x].value_counts().sort_index().values)
    sna = df[x].isna().sum()
    if sna > 0:
        pna = 1 - c.sum()
        b = b.append(pd.Series(['NA']))
        c = c.append(pd.Series([pna]))
        d = d.append(pd.Series([sna]))
    df = pd.DataFrame({'Cat': b, 'Freq': d, 'Prop': c})
    df['Cat'] = df['Cat'].astype(str)
    df['Prop'] = df['Prop'].round(decimals=4)
    df = df.sort_values(by = 'Cat', na_position='last')
    df = df.reset_index().drop('index', axis=1)
    return df

def wcrosstab(df, col1, col2):
    num = pd.DataFrame(df[[col1, col2, 'weight']].groupby([col1, col2]).sum()['weight'])
    den = df[[col1, 'weight']].groupby([col1]).sum().reset_index()
    num = num.reset_index().rename(columns={'weight': 'value'})
    val = num.merge(den, on=col1, how='inner')
    val['Prop'] = val['value'] / val['weight']
    q = df.groupby([col1, col2]).count()
    val['Freq'] = pd.Series(q['StartDatetime'].values)
    val.drop(['value', 'weight'], axis=1, inplace=True)
    val = val[[col1, col2, 'Freq', 'Prop']]
    return val

def wtab_by_date(df, col):
    num = pd.DataFrame(df[['StartDate', col, 'weight']].groupby(['StartDate', col]).sum()['weight'])
    den = df[['StartDate', 'weight']].groupby(['StartDate']).sum().reset_index()
    num = num.reset_index().rename(columns={'weight': 'value'})
    val = num.merge(den, on='StartDate', how='inner')
    val['Prop'] = val['value'] / val['weight']
    return val

def wcrosstab_by_date(df, col1, col2):
    num = pd.DataFrame(df[['StartDate', col1, col2, 'weight']].groupby(['StartDate', col1, col2]).sum()['weight'])
    den = df[['StartDate', col1, 'weight']].groupby(['StartDate', col1]).sum().reset_index()
    num = num.reset_index().rename(columns={'weight': 'value'})
    val = num.merge(den, on=['StartDate', col1], how='inner')
    val['Prop'] = val['value'] / val['weight']
    val.drop(['value', 'weight'], axis=1, inplace=True)
    return val

def wstats(df, col):
    num = df[col] * df['weight'].sum()
    den = df['weight'].sum()
    rat = num/den
    print(f'Total number of observations: {df[col].value_counts().values.sum()}')
    print(f'Weighted Mean: {rat.mean().round(decimals = 3)}')
    print(f'Standard Deviation: {rat.std().round(decimals = 3)}')
    print(f'Maximum: {rat.max().round(decimals = 3)}')
    print(f'Minimum: {rat.min().round(decimals = 3)}')
    pass

def missing_data_plot(x, y):
    plt.figure(figsize=(18,6))
    plt.bar(x, y)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xticks(rotation=90)
    plt.title('Missing data by survey item', fontsize=14)
    plt.grid(which='major', axis='y', color='#DDD', linestyle='--')
    plt.ylabel('Proportion missing')
    plt.show()
    pass

def trends_plot(d, col, bycol, col_labels, title):
    q = d.groupby([col, 'StartDate']).count()
    val = wtab_by_date(d, col)
    val['Freq'] = pd.Series(q['StartDatetime'].values)
    val = val[['StartDate', col, 'Freq', 'Prop']]
    a = val[col].value_counts().sort_index().index
    plt.figure(figsize=(8,6))
    for i in range(len(a)):
        plt.plot(val.loc[val[col]==a[i], 'StartDate'], val.loc[val[col]==a[i], bycol], label=col_labels[i])
        plt.xticks(rotation=45)
        plt.legend()
    plt.title(title)
    if bycol == 'Prop':
        plt.ylabel('Proportion (weighted)')
    plt.grid(b=True,which='major', color='#DDD', linestyle='--')
    plt.show()
    pass

def df_stats(df, w=None):
    print(f'Total number of responses: {len(df)}')
    if w:
        print(f'Range of dates for wave {w} is from {df.StartDate.min()} to {df.EndDate.max()}')
    else:
        print(f'Range of dates is from {df.StartDate.min()} to {df.EndDate.max()}')
    print(f'This dataframe includes waves: {sorted(tab(df, "wave").Cat.unique())}')
    pass

def cross_plot(d, a, col, suptitle):
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    for i, resp in enumerate(d[col].unique()):
        ax[0].scatter(d.loc[d[col]==resp, 'D1'], d.loc[d[col]==resp, 'Prop'], label=resp, marker=markers[i], c=colors[i], s=30)
    ax[0].legend()
    ax[0].grid(which='major', axis='y', color='#DDD', linestyle='--')
    ax[0].set_yticks(np.arange(0,1.1,0.2))
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    for i, resp in enumerate(a[col].unique()):
        ax[1].scatter(a.loc[a[col]==resp, 'D2'], a.loc[a[col]==resp, 'Prop'], label=resp, marker=markers[i], c=colors[i], s=30)
    ax[1].legend()
    ax[1].grid(which='major', axis='y', color='#DDD', linestyle='--')
    ax[1].set_yticks(np.arange(0,1.1,0.2))
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    plt.suptitle(suptitle, fontsize=14)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Proportion (weighted)')
    plt.show()
    pass

def cross_trends_plot(d, col1, col2, col3, title):
    # By gender
    q = d.groupby([col1, col2, 'StartDate']).count()
    val = wcrosstab_by_date(d, col1, col2)
    val['Freq'] = pd.Series(q['StartDatetime'].values)
    val = val.loc[val[col2] == "Yes"]
    a = val[col1].value_counts().sort_index().index
    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    for i in range(len(a)):
        ax[0].plot(val.loc[val[col1]==a[i], 'StartDate'], val.loc[val[col1]==a[i], 'Prop'], label=a[i])
    ax[0].set_title('By gender')
    ax[0].legend()
    ax[0].set_ylabel('Proportion (weighted)')
    ax[0].grid(b=True,which='major', color='#DDD', linestyle='--')
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(45)
    # By age groups
    q2 = d.groupby([col3, col2, 'StartDate']).count()
    val2 = wcrosstab_by_date(d, col3, col2)
    val2['Freq'] = pd.Series(q2['StartDatetime'].values)
    val2 = val2.loc[val2[col2] == "Yes"]
    a2 = val2[col3].value_counts().sort_index().index
    for i in range(len(a2)):
        ax[1].plot(val2.loc[val2[col3]==a2[i], 'StartDate'], val2.loc[val2[col3]==a2[i], 'Prop'], label=a2[i])
    ax[1].set_title('By age groups')
    ax[1].legend()
    ax[1].set_ylabel('Proportion (weighted)')
    ax[1].grid(b=True,which='major', color='#DDD', linestyle='--')
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    plt.suptitle(title, fontsize=14)
    plt.show()
    pass
