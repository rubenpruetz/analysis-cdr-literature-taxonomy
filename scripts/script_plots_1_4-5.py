#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Data visualisation script for review of cdr side effects (figures 1, 4-5) """

# import libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filepath = ''  # specify filepath to data set
filename = 'Prütz_et_al_2024_CDR_side_effects_data_set_v1.0.xlsx'

def disaggregat_unique_effect_counter(input_df, input_column, doc_id):  # give input_column as string | 'pub_tit' as doc_id
    """ function to count unique side effects per article and cdr option """
    df_freq = input_df.drop_duplicates(subset=[doc_id, 'cdr_option', input_column], keep='first')
    df_freq = df_freq.groupby([input_column, 'cdr_option'])[input_column].count()
    df_freq = df_freq.to_frame().rename(columns={input_column: 'count'})
    df_freq.sort_values(by='count', ascending=False, inplace=True)
    df_freq = df_freq.reset_index()
    uniq_input = input_df[input_column].unique()
    cdr_options = input_df['cdr_option'].unique()
    all_combi = pd.MultiIndex.from_product([uniq_input,
                                            cdr_options],
                                           names=[input_column,
                                                  'cdr_option'])
    df_freq = df_freq.set_index([input_column, 'cdr_option'])
    df_freq = df_freq.reindex(all_combi, fill_value=0)
    df_freq = df_freq.reset_index()
    return df_freq

def disaggregat_unique_effect_counter2(input_df, input_column, doc_id):  # give input_column as string | 'pub_tit' as id
    """ function to count articles per category 1 and another variable (e.g., world region) """
    df_freq = input_df.drop_duplicates(subset=[doc_id, 'category_l1', input_column], keep='first')
    df_freq = df_freq.groupby([input_column, 'category_l1'])[input_column].count()
    df_freq = df_freq.to_frame().rename(columns={input_column: 'count'})
    df_freq.sort_values(by='count', ascending=False, inplace=True)
    df_freq = df_freq.reset_index()
    categories = input_df[input_column].unique()
    cat_options = input_df['category_l1'].unique()
    all_combi = pd.MultiIndex.from_product([categories,
                                            cat_options],
                                           names=[input_column,
                                                  'category_l1'])
    df_freq = df_freq.set_index([input_column, 'category_l1'])
    df_freq = df_freq.reindex(all_combi, fill_value=0)
    df_freq = df_freq.reset_index()
    return df_freq

def disaggregat_unique_effect_counter3(input_df, input_column, doc_id):  # give input_column as string | 'pub_tit' as id
    """ function to count articles per category 1, effect type, and another variable (e.g., cdr option) """
    df_freq = input_df.drop_duplicates(subset=[doc_id, 'category_l1', 'effect_type', input_column], keep='first')
    df_freq = df_freq.groupby([input_column, 'category_l1', 'effect_type'])[input_column].count()
    df_freq = df_freq.to_frame().rename(columns={input_column: 'count'})
    df_freq.sort_values(by='count', ascending=False, inplace=True)
    df_freq = df_freq.reset_index()
    categories = input_df[input_column].unique()
    cat_options = input_df['category_l1'].unique()
    e_t_options = input_df['effect_type'].unique()
    all_combi = pd.MultiIndex.from_product([categories,
                                            cat_options,
                                            e_t_options],
                                           names=[input_column,
                                                  'category_l1',
                                                  'effect_type'])
    df_freq = df_freq.set_index([input_column, 'category_l1', 'effect_type'])
    df_freq = df_freq.reindex(all_combi, fill_value=0)
    df_freq = df_freq.reset_index()
    return df_freq

def effect_direct_plot_aggregate_artic(input_df, effect_category, select_cdr_opt):
    """
    This function counts (relative and absolute) how many articles discuss positve, negative, neutral
    and unclear effects. Desirability is not aggregated. Double counting of articles is possible
    when articles discuss both positive and negative effects for a given CDR option and effect category.
    """
    output_df = disaggregat_unique_effect_counter3(input_df, 'cdr_option', 'pub_tit')
    cdr_option = output_df.loc[output_df['cdr_option'].isin(select_cdr_opt)]
    unique_categories = input_df[effect_category].unique()

    fig, axes = plt.subplots(figsize=(20, 20),  # specify subplots
                             nrows=2,
                             ncols=len(select_cdr_opt),
                             dpi=300,
                             gridspec_kw={'width_ratios': [1] * len(select_cdr_opt)},
                             sharey=True)

    for i, cdr_opt in enumerate(select_cdr_opt):  # go through each selected cdr option to perform the following steps
        cdr_option_data = cdr_option[cdr_option['cdr_option'] == cdr_opt]

        # prepare df containing absolute information
        count_df = cdr_option_data.pivot(index=['cdr_option',
                                                effect_category],
                                         columns=['effect_type'],
                                         values='count')
        count_df.reset_index(inplace=True)
        count_df = count_df.set_index(effect_category)
        count_df = count_df.sort_index(ascending=False, key=lambda col: col.str.lower())
        count_df = count_df.filter(['negative', 'unclear', 'neutral', 'positive'])

        # prepare df containing relative information
        article_counts = disaggregat_unique_effect_counter(input_df, effect_category, 'pub_tit')
        article_counts = article_counts.rename(columns={'count': 'article_count'})
        cdr_option_data = cdr_option_data.merge(article_counts, on=['cdr_option', effect_category])
        cdr_option_data['count'] = cdr_option_data['count'] / cdr_option_data['article_count'] * 100

        shares_df = cdr_option_data.pivot(index=['cdr_option',
                                                 effect_category],
                                          columns=['effect_type'],
                                          values='count')
        shares_df.reset_index(inplace=True)
        shares_df = shares_df.set_index(effect_category)
        shares_df = shares_df.sort_index(ascending=False, key=lambda col: col.str.lower())
        binary_df = shares_df.filter(['negative', 'positive'])
        binary_df = binary_df.fillna(0)

        (l1, negative), (l2, positive) = binary_df.items()
        y = range(len(negative))
        labels = binary_df.index.tolist()

        # first row of subplots showing relative information on desirability
        ax1 = axes[0, i]
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels)
        ax1.barh(y=y, height=0.6, width=-negative, color='crimson', label=l1)
        ax1.barh(y=y, height=0.6, width=positive, color='mediumseagreen', label=l2)
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='y', length=0, labelsize=19)
        ax1.tick_params(axis='x', labelsize=19)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax1.set_xlim(-125, 125)
        ax1.set_yticklabels(unique_categories)
        ax1.set_ylabel('Articles on positive vs negative effects [%]', fontsize=19)

        # second row of subplots showing absolute information on desirability
        ax2 = axes[1, i]
        count_df.plot(kind='barh',
                      stacked=True,
                      width=0.6,
                      legend=False,
                      color={'negative': 'crimson',
                             'unclear': 'darkgray',
                             'neutral': 'deepskyblue',
                             'positive': 'mediumseagreen'},
                      ax=ax2)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.tick_params(axis='y', length=0, labelsize=19)
        ax2.tick_params(axis='x', labelsize=19)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax2.set_xlim(0, 40) # ensure upper limit is not to low
        ax2.set_xlabel(cdr_opt, fontsize=19)
        ax2.set_ylabel('Article count per effect desirability', fontsize=19)

        if i == 0:
            ax2.legend(loc='upper right', fontsize=19, bbox_to_anchor=(8.3, 2.2),
            ncol=4)

    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(wspace=0.2)
    plt.minorticks_off()
    plt.show()




##### load extracted literature data #####

df = pd.read_excel(filepath+filename,
                   sheet_name='D1. Literature data')
df_objects = df.select_dtypes(['object'])
df[df_objects.columns] = df_objects.apply(lambda x: x.str.strip())  # remove leading and trailing spaces
df = df[df['authors'].notna()]  # remove empty bottom cells
df = df.loc[df['excluded'].isin(['no'])]  # keep non excluded rows

# filter for research articles
pub_type = ['Article']  # choose between "Article" and/or "Review"
df = df.loc[df['pub_type'].isin(pub_type)]

# rename long names
df_renamed = df.copy()
df_renamed['cdr_option'].replace(['Unspecified/multiple CDR'], ['Unspec/multi CDR'], inplace=True)
df_renamed['category_l1'].replace(['Air quality & condition'], ['Air qual & cond'], inplace=True)


##### plot stacked bars (figure 4) with "effect_direct_plot_aggregate_artic" #####

select_cdr_opt = ['AR',
                  'BECCS',
                  'Biochar',
                  'DACCS',
                  'EW',
                  'SCS',
                  'Unspec/multi CDR']
effect_direct_plot_aggregate_artic(df_renamed, 'category_l1', select_cdr_opt)



##### prepare and plot heatmap (figure 5) #####

# specify and filter considered world regions
region = ['Africa',
          'North America',
          'South America',
          'Asia',
          'Europe',
          'Oceania',
          'Multiple regions']
df_reg = df_renamed.loc[df_renamed['world_reg0'].isin(region)]

# create sub-dfs for positive and negative information on effect categories
positive = ['positive']
negative = ['negative']
df_pos = df_reg.loc[df_reg['effect_type'].isin(positive)]
df_neg = df_reg.loc[df_reg['effect_type'].isin(negative)]

# count articles per effect cdr option and world region and adjust format for plotting
all_artic_cdr = disaggregat_unique_effect_counter(df_reg, 'world_reg0', 'pub_tit')
pos_artic_cdr = disaggregat_unique_effect_counter(df_pos, 'world_reg0', 'pub_tit')
neg_artic_cdr = disaggregat_unique_effect_counter(df_neg, 'world_reg0', 'pub_tit')

all_artic_cdr = all_artic_cdr.pivot(index='cdr_option', columns='world_reg0', values='count')
pos_artic_cdr = pos_artic_cdr.pivot(index='cdr_option', columns='world_reg0', values='count')
neg_artic_cdr = neg_artic_cdr.pivot(index='cdr_option', columns='world_reg0', values='count')

# count articles per effect category 1 and world region and adjust format for plotting
all_articles = disaggregat_unique_effect_counter2(df_reg, 'world_reg0', 'pub_tit')
pos_articles = disaggregat_unique_effect_counter2(df_pos, 'world_reg0', 'pub_tit')
neg_articles = disaggregat_unique_effect_counter2(df_neg, 'world_reg0', 'pub_tit')

all_articles = all_articles.pivot(index='category_l1', columns='world_reg0', values='count')
pos_articles = pos_articles.pivot(index='category_l1', columns='world_reg0', values='count')
neg_articles = neg_articles.pivot(index='category_l1', columns='world_reg0', values='count')

# ensure categories on y axis are in alphabetic order
all_articles = all_articles.sort_index(ascending=True, key=lambda col: col.str.lower())
pos_articles = pos_articles.sort_index(ascending=True, key=lambda col: col.str.lower())
neg_articles = neg_articles.sort_index(ascending=True, key=lambda col: col.str.lower())

# adjust columns of plot input dfs to have regions in desired order
all_artic_cdr = all_artic_cdr[['Africa',
                               'North America',
                               'South America',
                               'Asia',
                               'Europe',
                               'Oceania',
                               'Multiple regions']]
pos_artic_cdr = pos_artic_cdr[['Africa',
                               'North America',
                               'South America',
                               'Asia',
                               'Europe',
                               'Oceania',
                               'Multiple regions']]
neg_artic_cdr = neg_artic_cdr[['Africa',
                               'North America',
                               'South America',
                               'Asia',
                               'Europe',
                               'Oceania',
                               'Multiple regions']]

all_articles = all_articles[['Africa',
                             'North America',
                             'South America',
                             'Asia',
                             'Europe',
                             'Oceania',
                             'Multiple regions']]
pos_articles = pos_articles[['Africa',
                             'North America',
                             'South America',
                             'Asia',
                             'Europe',
                             'Oceania',
                             'Multiple regions']]
neg_articles = neg_articles[['Africa',
                             'North America',
                             'South America',
                             'Asia',
                             'Europe',
                             'Oceania',
                             'Multiple regions']]

# create subplots
fig, ax = plt.subplots(nrows=2,
                       ncols=3,
                       figsize=(8, 12),
                       dpi=600,
                       sharex='col',
                       sharey='row')

g01 = sns.heatmap(all_artic_cdr,
                  annot=True,
                  linewidths=.5,
                  square=True,
                  cmap='Blues',
                 cbar_kws={'shrink': 0.27,
                           'aspect': 18.5,
                           'pad': 0.03,
                           'ticks': [0, all_artic_cdr.max(numeric_only=True).max()]}, ax=ax[0, 0])

g02 = sns.heatmap(neg_artic_cdr,
                  annot=True,
                  linewidths=.5,
                  square=True,
                  cmap='Reds',
                 cbar_kws={'shrink': 0.27,
                           'aspect': 18.5,
                           'pad': 0.03,
                           'ticks': [0, neg_artic_cdr.max(numeric_only=True).max()]}, ax=ax[0, 1])

g03 = sns.heatmap(pos_artic_cdr,
                  annot=True,
                  linewidths=.5,
                  square=True,
                  cmap='Greens',
                 cbar_kws={'shrink': 0.27,
                           'aspect': 18.5,
                           'pad': 0.03,
                           'ticks': [0, pos_artic_cdr.max(numeric_only=True).max()]}, ax=ax[0, 2])

g1 = sns.heatmap(all_articles,
                 annot=True,
                 linewidths=.5,
                 square=True,
                 cmap='Blues',
                 cbar_kws={'shrink': 0.695,
                           'aspect': 50,
                           'pad': 0.03,
                           'ticks': [0, all_articles.max(numeric_only=True).max()]}, ax=ax[1, 0])

g2 = sns.heatmap(neg_articles,
                 annot=True,
                 linewidths=.5,
                 square=True,
                 cmap='Reds',
                 cbar_kws={'shrink': 0.695,
                           'aspect': 50,
                           'pad': 0.03,
                           'ticks': [0, neg_articles.max(numeric_only=True).max()]}, ax=ax[1, 1])

g3 = sns.heatmap(pos_articles,
                 annot=True,
                 linewidths=.5,
                 square=True,
                 cmap='Greens',
                 cbar_kws={'shrink': 0.695,
                           'aspect': 50,
                           'pad': 0.03,
                           'ticks': [0, pos_articles.max(numeric_only=True).max()]}, ax=ax[1, 2])

g01.set_xlabel('')
g01.set_ylabel('')
g02.set_xlabel('')
g02.set_ylabel('')
g03.set_xlabel('')
g03.set_ylabel('')
g1.set_xlabel('Articles on all effect types')
g1.set_ylabel('')
g2.set_xlabel('Articles on negative effects')
g2.set_ylabel('')
g3.set_xlabel('Articles on positive effects')
g3.set_ylabel('')

g01.tick_params(length=0)
g02.tick_params(length=0)
g03.tick_params(length=0)
g1.tick_params(length=0)
g2.tick_params(length=0)
g3.tick_params(length=0)

plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.setp(ax[1, 2].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

plt.subplots_adjust(hspace=-0.45)
plt.subplots_adjust(wspace=0.1)

def annotate_zeros(val, text):
    if val == 0:
        text.set_text('x')
    else:
        text.set_text('')

for text in g01.texts:
    annotate_zeros(float(text.get_text()), text)

for text in g02.texts:
    annotate_zeros(float(text.get_text()), text)

for text in g03.texts:
    annotate_zeros(float(text.get_text()), text)

for text in g1.texts:
    annotate_zeros(float(text.get_text()), text)

for text in g2.texts:
    annotate_zeros(float(text.get_text()), text)

for text in g3.texts:
    annotate_zeros(float(text.get_text()), text)

plt.show()



##### plot literature development (figure 1) #####

# import data
rel_docs = pd.read_excel(filepath+filename,
                         sheet_name='D4. Docs (n=982)')

# prepare dfs for plotting
plot_df = rel_docs.groupby(['year'])['title'].count().reset_index()
plot_df['cumsum'] = plot_df['title'].cumsum()

plot_shares = rel_docs.groupby(['year',
                                'method_group'])['method_group'].count()
plot_shares = plot_shares.reset_index(name='count')
plot_shares = plot_shares.pivot(index='year', columns='method_group')['count']
plot_shares = plot_shares.fillna(0)
plot_shares = plot_shares.div(plot_shares.sum(axis=1), axis=0) * 100

# fill missing years with zeros
missing_years = pd.DataFrame({'year': range(1991, 2023)})
plot_df = pd.merge(missing_years,
                  plot_df,
                  on='year',
                  how='left').fillna(0)
plot_df = plot_df.fillna(0)
plot_shares = pd.merge(missing_years,
                  plot_shares,
                  on='year',
                  how='left')
plot_shares = plot_shares.fillna(0)
plot_shares.set_index('year', inplace=True)

# plot data
plt.rcParams.update({'figure.dpi': 300})
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
sns.barplot(plot_df,
            x='year',
            y='title',
            color='dimgrey',
            alpha=0.8,
            width=0.7,
            ax=ax1)

plot_shares.plot(kind='bar',
                 stacked=True,
                 figsize=(11, 4),
                 color=['mediumpurple',
                        'chocolate',
                        'crimson',
                        'royalblue',
                        'limegreen'],
                 alpha=0.8,
                 width=0.7,
                 ax=ax2)

sns.barplot(plot_df,
             x='year',
             y='cumsum',
             color='dimgrey',
             alpha=0.8,
             width=0.7,
             ax=ax3)

ax1.set_ylabel('Article\ncount', fontsize=12)
ax2.set_ylabel('Method\nshares', fontsize=12)
ax3.set_xlabel('', fontsize=12)
ax3.set_ylabel('Article\nsum', fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)

ax1.yaxis.labelpad = 18
ax2.yaxis.labelpad = 18
ax3.yaxis.labelpad = 10

ax1.set_yticks((0, 150))

for ax in [ax1, ax2, ax3]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.xticks([0, 10, 20, 30])
plt.xticks(fontsize=12)
plt.subplots_adjust(hspace=0.175)

ax2.legend(bbox_to_anchor=([-0.0105, -1.6]),
           ncol=3,
           loc='upper left',
           fontsize=12)

plt.text(0.15, 3250,
         'Cut off: May 20, 2022',
         fontsize=12,
         ha='left',
         va='center',
         color='black')