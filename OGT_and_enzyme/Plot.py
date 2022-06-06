import os
from os.path import join, exists
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

from utils.utils_json import read_json_lines
from utils.utils_plot import draw_grid, histplot_dif, set_plot
from utils.utils_sent_split import sent_slipter


def make_a_graph(df):
    df_relation = df.copy()
    # df_relation = df_relation[['organism', 'ogt', 'topt']]
    # sns.histplot(data=df_relation["ogt"], color='#3426AE',ax=ax)
    # g = sns.boxplot(data=df_relation,x="ogt",ax=ax)
    # sns.catplot(
    #     kind="box",
    #     x="ogt",
    #     y="domain",
    #     data=df_relation
    # )

    # g = sns.JointGrid(data = df_relation,x="ogt" , space=0, ratio=8)
    # sns.catplot(data=df_relation, x="ogt", hue='domain', ax=g.ax_joint)
    # sns.boxplot(data=df_relation, x="ogt", ax=g.ax_marg_x,linewidth=.3, fliersize=1, notch=True)
    # g.set_axis_labels('Optimal growth temperature($^\circ$C)', 'Frequency', fontsize=24)
    pal = {0: "#C6C4E1", 1: "#E7CECC", 2: '#E8F5E9'}
    g = sns.FacetGrid(df_relation, col="domain")
    g.map(sns.histplot, "ogt", color='#3426AE')
    g.set(xlabel='OGT ($^\circ$C)', ylabel='Frequency')
    
    return g


def make_b_graph(df):
    df_relation = df.copy()
    df_relation = df_relation[['organism', 'ogt', 'topt']]
    df_relation['minus'] = df_relation['ogt'] - df_relation['topt']
    g = sns.jointplot("ogt", "minus", data=df_relation, color='#3426AE', space=0, kind='kde', levels=60, fill=True,
                      alpha=0.6, cut=2, ratio=15)
    g.plot_joint(sns.scatterplot, color="b", alpha=.7)
    # g.plot_marginals(sns.boxplot, data=df_relation, linewidth=.3, fliersize=1, notch=True)
    # g.ax_marg_x.remove()
    # g.ax_joint.axhline(0,  lw=0.3, c="black", alpha=.2)
    # g.ax_joint.axvline(0,  lw=0.3, c="black", alpha=.2)
    g.ax_joint.axvline(x=32, lw=3, ls="--", )
    g.ax_joint.axhline(y=-2, lw=3, ls="--", )
    g.set_axis_labels('Optimal growth temperature($^\circ$C)', 'Difference between OGT and Topt($^\circ$C)',
                      fontsize=24)
    
    return g
    # sns.scatterplot(data=df_relation, ax=ax_b, x="ogt", y="minus")
    # sns.kdeplot(
    #     data=df_relation,
    #     x="ogt",
    #     y="minus",
    #     levels=5,
    #     fill=True,
    #     alpha=0.6,
    #     cut=2,
    #     ax=ax_b,
    # )
    # ax_b.set(xlabel='Optimal growth temperature', ylabel='Difference between OGT and Topt($^\circ$C)')
    # myPlot = myPlot.map_dataframe(plt.plot, [-20, 120], [0, 0], 'r-').add_legend().set_axis_labels("x", "y")
    #
    # df_lines = pd.DataFrame([])
    # df_lines = df_lines.append([{'x': -20, 'y': 20}])
    # df_lines = df_lines.append([{'x': -70, 'y': 70}])
    # sns.lineplot(data=df_lines, x='x', y='y', color='g', ax=ax_b,linestyle="dashed")
    # df_lines = pd.DataFrame([])
    # df_lines = df_lines.append([{'x': 120, 'y': -20}])
    # df_lines = df_lines.append([{'x': -20, 'y': 120}])
    # sns.lineplot(data=df_lines, x='x', y='y', color='g', ax=ax_b,linestyle="dashed")


def make_c_graph(df: pd.DataFrame):
    df_relation = df.copy()
    print(df_relation)
    medians = df_relation.groupby('domain')['ogt'].median()
    pal = {'Archaea': "#C6C4E1", 'Eukarya': "#E7CECC", 'Bacteria': '#E8F5E9'}
    g = sns.JointGrid(data=df_relation, x="domain", y="ogt", space=0, ratio=40)
    box = sns.boxplot(data=df_relation, x='domain', y="ogt", palette="mako", ax=g.ax_joint)
    g.set_axis_labels('Domain', 'Boxplot of OGT($^\circ$C)', fontsize=24)
    vertical_offset = df_relation['ogt'].median() * 0.005  # offset from median for display

    for xtick in box.get_xticks():
        box.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                 horizontalalignment='center', size='small', color='w', weight='semibold')
    # g = sns.FacetGrid(df_relation, col='domain')
    # g.map(sns.boxplot, 'ogt', palette=pal)
    # g.set(xlabel='Domain', ylabel='OGT($^\circ$C)')
    
    return g


def make_d_graph(df: pd.DataFrame):
    df_relation = df.copy()
    df_relation = df_relation[['organism', 'ogt', 'topt']]
    df_relation['new'] = ""
    df_Topt_averaged = df_relation.groupby(['organism', 'ogt'], as_index=False)['topt'].agg(['count', 'mean'])
    df_Topt_averaged = pd.DataFrame(df_Topt_averaged)
    df_Topt_averaged.rename(columns={'mean': 'Avg_Temperature'}, inplace=True)
    df_Topt_averaged.reset_index(inplace=True)
    df_Topt_averaged = df_Topt_averaged[df_Topt_averaged['count'] >= 5]
    x = np.array(df_Topt_averaged["ogt"])
    y = np.array(df_Topt_averaged["Avg_Temperature"])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)(values)
    g = sns.JointGrid("ogt", "Avg_Temperature", df_Topt_averaged, space=0, ratio=15)
    sns.regplot(data=df_Topt_averaged, x="ogt", y="Avg_Temperature", ax=g.ax_joint)
    sns.scatterplot(data=df_Topt_averaged, x="ogt", y="Avg_Temperature", c=kernel, cmap='viridis', ax=g.ax_joint)
    r, p = stats.pearsonr(x, y)
    # if you choose to write your own legend, then you should adjust the properties then
    phantom, = g.ax_joint.plot([], [], linestyle="solid", alpha=0.0)
    # here graph is not a ax but a joint grid, so we access the axis through ax_joint method
    # legend_properties = {'weight': 'bold', 'size': 14}
    g.ax_joint.legend([phantom], ['r={:.4f}, n={:d},p<0.05'.format(r, len(df_Topt_averaged))], loc=2, )
    plt.setp(g.ax_joint.get_legend().get_texts(), fontsize='25')  # for legend text
    g.set_axis_labels('Optimal growth temperature($^\circ$C)', 'Averaged enzyme optima', fontsize=24)
    return g


def make_e_graph(df: pd.DataFrame, sampled_n=50, repeated=5):
    df_relation = df.copy()
    df_relation = df_relation[['organism', 'ogt', 'topt']]
    df_pearson = pd.DataFrame([])
    for i in range(sampled_n):
        pearson_list = []
        samples = i + 1
        print(samples)
        for j in range(repeated):
            df_Topt_averaged = df_relation.groupby('organism', as_index=False).apply(
                lambda x: float(np.mean([random.sample(list(x['topt']), 1) for z in range(samples)])))
            df_Topt_averaged = pd.DataFrame(df_Topt_averaged, columns=['organism', 'ogt', 'Topt_Avg'])
            pearson_coef = df_Topt_averaged['ogt'].corr(df_Topt_averaged['Topt_Avg'], method='pearson')
            pearson_list.append(pearson_coef)
        df_pearson = df_pearson.append([{'n': samples, "corr": np.mean(pearson_list)}])
        df_pearson.to_excel('pearson.xlsx')
    g = sns.scatterplot("n", "corr", data=df_pearson, color='#E7CECC')
    g.set(xlabel='Number of averaged enzymes', ylabel='Pearson correlation')
    
    return g


def delete_useless():
    file_path = r'sources/enzyme_ogt_topt.tsv'
    df_relation = pd.read_csv(file_path, sep='\t', engine='python')
    print(df_relation.head())
    del df_relation['uniprot_id']
    del df_relation['ec']
    df_relation = df_relation[df_relation['topt_source'] == 'experimental']
    del df_relation['topt_source']
    del df_relation['ogt_source']
    print(len(df_relation))
    df_relation = pd.DataFrame(df_relation)
    df_relation.to_excel('sources/enzyme_ogt_topt.xlsx', index=False)
    return df_relation


def get_sent_len(file_path):
    items = read_json_lines(file_path)
    tokenizer = sent_slipter()
    df = pd.DataFrame([])
    for item in items:
        context = item['context']
        sent_len = len(tokenizer.tokenize(context)) - 1
        answers = item['answers']['text']
        flag = 'Answerable' if len(answers) != 0 else 'Unanswerable'
        items_appended = [{'id': item['id'], 'sent_len': sent_len, 'flag': flag}]
        df = df.append(items_appended, ignore_index=True)
    return df


def draw_sent_len_frequent(df, ax):
    df_relation = df.copy()
    g = sns.histplot(legend=False,data=df_relation, y="flag",hue='flag',shrink=.7,palette='mako', ax=ax)
    # g.set_axis_labels('Count', 'Category', fontsize=24)
    g.set(xlabel='Count', ylabel='Category')

    initialx = 0
    for p in ax.patches:
        if p.get_width() != 0:
            ax.text(p.get_width()-60, p.get_y() + p.get_height() / 2,"{:1.0f}".format(p.get_width()))
    return g


def draw_sent_len_displot(df, ax):
    df_relation = df.copy()
    g = sns.histplot(data=df_relation, x="sent_len", ax=ax, color ='#3426AE')
    g.set(xlabel='Context size (Number of sentences)', ylabel='Frequency')
    return g


def draw_sentence_len():
    dst_img = '../result_img/sent_len/'
    set_plot(2)
    df_relation = get_sent_len('../unanswerable/QA_v2_1006_time_05_26_19.json')
    f, axarr = plt.subplots(1, 2, figsize=(25, 12))
    draw_sent_len_frequent(df_relation, axarr[0])
    draw_sent_len_displot(df_relation, axarr[1])
    # [ax.set_axis_off() for ax in axarr.ravel()]
    if not exists(dst_img):
        os.mkdir(dst_img)
    plt.subplots_adjust(hspace=0.8, wspace=0.6,left=0.2, right=0.9, top=0.9, bottom=0.1)
    # plt.tight_layout()
    f.savefig(dst_img + "sent_len.png", format='png', dpi=100, bbox_inches='tight')
    


def merge_the_OGT_and_Topt():
    file_path = r'sources/enzyme_ogt_topt.xlsx'
    df_relation = pd.read_excel(file_path)
    print(df_relation.columns)
    file_path = r'sources/OGT_descriptions_of_1224_extracted_from_literature.xlsx'
    df_extracted = pd.read_excel(file_path)
    print(df_extracted.columns)
    df_extracted['scientific_name'] = df_extracted['scientific_name'].apply(lambda x: x.replace(' ', '_'))
    df_extracted['scientific_name'] = df_extracted['scientific_name'].apply(lambda x: x.lower())
    df_relation = pd.merge(df_extracted, df_relation, left_on='scientific_name', right_on='organism', how='inner')
    df_relation=df_relation.sort_values(by='domain')
    print(df_relation.head())
    df_relation.to_excel('sources/OGT_and_Topt_of_{}.xlsx'.format(len(df_relation)),index=False)
    return df_relation

def draw_ogt_topt(df_relation):
    set_plot(2.2)
    # file_path = r'sources/enzyme_ogt_topt.xlsx'
    # df_relation = pd.read_excel(file_path)

    # df_relation = delete_useless()
    ga = make_a_graph(df_relation)
    set_plot(2.2)
    gb = make_b_graph(df_relation)
    set_plot(2.2)
    gc = make_c_graph(df_relation)
    set_plot(2.2)
    gd = make_d_graph(df_relation)
    dst_img = '../result_img/ogt_topt/'
    if not exists(dst_img):
        os.mkdir(dst_img)
    draw_grid(ga, gb, gc, gd, dst_img)


if __name__ == '__main__':
    # merge_the_OGT_and_Topt()
    df_relation = merge_the_OGT_and_Topt()
    draw_ogt_topt(df_relation)
    # draw_joint_with_box(df_relation)

    # draw_sentence_len()

# make_a_graph()#生长温度数据集中所有生物体的生长温度分布
# make_d_graph()#单个酶温度与生长温度最优值的距离评分
# make_e_graph()#酶温度最优值与生长温度之间 Pearson 相关系数的灵敏度图
# make_f_graph()#一个散点图，比较了具有五个以上报告的酶温度最差值的生物体的生长温度与该生物体报告的所有酶的平均温度最佳值
