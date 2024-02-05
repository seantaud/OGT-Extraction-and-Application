import os
from os.path import join, exists
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D

from utils.utils_plot import get_color, set_plot
from utils.utils_temperature_process import func_replace, get_temperature

from utils.utils_json import read_json_single_but_multi_lines, read_json_lines, write_json, read_json_std


def draw_metric(df, ax, metric):
    sns.set_palette(["#3498db" ,"b","#9b59b6",])
    g = sns.barplot(data=df, x='Model', y=metric, hue='Method', ax=ax,
                    # palette='PuBu'
                    )
    g.set(xlabel='Model', ylabel=metric + ' score', )
    # ax.set_ylabel(metric+' score', fontsize=40)  # 设置Y坐标轴标签字体
    # ax.set_xlabel('Model', fontsize=40)
    # g.legend(fontsize=40)
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    for p in ax.patches:
        if p.get_height() != 0:
            ax.text(p.get_x(), p.get_y() + p.get_height(), "{:1.2f}".format(p.get_height()), fontsize=18)


def draw_models(df_results):
    # sns.lineplot(data=settings_df,x='context_size',y='rate_org')
    df = df_results.copy()

    dst_img = '../result_img/'
    set_plot(2.5)
    f, axarr = plt.subplots(2, 1, figsize=(25, 16))
    draw_metric(df=df, ax=axarr[0], metric='EM')
    draw_metric(df=df, ax=axarr[1], metric='F1')
    # [ax.set_axis_off() for ax in axarr.ravel()]
    if not exists(dst_img):
        os.mkdir(dst_img)
    plt.subplots_adjust(hspace=0.8, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.tight_layout()
    f.savefig(dst_img + "models_comp.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()


def search_result(file_path):
    curren = file_path
    methods = ['fine_tuning', 'P-tuning-v2']
    F1_score = dict(zip(methods, [0, 0]))
    EM_score = dict(zip(methods, [0, 0]))
    for method in methods:
        dst = join(curren, method)
        max_EM = 0
        max_F1 = 0
        for root, dirs, files in os.walk(dst):
            if root.find('dev') != -1 or root.find('validation') != -1:

                for file in files:
                    print(file)
                    if file.find('trainer_state') != -1:
                        src = join(root, file)
                        json_items = read_json_single_but_multi_lines(src)
                        history = json_items["log_history"]
                        for trace in history:
                            if method == 'P-tuning-v2' and 'best_epoch' in trace:
                                max_EM = max(max_EM, trace['best_eval_exact_match'])
                                max_F1 = max(max_F1, trace['best_eval_f1'])
                            if method == 'fine_tuning' and 'eval_HasAns_exact' in trace:
                                # print(trace)
                                max_EM = max(max_EM, trace['eval_best_exact'])
                                max_F1 = max(max_F1, trace['eval_best_f1'])
        F1_score[method] = max_F1
        EM_score[method] = max_EM
    return F1_score, EM_score


if __name__ == '__main__':
    models_information = read_json_std('./models.json')
    df_results = pd.DataFrame([])
    df_show = pd.DataFrame([])
    for item in models_information:
        model_name = item['model_name']
        model_path_in_Huggingface = item['model_path']
        month = item['month']
        year = item['year']
        size = item['model_size']
        file_path = './B-models_results/' + model_name
        df_show = df_show.append([{'Model': model_name,
                                         'Model_path_in_Huggingface': model_path_in_Huggingface,
                                         'Month': month,
                                         'Year': year,
                                         'Size':size,}], ignore_index=True)
        df_results = df_results.append([{'Model': model_name,
                                         'Model_path_in_Huggingface': model_path_in_Huggingface,
                                         'Month': month,
                                         'Year': year,
                                         'Size':size,
                                         'F1': item['F1']*100,
                                         'EM': item['EM']*100,
                                         'Method': 'Only pretrained'
                                         }], ignore_index=True)
        F1_score, EM_score = search_result(file_path)
        if model_name =='PubMedBERT':
            print(F1_score, EM_score)
        df_results = df_results.append([{'Model': model_name,
                                         'Model_path_in_Huggingface': model_path_in_Huggingface,
                                         'Month': month,
                                         'Year': year,
                                         'Size':size,
                                         'F1': F1_score['P-tuning-v2'],
                                         'EM': EM_score['P-tuning-v2'],
                                         'Method': 'P-tuning-v2'
                                         }],ignore_index=True)
        df_results = df_results.append([{'Model': model_name,
                                         'Model_path_in_Huggingface': model_path_in_Huggingface,
                                         'Month': month,
                                         'Year': year,
                                         'Size':size,
                                         'F1': F1_score['fine_tuning'],
                                         'EM': EM_score['fine_tuning'],
                                         'Method': 'Fine-tuning'
                                         }],ignore_index=True)
    draw_models(df_results)
    df_results.to_excel('./models_comparison_for_plot.xlsx', index=False)
    df_show.to_excel('./models_comparison_for_show.xlsx', index=False)
