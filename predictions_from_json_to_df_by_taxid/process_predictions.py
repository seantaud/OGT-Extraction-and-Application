import os
from os.path import join, exists
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D

from utils.utils_plot import get_color
from utils.utils_temperature_process import func_replace, get_temperature

from utils.utils_json import read_json_single_but_multi_lines, read_json_lines, write_json


def drop_valid(df: pd.DataFrame, tolerance):
    series_groups = df[['taxonomy_id', 'doi', 'temperature_mid']].groupby(['taxonomy_id', 'doi'], as_index=False)[
        'temperature_mid'].agg(
        ['count'])
    df_groups = pd.DataFrame(series_groups)
    # cnt_list = df_groups['count'].tolist()
    doi_num = len(df_groups)
    series_groups = df[['taxonomy_id', 'temperature_mid']].groupby('taxonomy_id', as_index=False)[
        'temperature_mid'].agg(
        ['max', 'min', 'mean'])
    df_groups = pd.DataFrame(series_groups)
    df_groups.rename(columns={'mean': 'Avg_Temperature', 'max': 'Max_Temperature', 'min': 'Min_Temperature'},
                     inplace=True)
    df_groups.reset_index(inplace=True)
    df_groups['Range'] = df_groups['Max_Temperature'] - df_groups['Min_Temperature']
    number_of_organisms = len(df_groups)
    df_groups = df_groups[df_groups['Range'] <= tolerance]
    number_of_valid_ones = len(df_groups)
    valid_list = df_groups['taxonomy_id'].tolist()

    return df_groups, number_of_organisms, number_of_valid_ones, number_of_valid_ones / number_of_organisms, doi_num


def process_one_split(file_name):
    predict_predictions_path = join(file_name, 'eval_predictions.json')
    if not exists(predict_predictions_path):
        return [], -1
    json_answer = read_json_lines(predict_predictions_path)

    ans_dict = dict([])
    for ans in json_answer:
        ans_dict[str(list(ans.keys())[0])] = list(ans.values())[0]
    # print(len(ans_dict))
    dev_path = join(file_name, 'QA_v2_dev.json')
    json_dev = read_json_lines(dev_path)
    items_right = []
    id_doi_set = set([])
    wrong_flag = 0
    for item in json_dev:
        id = str(item['id'])
        doi = str(item['doi'])
        id_doi_set.add(doi)
        ans_str = ans_dict[id]
        if len(ans_str) == 0:
            continue
        temperature_mid = get_temperature(ans_str)
        if temperature_mid == 'Drop':
            continue
        if temperature_mid == 'Wrong':
            # print("Wrong!!!", file_name, id, ans_str, func_replace(ans_str))
            wrong_flag = 1
            break
        item_appended = {
            'id': item['id'],
            'taxonomy_id': item['taxonomy_id'],
            'scientific_name': item['scientific_name'],
            'question': item['question'],
            'context': item['context'],
            'doi': item['doi'],
            'answer_text': ans_str,
            'temperature_mid': float(temperature_mid)
        }
        items_right.append(item_appended)
    return items_right, wrong_flag, id_doi_set,len(json_dev)


def check_context_size(size):
    dir_list = []
    curren = 'predictions_all'
    # print(curren)
    for i in range(2):
        for j in [40, 60, 80, 100]:
            if j <= size:
                dst = join(curren, '{}\json_{}'.format(i, j))
                for root, dirs, files in os.walk(dst):
                    for dir in dirs:
                        if dir.find('split') != -1:
                            dir_list.append(join(dst, dir))
    df = pd.DataFrame([])
    wrong_flag = 0
    id_doi_set_per = set([])
    total_contexts = 0
    for dir in dir_list:
        # print(dir)
        items_right, wrong_flag, id_doi_set,number_of_contexts = process_one_split(dir)
        total_contexts+=number_of_contexts
        if wrong_flag:
            break
        df = df.append(items_right, ignore_index=True)
        id_doi_set_per = id_doi_set_per | id_doi_set
    print(len(df))

    if wrong_flag != 1:
        df.to_excel('size_{}_temperatures_with_{}.xlsx'.format(size, len(df)), index=False)

        df_groups, number_of_organisms, number_of_valid_ones, rate, doi_num = drop_valid(df, 4)
        return number_of_organisms, number_of_valid_ones, rate, doi_num, total_contexts,total_contexts
    return 0,0,0,0,0,0

def draw_context_size(settings, dst_img):
    # sns.lineplot(data=settings_df,x='context_size',y='rate_org')
    df = settings.copy()
    df['Self_consistency']=df['rate_org']
    df['Efficiency'] = df['rate_doi']
    g = sns.lineplot(data=df, x='context_size', y='Self_consistency', color=get_color(0))
    sns.lineplot(data=df, x='context_size', y='Efficiency', color=get_color(-1), ax=g.axes.twinx())
    g.legend(handles=[Line2D([], [], marker='_', color=get_color(0), label='Self-consistency'),
                      Line2D([], [], marker='_', color=get_color(-1), label='Efficiency')])
    g.xaxis.set_major_locator(ticker.MultipleLocator(20))
    g.xaxis.set_major_formatter(ticker.ScalarFormatter())
    g.set_xlabel('Context size(Number of sentences)')
    plt.savefig(join(dst_img , "context_size.png"), format='png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    file_path = 'context_size.xlsx'
    if exists(file_path):
        settings_df = pd.read_excel(file_path)
    else:
        settings_df = pd.DataFrame([])
        for size in [40, 60, 80, 100]:
            number_of_organisms, number_of_valid_ones, rate, doi_num, doi_tot,total_contexts = check_context_size(size)
            settings_df = settings_df.append([{'context_size': size,
                                               'organisms': number_of_organisms,
                                               'tolerated': number_of_valid_ones,
                                               'rate_org': rate,
                                               'doi_num': doi_num,
                                               'doi_tot': doi_tot,
                                               'rate_doi': doi_num / doi_tot,
                                               'total_contexts':total_contexts
                                               }],ignore_index=True)
        settings_df.to_excel('context_size.xlsx', index=False)

    draw_context_size(settings_df, '../OGT_and_enzyme/result_img')
