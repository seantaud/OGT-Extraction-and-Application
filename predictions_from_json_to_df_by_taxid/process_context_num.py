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
    # print(df.columns)
    series_groups = df[['taxonomy_id', 'doi', 'temperature_mid']].groupby(['taxonomy_id', 'doi'], as_index=False).agg(
        ['count'])
    df_groups = pd.DataFrame(series_groups)
    # cnt_list = df_groups['count'].tolist()
    # print(df_groups.head())
    doi_num = len(df_groups)
    series_groups = df[['taxonomy_id','scientific_name', 'temperature_mid']].groupby(['taxonomy_id','scientific_name'], as_index=False)[
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
    df_groups.to_excel('../OGT_and_enzyme/sources/OGT_descriptions_of_{}_extracted_from_literature.xlsx'.format(len(df_groups)),index=False)

    return df_groups, number_of_organisms, number_of_valid_ones, number_of_valid_ones / number_of_organisms, doi_num


def process_one_split(file_name):
    predict_predictions_path = join(file_name, 'eval_predictions.json')
    if not exists(predict_predictions_path):
        return [], -1
    try :
        json_answer = read_json_lines(predict_predictions_path)

        ans_dict = dict([])
        for ans in json_answer:
            ans_dict[str(list(ans.keys())[0])] = list(ans.values())[0]

    except:
        json_answer = read_json_single_but_multi_lines(predict_predictions_path)
        ans_dict = dict(json_answer)


    # print(len(ans_dict))
    dev_path = join(file_name, 'QA_v2_dev.json')
    json_dev = read_json_lines(dev_path)
    total = len(json_dev)
    items_right = []
    id_doi_set = set([])
    wrong_flag = 0
    for item in json_dev:
        id = str(item['id'])
        doi = str(item['doi'])
        id_doi_set.add(doi)
        if id not in ans_dict:
            continue
        ans_str = ans_dict[id]
        if len(ans_str) == 0:
            continue
        temperature_mid = get_temperature(ans_str)
        if temperature_mid == 'Drop':
            continue
        if temperature_mid == 'Wrong':
            print("Wrong!!!", file_name, id, ans_str, func_replace(ans_str),item['context'])
            wrong_flag = 1
            break
        item_appended = {
            'id': item['id'],
            'taxonomy_id': item['taxonomy_id'],
            'scientific_name': item['scientific_name'],
            'question': item['question'],
            'context': item['context'],
            'start':item['context'].find(ans_str),
            'doi': item['doi'],
            'answer_text': ans_str,
            'temperature_mid': float(temperature_mid)
        }
        items_right.append(item_appended)
    return items_right, wrong_flag, id_doi_set,total


def check_context_size(size):
    dir_list = []
    curren = 'predictions_all'
    # print(curren)
    for i in range(7):
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
    number_of_contexts = 0
    total_context_count = dict(zip([str(i) for i in range(7)],[0 for i in range(7)]))
    total_doi_count = dict(zip([str(i) for i in range(7)],[0 for i in range(7)]))
    for dir in dir_list:
        print(dir)
        items_right, wrong_flag, id_doi_set,total = process_one_split(dir)
        number_of_contexts += total
        if wrong_flag:
            break
        df = df.append(items_right, ignore_index=True)
        id_doi_set_per = id_doi_set_per | id_doi_set
        name = dir.replace('predictions_all','')
        print(name)
        total_context_count[name[1]] += total
        total_doi_count[name[1]] += len(id_doi_set)

    if wrong_flag != 1:
        df=df.drop_duplicates(subset=['taxonomy_id','start','answer_text'])
        new_df = df.copy()
        del new_df['start']
        del new_df['id']
        df.to_excel('../QA_results/OGT_descriptions_of_{}_extracted_from_literature.xlsx'.format(len(df)), index=False)
        print(total_context_count)
        print(total_doi_count)
        df_groups, number_of_organisms, number_of_valid_ones, rate, doi_num = drop_valid(df, 5)
        return number_of_organisms, number_of_valid_ones, rate, doi_num, number_of_contexts
    return 0,0,0,0,0


if __name__ == '__main__':

    settings_df = pd.DataFrame([])
    size = 60
    number_of_organisms, number_of_valid_ones, rate, doi_num, doi_tot = check_context_size(size)
    print(number_of_organisms, number_of_valid_ones, rate, doi_num, doi_tot)

