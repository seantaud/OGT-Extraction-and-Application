import os
from os.path import join, exists
import numpy as np
import pandas as pd
import json

from utils.utils_temperature_process import func_replace, get_temperature

from utils.utils_json import read_json_single_but_multi_lines, read_json_lines, write_json


def process_one_split(file_name):
    print(file_name)
    num = 0
    predict_predictions_path = file_name + "/predict_predictions.json"

    if not exists(predict_predictions_path):
        return [], -1
    with open(predict_predictions_path, 'r') as json_file:
        json_answer = json.load(json_file)


    ans_dict = dict([])
    for index, value in json_answer.items():
        ans_dict[str(index)] = value

    # print(len(ans_dict))
    dev_path = file_name + "/journal_raw.json"
    journal_raw = read_json_lines(dev_path)
    items_right = []
    id_doi_set = set([])
    wrong_flag = 0
    for item in journal_raw:
        num += 1
        id = str(item['id'])
        doi = str(item['doi'])
        id_doi_set.add(doi)
        ans_str = ans_dict[id]
        name_in_context = item['name_in_context']
        if len(ans_str) == 0:
            continue
        temperature_mid = get_temperature(ans_str)
        if temperature_mid == 'Drop':
            continue
        if temperature_mid == 'Wrong':
            print("Wrong!!!", file_name, id, ans_str, func_replace(ans_str), "Wrong!!!")
            wrong_flag = 1
            continue

        item_appended = {
            'id': item['id'],
            'taxonomy_id': item['taxonomy_id'],
            'scientific_name': str(item['scientific_name']),
            'name_in_context': str(name_in_context),
            'question': str(item['question']),
            'context': str(item['context']),
            'doi': item['doi'],
            'answer_text': ans_str,
            'temperature_mid': float(temperature_mid),
            'source':file_name
        }
        items_right.append(item_appended)

    return pd.DataFrame(items_right), wrong_flag, id_doi_set, len(journal_raw), num

def check_context_size():
    dir_list = []
    filepath = 'Result'
    for fn in os.listdir(filepath):
        fpath = 'Result/' + fn
        dir_list.append(fpath)

    df = pd.DataFrame([])
    all_count = 0
    wrong_flag = 0
    total_contexts = 0
    for dir in dir_list:
        items_right, wrong_flag, id_doi_set,number_of_contexts, num = process_one_split(dir)
        total_contexts+=number_of_contexts
        df = pd.concat([df,items_right])
        all_count += num
    
    df.to_csv(("Predicted_results.csv"), index=False)
    print(len(df))
    print("number of raw dataset:{}".format(all_count))
check_context_size()