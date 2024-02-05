import json
import os
import random
import re
import pandas as  pd


def read_json_single_but_multi_lines(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    str = ''
    for line in lines:
        str += line
    json_file = json.loads(str)
    return json_file


def read_json_std(file_path):
    f = open(file_path)
    json_file = json.load(f)
    return json_file


def read_json_lines(file_path):
    json_file = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item_appended = json.loads(line)
            json_file.append(item_appended)
    return json_file


def write_json(file_path, json_file, sorted=True):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in json_file:
            item_appended = {
                'id': item['id'],
                'title': item['title'],
                'question': item['question'],
                'context': item['context'],
                'answers': item['answers']
            }
            str_item = json.dumps(item_appended, ensure_ascii=False, sort_keys=sorted)
            f.write(str_item + '\n')


def split_json(file_path, ratio):
    json_file = read_json_lines(file_path)
    random.shuffle(json_file)
    p = int(len(json_file) * ratio)
    json_v2_train = json_file[:p]
    json_v2_validation = json_file[p + 1:]
    write_json("QA_v2_train.json", json_v2_train)
    write_json("QA_v2_validation.json", json_v2_validation)


def contain_temperature(s):
    result = re.findall(r"\b12[0-1]\s?EC\b", s)
    result += re.findall(r"\b1[01][0-9]\s?EC\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?EC\b", s)
    result += re.findall(r"\b[0-9]\s?EC\b", s)

    result += re.findall(r"\b12[0-1]\s?C\b", s)
    result += re.findall(r"\b1[01][0-9]\s?C\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?C\b", s)
    result += re.findall(r"\b[0-9]\s?C\b", s)

    result += re.findall(r"\b12[0-1]\s?oC\b", s)
    result += re.findall(r"\b1[01][0-9]\s?oC\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?oC\b", s)
    result += re.findall(r"\b[0-9]\s?oC\b", s)

    result += re.findall(r"\b12[0-1]\s?degrees\b", s)
    result += re.findall(r"\b1[01][0-9]\s?degrees\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?degrees\b", s)
    result += re.findall(r"\b[0-9]\s?degrees\b", s)

    result += re.findall(r"\b12[0-1]\s?[(]deg[)]C\b", s)
    result += re.findall(r"\b1[01][0-9]\s?[(]deg[)]C\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?[(]deg[)]C\b", s)
    result += re.findall(r"\b[0-9]\s?[(]deg[)]C\b", s)

    result += re.findall(r"\b12[0-1]\s?°C\b", s)
    result += re.findall(r"\b1[01][0-9]\s?°C\b", s)
    result += re.findall(r"\b[0-9][0-9]\s?°C\b", s)
    result += re.findall(r"\b[0-9]\s?°C\b", s)
    if len(result):
        return True
    return False


def is_number(str):

    try:
        if str == 'NaN':
            return False
        s = float(str)
        # if str == '80':
        #     print("asdasd", str)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    os.system('ls')
    f = open('predict_nbest_predictions.json')
    json_nbest = json.load(f)
    df = pd.DataFrame([])
    json_answer = read_json_single_but_multi_lines('predict_predictions.json')
    ans_dict = dict(json_answer)
    json_dev = read_json_lines('../literature_procoss_to_dev_json/QA_v2_dev.json')

    items_unanswerable = []
    for item in json_dev:
        id = str(item['id'])
        ans_str = ans_dict[id]
        ans_score = json_nbest[id][0]['probability']

        if len(ans_str) == 0:
            continue
        if ans_score < 0.7:
            continue
        # if contain_temperature(ans_str):
        #     continue
        # if is_number(ans_str):
        #     continue
        # if (ans_str.startswith('−') ) and is_number(ans_str[1:]):
        #     # if ans_str == '−80':
        #     #     print("ASdasdasdasd")
        #     continue
        # if ans_str == '−80':
        #     print(len(ans_str))
        #     print(is_number(ans_str[1:]))
        #     print(is_number(ans_str))
        # print(s)
        ans_start = item['context'].find(ans_str)
        item_appended = {
            'id': item['id'],
            'title': item['title'],
            'question': item['question'],
            'context': item['context'],
            'answers': {
                'text':ans_str,
                'answer_start':ans_start
            },
        }
        df_item_appended = [{
            'id': item['id'],
            'title': item['title'],
            'question': item['question'],
            'context': item['context'],
            'text': ans_str,
            'answer_start': ans_start
        }]
        df = df.append(df_item_appended,ignore_index=True)
        items_unanswerable.append(item_appended)
    write_json('maybe_{}.json'.format(len(items_unanswerable)), items_unanswerable)
    df.to_excel('maybe_{}.xlsx'.format(len(items_unanswerable)))

