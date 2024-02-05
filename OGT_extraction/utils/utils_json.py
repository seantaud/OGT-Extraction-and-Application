import json
import pandas as pd


def get_df_from_json(file_path):
    items = read_json_lines(file_path)
    df = pd.DataFrame([])
    for item in items:
        id = str(item['id'])
        df_item_appended = [{
            'id': item['id'],
            'title': item['title'],
            'question': item['question'],
            'context': item['context'],
            'answers': item['answers']
        }]
        df = df.append(df_item_appended, ignore_index=True)
    file_name = file_path.replace('json','xlsx')
    df.to_excel(file_name)


def read_json_single_but_multi_lines(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    str = ''
    for line in lines:
        str += line
    json_file = json.loads(str)
    return json_file


def read_json_std(file_path):
    f = open(file_path, 'r', encoding='utf-8')
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

            str_item = json.dumps(item, ensure_ascii=False, sort_keys=sorted)
            f.write(str_item + '\n')