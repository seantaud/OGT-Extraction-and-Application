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

def read_json(file_path):
    return read_json_lines(file_path)


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


def get_json_from_df(df_unanswerable, start, min_score, answerable_list, invalid_list):
    cnt = start
    train_unanswerable = []
    for row in df_unanswerable.iterrows():
        item = row[1]

        if item['score'] < min_score:
            continue
        id = item['id']
        context = item['context']
        answer_predict = item['answer_predict']
        start_predict = item['start_predict']
        end_predict = item['end_predict']
        text = []
        answer_start = []
        length = len(context)
        if id in invalid_list:
            continue
        if id in answerable_list:
            start = context.find(answer_predict)
            text = [answer_predict]
            answer_start = [start]
        cnt += 1
        item_appended = {
            'id': cnt,
            'title': item['title'],
            'question': item['question'],
            'context': item['context'],
            'synonyms_desc': item['synonyms_desc'],
            'answers': {
                'text': text,
                'answer_start': answer_start
            }
        }
        train_unanswerable.append(item_appended)
    write_json("QA_v2_dev.json", train_unanswerable)
    return train_unanswerable


def re_index_json(file_path,start=0):
  json_file=read_json(file_path)
  cnt = start
  ans = []
  for item in json_file:
    cnt += 1
    item_appended ={
          'id':cnt,
          'title':item['title'],
          'question':item['question'],
          'context':item['context'],
          'answers':item['answers']
        }
    ans.append(item_appended)
  return ans

def transform(file_path):
  json_file=read_json(file_path)
  ans = []
  cnt_unanswerable = 0
  cnt_answerable = 0
  for item in json_file:
    text = item['answers']['text']
    answer_start = item['answers']['answer_start']
    if len(text) == 0:
      cnt_unanswerable += 1
      new_text = []
      new_answer_start=[]
    else:
      cnt_answerable += 1
      new_text = [str(item) for item in text]
      new_answer_start = [int(item) for item in answer_start]
    item_appended ={
          'id':item['id'],
          'title':item['title'],
          'question':item['question'],
          'context':item['context'],
          'answers':{
              'text':new_text,
              'answer_start':new_answer_start
          },
          'synonym_description':item['synonym_description'],
        }
    ans.append(item_appended)
  # print("answerable/unanswerable:",cnt_answerable/cnt_unanswerable)
  print("cnt_answerable:",cnt_answerable)
  print("unanswerable:",cnt_unanswerable)
  print("total:",cnt_answerable+cnt_unanswerable)
  write_json(file_path,ans)