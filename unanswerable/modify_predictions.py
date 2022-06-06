import json
import os
import random
import re
import pandas as  pd
from find_unanswerable import write_json,read_json_lines

def get_json_from_df_labeled(df_unanswerable,start,time):
  cnt = start
  train_unanswerable = []
  for row in df_unanswerable.iterrows():
    item = row[1]
    flag = int(item['flag'])
    text = [item['text']]
    answer_start = [item['answer_start']]
    if flag ==2:

        text=[]
        answer_start = []

    cnt += 1
    item_appended ={
      'id':cnt,
      'title':item['title'],
      'question':item['question'],
      'context':item['context'],
      # 'synonyms_desc':item['synonyms_desc'],
      'answers':{
          'text':text,
          'answer_start':answer_start
      }
    }
    train_unanswerable.append(item_appended)
  write_json("{}_labeled_{}.json".format(len(train_unanswerable),time),train_unanswerable)
  return train_unanswerable

if __name__ == '__main__':

    json_dev = read_json_lines('QA_v2_983_time_05_24_21.json')
    print(len(json_dev))
    df = pd.read_excel('maybe_24.xlsx')
    train_json = get_json_from_df_labeled(df,983,'05_26_12')
    json_dev += train_json

    write_json('QA_v2_{}_time_05_26_12.json'.format(len(json_dev)),json_dev)

