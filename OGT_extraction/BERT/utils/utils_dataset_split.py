import json
import random
from utils.utils_json import read_json, write_json
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
          }
        }
    ans.append(item_appended)
  # print("answerable/unanswerable:",cnt_answerable/cnt_unanswerable)
  print("cnt_answerable:",cnt_answerable)
  print("unanswerable:",cnt_unanswerable)
  print("total:",cnt_answerable+cnt_unanswerable)
  write_json(file_path,ans)

def split_for_CV(file_path,dst,fold=10,seed=42):
  json_file=read_json(file_path)
  json_file_with_answer = []
  json_file_without_answer = []
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
          }
        }
    if len(text) == 0:
      json_file_without_answer.append(item_appended)
    else:
      json_file_with_answer.append(item_appended)
  random.Random(seed).shuffle(json_file_with_answer)
  random.Random(seed).shuffle(json_file_without_answer)
  ratio = 1.0/fold
  point_with_answer = int(len(json_file_with_answer) * ratio)
  point_without_answer = int(len(json_file_without_answer) * ratio)
  for start in range(fold-1):
    end = start +1 
    json_CV = json_file_with_answer[start * point_with_answer : end * point_with_answer] + json_file_without_answer[start * point_without_answer : end * point_without_answer]
    write_json(dst+"/CV{}.json".format(start+1),json_CV)
  json_CV = json_file_with_answer[(fold-1) * point_with_answer:] + json_file_without_answer[(fold-1) * point_without_answer:]
  write_json(dst+"/CV{}.json".format(fold),json_CV)



def split_json_3(file_path,ratio,ratio2,seed=11):
  json_file=read_json(file_path)
  random.Random(seed).shuffle(json_file)
  p = int(len(json_file) * ratio)
  p2 = int(len(json_file) * (ratio+ratio2))
  json_v2_train =json_file[:p]
  json_v2_validation = json_file[p:p2]
  json_v2_dev = json_file[p2:]
  write_json("QA_v2_train.json",json_v2_train)
  write_json("QA_v2_validation.json",json_v2_validation)
  write_json("QA_v2_dev.json",json_v2_dev)

def split_datasets(Datapath,file_path,ratio,ratio2,seed=42):
  
  json_file=read_json(Datapath+file_path)
  json_file_with_answer = []
  json_file_without_answer = []
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
    if len(text) == 0:
      json_file_without_answer.append(item_appended)
    else:
      json_file_with_answer.append(item_appended)
  random.Random(seed).shuffle(json_file_with_answer)
  p = int(len(json_file_with_answer) * ratio)
  p2 = int(len(json_file_with_answer) * (ratio+ratio2))
  json_v2_train =json_file_with_answer[:p]
  json_v2_validation = json_file_with_answer[p:p2]
  json_v2_dev = json_file_with_answer[p2:]

  random.Random(seed).shuffle(json_file_without_answer)
  p = int(len(json_file_without_answer) * ratio)
  p2 = int(len(json_file_without_answer) * (ratio+ratio2))
  json_v2_train +=json_file_without_answer[:p]
  json_v2_validation += json_file_without_answer[p:p2]
  json_v2_dev += json_file_without_answer[p2:]
  
  write_json(Datapath+"QA_v2_train.json",json_v2_train)
  write_json(Datapath+"QA_v2_validation.json",json_v2_validation)
  write_json(Datapath+"QA_v2_dev.json",json_v2_dev)



