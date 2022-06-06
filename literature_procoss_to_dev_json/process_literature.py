import json
import os
import random

from os.path import exists
# import rarfile

from unanswerable.find_unanswerable import write_json


def get_literature_json_to_squadv2_dataset(file_path, new_file, num, rd, maxlen, max_lenstr):
    f = open(file_path, 'r', encoding='utf-8')
    train_json_unanswer = json.load(f)
    print(len(train_json_unanswer))

    items_train = []
    items_train_with_syn = []
    last = ''
    cnt = 0
    new_context = """Campylobacter fetus is also known as C. fetus. In addition to the abovementioned studies, another research group reported only C. fetus in chelonians from Taiwan (Wang et al., 2013).Taken together, present and previous data suggest that Chelonii, besides being infrequently colonized by Campylobacter spp., are mainly carriers of reptilian-associated campylobacters, rather than clinically relevant, thermophilic species.Therefore, we speculate that some of these species, like C. geochelonis, may be markedly host-restricted to chelonians.However, C. fetus subsp.testudinum seems to possess a broader host range (reptiles and humans), while C. iguaniorum has also been isolated from an alpaca (Vicugna pacos) (Miller et al., 2016).The different body temperatures among mammals, birds and reptiles could explain the heterogeneous distribution of Campylobacter species among hosts.In fact, the optimal growth temperature for most campylobacters ranges between 30 and 45 °C (Vandamme et al., 2015), but these new strains isolated from reptiles show optimal growth at 25 °C (Fitzgerald et al., 2014; Gilbert et al., 2014; Piccirillo et al., 2016), which could be attributed to an evolutionary adaptation to these animals (Gilbert et al., 2015, 2016)."""
    new_text = '25 °C'
    new_start = new_context.find(new_text)
    cnt += 1
    item_appended_with_syn = {
        'id': cnt,
        'title': '196+Campylobacter fetus',
        'question': "what is the optimal growth temperature of Campylobacter fetus?",
        'context': new_context,
        'synonyms_desc': 'Campylobacter fetus is also known as C. fetus.',
        'answers': {
            'text': [new_text],
            'answer_start': [new_start]
        }
    }
    item_appended = {
        'id': cnt,
        'title': '196+Campylobacter fetus',
        'question': "what is the optimal growth temperature of Campylobacter fetus?",
        'context': new_context,
        'answers': {
            'text': [new_text],
            'answer_start': [new_start]
        }
    }
    items_train_with_syn.append(item_appended_with_syn)
    items_train.append(item_appended)
    len_json = len(train_json_unanswer)
    st = int(len_json/2)
    doi_dict = dict([])
    for item in train_json_unanswer:
        if item['doi'] not in doi_dict:
            doi_dict[item['doi']] = 1
        length = item['length']
        if length == 1:
            continue
        if cnt >= num:
            continue
        taxid = item['taxid']
        if taxid == last:
            if rd > 0 and random.random() > rd:
                continue

        name_in_context = item['name_context']
        if name_in_context == '':
            continue
        if len(item['context']) > max_lenstr:
            continue
        cnt += 1
        id = cnt
        last = taxid

        scientific_name = item['scientific_name']

        question = 'what is the optimal growth temperature of ' + scientific_name + '?'
        context = item['context']

        if scientific_name == name_in_context:
            synonyms_desc = ''
        else:
            name_in_context = name_in_context.replace(',',', ')
            synonyms_desc = scientific_name + ' is also known as ' + name_in_context + '. '

        item_appended_with_syn = {
            'id': id,
            'title': str(taxid) + '+' + scientific_name,
            'question': question,
            'context': synonyms_desc + context,
            'synonyms_desc': synonyms_desc,
            'answers': {
                'text': [],
                'answer_start': []
            }
        }
        item_appended = {
            'id': id,
            'title': str(taxid) + '+' + scientific_name,
            'question': question,
            'context': synonyms_desc + context,
            'answers': {
                'text': [],
                'answer_start': []
            }
        }
        items_train_with_syn.append(item_appended_with_syn)
        items_train.append(item_appended)

    # with open(new_file,'w',encoding='utf-8') as f:
    #   for item in items_train:
    #     str_item = json.dumps(item,ensure_ascii=False,sort_keys=True)
    #     f.write(str_item+'\n')

    write_json("with_syn_" + new_file, items_train_with_syn)
    write_json(new_file, items_train)
    print("total articles: ",len(doi_dict))
    return items_train_with_syn


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


# def unrar(base_path):
#     full_path = base_path
#     print(full_path)
#     z = rarfile.RarFile(full_path)
#     z.extractall('/journals')
#     z.close()
#     os.remove(full_path)


if __name__ == '__main__':
    os.system('ls')

    # base_path = r'/journals/final0.rar'

    # unrar(base_path)
    literature_file = 'journals_all/final0.json'

    items_train = get_literature_json_to_squadv2_dataset(literature_file, 'QA_v2_dev.json', 500, 0.1, 20, 3000)
    print(len(items_train))
