import os.path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_json import read_json_std, write_json
from utils.utils_sent_split import sent_slipter


def plot_frequncy(file_path):
    pd.options.display.notebook_repr_html = False  # 表格显示
    plt.rcParams['figure.dpi'] = 75  # 图形分辨率
    sns.set_theme(style='darkgrid')  # 图形主题
    sent_df = pd.read_excel(file_path)
    stats = np.array(sent_df['sent_len'].tolist())
    print(np.percentile(stats, 25))
    print(np.percentile(stats, 75))
    sns.histplot(data=sent_df, x='sent_len', kde=True, bins=200)
    plt.show()


def process_all(file_path, number, context_size, split=1000):
    root_path = os.curdir
    tokenizer = sent_slipter()
    train_json_unanswer = read_json_std(file_path)
    id = 0
    cnt = [0 for i in range(len(context_size))]
    cnt_split = [0 for i in range(len(context_size))]

    items_split = [[], [], []]
    items_split_with_doi = [[], [], []]
    for item in train_json_unanswer:
        context = item['context']
        sent_len = len(tokenizer.tokenize(context))
        if sent_len <= 80 or sent_len > 100:
            continue
        taxid = item['taxid']
        doi = item['doi']
        name_in_context = item['name_context']
        if name_in_context == '':
            continue

        name_in_context = name_in_context.replace(',', ', ')
        scientific_name = item['scientific_name']
        question = 'what is the optimal growth temperature of ' + scientific_name + '?'
        context = item['context']
        if scientific_name == name_in_context:
            synonyms_desc = ''
        else:
            synonyms_desc = scientific_name + ' is also known as ' + name_in_context + '. '

        id = id + 1
        item_appended_with_doi = {
            'id': id,
            'taxonomy_id': str(taxid),
            'scientific_name': scientific_name,
            'question': question,
            'context': synonyms_desc + context,
            'doi': doi,
            'sent_len': sent_len,
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
        for j in range(3):
            if sent_len <= context_size[j]:
                cnt[j] += 1
                items_split[j].append(item_appended)
                items_split_with_doi[j].append(item_appended_with_doi)
                if cnt[j] % split == 0:
                    cnt_split[j] += 1
                    new_file = os.path.join(root_path,"text_from_journals/{}/json_{}/split_{}/".format(number, context_size[j], cnt_split[j]))
                    print(new_file)
                    if not os.path.exists(new_file):
                        os.makedirs(new_file)
                    write_json(new_file + 'QA_v2_dev.json', items_split[j])
                    new_file = os.path.join(root_path,"../predictions_from_json_to_df_by_taxid/predictions_all/{}/json_{}/split_{}/".format(
                        number, context_size[j], cnt_split[j]))
                    if not os.path.exists(new_file):
                        os.makedirs(new_file)
                    write_json(new_file + 'QA_v2_dev.json', items_split_with_doi[j])
                    items_split[j] = []
                    items_split_with_doi[j] = []
                break

    for j in range(1):
        if cnt[j] % split != 0:
            cnt_split[j] += 1
            new_file = os.path.join(root_path, "text_from_journals/{}/json_{}/split_{}/".format(number, context_size[j],
                                                                                                cnt_split[j]))
            if not os.path.exists(new_file):
                os.makedirs(new_file)
            write_json(new_file + 'QA_v2_dev.json', items_split[j])
            new_file = os.path.join(root_path,
                                    "../predictions_from_json_to_df_by_taxid/predictions_all/{}/json_{}/split_{}/".format(
                                        number, context_size[j], cnt_split[j]))
            if not os.path.exists(new_file):
                os.makedirs(new_file)
            write_json(new_file + 'QA_v2_dev.json', items_split_with_doi[j])
            items_split[j] = []
            items_split_with_doi[j] = []

    return cnt_split


if __name__ == '__main__':
    for i in range(7):
        num = process_all('./journals/final{}.json'.format(i), i, [100], 10000)
        print(num)

    # plot_frequncy()
