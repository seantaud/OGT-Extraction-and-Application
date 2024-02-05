import json
import random
import itertools
import pandas as pd
from tqdm import tqdm
import datetime
import re
import os
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from transformers import AutoTokenizer
from datasets import load_dataset, Features, Value, Sequence, DatasetDict, Dataset

# Load the sentence slipter to do cropping and sampling to the raw datasets
def sent_slipter():
    punkt_param = PunktParameters()
    abbreviation = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "et al",
        "spp",
        "sp",
        "fig",
        "nov",
        "e.g",
        "Dr",
    ]
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)
    return tokenizer


Sent_slipter = sent_slipter()



def read_json(file_path):
    return read_json_lines(file_path)


def read_json_std(file_path):
    f = open(file_path, "r", encoding="utf-8")
    json_file = json.load(f)
    return json_file


def read_json_lines(file_path):
    json_file = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            item_appended = json.loads(line)
            json_file.append(item_appended)
    return json_file


def write_json(file_path, json_file, sorted=True):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in json_file:
            str_item = json.dumps(item, ensure_ascii=False, sort_keys=sorted)
            f.write(str_item + "\n")


def split_datasets(json_file, Datapath, ratio, ratio2, seed=42,write=True):
    json_file_with_answer = []
    json_file_without_answer = []
    cnt_unanswerable = 0
    cnt_answerable = 0
    for item in json_file:
        text = item["answers"]["text"]
        answer_start = item["answers"]["answer_start"]
        # ogt = item['ogt']
        synonym_description = item["synonym_description"]
        doi = item["doi"]
        if len(text) == 0:
            cnt_unanswerable += 1
            new_text = []
            new_answer_start = []
        else:
            cnt_answerable += 1
            new_text = [str(item) for item in text]
            new_answer_start = [int(item) for item in answer_start]
        item_appended = {
            "id": item["id"],
            "title": item["title"],
            "question": item["question"],
            "context": item["context"],
            "answers": {"text": new_text, "answer_start": new_answer_start},
            "synonym_description": synonym_description,
        }
        if len(text) == 0:
            json_file_without_answer.append(item_appended)
        else:
            json_file_with_answer.append(item_appended)
    random.Random(seed).shuffle(json_file_with_answer)
    p = int(len(json_file_with_answer) * ratio)
    p2 = int(len(json_file_with_answer) * (ratio + ratio2))
    json_OGT_train = json_file_with_answer[:p]
    json_OGT_validation = json_file_with_answer[p:p2]
    json_OGT_dev = json_file_with_answer[p2:]

    random.Random(seed).shuffle(json_file_without_answer)
    p = int(len(json_file_without_answer) * ratio)
    p2 = int(len(json_file_without_answer) * (ratio + ratio2))
    json_OGT_train += json_file_without_answer[:p]
    json_OGT_validation += json_file_without_answer[p:p2]
    json_OGT_dev += json_file_without_answer[p2:]
    if write:
      write_json(Datapath + "OGT_QA_train.json", json_OGT_train)
      write_json(Datapath + "OGT_QA_validation.json", json_OGT_validation)
      write_json(Datapath + "OGT_QA_dev.json", json_OGT_dev)
    return json_OGT_train,json_OGT_validation,json_OGT_dev


def get_json_from_df(DATA_PATH, df_meregd,dataset_name, check=0,write=True):
    cnt = 0
    train_json = []
    for row in df_meregd.iterrows():
        # print(row[0])
        item = row[1]
        id = item["taxonomy_id"]
        question = str(item["question"])
        doi = str(item["doi"])
        if doi == "nan":
            doi = ""
        synonym_description = str(item["synonym_description"])
        if synonym_description.lower() == "nan":
            synonym_description = ""
        # ogt = item['new_answer_text']
        # if str(ogt) == "nan":
        #   ogt=""
        # print(ogt)
        sn = question.replace("what is the optimal growth temperature of ", "")
        name = item["name"]
        sn = sn[:-1]
        # if check == 1:
        #   scientificName = check_id_name(id)
        scientificName = sn
        title = str(id)
        title += "+" + sn
        context = item["context"]
        if item["answer"] < -1:
            continue
        elif item["answer"] == -1:
            text = []
            answer_start = []
        else:
            text = [item["answer_text"]]
            new_start = int(item["answer"])

            if (
                context[new_start : new_start + len(item["answer_text"])]
                != item["answer_text"]
            ):
                # print("syn desc "+synonym_description)
                # print(len(item['answer_text']))
                print(len(synonym_description))
                print(new_start - context.find(item["answer_text"]))
                print(
                    "error answer start "
                    + str(id)
                    + " "
                    + scientificName
                    + " "
                    + context[new_start - 100 : new_start + len(item["answer_text"])]
                    + " "
                    + item["answer_text"]
                    + " "
                    + str(row[0])
                )

            answer_start = [new_start]

        cnt += 1
        if check == 1:
            raw_context = item["context"]
            if raw_context.find(" is also known as ") != -1:
                print(
                    "error context has syn"
                    + str(id)
                    + " "
                    + scientificName
                    + " "
                    + str(row[0])
                )
            if scientificName != "" and scientificName != sn:
                print(
                    "name wrong! "
                    + str(id)
                    + " "
                    + scientificName
                    + " "
                    + sn
                    + " "
                    + str(row[0])
                )
            if scientificName != "" and scientificName != name:
                print(
                    "error "
                    + str(id)
                    + " "
                    + scientificName
                    + " "
                    + name
                    + " "
                    + str(row[0])
                )

            name_in_context = [scientificName]
            if len(synonym_description) != 0:
                if synonym_description[-1] != " ":
                    print(
                        "error desc incomplete "
                        + str(id)
                        + " "
                        + scientificName
                        + " "
                        + " "
                        + str(row[0])
                    )
                items = synonym_description.split(" is also known as ")
                name_in_desc = items[0]
                name_in_context = items[1]
                name_in_context = name_in_context[:-2]
                # name_in_desc = re.search('[A-Z][^,]* is also known', synonym_description).group()[:-16]
                if scientificName != "" and name_in_desc != scientificName:
                    print(
                        "error desc is not consistent with SN "
                        + str(id)
                        + " "
                        + scientificName
                        + " "
                        + name_in_desc
                        + " "
                        + str(row[0])
                    )
                if name_in_context.find(" or ") != -1:
                    name_in_context = name_in_context.split(" or ")
                else:
                    name_in_context = [name_in_context]
            flag = 1
            for one_name_in_context in name_in_context:
                if raw_context.find(one_name_in_context) != -1:
                    flag = 0
                    break
            if flag:
                print(
                    "error name_in_context not found in context"
                    + str(id)
                    + " "
                    + scientificName
                    + " "
                    + " "
                    + str(row[0])
                )
            # sn = scientificName
            # question = question.replace(sn,scientificName)
        item_appended = {
            "id": cnt,
            "title": title,
            "question": question,
            "context": context,
            "answers": {"answer_start": answer_start, "text": text},
            "doi": doi,
            # 'ogt':ogt,
            "synonym_description": synonym_description,
        }

        train_json.append(item_appended)
    if write:
      write_json(os.path.join(DATA_PATH, f"{dataset_name}.json"), train_json)
    return train_json






def find_centigrade(context):
    centigrade_syms = [
        r"\b12[0-1]\s?EC\b",
        r"\b1[01][0-9]\s?EC\b",
        r"\b[0-9][0-9]\s?EC\b",
        r"\b[0-9]\s?EC\b",
        r"\b12[0-1]\s?C\b",
        r"\b1[01][0-9]\s?C\b",
        r"\b[0-9][0-9]\s?C\b",
        r"\b[0-9]\s?C\b",
        r"\b12[0-1]\s?oC\b",
        r"\b1[01][0-9]\s?oC\b",
        r"\b[0-9][0-9]\s?oC\b",
        r"\b[0-9]\s?oC\b",
        r"\b12[0-1]\s?degrees\b",
        r"\b1[01][0-9]\s?degrees\b",
        r"\b[0-9][0-9]\s?degrees\b",
        r"\b[0-9]\s?degrees\b",
        r"\b12[0-1]\s?[(]deg[)]C\b",
        r"\b1[01][0-9]\s?[(]deg[)]C\b",
        r"\b[0-9][0-9]\s?[(]deg[)]C\b",
        r"\b[0-9]\s?[(]deg[)]C\b",
        r"\b12[0-1]\s?°C\b",
        r"\b1[01][0-9]\s?°C\b",
        r"\b[0-9][0-9]\s?°C\b",
        r"\b[0-9]\s?°C\b",
    ]
    for sym in centigrade_syms:
        if len(re.findall(sym, context)) > 0:
            return True

    return context.find("°C") != -1


def find_name(seq, names):
    return any(one_name in seq for one_name in names)


def text_processing_from_json(examples,tokenizer, max_token_length=1024, operation="Crop",reset_ids=True):

    drop_data = 0
    samples = []
    samples_id = 0
    for item in tqdm(examples):
        id = item["id"]
        question = str(item["question"])
        sn = question.replace("what is the optimal growth temperature of ", "")
        sn = sn[:-1]

        synonym_description = str(item["synonym_description"])
        if synonym_description == "":
            synonym_description = ""
            name_in_context = [sn]
        else:
            name_in_context = synonym_description.split(" is also known as ")
            name_in_context = name_in_context[1]
            name_in_context = name_in_context[:-2]
            if name_in_context.find(" or ") != -1:
                name_in_context = name_in_context.split(" or ")
            else:
                name_in_context = [name_in_context]
            name_in_context.append(sn)
        if len(item["answers"]["answer_start"]) == 0:
            answer = -1
        else:
            answer = int(item["answers"]["answer_start"][0])
        context = item["context"]
        seqs = Sent_slipter.tokenize(context)
        seqs_length = [len(seq) + 1 for seq in seqs]
        seqs_length_persum = [
            sum(seqs_length[: i + 1]) for i in range(len(seqs_length))
        ]
        seqs_length_persum.append(0)
        if answer != -1:
            index_answer = next(
                i for i, persum in enumerate(seqs_length_persum) if persum > answer
            )
        else:
            index_answer = next(i for i, seq in enumerate(seqs) if find_centigrade(seq))

        indexes_name = [
            i
            for i, seq in enumerate(seqs)
            if i <= index_answer and find_name(seq, name_in_context)
        ]
        if len(indexes_name) == 0:
            indexes_start = [index_answer]
            try:
                index_name = next(
                    i
                    for i, seq in enumerate(seqs)
                    if i > index_answer and find_name(seq, name_in_context)
                )
            except:
                print(context)
                print(id)
                print(sn)
                print(item["id"])
                print(seqs_length_persum)
                print(answer)
                if answer != -1:
                    print("Except!!!!")
                continue
            indexes_end = range(index_name, len(seqs_length))
        else:
            indexes_start = range(indexes_name[-1] + 1)

            indexes_end = range(index_answer, len(seqs_length))

        cnt = 0
        max_context_length = max_token_length-len(tokenizer(synonym_description+question)['input_ids'])-4
        all_new_indexes = list(itertools.product(indexes_start,reversed(indexes_end)))

        '''if operation=="Augment":
            random.shuffle(all_new_indexes)'''

        for (index_start, index_end ) in all_new_indexes:
                new_context = " ".join(seqs[index_start : index_end + 1])
                tokenized = tokenizer(new_context)
                if len(tokenized['input_ids']) > max_context_length:
                    continue
                if operation=="Crop" and cnt > 0:
                    break
                if operation=="Augment" and cnt > 16: # 数据增强
                    break
                cnt += 1
                samples_id += 1
                new_start = answer

                if new_start != -1:
                    new_start -= seqs_length_persum[index_start - 1]

                item_appended = item.copy()
                if new_context.find(sn) != -1:
                    item_appended["synonym_description"] = ""

                item_appended["answers"]["answer_start"] = [new_start]

                if new_start == -1:
                    item_appended["answers"]["text"] = []
                    item_appended["answers"]["answer_start"] = []
                else:
                    true_answer_text = item_appended["answers"]["text"][0]
                    new_answer_text = new_context[new_start:new_start+len(true_answer_text)]
                    assert new_answer_text==true_answer_text,f"answer {new_answer_text} is not matched with {true_answer_text}"
                item_appended["context"] = new_context
                if reset_ids:
                    item_appended['id'] = samples_id
                samples.append(item_appended)

        if cnt == 0:
            print()
            print(indexes_start)
            print(id)
            print(sn)
            print(item["id"])
            print(seqs_length_persum)
            print(answer)
            if answer != -1:
                print("Attention!!!!")
            drop_data += 1
    print(f"Drop data: {drop_data}")

    
    return samples

def convrt_List_of_Dict_to_Dict_of_List(samples):
    samples_dict = {k:[] for k in samples[0].keys()}
    for dic in samples:
        for k, v in dic.items():
            samples_dict[k].append(v)
    return samples_dict

feat = Features(
        {
            "id": Value(dtype="int32", id=None),
            "title": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
            "synonym_description": Value(dtype="string", id=None),
            "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
        }
    )

def load_examples():
    data_files = {
        "train": "data/OGT_QA_train.json",
        "validation": "data/OGT_QA_validation.json",
        "test": "data/OGT_QA_dev.json",
    }
    raw_datasets = load_dataset("json", data_files=data_files, features=feat, cache_dir="/path/to/nonexistent/directory")
    return raw_datasets


def load_samples(ckpt_dir):
    data_files = {
        "train": f"{ckpt_dir}/train.json",
        "validation": f"{ckpt_dir}/validation.json",
        "test": f"{ckpt_dir}/dev.json",
    }
    datasets = load_dataset("json", data_files=data_files, features=feat)
    return datasets


def process_examples(examples, tokenizer, max_length, test_length,ckpt_dir):
    new_datasets = {}
    # new_datasets["test"] = convrt_List_of_Dict_to_Dict_of_List(examples["test"])

    samples = text_processing_from_json(
        examples["train"], tokenizer, max_length, operation="Augment"
    )
    print(f"process {len(examples['train'])} Training examples to {len(samples)} samples")
    write_json(f"{ckpt_dir}/train.json", samples)
    new_datasets["train"] =  convrt_List_of_Dict_to_Dict_of_List(samples)

    samples = text_processing_from_json(
        examples["validation"], tokenizer, max_length, operation="Augment"
    )
    print(f"process {len(examples['validation'])} Validation examples to {len(samples)} samples")
    write_json(f"{ckpt_dir}/validation.json", samples)
    new_datasets["validation"] = convrt_List_of_Dict_to_Dict_of_List(samples)

    samples = text_processing_from_json(
        examples["test"], tokenizer,test_length
    )
    print(f"Crop {len(examples['test'])} test examples to {len(samples)} samples")
    write_json(f"{ckpt_dir}/dev.json", samples)
    new_datasets["test"] = convrt_List_of_Dict_to_Dict_of_List(samples)
    return DatasetDict(
        {data: Dataset.from_dict(new_datasets[data]) for data in new_datasets}
    )

if __name__ == "__main__":
    context = pd.read_excel(
        "data/OGT_QA.xlsx", converters={"taxonomy_id": int, "answer_text": str}
    )
    new_data = context[~context["answer"].isna()]
    DATA_PATH = "data/"
    OGT_QA_rawdata = get_json_from_df(DATA_PATH, new_data,"OGT_QA_rawdata", check=1,write=False)
    print(f"Raw data : {len(OGT_QA_rawdata)}")

    
    # Load the Biogpt tokenizer to preprocess the datasets
    model_name_or_path = "PrLM/BioGPT-Large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    OGT_QA = text_processing_from_json(examples=OGT_QA_rawdata,tokenizer=tokenizer,max_token_length=1024,operation="Crop")
    write_json(DATA_PATH+"OGT_QA.json",OGT_QA)
    print(f"New data : {len(OGT_QA)}")
    json_OGT_train,json_OGT_validation,json_OGT_dev = split_datasets(OGT_QA,DATA_PATH, 0.6, 0.2)
    json_OGT_train = read_json(os.path.join(DATA_PATH, "OGT_QA_train.json"))
    samples = text_processing_from_json(examples=json_OGT_train,tokenizer=tokenizer, max_token_length=384,operation="Augment")
    train_dataset = Dataset.from_dict(convrt_List_of_Dict_to_Dict_of_List(samples)) 
    # text_processing_from_json(examples=json_OGT_validation,tokenizer=tokenizer, max_token_length=384,operation="Augment")
