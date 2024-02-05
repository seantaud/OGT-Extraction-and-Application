from datasets import DatasetDict, Dataset
POSITIVE_ANSWER = "mentioned"
NEGATIVE_ANSWER = "not mentioned"

def preprocess_dataset(raw_datasets, tokenizer, purpose="extraction"):
        return DatasetDict(
            {
                data: Dataset.from_dict(preprocess_data(raw_datasets[data], tokenizer))
                for data in raw_datasets
            }
        )



def preprocess_data(samples, tokenizer):
    inputs = []
    outputs = []

    for sample in samples:
        context_tokenized = tokenizer(sample["context"])
        input_text = "Question:" + sample["question"] + sample["synonym_description"] + "Context:"+sample["context"]+tokenizer.sep_token + "Answer:"
        output_text = NEGATIVE_ANSWER
        answer = sample["answers"]["text"]
        if len(answer):
            output_text = func_replace(answer[0])

        inputs.append(input_text)
        outputs.append(output_text)

    return {"input": inputs, "output": outputs}

import re

dict_number = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def is_number(str):
    try:
        if str == "NaN":
            return False
        s = float(str)
        return True
    except ValueError:
        return False


def func_replace(answer_predict):
    answer_predict = str(answer_predict)
    answer_predict = strQ2B(answer_predict)
    answer_predict = answer_predict.replace("Â°C", "")
    answer_predict = answer_predict.replace("degrees Celsius", "")
    answer_predict = answer_predict.replace("degrees C", "")
    answer_predict = answer_predict.replace("0c", "")
    answer_predict = answer_predict.replace("0C", "")
    answer_predict = answer_predict.replace("˚C", "")
    answer_predict = answer_predict.replace("°C", "")
    answer_predict = answer_predict.replace("oC", "")
    answer_predict = answer_predict.replace("C", "")
    answer_predict = answer_predict.replace("°", "")
    answer_predict = answer_predict.replace("°/oo", "")
    answer_predict = answer_predict.replace("ºC", "")
    answer_predict = answer_predict.replace("º", "")
    answer_predict = answer_predict.replace("℃", "")
    answer_predict = answer_predict.replace(" ", "")  # special space
    answer_predict = answer_predict.replace(
        " ", ""
    )  # special space ,or something unvisible
    answer_predict = answer_predict.replace(
        " ", ""
    )  # special space ,or something unvisible
    answer_predict = answer_predict.replace(" ", "")
    answer_predict = answer_predict.replace(" ", "")
    answer_predict = answer_predict.replace(" ", "")
    answer_predict = answer_predict.replace(" ", "")

    r = re.compile(r"between[0-9.-]+and[0-9.-]+")
    if r.search(answer_predict):
        ans = r.search(answer_predict)[0]
        answer_predict = ans[7:]
        answer_predict = answer_predict.replace("and", "-")

    r = re.compile(r"from[0-9.-]+to[0-9.-]+")
    if r.search(answer_predict):
        ans = r.search(answer_predict)[0]
        answer_predict = ans[4:]
        answer_predict = answer_predict.replace("to", "-")

    r = re.compile(r"[0-9.-]+~[0-9.-]+")
    if r.search(answer_predict):
        answer_predict = r.search(answer_predict)[0]
        answer_predict = answer_predict.replace("~", "-")
        answer_predict = answer_predict.replace("∼", "-")

    answer_predict = answer_predict.replace("or", "-")
    answer_predict = answer_predict.replace("and", "-")
    answer_predict = answer_predict.replace("approximately", "")
    answer_predict = answer_predict.replace("about", "")
    answer_predict = answer_predict.replace("≈", "")
    answer_predict = answer_predict.replace(">", "")
    answer_predict = answer_predict.replace("<", "")
    answer_predict = answer_predict.replace("below", "")
    answer_predict = answer_predict.replace("above", "")

    answer_predict = answer_predict.replace("between", "")
    answer_predict = answer_predict.replace("to", "-")

    answer_predict = answer_predict.replace("~", "")
    answer_predict = answer_predict.replace("∼", "")
    answer_predict = answer_predict.replace("∘", "")
    answer_predict = answer_predict.replace("–", "-")  # special -
    answer_predict = answer_predict.replace("－", "-")
    answer_predict = answer_predict.replace("‒", "-")
    answer_predict = answer_predict.replace("—", "-")
    answer_predict = answer_predict.replace("−", "-")

    answer_predict = answer_predict.replace("·", ".")
    answer_predict = answer_predict.replace(".0", "")

    return answer_predict
