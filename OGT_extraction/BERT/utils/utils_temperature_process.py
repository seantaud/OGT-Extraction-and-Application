import re
dict_number={

'zero': 0,

'one': 1,

'two': 2,

'three': 3,

'four': 4,

'five': 5,

'six': 6,

'seven': 7,

'eight': 8,

'nine': 9,

'ten': 10,

'eleven': 11,

'twelve': 12,

'thirteen': 13,

'fourteen': 14,

'fifteen': 15,

'sixteen': 16,

'seventeen': 17,

'eighteen': 18,

'nineteen': 19,

'twenty': 20,

'thirty': 30,

'forty': 40,

'fifty': 50,

'sixty': 60,

'seventy': 70,

'eighty': 80,

'ninety': 90

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
        if str == 'NaN':
            return False
        s = float(str)
        return True
    except ValueError:
        return False


def func_replace(answer_predict):
    answer_predict = str(answer_predict)
    answer_predict = strQ2B(answer_predict)
    answer_predict = answer_predict.replace('Â°C', '')
    answer_predict = answer_predict.replace('degrees Celsius', '')
    answer_predict = answer_predict.replace('degrees C', '')
    answer_predict = answer_predict.replace('0c', '')
    answer_predict = answer_predict.replace('˚C', '')
    answer_predict = answer_predict.replace('°C', '')
    answer_predict = answer_predict.replace('oC', '')
    answer_predict = answer_predict.replace('C', '')
    answer_predict = answer_predict.replace('°', '')
    answer_predict = answer_predict.replace('°/oo', '')
    answer_predict = answer_predict.replace('ºC', '')
    answer_predict = answer_predict.replace('º', '')
    answer_predict = answer_predict.replace('℃', '')
    answer_predict = answer_predict.replace(' ', '')  # special space
    answer_predict = answer_predict.replace(' ', '')  # special space ,or something unvisible
    answer_predict = answer_predict.replace(' ', '')  # special space ,or something unvisible
    answer_predict = answer_predict.replace(' ', '')
    answer_predict = answer_predict.replace(' ', '')
    answer_predict = answer_predict.replace(' ', '')
    answer_predict = answer_predict.replace(' ', '')

    r = re.compile(r'between[0-9.-]+and[0-9.-]+')
    if r.search(answer_predict):
        ans = r.search(answer_predict)[0]
        answer_predict = ans[7:]
        answer_predict = answer_predict.replace('and', '-')

    r = re.compile(r'from[0-9.-]+to[0-9.-]+')
    if r.search(answer_predict):
        ans = r.search(answer_predict)[0]
        answer_predict = ans[4:]
        answer_predict = answer_predict.replace('to', '-')

    r = re.compile(r'[0-9.-]+~[0-9.-]+')
    if r.search(answer_predict):
        answer_predict = r.search(answer_predict)[0]
        answer_predict = answer_predict.replace('~', '-')
        answer_predict = answer_predict.replace('∼', '-')

    answer_predict = answer_predict.replace('or', '-')
    answer_predict = answer_predict.replace('and', '-')
    answer_predict = answer_predict.replace('approximately', '')
    answer_predict = answer_predict.replace('about', '')
    answer_predict = answer_predict.replace('≈', '')
    answer_predict = answer_predict.replace('>', '')
    answer_predict = answer_predict.replace('<', '')
    answer_predict = answer_predict.replace('below', '')
    answer_predict = answer_predict.replace('above', '')

    answer_predict = answer_predict.replace('between', '')
    answer_predict = answer_predict.replace('to', '-')

    answer_predict = answer_predict.replace('~', '')
    answer_predict = answer_predict.replace('∼', '')
    answer_predict = answer_predict.replace('∘', '')
    answer_predict = answer_predict.replace('–', '-')  # special -
    answer_predict = answer_predict.replace('－', '-')
    answer_predict = answer_predict.replace('‒', '-')
    answer_predict = answer_predict.replace('—', '-')
    answer_predict = answer_predict.replace('−', '-')


    answer_predict = answer_predict.replace('·', '.')
    answer_predict = answer_predict.replace('.0', '')

    return answer_predict

def get_strict_temperature(ans_str):
    new_str = func_replace(ans_str)
    new_str = new_str.lower()
    if new_str.find('k') != -1:
        print("K:" + new_str)
        try:
            T = str(float(new_str[:-1]) - 273.15)
            if T < -20 or T > 120:
                return "Drop"
            return T
        except:
            return "Drop"
    r = re.compile(r'[0-9.]+±[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+±[0-9.]+')
            return 'Drop'
        return float(ans[0])
    r = re.compile(r'-[0-9.]+±[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"-[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) == 1:
            return -float(ans[0][1:])
        print('-[0-9.]+±[0-9.]+')
        return 'Drop'
    if new_str in dict_number:
        return dict_number[new_str]

    if is_number(new_str):
        T = float(new_str)
        if T <-20 or T > 120:
            return "Drop"
        return float(new_str)

    if (new_str.startswith('−')) and is_number(new_str[1:]):
        return -float(new_str[1:])

    return "Drop"

def get_temperature_medium(ans_str, ):
    # if ans_str =='16°C for A. parricida and 18°C':
    #     return "Drop"
    new_str = func_replace(ans_str)
    new_str = new_str.lower()
    # if new_str =='37,45-55' or new_str =='37,45,55,60,65,70-75' or new_str=='25,-45':
    #     return "Drop"
    if new_str.find('ph') != -1:
        return "Drop"
    if new_str.find('aw') != -1:
        return "Drop"


    if new_str.find('k')!=-1:
        print("K:"+new_str)
        try:
            T = str(float(new_str[:-1]) - 273.15)
            return T
        except:
            return "Wrong"
    

    r = re.compile(r'[0-9.]+±[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+±[0-9.]+')
            return 'Wrong'
        return float(ans[0])

    r = re.compile(r'[0-9.]+versus[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+versus[0-9.]+')
            return 'Wrong'
        return float(ans[1])

    r = re.compile(r'[0-9.]+,[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+,[0-9.]+')
            print(new_str)
            return 'Wrong'
        return float(ans[1])

    r = re.compile(r'[0-9.]+/[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+/[0-9.]+')
            return 'Wrong'
        return float(ans[1])

    r = re.compile(r'-[0-9.]+±[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"-[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) == 1:
            return -float(ans[0][1:])
        print('-[0-9.]+±[0-9.]+')
        return 'Wrong'

    r = re.compile(r'-[0-9.]+--[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"-[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]

        if len(ans) != 2:
            print(ans)
            print('-[0-9.]+--[0-9.]+')
            return 'Wrong'
        return (-float(ans[0][1:]) - float(ans[1][1:])) / 2.0

    r = re.compile(r'-[0-9.]+-[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"-[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('-[0-9.]+-[0-9.]+')
            return 'Wrong'
        return (float(ans[0]) - float(ans[1])) / 2.0

    r = re.compile(r'[0-9.]+-[0-9.]+')
    if r.search(new_str):
        ans = re.findall(r"[0-9.]+", new_str)
        ans = [item for item in ans if len(item) != 0]
        if len(ans) != 2:
            print('[0-9.]+-[0-9.]+')
            return 'Wrong'
        return (float(ans[0]) + float(ans[1])) / 2.0
    
    if new_str in dict_number:
        return dict_number[new_str]
    
    if is_number(new_str):
        return float(new_str)

    if (new_str.startswith('−')) and is_number(new_str[1:]):
        return -float(new_str[1:])

    return 'Wrong'


def get_temperature_standard(ans_str, ):
    std_str = func_replace(ans_str)
    new_str = std_str.lower()

    if new_str in dict_number:
        return dict_number[new_str]
    
    if is_number(new_str):
        return std_str

    if (new_str.startswith('−')) and is_number(new_str[1:]):
        return std_str

    patterns = [
        r'[0-9.]+k',
        r'[0-9.]+±[0-9.]+',
        r'-[0-9.]+±[0-9.]+',
        r'-[0-9.]+--[0-9.]+',
        r'-[0-9.]+-[0-9.]+',
        r'[0-9.]+-[0-9.]+'
    ]
    for p in patterns:
        r = re.compile(p)
        if r.search(new_str):
            return std_str
        
    return 'Wrong'


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


if __name__ == '__main__':
    # ans_str = '55−60'
    # ans = func_replace(ans_str)
    # print(ans)
    # ans = get_temperature(ans_str)
    # print(ans)
    #
    # ans_str = '-12.1 ± 0.2 °C'
    # ans = func_replace(ans_str)
    # print(ans)
    # ans = get_temperature(ans_str)
    # print(ans)

    ans = get_temperature('K:5gkg-1')
    print(ans)
