# coding:utf-8
# After getting fulltxt, the next step is context locating

#First, character conversion.
import os
def all_to_half(all_string):
    """Full-width turn half-angle"""
    half_string = ""
    for char in all_string:
        inside_code = ord(char)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        half_string += chr(inside_code)
    return half_string

for root,dirs,files in os.walk(r"fulltxt"):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        f2 = open(file_path, 'r', encoding='utf-8')
        s = f2.read()
        s=all_to_half(s)
        f3 = open(file_path, 'w', encoding='utf-8')
        f3.write(s)

#Next, context locating.
# Import libraries
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import os
import re
import linecache
from matplotlib import pyplot as plt

# Download NLTK punkt data
nltk.download('punkt')

# Set up PunktSentenceTokenizer with specific abbreviations
punkt_param = PunktParameters()
abbreviation = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','al','spp','sp','fig','nov','e.g']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)

# Function to draw histogram
def draw_hist(myList, Title, Xlabel, Ylabel,Xmin,Xmax):
    plt.hist(myList, 100)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.xlim(Xmin, Xmax)
    plt.show()
test_num = 400

path1=r"taxid.txt"
path2=r"fulltxt\\"
path3=r"elsevier_keywords_doi_results\\"
i=0
cnt = 0
ans = 0
areaList=[]
for i in range(1,18854):
    taxid=linecache.getline(path1, i).strip()
    filename2=path2+taxid
    print(filename2)
    filename3 = path3 + taxid+'.txt'
    print(filename3)
    f0=open(filename3,encoding = 'utf -8')
    txt = []
    for line in f0.readlines():
        curline = line[:-1]
        if (curline != ""):
            txt.append(curline[:])

    for root, dirs, files in os.walk(filename2):
        for file in files:
            file_path = os.path.join(root,file)

            # find temperature symbols , use regular expression
            with open(file_path,encoding = 'utf -8') as f:
               sentences_selected = []
               sentences_selected2 = []
               lines = f.readlines()
               f11=open(file_path,encoding = 'utf -8')
               s = f11.read()
               res1 = re.findall(r"\b12[0-1]\s?EC\b", s)
               res2 = re.findall(r"\b1[01][0-9]\s?EC\b", s)
               res3 = re.findall(r"\b[0-9][0-9]\s?EC\b", s)
               res4=re.findall(r"\b[0-9]\s?EC\b", s)
               res5 = re.findall(r"\b12[0-1]\s?C\b", s)
               res6 = re.findall(r"\b1[01][0-9]\s?C\b", s)
               res7 = re.findall(r"\b[0-9][0-9]\s?C\b", s)
               res8 = re.findall(r"\b[0-9]\s?C\b", s)
               res9 = re.findall(r"\b12[0-1]\s?oC\b", s)
               res10 = re.findall(r"\b1[01][0-9]\s?oC\b", s)
               res11 = re.findall(r"\b[0-9][0-9]\s?oC\b", s)
               res12 = re.findall(r"\b[0-9]\s?oC\b", s)
               res13 = re.findall(r"\b12[0-1]\s?degrees\b", s)
               res14 = re.findall(r"\b1[01][0-9]\s?degrees\b", s)
               res15 = re.findall(r"\b[0-9][0-9]\s?degrees\b", s)
               res16 = re.findall(r"\b[0-9]\s?degrees\b", s)
               res17 = re.findall(r"\b12[0-1]\s?[(]deg[)]C\b", s)
               res18 = re.findall(r"\b1[01][0-9]\s?[(]deg[)]C\b", s)
               res19 = re.findall(r"\b[0-9][0-9]\s?[(]deg[)]C\b", s)
               res20 = re.findall(r"\b[0-9]\s?[(]deg[)]C\b", s)
               res21 = re.findall(r"\b12[0-1]\s?°C\b", s)
               res22 = re.findall(r"\b1[01][0-9]\s?°C\b", s)
               res23 = re.findall(r"\b[0-9][0-9]\s?°C\b", s)
               res24 = re.findall(r"\b[0-9]\s?°C\b", s)
               result = res1 + res2 + res3+res4+res5+res6+res7+res8+res9+res10+res11+res12+res13+res14+res15+res16+res17+res18+res19+res20+res21+res22+res23+res24

               f1=open(file_path, 'w', encoding='utf -8')
               f1.close()
               for line in lines:
                  sentences = tokenizer.tokenize(line)
                  n = len(sentences)
                  indexming=[]
                  final=[]
                  for i in range(n):
                      sentence = sentences[i]
                      for name in txt:
                          if sentence.find(name) > 0 :
                              indexming.append(i)
                  for i in range(n):
                      sentence = sentences[i]
                      x0=0
                      for m in result:
                          if sentence.find(m)>0:
                              x0 =sentence.find(m)
                              break
                      if x0 > 0:
                         indexdu=[x - i for x in indexming]
                         test=[]
                         for j in indexdu:
                             if(j<=0):
                                test.append(j)
                         if(len(test)!=0):
                              testmax=max(test)
                              areaList.append((abs(testmax)))
                              xiabiao=test.index(testmax)
                              qishi=indexming[xiabiao]
                              mowei=i
                              final.append([qishi,mowei])

                  print(final)
                  f3 = open(file_path, 'a', encoding='utf -8')
                  for j in range(len(final)):
                      a=final[j][0]
                      b=final[j][1]
                      m=b+1-a
                      sentences_selected = []
                      sentences_selected1=[]
                      sentences_selected2 = []
                      setences_selected3=[]
                      if((m>6)and(m<=160)):
                          sentences_selected=sentences[a:b+1]
                      elif(m<6):
                          sentences_selected=sentences[b-6:b + 1]
                      else:
                          setences_selected2=sentences[a-1:b+1]
                          sentences_selected3 = ' '.join(sentences_selected2)
                      if(len(sentences_selected)!=0):
                          sentences_selected1 = ' '.join(sentences_selected)
                          f3.write(str(sentences_selected1))
                          f3.write('\n')

orgin_list=[[i,areaList[i]] for i in range(len(areaList))]
result = open('data11.xls', 'w', encoding='gbk')
result.write('X\tY\n')
for m in range(len(orgin_list)):
    for n in range(len(orgin_list[m])):
        result.write(str(orgin_list[m][n]))
        result.write('\t')
    result.write('\n')
result.close()



# Because we want to make all context locating results in one file and we want to record the positions of these names and temperatures within the corresponding sentence fragments
# So, the final step is below.
import os
import re
import json
import linecache
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
abbreviation = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','al','spp','sp','fig','nov','e.g']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)

path1=r"taxid.txt"
path2=r"fulltxt\\"
path3=r"elsevier_keywords_doi_results\\"

i=0
dict_list = []
for i in range(1,18854):
    taxid=linecache.getline(path1, i).strip()
    print(taxid)
    filename2=path2+taxid
    print(filename2)
    filename3 = path3 + taxid + '.txt'
    print(filename3)
    f0 = open(filename3, encoding='utf -8')
    txt = []
    for line in f0.readlines():
        curline = line[:-1]
        if (curline != ""):
            txt.append(curline[:])
    print(txt)
    for root, dirs, files in os.walk(filename2):
        for file in files:
            flag=0
            file_path = os.path.join(root, file)
            replacement=r'fulltxt\\'+taxid+'\\'
            doi=file_path.replace(replacement,'').replace('.txt','').replace('_','/')
            filename1 = r'elsevier_keywords_doi_results\\' + taxid + '.txt'
            f1 = open(filename1, "r", encoding='utf-8')
            scientificname=f1.read().replace('\n','')
            f2 = open(file_path, 'r', encoding='utf-8')
            for line in f2:
                line = line[:-1]
                if (line != ""):
                    synom=[]
                    for name in txt:
                        if line.find(name)>0:
                            synom.append(name)
                    synom1= ','.join(synom)
                    res1 = re.findall(r"\b12[0-1]\s?EC\b", line)
                    res2 = re.findall(r"\b1[01][0-9]\s?EC\b", line)
                    res3 = re.findall(r"\b[0-9][0-9]\s?EC\b", line)
                    res4 = re.findall(r"\b[0-9]\s?EC\b", line)
                    res5 = re.findall(r"\b12[0-1]\s?C\b", line)
                    res6 = re.findall(r"\b1[01][0-9]\s?C\b", line)
                    res7 = re.findall(r"\b[0-9][0-9]\s?C\b", line)
                    res8 = re.findall(r"\b[0-9]\s?C\b", line)
                    res9 = re.findall(r"\b12[0-1]\s?oC\b", line)
                    res10 = re.findall(r"\b1[01][0-9]\s?oC\b", line)
                    res11 = re.findall(r"\b[0-9][0-9]\s?oC\b", line)
                    res12 = re.findall(r"\b[0-9]\s?oC\b", line)
                    res13 = re.findall(r"\b12[0-1]\s?degrees\b", line)
                    res14 = re.findall(r"\b1[01][0-9]\s?degrees\b", line)
                    res15 = re.findall(r"\b[0-9][0-9]\s?degrees\b", line)
                    res16 = re.findall(r"\b[0-9]\s?degrees\b", line)
                    res17 = re.findall(r"\b12[0-1]\s?[(]deg[)]C\b", line)
                    res18 = re.findall(r"\b1[01][0-9]\s?[(]deg[)]C\b", line)
                    res19 = re.findall(r"\b[0-9][0-9]\s?[(]deg[)]C\b", line)
                    res20 = re.findall(r"\b[0-9]\s?[(]deg[)]C\b", line)
                    res21 = re.findall(r"\b12[0-1]\s?°C\b", line)
                    res22 = re.findall(r"\b1[01][0-9]\s?°C\b", line)
                    res23 = re.findall(r"\b[0-9][0-9]\s?°C\b", line)
                    res24 = re.findall(r"\b[0-9]\s?°C\b", line)
                    temp1 = res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13 + res14 + res15 + res16 + res17 + res18 + res19 + res20 + res21 + res22 + res23 + res24
                    temp = ' '.join(temp1)
                    try:
                        temp2=temp1[0]
                        xstr = str(temp1[0])
                        zstr =line
                        n = zstr.count(xstr)
                        xnum = 0
                        for i in range(n):
                            xnum = zstr.find(xstr, xnum)
                            start=xnum
                            end=xnum + len(xstr)
                            xnum = end
                    except:
                        temp2=0
                        start=0
                        end=0
                    sentences = tokenizer.tokenize(line)
                    n = len(sentences)
                    if(n!=1):
                        dict_obj = {'taxid': taxid, 'scientific_name':scientificname,'name_context':synom1,'context': line, 'temp':temp2,'start':start,'end':end,'length':n,'doi': doi}
                        dict_list.append(dict_obj)


# Finally, generate the json which contains the all context locating results.
file_name1 = r'final_results\\context_locating40.json'
f=open(file_name1, 'w', encoding='utf-8')
json.dump(dict_list, f, ensure_ascii=False, indent=2)
