from os.path import isfile
import json
import re
import os
import time
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch


con_file = open("config.json")
config = json.load(con_file)
print(config)
con_file.close()
client = ElsClient(config['apikey'])
client.inst_token = config['insttoken']   #API

f0 = open(r'E:.txt', 'r')  #txt contains taxid
lines=f0.readlines()
for line in lines:
    taxid = line.strip('\n')
    filename = r'E:\\elsevierapisearch\\txt2\\' + taxid + '.txt'      #txt2  txt --keyword searching
    filenam1 =r'E:\\elsevierapisearch\\TAXID\\' + taxid + '.txt'      #TAXID  txt --scientific name or another names
    if isfile(filename) is False:
        f4=open(filenam1,'r')
        f1 = open(filename, 'a', encoding='utf-8')
        for line in f4.readlines():
            guanjianci = line[:-1]
            m = guanjianci+' optimal growth temperature'
            try:
                doc_srch = ElsSearch(m, 'sciencedirect')
                doc_srch.execute(client, get_all=True)
                f1.write(str(doc_srch.results))
                time.sleep(1)
            except:
                pass
        f1.close()
    else:
        print('done')
        pass


f0 = open(r'E:.txt', 'r')  #txt contains taxid
for line in f0.readlines():
    line = line[:-1]
    path1 = r'E:\\elsevierapisearch\\txt2\\'
    filename=path1 + line +'.txt'
    f2=open(filename,'r',encoding='utf-8')
    a = f2.read()
    patterns=r'DOI:(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)'   #Regular Expression  mat out doi
    tt= re.compile(patterns)
    result = tt.findall(a)
    list1 = []
    f3= open(filename, 'w', encoding='utf-8')
    for i in result:
        if i != ',':
            list1.append(i)
            f3.write(i)
            f3.write('\n')
            list1.pop()
    f3.close()



path0=r'E:\\elsevierapisearch\\txt1\\'      #txt1 --fulltext
path1=r'E:\\elsevierapisearch\\txt2\\'
f0 = open(r'E:.txt', 'r')  #txt contains taxid
for line in f0.readlines():
    full1_path=path1+line[:-1]+ '.txt'
    full0_path = path0 + line[:-1]+ '.txt'
    taxid = line[:-1]
    print(taxid)
    f1= open(full1_path, "r", encoding="utf-8")
    f2 =open(full0_path, "w", encoding="utf-8")
    i=0
    for line in f1.readlines():
        line = line[:-1]
        doi=line
        i+=1
        path2=r'E:\\elsevierapisearch\\txt1\\%s\\'%taxid
        path3 = r'E:\\elsevierapisearch\\error\\%s' % taxid      #doi that cannot get fulltext
        if not os.path.exists(path2):
            os.mkdir(path2)
        filename1=path2 + line.replace("/","_") + '.txt'
        filename2=path3 + '.txt'
        f3 = open(filename2, 'a', encoding='utf-8')
        if isfile(filename1) is False:
            count=0
            flag=0
            f1 = open(filename1, 'w', encoding='utf-8')
            doi_doc=FullDoc(doi=doi)
            if doi_doc.read(client):
                try:
                    f1.write(doi_doc.data['originalText'])
                except:
                    count=1
                f1.close()
                if(count==1):
                    print('none')
                    os.remove(filename1)
                    f3.write(line + " occurs error!\n")
                f3.close()
            else:
                flag=1
                pass
            if(flag):
                os.remove(filename1)
                f3.write(line + " occurs error!\n")


"""character conversion"""

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

for root,dirs,files in os.walk(r"E:\elsevierapisearch\txt1"):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        f2 = open(file_path, 'r', encoding='utf-8')
        s = f2.read()
        s=all_to_half(s)
        f3 = open(file_path, 'w', encoding='utf-8')
        f3.write(s)
        #print(s)

"""context location
"""
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import os
import re
import json
import linecache
punkt_param = PunktParameters()
abbreviation = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','al','spp','sp','fig','nov','e.g']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)
from matplotlib import pyplot as plt
def draw_hist(myList, Title, Xlabel, Ylabel,Xmin,Xmax):
    plt.hist(myList, 100)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.xlim(Xmin, Xmax)
    #plt.xticks([0, 5,  10, 15,20,25,30,40,50,60,70,80,100])
    plt.show()
test_num = 400
cnt = 0
ans = 0
path1=r"E:\\elsevierapisearch\\taxid.txt"
path2=r"E:\\elsevierapisearch\\txt1\\"
path3=r"E:\\elsevierapisearch\\TAXID\\"
i=0
areaList=[]
for i in range(17517,17854):
    taxid=linecache.getline(path1, i).strip()
    #print(taxid)
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
            #print(file_path)

            with open(file_path,encoding = 'utf -8') as f:
               sentences_selected = []
               sentences_selected2 = []
               #print(1)
               lines = f.readlines()
               f11=open(file_path,encoding = 'utf -8')
               s = f11.read()
               #print(s)
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
               #find temperature
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
                         #print(test)
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



import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import os
import re
import json
import linecache
punkt_param = PunktParameters()
abbreviation = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','al','spp','sp','fig','nov','e.g']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)
import os
import re
import json
import linecache
path1=r"E:\\elsevierapisearch\\taxid.txt"
path2=r"E:\\elsevierapisearch\\txt1\\"
path3=r"E:\\elsevierapisearch\\TAXID\\"
i=0
dict_list = []
for i in range(10001,17854):
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
            print(file_path)
            replacement=r'E:\\elsevierapisearch\\text1\\'+taxid+'\\'
            doi=file_path.replace(replacement,'').replace('.txt','').replace('_','/')
            filename1 = 'E:\\elsevierapisearch\\TAXID\\' + taxid + '.txt'
            f1 = open(filename1, "r", encoding='utf-8')
            scientificname=f1.read().replace('\n','')
            f2 = open(file_path, 'r', encoding='utf-8')
            for line in f2:
                line = line[:-1]
                if (line != ""):
                    synom=[]
                    for name in txt:
                        if line.find(name)>0:
                            #print(name)
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

file_name1 = r'E:\\elsevierapisearch\\final\\f0.json'
f=open(file_name1, 'w', encoding='utf-8')
json.dump(dict_list, f, ensure_ascii=False, indent=2)
