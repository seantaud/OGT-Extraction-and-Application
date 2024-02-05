#This Python script interacts with the Elsevier API (ScienceDirect) to perform searches based on keywords obtained from a file containing taxids.
#Import modules
from os.path import isfile
import json
import re
import os
import time
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch

# Load API configuration from JSON file
# Please write your own apikey
con_file = open("config.json")
config = json.load(con_file)
print(config)
con_file.close()

# Initialize Elsevier API client
client = ElsClient(config['apikey'])
client.inst_token = config['insttoken']


# Read taxids from a file, here is taxid.txt.
# First, get doi number.
f0 = open(r'taxid.txt', 'r')
lines=f0.readlines()
for line in lines:
    taxid = line.strip('\n')

    # Define file paths
    filename = r'elsevier_keywords_doi_results\\' + taxid + '.txt'      #which restores doi results
    filenam1 =r'keyword_organisms\\' + taxid + '.txt'      #which retores the taxid corresponding the organism name

    # Check if the result file already exists
    if isfile(filename) is False:
        # Read keywords from the TAXID file
        f4=open(filenam1,'r')
        f1 = open(filename, 'a', encoding='utf-8')

        for line in f4.readlines():
            organism_name = line[:-1]
            m = organism_name+' optimal growth temperature'

            # Query ScienceDirect API with keywords
            try:
                doc_srch = ElsSearch(m, 'sciencedirect')
                doc_srch.execute(client, get_all=True)
                f1.write(str(doc_srch.results))
                time.sleep(1)
            except:
                pass
        # Close files
        f1.close()
    else:
        # Print message if the result file already exists
        print('done')
        pass

# This step is to mat out doi number
f0 = open(r'taxid.txt', 'r')
for line in f0.readlines():
    line = line[:-1]
    path1 = r'elsevier_keywords_doi_results\\' + taxid + '.txt'
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

# Next, use doi number to get fulltxt
f0 = open(r'taxid.txt', 'r')
#which retore doi number
path1=r'elsevier_keywords_doi_results\\'
for line in f0.readlines():

    full1_path = path1 + line[:-1] + '.txt'
    taxid = line[:-1]
    print(taxid)
    f1= open(full1_path, "r", encoding="utf-8")
    i=0
    for line in f1.readlines():
        line = line[:-1]
        doi=line
        i+=1
        path2=r'fulltxt\\%s\\'%taxid      #which retores fulltxt results
        path3 = r'error\\%s' % taxid      #which restores doi that cannot get fulltext
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