import pandas as pd
import xlwt

def Take_the_intersection(path,path1,path2):
    file_path = path  #file which only includes taxid of organism the Martin offer
    raw_data = pd.read_excel(file_path, header=0)
    data = raw_data.values

    file_path1 = path1  # file includes topt ec
    raw_data1 = pd.read_excel(file_path1, header=0)
    data1 = raw_data1.values

    n=[]
    z=set(data[:,0])
    for i in range(0,len(data1[:,2])):  #check taxid
        x=set([data1[i,2]])
        if (x.isdisjoint(z)==False):
            n.append(i)                 #If the taxid is the same, record the number of downlines

    workbook=xlwt.Workbook(encoding='utf-8')
    booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    q=0
    for i in range(0,len(n)):
       m=n[i]
       #print(data1[m,5])
       if (data1[m,5]!='[]'):  #when uniport_id is not none,save
           for j in range (0,6):
               booksheet.write(q, j, data1[m,j])
               workbook.save(path2)
            q += 1

Take_the_intersection('taxid_organism_ogt.xls','result.xls','TOPT.xls')


