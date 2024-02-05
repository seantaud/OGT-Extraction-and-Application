import xlwt
import pandas as pd

def addition(path,path1):
    #additional information
    file_path = path
    raw_data = pd.read_excel(file_path, header=0)
    #print(raw_data)
    row=len(raw_data.index.values)
    column=raw_data.columns.values
    #print(column)
    data = raw_data.values

    useless=[]
    for i in range(0,row):
        if (data[i,4]!='additional information'):
            useless.append(i)

    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    q = 0
    for i in range(0, len(useless)):
        m = useless[i]
        # print(data1[m,5])
        for j in range(0, 6):
           booksheet.write(q, j, data[m, j])
           workbook.save(path1)
        q += 1

def aver_drop(path,path1):
    # average  #  example:20-35 ,we need to take an average
    file_path = path
    raw_data = pd.read_excel(file_path, header=0)
    # print(raw_data)
    row = len(raw_data.index.values)
    column = raw_data.columns.values
    # print(column)
    data = raw_data.values

    need_average=[]
    for i in range(0, row):
        if (str(data[i,4]).find('-')!=-1):
            need_average.append(i)

    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    q = 0
    for i in range(0, row):
        if (i.isdisjoint(need_average)==True):
            for j in range(0, 6):
                booksheet.write(q, j, data[q, j])
                workbook.save(path1)
        else:
            m = str(data[i, 4]).find('-')
            x = float(str(data[i, 4])[0:m - 1])
            y = float(str(data[i, 4])[m + 2:7])
            for j in range(0, 6):
                if (j!=4):
                    booksheet.write(q, j, data[q, j])
                    workbook.save(path1)
                else:
                    booksheet.write(q, j, (x+y)/2)
                    workbook.save(path1)
        q += 1

def Deduplication(path,path1):
    #if ec and uniport id all same,need to take an average
    file_path = path
    raw_data = pd.read_excel(file_path, header=0)
    # print(raw_data)
    row = len(raw_data.index.values)
    column = raw_data.columns.values
    # print(column)
    data = raw_data.values

    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    q = 0
    for i in range(0, row):
        data_de=float(data[i,4])
        count=0
        for j in range(0,row):
            if(data[j,5]==data[i,5]):
                if (data[j,2]==data[i,2]): #if ec and uniport id
                    count=count+1
                    data_de=data_de+float(data[j,2])
        q+=1
        if (count!=0):
            for j in range(0, 6):
                if (j != 4):
                    booksheet.write(q, j, data[q, j])
                    workbook.save(path1)
                else:
                    booksheet.write(q, j, int(data_de / count))
                    workbook.save(path1)

        else:
            for j in range(0, 6):
                booksheet.write(q, j, data[q, j])
                workbook.save(path1)

def final_de(path,path1):
    #Remove duplicate rows
    data = pd.DataFrame(pd.read_excel(path, 'Sheet1'))
    #print(data)
    re_row = data.duplicated()
    #print(re_row)
    no_re_row = data.drop_duplicates()
    no_re_row.to_excel(path1)


addition('TOPT.xls','TOPTwithoutaddition.xls')
aver_drop('TOPTwithoutaddition.xls','TOPTaverage')
Deduplication('TOPTaverage.xls','TOPTde.xls')
final_de('TOPTde.xls','TOPTfinal.xls')

