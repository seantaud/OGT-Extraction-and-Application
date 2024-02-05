import pandas as pd

def clear_Topt(df):
    list = []
    for i in range(0, len(df)):
        uni_id = str(df.iloc[i]['uniport_id']).replace('"', '').replace('[', '').replace("'", '').replace(']','').replace(' ', '')
        if ',' in uni_id:
            for id in uni_id.split(','):
                item = {
                    'EC_number':df.iloc[i]['EC_number'],
                    'taxonomy_id': df.iloc[i]['taxonomy_id'],
                    'organism': df.iloc[i]['scientific_name'],
                    'Topt': df.iloc[i]['topt'],
                    'uniprot_id': id,
                    'commentary': df.iloc[i]['commentary']
                }
                list.append(item)
        else:
            item = {
                'EC_number': df.iloc[i]['EC_number'],
                'taxonomy_id': df.iloc[i]['taxonomy_id'],
                'organism': df.iloc[i]['scientific_name'],
                'Topt': df.iloc[i]['topt'],
                'uniprot_id': uni_id,
                'commentary': df.iloc[i]['commentary']
            }
            list.append(item)
    df_topt = pd.DataFrame([])
    df_topt = df_topt.append(list)
    print('Done')
    return df_topt

df = pd.read_excel('result_with_comment_of_all_need_deal.xls')
df_Topt = clear_Topt(df)
df_Topt.to_excel('result_ofall.xls', index=False)