import re
from pandas.core.frame import DataFrame

def txt_to_col_name(txt):
    pattern = r"\*\*([A-Za-z]+)\*\*"
    matches = re.findall(pattern, txt)
    return [re.sub(r'\W+', '', word.lower()) for word in matches]

def replace_w(col_name):
    results = []
    for i in range(1, 6):
        try:
            result = col_name.replace('w', str(i))
            results.append(result)
        except:
            pass
    return results

def fill_years(txt):
    years = [str(year) for year in range(2000, 2020)]
    
    txt_modified = re.sub(r'yyyy', '{}', txt.lower())
    
    txts_final = [txt_modified.format(year) for year in years]
    
    return txts_final


def filter_columns_name(df:DataFrame, col:list):
    col_exist = [col for col in col if col in list(df.columns)]
    df_filtrado = df[col_exist]
    return df_filtrado

def print_list(x):
    print(f'example:{x[0:3]}\nlen: {len(x)}')

DISEASE  =['rwhibpe', 'rwcancre', 'rwlunge_m', 'rwrthatte', 'rwhearte', 'rwstroke', 'rwarthre']
EXPENSES =['rwoophos1y', 'rwoopfhho1y', 'rwoopden1y', 'rwooposrg1y', 'rwoopdoc1y', 'rwoopmd1y']
SUPPORT  =['rwmealhlp', 'rwshophlp', 'rwmedhlp', 'rwmoneyhlp','rwrafaany', 'rwrifaany','hwfcany', 'rwhigov']
BEHAVIOR =['rwsmokev', 'rwsmoken', 'rwsmokef', 'rwstrtsmok','rwoangry', 'rwosleep', 'rwodngr', 'rwopace', 'rwoplot', 'rwoalchl']
PLUS     =['rwwthh', 'rwagey', 'hhwctot1m', 'rwmomage', 'rwdadage', 'rwprmem', 'rwrjudg', 'rwrorgnz','rwhosp1y']

ID = 'rahhidnp'

YEAR = 'CyyyyCPINDEX'

COLUMN_LIST = DISEASE + EXPENSES + SUPPORT + BEHAVIOR + PLUS