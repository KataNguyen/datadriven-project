from request_phs import *

info_path = join(dirname(dirname(__file__)),
                 'database', 'customer', 'personal_info.xlsx')

info_table = pd.read_excel(info_path)
info_table.columns = ['TRADING CODE', 'DOB', 'GENDER', 'ADDRESS', 'AOD', 'FTD']

def ytime(datetime_obj):
    try:
        year = datetime_obj.year
    except AttributeError:
        year = pd.NaT
    return year

info_table['YOB'] = info_table['DOB'].apply(ytime)

yob_analysis = pd.pivot_table(info_table,
                              values='TRADING CODE',
                              index=['YOB'], aggfunc='count')

