from request_phs import *

info_path = join(dirname(dirname(__file__)),
                 'database', 'customer', 'personal_info.xlsx')

info_table = pd.read_excel(info_path, index_col=[0])
info_table.columns = ['DOB', 'GENDER', 'ADDRESS', 'AOD', 'FTD']

dob = info_table[['DOB']]
dob['YEAR'] = pd.NaT
for account in dob.index:
    if isinstance(dob.loc[account,'DOB'], str):
        pass
    else:
        dob['YEAR'] = dob['DOB'].datetime.year