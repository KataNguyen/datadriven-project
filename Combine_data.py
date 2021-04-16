import os
import pandas as pd
import csv



bs_path = join(dirname(dirname(dirname(realpath(__file__)))), 'So_huu')
files = os.listdir(bs_path)

bs_table = pd.DataFrame()
for file in files:

    excel = Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(join(bs_path, file))
    time.sleep(3)  # suspend 3 secs for excel to catch up python
    excel.Range("A1:XFD1048576").Select()
    excel.Selection.Copy()
    excel.Selection.PasteSpecial(Paste=-4163)
    excel.ActiveWorkbook.Save()
    excel.ActiveWorkbook.Close()
    raw_fiinpro \
        = openpyxl.load_workbook(join(bs_path, file)).active
    # delete StoxPlux Sign
    raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row - 21,
                            amount=1000)
    # delete headers
    raw_fiinpro.delete_rows(idx=0, amount=7)
    raw_fiinpro.delete_rows(idx=2, amount=1)

    # import
    clean_data = pd.DataFrame(raw_fiinpro.values)
    clean_data.set_index([0], drop=True, inplace=True)
    clean_data.columns = clean_data.iloc[0]

    clean_data.drop(clean_data.index[:1], inplace=True)


    # process
    clean_data.insert(loc=2, column='Year', value=file.split('.')[0][-4:])

    clean_data.columns = [col.split('Ngày công bố:')[0] for col in clean_data.columns]

    # merge
    bs_table = pd.concat([bs_table, clean_data], axis=0, ignore_index= True )

    bs_table.to_csv(r'D:\Data\CCSH.csv', index=False, header=True)


