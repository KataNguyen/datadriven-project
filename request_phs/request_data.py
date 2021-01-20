import numpy as np
import pandas as pd
import openpyxl
import os
import sys
from os import listdir
from os.path import isfile, isdir, join
from win32com.client import Dispatch
import time
from datetime import datetime, timedelta
import requests
import json
import holidays
from typing import Union

database_path \
    = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
           'database')


def request_fstype() -> list:

    """
    This function returns valid financital statements

    :param: None
    :return: list
    """

    folders = [f for f in listdir(database_path) if f.startswith('fs_')]
    fs_types = []
    for folder in folders:
        fs_types = [name.split('_')[0]
                    for name in listdir(join(database_path, folder))
                    if isfile(join(database_path, folder, name))]
        fs_types = list(dict.fromkeys(fs_types))
    fs_types = [x for x in fs_types if not x.startswith('~$')]

    return fs_types


def request_segment() -> list:

    """
    This function returns the names of segments

    :param: None
    :return: list
    """

    folders = [f for f in listdir(database_path) if f.startswith('fs_')]
    segments = [x.split('_')[1] for x in folders]

    return segments


def request_financial_ticker(sector_break=False) \
        -> Union[list, dict[str, list]]:

    """
    This function returns all tickers of financial segments

    :param sector_break: False: ignore sectors, True: show sectors
    :return: list (sector_break=False), dictionary (sector_break=True)
    """

    financials = ['bank', 'sec', 'ins']
    periods = request_period()
    periods.sort()
    latest_period = periods[-1]

    tickers = []
    tickers_ = dict()
    for segment in financials:
        folder = 'fs_' + segment + '_industry'
        file = 'is_' + latest_period[:4] + 'q' + latest_period[-1] + '.xlsm'
        raw_fiinpro \
            = openpyxl.load_workbook(
            os.path.join(database_path, folder, file)).active
        # delete StoxPlux Sign
        raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row-21,
                                amount=1000)
        # delete headers
        raw_fiinpro.delete_rows(idx=0, amount=7)
        raw_fiinpro.delete_rows(idx=2, amount=1)
        # import
        clean_data = pd.DataFrame(raw_fiinpro.values)
        clean_data.drop(index=[0], inplace=True)
        # remove OTC
        a = clean_data.loc[:, 3] != 'OTC'
        if sector_break is False:
            tickers += clean_data.loc[:,1][a].tolist()
        else:
            tickers_[segment] = clean_data.loc[:,1][a].tolist()
            tickers = tickers_

    return tickers


def reload() -> None:

    """
    This function handles cached data in newly-added files

    :param: None
    :return: None
    """

    global database_path
    folder_names = [folder
                    for folder in listdir(database_path)
                    if isdir(join(database_path, folder))]

    for folder in folder_names:
        file_names = [file
                      for file in listdir(join(database_path, folder))
                      if isfile(join(database_path, folder, file))]

        for file in file_names:
            excel = Dispatch("Excel.Application")
            excel.Visible = True
            excel.Workbooks.Open(os.path.join(database_path, folder, file))
            time.sleep(3) # suspend 3 secs for excel to catch up python
            excel.Range("A1:XFD1048576").Select()
            excel.Selection.Copy()
            excel.Selection.PasteSpecial(Paste=-4163)
            excel.ActiveWorkbook.Save()
            excel.ActiveWorkbook.Close()


def request_fs_period(year=int, quarter=int, segment=str, fs_type=str) \
        -> pd.DataFrame:

    """
    This function extracts data from Github server, clean up
    and make it ready for use

    :param year: reported year
    :param quarter: reported quarter
    :type year: int
    :param segment: allow values in request_segment()
    :type quarter: int
    :type segment: str
    :param fs_type: allow valeus in request_fstype()
    :type fs_type: str

    :return: pandas.DataFrame
    :raise ValueError: this function yet supported cashflow for
    securities companies
    """

    global database_path
    segments = request_segment()
    fs_types = request_fstype()

    if segment not in segments:
        raise ValueError(f'sector must be in {segments}')

    if fs_type not in fs_types:
        raise ValueError(f'sector must be in {fs_types}')

    folder = 'fs_' + segment + '_industry'
    file = fs_type + '_' + str(year) + 'q' + str(quarter) + '.xlsm'

    # create Workbook object, select active Worksheet
    raw_fiinpro \
        = openpyxl.load_workbook(
        os.path.join(database_path, folder, file)).active

    # delete StoxPlux Sign
    raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row-21,
                            amount=1000)

    # delete header rows
    raw_fiinpro.delete_rows(idx=0, amount=7)
    raw_fiinpro.delete_rows(idx=2, amount=1)

    # import to DataFrame, no column labels, no index
    clean_data = pd.DataFrame(raw_fiinpro.values)

    # assign column labels and index
    clean_data.columns = clean_data.iloc[0,:]
    clean_data.drop(index=[0], inplace=True)
    clean_data.index \
        = pd.MultiIndex.from_arrays([[year]*len(clean_data),
                                     [quarter]*len(clean_data),
                                     clean_data['Ticker'].tolist()])
    clean_data.index.set_names(['year', 'quarter', 'ticker'], inplace=True)

    # rename 2 columns
    clean_data.rename(columns=
                      {'Name':'full_name', 'Exchange':'exchange'},
                      inplace=True)

    # drop unwanted columns and index
    clean_data.drop(columns=['No', 'Ticker'], inplace=True)

    # fill na with 0s
    clean_data.fillna(0, inplace=True)

    # remove OTC
    clean_data = clean_data.loc[clean_data['exchange'] != 'OTC']


    if segment == 'bank':
        if fs_type == 'bs':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('A.','B.','C.','D.','E.','F.','G.')):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 :clean_data.columns[col].split()[0]},
                        inplace=True)
                    header.append(clean_data.columns[col])
                else:
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 :header[-1]
                                  + clean_data.columns[col].split()[0]},
                        inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += '_'
            clean_data.columns = col_list

            subheader = list('I')
            for col in range(2, len(clean_data.columns)):
                l = clean_data.columns[col].split('.')
                a = l[1]
                if a in ['I','II','III','IV','V','VI','VII']:
                    subheader.append(a)
                else:
                    name_new = l
                    name_new.insert(1,subheader[-1])
                    name_new = '.'.join(name_new)
                    clean_data.rename(
                        columns={clean_data.columns[col]:name_new},
                        inplace=True)

            clean_data.rename(axis=1, mapper=
                lambda x: x.rstrip('_'), inplace=True)

        elif fs_type == 'is':
            for col in range(2, len(clean_data.columns)):
                clean_data.rename(
                    columns={clean_data.columns[col]
                             :clean_data.columns[col].split()[0]},
                    inplace=True)

        elif fs_type == 'cfi':
            header = list()
            for col in range(len(clean_data.columns)-1,1,-1):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : header[-1]
                                      + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)

        elif fs_type == 'cfd':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : header[-1]
                                  + clean_data.columns[col].split()[0]},
                        inplace=True)
            col_list = clean_data.columns.tolist()
            duplicated = clean_data.columns.duplicated()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += 'b.'
            clean_data.columns = col_list


    elif segment == 'gen':
        # remove financial
        fin = set(request_financial_ticker()) \
              & set(clean_data.index.get_level_values(2))
        clean_data.drop(index=fin, level=2, inplace=True)
        if fs_type == 'bs':
            header = list()
            for col in range(len(clean_data.columns)-1,1,-1):
                if clean_data.columns[col]\
                    .startswith(('A.','B.','C.','D.','E.','F.','G.')):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 :clean_data.columns[col].split()[0]},
                        inplace=True)
                    header.append(clean_data.columns[col])
                else:
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 :header[-1]
                                  + clean_data.columns[col].split()[0]},
                        inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += '_'
            clean_data.columns = col_list

            subheader = list('I')
            for col in range(2, len(clean_data.columns)):
                l = clean_data.columns[col].split('.')
                a = l[1]
                if a in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    subheader.append(a)
                else:
                    name_new = l
                    name_new.insert(1, subheader[-1])
                    name_new = '.'.join(name_new)
                    clean_data.rename(
                        columns={clean_data.columns[col]: name_new},
                        inplace=True)

            clean_data.rename(axis=1, mapper=
            lambda x: x.rstrip('_'), inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i] and col_list[i].split('.',1)[1] \
                        not in ['I.','II.','III.','IV','V.','VI.','VII.']:
                    col_list[i] += 'b.'
            clean_data.columns = col_list

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                l = col_list[i].split('.')
                if duplicated[i] and l[1]\
                    in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    col_list[i] = l[0] + '.'
            clean_data.columns = col_list

        elif fs_type == 'is':
            for col in range(2, len(clean_data.columns)):
                clean_data.rename(
                    columns={clean_data.columns[col]
                             :clean_data.columns[col].split()[0]},
                    inplace=True)

        elif fs_type == 'cfi':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : header[-1]
                                      + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)

        elif fs_type == 'cfd':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : header[-1]
                                      + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)


    elif segment == 'ins':
        if fs_type == 'bs':
            header = list()
            for col in range(len(clean_data.columns)-1,1,-1):
                if clean_data.columns[col]\
                    .startswith(('A.','B.','C.','D.','E.','F.','G.')):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 :clean_data.columns[col].split()[0]},
                        inplace=True)
                    header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     :header[-1]
                                      + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += '_'
            clean_data.columns = col_list

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += '_'
            clean_data.columns = col_list

            subheader = list('I')
            for col in range(2, len(clean_data.columns)):
                l = clean_data.columns[col].split('.')
                a = l[1]
                if a in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    subheader.append(a)
                else:
                    name_new = l
                    name_new.insert(1, subheader[-1])
                    name_new = '.'.join(name_new)
                    clean_data.rename(
                        columns={clean_data.columns[col]: name_new},
                        inplace=True)

            clean_data.rename(axis=1, mapper=
                lambda x: x.rstrip('__').rstrip('_'), inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                l = col_list[i].split('.')
                if duplicated[i] and l[2] != '':
                    col_list[i] += 'b.'
            clean_data.columns = col_list

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                l = col_list[i].split('.')
                if duplicated[i] and l[1]\
                    in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    col_list[i] = l[0] + '.'
            clean_data.columns = col_list

            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                a = col_list[i].split('.')[1]
                if col_list[i].startswith(('1','2','3','4','5')):
                    col_list[i] = col_list[i].replace(a + '.', '')
            clean_data.columns = col_list

        elif fs_type == 'is':
            for col in range(2, len(clean_data.columns)):
                clean_data.rename(
                    columns={clean_data.columns[col]
                             :clean_data.columns[col].split()[0]},
                    inplace=True)
            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += 'b.'
                elif col_list[i] == '2201.':
                    col_list[i] = '20.1.'
            clean_data.columns = col_list

        elif fs_type == 'cfi':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : header[-1]
                                  + clean_data.columns[col].split()[0]},
                        inplace=True)

            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                a = col_list[i].split('.')
                if len(a[-1]) >= 5:
                    col_list[i] = '.'.join(a[:2]) + '.'
            clean_data.columns = col_list

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += 'b.'
                    break
            clean_data.columns = col_list

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += 'c.'
            clean_data.columns = col_list

        elif fs_type == 'cfd':
            header = list()
            for col in range(2, len(clean_data.columns)):
                if clean_data.columns[col]\
                    .startswith(('I','II','III','IV','V','VI','VII')):
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
                        header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : header[-1]
                                      + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     : clean_data.columns[col].split()[0]},
                            inplace=True)
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                a = col_list[i].split('.')
                if len(a[-1]) >= 1:
                    col_list[i] = '.'.join(a[:-1]) + '.'
            clean_data.columns = col_list


    elif segment == 'sec':
        if fs_type == 'bs':
            header = list()
            for col in range(len(clean_data.columns)-1,1,-1):
                if clean_data.columns[col] \
                        .startswith(
                    ('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.')):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : clean_data.columns[col].split()[0]},
                        inplace=True)
                    header.append(clean_data.columns[col])
                else:
                    try:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     :header[-1]
                                       + clean_data.columns[col].split()[0]},
                            inplace=True)
                    except IndexError:
                        clean_data.rename(
                            columns={clean_data.columns[col]
                                     :clean_data.columns[col].split()[0]},
                            inplace=True)


            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                if duplicated[i]:
                    col_list[i] += '_'
            clean_data.columns = col_list

            subheader = list('I')
            for col in range(2, len(clean_data.columns)):
                l = clean_data.columns[col].split('.')
                a = l[1]
                if a in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    subheader.append(a)
                else:
                    name_new = l
                    name_new.insert(1, subheader[-1])
                    name_new = '.'.join(name_new)
                    clean_data.rename(
                        columns={clean_data.columns[col]: name_new},
                        inplace=True)

            clean_data.rename(axis=1, mapper=
            lambda x: x.rstrip('_'), inplace=True)

            duplicated = clean_data.columns.duplicated()
            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                l = col_list[i].split('.')
                if duplicated[i] and l[1]\
                    in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    col_list[i] = l[0] + '.'
            clean_data.columns = col_list

            col_list = clean_data.columns.tolist()
            for i in range(2, len(clean_data.columns)):
                l = col_list[i].split('.')
                if l[0] in ['1','2','3','4','5'] and l[1]\
                    in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                    col_list[i] = col_list[i].replace('.'+l[1], '')
            clean_data.columns = col_list

        elif fs_type == 'is':
            for col in range(2, len(clean_data.columns)):
                clean_data.rename(
                    columns={clean_data.columns[col]
                             :clean_data.columns[col].split()[0]},
                    inplace=True)

        elif fs_type == 'cfi':
            pass
#            raise ValueError('This functionality was not '
#                             'written in "request_phs"')

        elif fs_type == 'cfd':
            pass
#            raise ValueError('This functionality was not '
#                             'written in "request_phs"')


    clean_data.columns \
        = pd.MultiIndex.from_product([[segment], [fs_type],
                                      clean_data.columns.tolist()],
                                     names = ['segment', 'fs_type', 'item'])
    print('Extracting...')
    return clean_data


def request_fs_all(segment='gen') -> pd.DataFrame:

    """
    This function returns all financial statements
    of all companies in all periods

    :param segment: allow values in request_segment()
    :return: pandas.DataFrame
    """

    global database_path
    fs_types = request_fstype()
    periods = request_period()

    frames = list()
    for period in periods:
        for fs_type in fs_types:
            try:
                frames.append(
                    request_fs_period(int(period[:4]),
                                      int(period[-1]),
                                      segment, fs_type))
            except FileNotFoundError:
                continue

    df = pd.concat(frames, axis=1, join='outer')\
        .groupby(axis=1, level=[2],
                 dropna=False, sort=False, group_keys=True).transform(lambda x)\
        .sum(min_count=1)
    df.drop(columns=['exchange', 'full_name'], inplace=True)

    return df


def request_fs_ticker(ticker=str):

    global database_path
    folder = 'fs_gen_industry'

    file_names = [f for f in listdir(join(database_path, folder))
                  if isfile(join(database_path, folder, f))]

    refs = [(int(name[-11:-7]),int(name[-6]),name[:2]
            if name[2]=='_' else name[:3]) for name in file_names]

    fs_types \
        = list(dict.fromkeys([name.split('_')[0]
                              for name in listdir(join(database_path, folder))
                              if isfile(join(database_path, folder, name))]))
    fs_types.sort()

    for fs_type in fs_types:
        if fs_type.startswith('~$'):
            fs_types.remove(fs_type)

    inds = list()
    for fs_type in fs_types:
        inds.append(
            request_fs_period(year=refs[-1][0],
                              quarter=refs[-1][1],
                              fs_type=fs_type) \
                .xs(ticker, axis=0, level=2) \
                .drop(['full_name', 'exchange'], level=1, axis=1).T \
                .index.tolist())

    inds = dict([(fs_type, ind) for fs_type, ind in zip(fs_types, inds)])

    fs = pd.concat(
        [
            request_fs_period(year=ref[0], quarter=ref[1], fs_type=ref[2])
                .xs(ticker, axis=0, level=2)
                .drop(['full_name', 'exchange'], axis=1).T
                .set_index(pd.MultiIndex.from_product(
                    [[ref[2]], inds[ref[2]]]))
            for ref in refs
        ]
    )
    fs = fs.groupby(fs.index, sort=False).sum()

    print('Data Extracted!')
    return fs


def request_ticker():

    last_period = request_period()[-1]
    tickerlist \
        = request_fs_period(year=int(last_period[:4]),
                            quarter=int(last_period[-1]),
                            fs_type='is')\
        .index.get_level_values(level=2).tolist()

    return tickerlist


def request_crash_list(benchmark=-0.5, period=str):
    returns = request_return()
    crash = list()
    for ticker in returns.index:
        if returns.loc[ticker, period] <= benchmark:
            crash.append(ticker)
        else:
            continue
    return crash


def request_fs_variable(fs_type=str):

    last_period = request_period()[-1]
    variable_names \
        = request_fs_period(year=int(last_period[:4]),
                            quarter=int(last_period[-1]),
                            fs_type=fs_type)\
        .columns.get_level_values(level=1).tolist()
    try:
        variable_names.remove('full_name')
        variable_names.remove('exchange')
    except ValueError:
        pass

    return variable_names


def request_period():

    folder = 'fs_gen_industry'
    periods \
        = list(dict.fromkeys(
        [name[-11:-5] for name in listdir(join(database_path, folder))
         if isfile(join(database_path, folder, name))]))
    periods.sort()

    return periods


def request_industry(standard=str):

    global database_path
    standards = request_industry_standard()
    st_dict = dict()
    folder = 'industry_classification'

    for st in standards:
        # create Workbook object, select active Worksheet
        raw_bloomberg \
            = openpyxl.load_workbook(
            os.path.join(database_path, folder, st+'.xlsx')
        ).active
        # delete Bloomberg Sign
        raw_bloomberg.delete_rows(idx=raw_bloomberg.max_row)
        # delete headers
        raw_bloomberg.delete_rows(idx=0, amount=2)
        clean_data = pd.DataFrame(raw_bloomberg.values)
        clean_data.iloc[1] = clean_data.iloc[0]
        clean_data.drop(index=0, inplace=True)
        # set index and columns
        clean_data.index = clean_data.iloc[:,0]
        clean_data.index.rename('ticker')
        clean_data.columns = clean_data.iloc[0,:]
        clean_data.columns.rename('level')
        # remore unwanted columns, rows
        clean_data.drop(columns=['Ticker', 'Name'],
                        index=['Ticker'], inplace=True)
        # rename columns
        clean_data.columns\
            = pd.Index(
            data = [
                clean_data.columns[i].split()[0].lower()
                    .split(' ', maxsplit=1)[0] + '_l' + str(i+1)
                for i in range(clean_data.shape[1])
            ]
        )
        # rename index
        clean_data.index \
            = pd.Index(
            data=[clean_data.index[i].split()[0]
                for i in range(clean_data.shape[0])]
        )

        st_dict[st] = clean_data

    return st_dict[standard]


def request_industry_standard():
    global database_path
    folder = 'industry_classification'
    standards \
        = list(dict.fromkeys(
        [name[:-5] for name in listdir(join(database_path, folder))
         if isfile(join(database_path, folder, name))]))
    return standards


def request_industry_level(standard=str):
    levels = request_industry(standard).columns.tolist()
    return levels


def request_industry_list(standard=str, level=int):
    industries \
        = request_industry(standard)[standard + '_l' + str(level)]\
        .drop_duplicates().tolist()
    return industries


def request_return():

    periods = request_period()
    tickers = request_ticker()

    returns = pd.DataFrame(data=np.zeros((len(tickers),len(periods))),
                           columns=[periods[i]
                                    for i in range(len(periods))],
                           index=[tickers[j]
                                  for j in range(len(tickers))])

    date_input = {'q1':['01-01','03-31'], 'q2':['04-01','06-30'],
                  'q3':['07-01','09-30'], 'q4':['10-01','12-28']}
    # Note: because of the 'Bday' function structure,
    #       December must be adjusted to 28 instead of 31

    price_data = dict()
    for ticker in tickers:
        try:
            price_data[ticker] = request_trading_hist(ticker)
        except (KeyError, IndexError):
            continue

    def Bday(date=str):
        date_ = datetime(year=int(date.split('-')[0]),
                         month=int(date.split('-')[1]),
                         day=int(date.split('-')[2]))
        one_day = timedelta(days=1)
        while date_.weekday() in holidays.WEEKEND or date_ in holidays.VN():
            date_ = date_ + one_day

        return date_.strftime(format='%Y-%m-%d')

    for period in periods:
        for ticker in tickers:
            try:
                fromdate = Bday(period[:4] + '-' + date_input[period[-2:]][0])
                todate = Bday(period[:4] + '-' + date_input[period[-2:]][1])
            except (KeyError, IndexError):
                continue
            try:
                f = price_data[ticker]['trading_date'] == fromdate
                t = price_data[ticker]['trading_date'] == todate
                try:
                    open_price = price_data[ticker].loc[f]['close'].iloc[0]
                    close_price = price_data[ticker].loc[t]['close'].iloc[0]
                    returns.loc[ticker, period] = close_price / open_price - 1
                except (IndexError, KeyError, RuntimeWarning):
                    returns.loc[ticker, period] = np.nan
            except (IndexError, KeyError, RuntimeWarning):
                returns.loc[ticker, period] = np.nan

    returns.replace([np.inf, -np.inf, -1, 1], np.nan, inplace=True)

    return returns


def request_price():

    periods = request_period()
    tickers = request_ticker()

    prices = pd.DataFrame(data=np.zeros((len(tickers),len(periods))),
                          columns=[periods[i]
                                   for i in range(len(periods))],
                          index=[tickers[j]
                                 for j in range(len(tickers))])

    date_input = {'q1':'03-31', 'q2':'06-30',
                  'q3':'09-30', 'q4':'12-28'}
    # Note: because of the 'Bday' function structure,
    #       December must be adjusted to 28 instead of 31

    price_data = dict()
    for ticker in tickers:
        try:
            price_data[ticker] = request_trading_hist(ticker)
        except (KeyError, IndexError):
            continue

    def Bday(date=str):
        date_ = datetime(year=int(date.split('-')[0]),
                         month=int(date.split('-')[1]),
                         day=int(date.split('-')[2]))
        one_day = timedelta(days=1)
        while date_.weekday() in holidays.WEEKEND or date_ in holidays.VN():
            date_ = date_ + one_day

        return date_.strftime(format='%Y-%m-%d')

    for period in periods:
        for ticker in tickers:
            try:
                date = Bday(period[:4] + '-' + date_input[period[-2:]])
            except (KeyError, IndexError):
                continue
            try:
                t = price_data[ticker]['trading_date'] == date
                try:
                    prices.loc[ticker, period] \
                        = price_data[ticker].loc[t]['close'].iloc[0]
                except (IndexError, KeyError, RuntimeWarning):
                    prices.loc[ticker, period] = np.nan
            except (IndexError, KeyError, RuntimeWarning):
                prices.loc[ticker, period] = np.nan

    prices.replace([np.inf, -np.inf, -1, 1], np.nan, inplace=True)
    return prices


def ownership_structure():

    global database_path
    folder = 'ownership'
    file = [f for f in listdir(join(database_path, folder))
            if isfile(join(database_path, folder, f))][-1]

    excel = Dispatch("Excel.Application")
    excel.Visible = True
    for wb in [wb for wb in excel.Workbooks]:
        wb.Close(True)
    excel.Workbooks.Open(os.path.join(database_path, file))
    time.sleep(3) # suspend 3 secs for excel to catch up python
    excel.Range("A1:XFD1048576").Select()
    excel.Selection.Copy()
    excel.Selection.PasteSpecial(Paste=-4163)
    excel.ActiveWorkbook.Save()
    excel.ActiveWorkbook.Close()

    # create Workbook object, select active Worksheet
    raw_fiinpro = openpyxl.load_workbook(
        os.path.join(database_path, folder, file)
    ).active
    try:
        # delete StoxPlux Sign
        raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row-21, amount=1000)
    except IndexError:
        excel = Dispatch("Excel.Application")
        excel.Visible = True
        for wb in [wb for wb in excel.Workbooks]:
            wb.Close(True)
        excel.Workbooks.Open(os.path.join(database_path, file))
        time.sleep(3) # suspend 3 secs for excel to catch up python
        excel.Range("A1:XFD1048576").Select()
        excel.Selection.Copy()
        excel.Selection.PasteSpecial(Paste=-4163)
        excel.ActiveWorkbook.Save()
        excel.ActiveWorkbook.Close()
        # delete StoxPlux Sign
        raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row - 21,
                                amount=1000)

    # get result info
    date_of_extract = raw_fiinpro['B6'].value

    # delete header rows
    raw_fiinpro.delete_rows(idx=0, amount=7)
    raw_fiinpro.delete_rows(idx=2, amount=1)

    # import to DataFrame
    clean_data = pd.DataFrame(raw_fiinpro.values)

    # drop unwanted index and columns
    clean_data.drop(columns=[0], inplace=True)
    clean_data.drop(index=[0], inplace=True)

    # set ticker as index
    clean_data.index = clean_data.iloc[:,0]
    clean_data.drop(clean_data.columns[0], axis=1, inplace=True)

    # rename columns
    columns = ['full_name', 'exchange', 'state_share',
               'state_percent', 'frgn_share', 'frgn_percent',
               'other_share', 'other_percent', 'frgn_maxpercent',
               'frgn_maxshare','frgn_remainshare']
    clean_data.columns = columns

    return clean_data, date_of_extract


def request_trading_hist(ticker=str, fromdate=None, todate=None):
    address = 'https://api.phs.vn/market/utilities.svc/GetShareIntraday'
    pd.options.mode.chained_assignment = None
    if fromdate is not None and todate is not None:
        if datetime.strptime(fromdate, '%Y-%m-%d') < \
                datetime(year=2015, month=1, day=1):
            raise Exception('Only data since 2015/01/01 is reliable')
        else:
            try:
                r = requests.post(address,
                    data=json.dumps(
                        {'symbol': ticker,
                         'fromdate': fromdate,
                         'todate': todate}
                    ),
                    headers={'content-type': 'application/json'})
                history = pd.DataFrame(json.loads(r.json()['d']))
            except KeyError:
                raise Exception('Date Format Required: yyyy-mm-dd, yyyy/mm/dd')
    else:
        try:
            r = requests.post(address,
                              data=json.dumps(
                                  {'symbol': ticker,
                                   'fromdate': datetime(year=2015,
                                                        month=1,
                                                        day=1) \
                                       .strftime("%Y-%m-%d"),
                                   'todate': datetime.now()
                                       .strftime("%Y-%m-%d")}
                              ),
                              headers={'content-type': 'application/json'})
            history = pd.DataFrame(json.loads(r.json()['d']))
        except KeyError:
            try:
                r = requests.post(address,
                                  data=json.dumps(
                                      {'symbol': ticker,
                                       'fromdate': (datetime.now()
                                                    - timedelta(days=1000)) \
                                           .strftime("%Y-%m-%d"),
                                       'todate': datetime.now()
                                           .strftime("%Y-%m-%d")}
                                  ),
                                  headers={'content-type': 'application/json'})
                history = pd.DataFrame(json.loads(r.json()['d']))
            except KeyError:
                raise Exception('Date Format Required: yyyy-mm-dd, yyyy/mm/dd')

    history.rename(columns={'symbol':'ticker',
                            'prior_price':'ref',
                            'close_price':'close'}, inplace=True)

    def addzero(int_=str):
        if len(int_) == 1:
            int_ = '0' + int_
        else:
            pass
        return int_

    for i in range(history.shape[0]):
        history['trading_date'].iloc[i] \
            = history['trading_date'].iloc[i][:-12].split('/')[2] + "-"\
              + addzero(history['trading_date'].iloc[i][:-12] \
                        .split('/')[0]) + "-" \
              + addzero(history['trading_date'].iloc[i][:-12] \
                        .split('/')[1])

    history['ref'].iloc[0] = history['close'].iloc[0] # wait for 'open' from IT

    for col in ['ref', 'close', 'high', 'low']: # wait 'open' from IT
        for i in range(1, history.shape[0]):
            if history[col].iloc[i] == 0:
                history[col].iloc[i] = history[col].iloc[i-1]

    return history


def request_trading_intra(ticker=str, fromdate=None, todate=None):
    address = 'https://api.phs.vn/market/Utilities.svc/GetRealShareIntraday'
    pd.options.mode.chained_assignment = None
    if fromdate is not None and todate is not None:
        if datetime.strptime(todate, '%Y-%m-%d') \
                - datetime.strptime(fromdate, '%Y-%m-%d') > timedelta(days=60):
            raise Exception('Can\'t extract more than 60 days')
        else:
            try:
                r = requests.post(address,
                                  data=json.dumps(
                                      {'symbol': ticker,
                                       'fromdate': fromdate,
                                       'todate': todate}
                                  ),
                                  headers={'content-type': 'application/json'})
                #history = pd.DataFrame(json.loads(r.json()['d'])['histories'])
                intraday = pd.DataFrame(json.loads(r.json()['d'])['intradays'])
            except KeyError:
                raise Exception('Date Format Required: yyyy-mm-dd, yyyy/mm/dd')
    else:
        try:
            fromdate = (datetime.now()
                        - timedelta(days=60)).strftime("%Y-%m-%d")
            todate = (datetime.now()
                      - timedelta(days=0)).strftime("%Y-%m-%d")
            r = requests.post(address,
                              data=json.dumps(
                                  {'symbol': ticker,
                                   'fromdate': fromdate,
                                   'todate': todate}
                              ),
                              headers={'content-type': 'application/json'})
            #history = pd.DataFrame(json.loads(r.json()['d'])['histories'])
            intraday = pd.DataFrame(json.loads(r.json()['d'])['intradays'])
        except KeyError:
            raise Exception('Date Format Required: yyyy-mm-dd, yyyy/mm/dd')

    def datemod(date=str):
        def addzero(int_=str):
            if len(int_) == 1:
                int_ = '0' + int_
            else:
                pass
            return int_
        day = addzero(date.split('/')[0])
        month = addzero(date.split('/')[1])
        date = date.split('/')[2][:4] + '-' \
               + month + '-' \
               + day + ' ' \
               + date.split(" ",maxsplit=1)[1]
        return date

    for i in range(intraday.shape[0]):
        intraday['trading_time'].iloc[i] \
            = datemod(intraday['trading_time'].iloc[i])

    return intraday


def request_latest_close_price(ticker=str):
    close_price = request_trading_hist(ticker)['close'].iloc[-1]
    return close_price

