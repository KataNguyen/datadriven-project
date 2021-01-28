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


class fs:

    database_path \
        = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
               'database')

    def __init__(self, ticker=None, segment=None, exchange=None):

        folders = [f for f in listdir(self.database_path) if f.startswith('fs_')]
        fs_types = []
        for folder in folders:
            fs_types = [name.split('_')[0]
                        for name in listdir(join(self.database_path, folder))
                        if isfile(join(self.database_path, folder, name))]
            fs_types = list(dict.fromkeys(fs_types))
        self.fs_types = [x for x in fs_types if not x.startswith('~$')]
        self.fs_types.sort() # Returns all valid financial statements

        self.segments = [x.split('_')[1] for x in folders]
        self.segments.sort() # Returns all valid segments

        periods = []
        for folder in folders:
            periods \
                = list(set(
                [name[-11:-5] for name in listdir(join(self.database_path,
                                                       folder))
                 if isfile(join(self.database_path, folder, name))]))
            periods.sort()
        self.periods = periods # Returns all periods
        self.latest_period = periods[-1]

        self.ticker = ticker # Specified ticker
        self.segment = segment # Specified segment
        self.exchange = exchange # Specified exchange


    def reload(self) -> None:

        """
        This method handles cached data in newly-added files

        :param: None
        :return: None
        """

        folder_names = [folder
                        for folder in listdir(self.database_path)
                        if isdir(join(self.database_path, folder))]

        for folder in folder_names:
            file_names = [file
                          for file in listdir(join(self.database_path, folder))
                          if isfile(join(self.database_path, folder, file))]

            for file in file_names:
                excel = Dispatch("Excel.Application")
                excel.Visible = True
                excel.Workbooks.Open(os.path.join(self.database_path,
                                                  folder, file))
                time.sleep(3)  # suspend 3 secs for excel to catch up python
                excel.Range("A1:XFD1048576").Select()
                excel.Selection.Copy()
                excel.Selection.PasteSpecial(Paste=-4163)
                excel.ActiveWorkbook.Save()
                excel.ActiveWorkbook.Close()


    def fin_tickers(self, sector_break=False) \
            -> Union[list, dict]:

        """
        This function returns all tickers of financial segments

        :param sector_break: False: ignore sectors, True: show sectors
        :para exchange: allow values in request_
        :return: list (sector_break=False), dictionary (sector_break=True)
        """

        financials = ['bank', 'sec', 'ins']
        latest_period = self.periods[-1]

        tickers = []
        tickers_ = dict()
        for segment in financials:
            folder = 'fs_' + segment + '_industry'
            file = 'is_' + latest_period[:4] + 'q' + latest_period[
                -1] + '.xlsm'
            raw_fiinpro \
                = openpyxl.load_workbook(
                os.path.join(self.database_path, folder, file)).active
            # delete StoxPlux Sign
            raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row - 21,
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
                tickers += clean_data.loc[:, 1][a].tolist()
            else:
                tickers_[segment] = clean_data.loc[:, 1][a].tolist()
                tickers = tickers_

        return tickers


    def core(self, year, quarter, segment, fs_type, exchange='all') \
            -> pd.DataFrame:

        """
            This method extracts data from Github server, clean up
            and make it ready for use

            :param year: reported year
            :param quarter: reported quarter
            :param segment: allow values in request_segment_all()
            :param fs_type: allow values in request_fstype()
            :param exchange: allow values in ['HOSE', 'HNX', 'UPCOM'] or 'all'
            :type year: int
            :type quarter: int
            :type segment: str
            :type fs_type: str
            :type exchange: str

            :return: pandas.DataFrame
            :raise ValueError: this function yet supported cashflow for
            securities companies
        """

        if segment not in self.segments:
            raise ValueError(f'sector must be in {self.segments}')

        if fs_type not in self.segments:
            raise ValueError(f'sector must be in {self.fs_types}')

        folder = 'fs_' + segment + '_industry'
        file = fs_type + '_' + str(year) + 'q' + str(quarter) + '.xlsm'

        # create Workbook object, select active Worksheet
        raw_fiinpro \
            = openpyxl.load_workbook(
            os.path.join(self.database_path, folder, file)).active

        # delete StoxPlux Sign
        raw_fiinpro.delete_rows(idx=raw_fiinpro.max_row - 21,
                                amount=1000)

        # delete header rows
        raw_fiinpro.delete_rows(idx=0, amount=7)
        raw_fiinpro.delete_rows(idx=2, amount=1)

        # import to DataFrame, no column labels, no index
        clean_data = pd.DataFrame(raw_fiinpro.values)

        # assign column labels and index
        clean_data.columns = clean_data.iloc[0, :]
        clean_data.drop(index=[0], inplace=True)
        clean_data.index \
            = pd.MultiIndex.from_arrays([[year] * len(clean_data),
                                         [quarter] * len(clean_data),
                                         clean_data['Ticker'].tolist()])
        clean_data.index.set_names(['year', 'quarter', 'ticker'], inplace=True)

        # rename 2 columns
        clean_data.rename(columns=
                          {'Name': 'full_name', 'Exchange': 'exchange'},
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
                    if clean_data.columns[col] \
                            .startswith(
                        ('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.')):
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

            elif fs_type == 'is':
                for col in range(2, len(clean_data.columns)):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : clean_data.columns[col].split()[0]},
                        inplace=True)

            elif fs_type == 'cfi':
                header = list()
                for col in range(len(clean_data.columns) - 1, 1, -1):
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
                                           + clean_data.columns[col].split()[
                                               0]},
                                inplace=True)
                        except IndexError:
                            clean_data.rename(
                                columns={clean_data.columns[col]
                                         : clean_data.columns[col].split()[0]},
                                inplace=True)

            elif fs_type == 'cfd':
                header = list()
                for col in range(2, len(clean_data.columns)):
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
            fin = set(self.fin_tickers()) \
                  & set(clean_data.index.get_level_values(2))
            clean_data.drop(index=fin, level=2, inplace=True)
            if fs_type == 'bs':
                header = list()
                for col in range(len(clean_data.columns) - 1, 1, -1):
                    if clean_data.columns[col] \
                            .startswith(
                        ('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.')):
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
                    if duplicated[i] and col_list[i].split('.', 1)[1] \
                            not in ['I.', 'II.', 'III.', 'IV', 'V.', 'VI.',
                                    'VII.']:
                        col_list[i] += 'b.'
                clean_data.columns = col_list

                duplicated = clean_data.columns.duplicated()
                col_list = clean_data.columns.tolist()
                for i in range(2, len(clean_data.columns)):
                    l = col_list[i].split('.')
                    if duplicated[i] and l[1] \
                            in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                        col_list[i] = l[0] + '.'
                clean_data.columns = col_list

            elif fs_type == 'is':
                for col in range(2, len(clean_data.columns)):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : clean_data.columns[col].split()[0]},
                        inplace=True)

            elif fs_type == 'cfi':
                header = list()
                for col in range(2, len(clean_data.columns)):
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
                                           + clean_data.columns[col].split()[
                                               0]},
                                inplace=True)
                        except IndexError:
                            clean_data.rename(
                                columns={clean_data.columns[col]
                                         : clean_data.columns[col].split()[0]},
                                inplace=True)

            elif fs_type == 'cfd':
                header = list()
                for col in range(2, len(clean_data.columns)):
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
                                           + clean_data.columns[col].split()[
                                               0]},
                                inplace=True)
                        except IndexError:
                            clean_data.rename(
                                columns={clean_data.columns[col]
                                         : clean_data.columns[col].split()[0]},
                                inplace=True)


        elif segment == 'ins':
            if fs_type == 'bs':
                header = list()
                for col in range(len(clean_data.columns) - 1, 1, -1):
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
                                         : header[-1]
                                           + clean_data.columns[col].split()[
                                               0]},
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
                    if duplicated[i] and l[1] \
                            in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                        col_list[i] = l[0] + '.'
                clean_data.columns = col_list

                col_list = clean_data.columns.tolist()
                for i in range(2, len(clean_data.columns)):
                    a = col_list[i].split('.')[1]
                    if col_list[i].startswith(('1', '2', '3', '4', '5')):
                        col_list[i] = col_list[i].replace(a + '.', '')
                clean_data.columns = col_list

            elif fs_type == 'is':
                for col in range(2, len(clean_data.columns)):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : clean_data.columns[col].split()[0]},
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
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
                    if clean_data.columns[col] \
                            .startswith(
                        ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII')):
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
                                           + clean_data.columns[col].split()[
                                               0]},
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
                for col in range(len(clean_data.columns) - 1, 1, -1):
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
                                         : header[-1]
                                           + clean_data.columns[col].split()[
                                               0]},
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
                    if duplicated[i] and l[1] \
                            in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                        col_list[i] = l[0] + '.'
                clean_data.columns = col_list

                col_list = clean_data.columns.tolist()
                for i in range(2, len(clean_data.columns)):
                    l = col_list[i].split('.')
                    if l[0] in ['1', '2', '3', '4', '5'] and l[1] \
                            in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                        col_list[i] = col_list[i].replace('.' + l[1], '')
                clean_data.columns = col_list

            elif fs_type == 'is':
                for col in range(2, len(clean_data.columns)):
                    clean_data.rename(
                        columns={clean_data.columns[col]
                                 : clean_data.columns[col].split()[0]},
                        inplace=True)

            elif fs_type == 'cfi':
                pass

            elif fs_type == 'cfd':
                pass

        clean_data.columns \
            = pd.MultiIndex.from_product([[fs_type],
                                          clean_data.columns.tolist()],
                                         names=['fs_type', 'item'])
        if exchange != 'all':
            clean_data \
                = clean_data.loc[
                clean_data.loc[:, (fs_type, 'exchange')] == exchange]

        print('Extracting...')
        return clean_data


    def _segment(self) -> str:

        """
        This method returns the segment of a given ticker

        :return: str
        """

        segment = ''
        if self.ticker not in self.fin_tickers():
            segment = 'gen'
        else:
            for key in self.fin_tickers(True):
                if self.ticker not in self.fin_tickers(True)[key]:
                    pass
                else:
                    segment = key
                    break

        return segment


    def ticker(self, ticker) -> pd.DataFrame:

        """
        This method returns all financial statements
        of given ticker in all periods

        :param ticker: allow values in request_ticker_all()
        :return: pandas.DataFrame
        """

        folder = 'fs_' + self.segments + '_industry'
        files = listdir(join(self.database_path, folder))

        file_names = []
        for file in files:
            if isfile(join(self.database_path, folder, file)) \
                    and not file.startswith('~$'):
                file_names.append(file)
        file_names = list(set(file_names))
        file_names.sort()

        refs = [(int(name[-11:-7]), int(name[-6]), name[:2]
        if name[2] == '_' else name[:3]) for name in file_names]

        inds = list()
        for fs_type in self.fs_types:
            try:
                inds += self.core(refs[-1][0],
                                  refs[-1][1],
                                  self._segment(),
                                  fs_type) \
                    .xs(ticker, axis=0, level=2) \
                    .drop(['full_name', 'exchange'], level=1, axis=1) \
                    .columns.tolist()
            except KeyError:
                continue

        dict_ind = dict()
        for fs_type in self.fs_types:
            dict_ind[fs_type] = [x[1] for x in inds if x[0] == fs_type]

        fs = pd.concat(
            [
                self.core(ref[0], ref[1], self._segment, ref[2]) \
                    .xs(ticker, axis=0, level=2) \
                    .drop(['full_name', 'exchange'], level=1, axis=1).T \
                    .set_index(pd.MultiIndex.from_product(
                    [[self._segment], [ref[2]], dict_ind[ref[2]]]))
                for ref in refs
            ]
        )
        fs = fs.groupby(fs.index, sort=False).sum()

        print('Finished!')
        return fs


    def all(self, segment) -> pd.DataFrame:

        """
        This method returns all financial statements
        of all companies in all periods

        :param segment: allow values in all_segments()
        :return: pandas.DataFrame
        """

        frames = list()
        for period in self.periods:
            for fs_type in self.fs_types:
                try:
                    frames.append(
                        self.core(int(period[:4]),
                                  int(period[-1]),
                                  segment, fs_type))
                except FileNotFoundError:
                    continue

        df = pd.concat(frames, axis=1, join='outer')
        cols = df.columns.get_level_values(
            0) + '__' + df.columns.get_level_values(1)
        df.columns = cols
        df = df.groupby(by=cols, axis=1,
                        dropna=False, sort=False).sum(min_count=1)
        lvl_0 = [col.split('__')[0] for col in df.columns]
        lvl_1 = [col.split('__')[1] for col in df.columns]
        df.columns = pd.MultiIndex.from_arrays([lvl_0, lvl_1],
                                               names=['fs_type', 'item'])
        df.drop(columns=['exchange', 'full_name'], level=1, inplace=True)

        return df


    def exchanges(self) -> pd.DataFrame:

        """
        This method returns stock exchanges of all tickers

        :param: None
        :return: pandas.DataFrame
        """

        table = pd.DataFrame(columns=['exchange'])
        for segment in self.segments:
            a = self.core(int(self.latest_period[:4]),
                          int(self.latest_period[-1]),
                          segment, 'is', 'all')
            a = a.xs(key='exchange', axis=1, level=1)
            a = a.droplevel(level=['year', 'quarter'], )
            a.columns = ['exchange']
            table = pd.concat([table, a])

        return table


    def _exchange(self) -> str:

        """
        This method returns stock exchange of given stock

        :return: str
        """

        exchange_table = self.exchanges()
        exchange = exchange_table.loc[self.ticker].iloc[0]

        return exchange


    def tickers(self):

        """
        This function returns all tickers of given segment or exchange

        :return: list
        """

        if self.segment == 'gen':
            ticker_list \
                = self.core(int(self.latest_period[:4]),
                            int(self.latest_period[-1]),
                            self.segment, 'is') \
                .index.get_level_values(level=2).tolist()
        elif self.segment is not None:
            fin_dict = self.fin_tickers(True)
            ticker_list = fin_dict[self.segment]

        tickers = []
        for s in self.segments:
            tickers +=

        return ticker_list

