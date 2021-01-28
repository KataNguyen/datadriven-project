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

class fs:
    def __init__(self, year, quarter, segment, fs_type, exchange='all'):
        """
            This function extracts data from Github server, clean up
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

        global database_path
        segments = self.segment_all()
        fs_types = self.fstype()

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
            fin = set(request_financial_ticker()) \
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


    def segment_all() -> list:

        """
        This function returns the names of segments

        :param: None
        :return: list
        """

        folders = [f for f in listdir(database_path) if f.startswith('fs_')]
        segments = [x.split('_')[1] for x in folders]
        segments.sort()

        return segments