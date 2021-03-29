from phs import *
from request_phs import *
import request_phs


def tinnghiepvutochucphathanh_bs4(num_hours:int=48):

    url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg'

    num_hours = 48
    now = datetime.now()

    page_content = []
    time_news = []
    link_list = []
    headline_list = []

    fromtime = now
    num_page = 0
    while fromtime >= now + timedelta(hours=-num_hours):

        num_page += 1

        data = {'CurrentPage': 'changePage(9)'}
        page = requests.post(url, data=data)

        soup = BeautifulSoup(page.text, features='html.parser')

        page_tag = soup.body.div.main.div.div.div.div.ul
        page_li_list = page_tag.find_all('li')

        for tag in page_li_list:
            page_content += tag.find('h3')
            time_news += tag.find('div')

        for tag in page_content:
            link_list += [tag['href']]
            headline_list += [tag.string]

        fromtime_str = time_news[-1][-21:]
        fromtime = datetime.strptime(fromtime_str, '%d/%m/%Y - %H:%M:%S')

        print(url)
        print(time_news[-1][-21:])
        print(fromtime_str)
        print(now + timedelta(hours=-num_hours))


###################### RETRY WITH SELENIUM ######################


class vsd:
    def __init__(self):
        pass

    @staticmethod
    def tinnghiepvutochucphathanh(num_hours:int=48):

        num_hours = 48

        PATH = 'D:\Python\Selenium\chromedriver.exe'
        driver = webdriver.Chrome(PATH)

        url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg'
        driver.get(url)

        now = datetime.now()
        fromtime = now

        news_headlines = []
        news_urls = []
        news_time = []

        while fromtime >= now - timedelta(hours=num_hours):
            tags = driver.find_elements_by_xpath('/html/body/div/main'
                                                 '/div/div/div/div/ul/li')
            for tag_ in tags:
                wait_sec = np.random.random(1)[0] * 2
                time.sleep(wait_sec)
                h3_tag = tag_.find_element_by_tag_name('h3')
                news_headlines += [h3_tag.text]
                news_urls += [h3_tag.find_element_by_tag_name('a')
                                  .get_attribute('href')]
                news_time += [tag_.find_element_by_tag_name('div').text]

            fromtime_str = news_time[-1][-21:]
            fromtime = datetime.strptime(fromtime_str, '%d/%m/%Y - %H:%M:%S')

            # Turn Page
            page_buttons = driver.find_elements_by_xpath('/html/body/div/main/'
                                                         'div/div/div/div/'
                                                         'div/div/div/button')
            page_buttons = page_buttons[-2]
            page_buttons.click()
######IMPORT PACKAGE###
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime
from datetime import datetime
from datetime import timedelta
import numpy as np
import requests
import pandas as pd
from pandas import ExcelWriter
import io
from selenium.webdriver.support.select import Select
######
class hnx:
    def __init__(self):
        pass


def thongtincongbo(num_hours: int = 48):

    num_hours = 48

    PATH = 'D:\Python\Selenium\chromedriver.exe'
    driver = webdriver.Chrome(PATH)

    url = 'https://www.hnx.vn/thong-tin-cong-bo-ny-tcph.html'
    driver.get(url)

    now = datetime.now()
    fromtime = now

    news_time = []
    orders = []
###Expand to 50 rows
    driver.find_element_by_xpath("//select[@id='divNumberRecordOnPageTCPH']/option[text()='50']").click()
    time.sleep(0.8)
    while fromtime >= now - timedelta(hours=num_hours):
        tags = driver.find_elements_by_tag_name('tbody')
        del tags[0]
        cont = tags[0].text
###Create Data frame
        order = cont.replace("\n", "*")
        need = order.split("*")
        df = pd.DataFrame(need)
        df_split = df[0].str.split(' ',4)
        df['STT'] = df_split.str.get(0)
        df['Date'] = df_split.str.get(1)
        df['Time'] = df_split.str.get(2)
        df['Firm'] = df_split.str.get(3)
        df['Infor'] = df_split.str.get(-1)
        df["Date-Time"] = df["Date"] + ' ' + df["Time"]
        final = df[["STT", "Date-Time",'Firm','Infor']]

        news_time = df['Date-Time'].iloc[-1]

        ###Export
        writer = ExcelWriter('PythonExport.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()


        wait_sec = np.random.random(1)[0] * 2
        time.sleep(wait_sec)
        fromtime_str = news_time[-1][-21:]
        fromtime = datetime.strptime(fromtime_str, '%d/%m/%Y - %H:%M:%S')

        # Turn Page
        page_buttons = driver.find_elements_by_xpath('/html/body/div/main/'
                                                     'div/div/div/div/'
                                                     'div/div/div/button')
        page_buttons = page_buttons[-2]
        page_buttons.click()


class hose:
    def __init__(self):
        pass