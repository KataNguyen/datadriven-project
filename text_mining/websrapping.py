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


def thongtinTCPH(num_hours: int = 48):

    num_hours = 10
    PATH = 'D:\Python\Selenium\chromedriver.exe'
    driver = webdriver.Chrome(PATH)
    url = 'https://www.hnx.vn/thong-tin-cong-bo-ny-tcph.html'
    driver.get(url)
    now = datetime.now()
    fromtime = now
    news_time = []
    orders = []
    final = pd.DataFrame()
    att_file = []
    link_all = []
    box_text_ = []
    firm_ = []
    date_time_ = []
    title_ = []
    firm_name_ = []
    final_adj = []
    key_words = ['Cổ tức', 'Tạm ứng cổ tức',
                 'Tạm ứng', 'Chi trả', 'Bằng tiền',
                 'Cổ phiếu', 'Quyền mua',
                 'cổ tức', 'tạm ứng cổ tức',
                 'tạm ứng', 'chi trả', 'bằng tiền',
                 'cổ phiếu' ,'quyền mua']
###Expand to 50 rows
    driver.find_element_by_xpath\
        ("//select[@id='divNumberRecordOnPageTCPH']/option[text()='50']").click()
    time.sleep(2)
###Create raw dataframe
    for i in range(1,100):
        driver.find_element_by_xpath(fr"//li"
                                     fr"[@onclick='pageNextTinTCPH({i})']").click()
        time.sleep(2)
        driver.execute_script("window.scrollTo(0,100)")
        time.sleep(1)
        firm = []
        title = []
        date_time = []
        firm_name = []
        fromtime_ = []
        for each_inf in range(1,51):
            firm += driver.find_elements_by_xpath\
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/ div[2]/div[1]/table/tbody/tr[{each_inf}]/td[3]/a")
            date_time += driver.find_elements_by_xpath\
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/div[2]/div[1]/table/tbody/tr[{each_inf}]/td[2]")
            title += driver.find_elements_by_xpath\
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/div[2]/div[1]/table/tbody/tr[{each_inf}]/td[5]/a")
            firm_name += driver.find_elements_by_xpath\
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/div[2]/div[1]/table/tbody/tr[{each_inf}]/td[4]")
            time.sleep(1)
            driver.find_element_by_xpath \
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/div[2]/div[1]/table/tbody/tr[{each_inf}]/td[5]/a").click()
            time.sleep(2)
            att_file = driver.find_element_by_xpath \
                ("//div[@id='divViewDetailArticles']/div[2]/div[3]")
            content_file = att_file.text
            if content_file != "":
                box_text_ += ['']
                links = driver.find_elements_by_xpath \
                    ("/html/body/div[1]/div[2]/div[3]/div[6]/div/div[2]/div[3]/p/a")
                link_all += [[link.get_attribute('href') for link in links]]
            else:
                link_all += ['']
                box_text = driver.find_element_by_xpath \
                    ("/html/body/div[1]/div[2]/div[3]/div[6]/div/div[2]/div[2]")
                box_text_ += [box_text.text]
            driver.find_element_by_xpath \
                ("/html/body/div[1]/div[2]/div[3]/div[6]/div/div[1]/div/a").click()
            time.sleep(1)
            news_time = driver.find_elements_by_xpath\
                (fr"/html/body/div[1]/div[2]/div[3]/div[5]/div[2]/div[2]/div[1]/table/tbody/tr[{each_inf}]/td[2]")
            fromtime_ = str(news_time[0].text)
            fromtime = datetime.strptime(fromtime_, '%d/%m/%Y %H:%M')
            if fromtime <= now - timedelta(hours=num_hours):
                break
            else:
                continue
        break
    firm_ = [f.text for f in firm]
    date_time_ = [d.text for d in date_time]
    title_ = [t.text for t in title]
    firm_name_ = [fn.text for fn in firm_name]
    df = pd.DataFrame(list(zip(firm_, firm_name_, date_time_, title_,box_text_, link_all)),
                      columns=['Mã CK', 'Tên TCPH', 'Ngày đăng tin', 'Tiêu đề tin','Nội dung','File đính kèm'])
    ###Drop unrelated newsdf
    for row in df['Tiêu đề tin']:
        sub_check = []
        for kw in key_words:
            sub_check += [kw in row]
        if any(sub_check):
            pass
        else:
            df.drop(df[df['Tiêu đề tin'] == row].index, inplace=True, )
    df['Nội dung'] = df['Nội dung'].str.split('\n')
    df['Lý do và mục đích'] = ''
    df['Tỷ lệ thực hiện'] = ''
    df['Ngày đăng ký cuối cùng'] = ''
    df['Ngày giao dịch không hưởng quyền'] = ''
    df['Thời gian thực hiện'] = ''
    for each_row in range(df.shape[0]):
        noidung = df['Nội dung'].iloc[each_row]
        ndmd = np.array(noidung)[['*' in word for word in noidung]]
        for ndmd_ in ndmd:
            df['Lý do và mục đích'].iloc[each_row] += ndmd_.lstrip().lstrip('*') + '\n'
        tlth = np.array(noidung)[['Tỷ lệ thực hiện' in word for word in noidung]]
        for tlth_ in tlth:
            df['Tỷ lệ thực hiện'].iloc[each_row] += tlth_.lstrip().lstrip('-') + '\n'
        ndkcc = np.array(noidung)[['Ngày đăng ký cuối cùng' in word for word in noidung]]
        for ndkcc_ in ndkcc:
            df['Ngày đăng ký cuối cùng'].iloc[each_row] += ndkcc_+ '\n'
        ngdkhq = np.array(noidung)[['Ngày giao dịch không hưởng quyền' in word for word in noidung]]
        for ngdkhq_ in ngdkhq:
            df['Ngày giao dịch không hưởng quyền'].iloc[each_row] += ngdkhq_ + '\n'
        thth = np.array(noidung)[['Thời gian thực hiện' in word for word in noidung]]
        for thth_ in thth:
            df['Thời gian thực hiện'].iloc[each_row] += thth_.lstrip().lstrip('-') + '\n'
    df.drop(['Nội dung'],axis =1, inplace=True)
    ###Final dataframe
    ###Export
    df.to_excel(r'C:\Users\Admin\Desktop\PHS\HNX_TCPH.xlsx', index=False)
    driver.close()




class hose:
    def __init__(self):
        pass
