from request_phs import *

class vsd:

    def __init__(self):
        pass

    @staticmethod
    def tinnghiepvuTCPH(num_hours:int=48, fixed_mp:bool=False):

        start_time = time.time()
        now = datetime.now()

        report_time = now.strftime('%Y-%m-%d %H-%M-%S')

        fixedmp_path = join(dirname(realpath(__file__)),'input','fixedmp.xlsx')
        fixedmp_table = pd.read_excel(fixedmp_path, usecols=['Stock'])
        fixedmp_list = list(fixedmp_table['Stock'])

        destination_path = join(dirname(realpath(__file__)),
                                f'tinnghiepvutochucphathanh - {report_time}.xlsx')

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg'
        driver.get(url)

        fromtime = now

        keywords = ['Cổ tức',
                    'Tạm ứng cổ tức',
                    'Tạm ứng',
                    'Chi trả',
                    'Bằng tiền',
                    'Cổ phiếu',
                    'Quyền mua',
                    'Chuyển dữ liệu đăng ký',
                    'cổ tức',
                    'tạm ứng cổ tức',
                    'tạm ứng',
                    'chi trả',
                    'bằng tiền',
                    'cổ phiếu',
                    'quyền mua',
                    'chuyển dữ liệu đăng ký']

        output_table = pd.DataFrame()
        while fromtime >= now - timedelta(hours=num_hours):

            news_time = []
            news_headlines = []
            news_urls = []

            news_ratios = []
            news_recordsdate = []
            news_paymentdate = []

            tags = driver.find_elements_by_xpath('//*[@id="d_list_news"]/ul/li')
            tags.reverse()
            for tag_ in tags:
                wait_sec = np.random.random(1)[0] + 0.5
                time.sleep(wait_sec)
                h3_tag = tag_.find_element_by_tag_name('h3')

                if fixed_mp is True:
                    if h3_tag.text[:3] in fixedmp_list:
                        txt = h3_tag.text
                    else:
                        continue
                else:
                    txt = h3_tag.text

                check = [word in txt for word in keywords]

                if any(check):

                    news_headlines += [txt]

                    sub_url = h3_tag.find_element_by_tag_name('a') \
                        .get_attribute('href')
                    news_urls += [sub_url]

                    news_time_ = tag_.find_element_by_tag_name('div').text
                    news_time_ \
                        = datetime.strptime(news_time_[-21:],
                                            '%d/%m/%Y - %H:%M:%S')
                    news_time += [news_time_]


                    sub_driver = webdriver.Firefox(executable_path=PATH)
                    sub_driver.get(sub_url)

                    # handle no-heading-table news
                    info_heads \
                        = sub_driver.find_elements_by_xpath(
                        "//div[substring(@class,string-length(@class)"
                        "-string-length('item-info')+1)='item-info']")
                    if len(info_heads) == 0:
                        news_dict = dict()
                    else:
                        # get info from heading table
                        info_heads_text = [head.text[:-1]
                                           for head in info_heads]
                        info_tails \
                            = sub_driver.find_elements_by_xpath(
                            "//div[substring(@class,string-length(@class)"
                            "-string-length('item-info-main')+1)='item-info-main']")
                        info_tails_text = [tail.text for tail in info_tails]

                        news_dict = {h:t for h,t in
                                     zip(info_heads_text, info_tails_text)}

                    content_elem = sub_driver.find_element_by_xpath(
                        "//div[@style='text-align: justify;']")
                    content = content_elem.text
                    content_split = content.split('\n')

                    def f(full_list):
                        result = ''
                        if len(full_list) != 0:
                            for i in range(len(full_list)-1):
                                result += full_list[i] + '\n'
                            result += full_list[-1]
                        return result


                    dividend_kws = ['Chi trả cổ tức', 'Tỷ lệ thực hiện']
                    mask_dividend = []
                    for row in content_split:
                        mask_dividend += [any([keyword in row
                                               for keyword in dividend_kws])]

                    dividends \
                        = list(np.array(content_split)[np.array(mask_dividend)])
                    dividends = f(dividends)
                    news_dict['Tỷ lệ thực hiện'] = dividends


                    payment_kws = ['Thời gian thanh toán',
                                   'Ngày thanh toán',
                                   'Thời gian thực hiện']
                    mask_payment = []
                    for row in content_split:
                        mask_payment \
                            += [any([keyword in row for keyword in payment_kws])]

                    payments \
                        = list(np.array(content_split)[np.array(mask_payment)])
                    payments = f(payments)
                    news_dict['Thời gian thanh toán'] = payments

                    try:
                        news_ratios += [news_dict['Tỷ lệ thực hiện']]
                    except KeyError:
                        news_ratios += ['']
                    try:
                        news_recordsdate += [news_dict['Ngày đăng ký cuối cùng']]
                    except KeyError:
                        news_recordsdate += ['']
                    try:
                        news_paymentdate += [news_dict['Thời gian thanh toán']]
                    except KeyError:
                        news_paymentdate += ['']

                    sub_driver.quit()

                else:
                    pass

            output_table = pd.concat([output_table,
                                      pd.DataFrame(
                                          {'Thời gian': news_time,
                                           'Tiêu đề': news_headlines,
                                           'Tỷ lệ thực hiện': news_ratios,
                                           'Ngày đăng ký cuối cùng': news_recordsdate,
                                           'Ngày thanh toán': news_paymentdate,
                                           'Link': news_urls}
                                      )], ignore_index=True)

            # Turn Page
            nextpage_button = driver.find_elements_by_xpath(
                "//button[substring(@onclick,1,string-length('changePage'))"
                "='changePage']")[-2]
            nextpage_button.click()

            # Check time
            last_tag = driver.find_elements_by_xpath('//*[@id="d_list_news"]/ul/li')[0]
            fromtime = last_tag.find_element_by_tag_name('div').text
            fromtime = datetime.strptime(fromtime[-21:],
                                         '%d/%m/%Y - %H:%M:%S')

        output_table['Mã cổ phiếu'] = ''
        output_table['Lý do, mục đích'] = ''
        output_table['Chuyển từ sàn'] = ''
        output_table['Chuyển đến sàn'] = ''
        output_table['Ngày giao dịch không hưởng quyền'] = ''

        for row in range(output_table.shape[0]):


            output_table['Tiêu đề'].iloc[row] \
                = output_table['Tiêu đề'].iloc[row].split(': ')
            output_table['Mã cổ phiếu'].iloc[row] \
                = output_table['Tiêu đề'].iloc[row][0]
            output_table['Lý do, mục đích'].iloc[row] \
                = output_table['Tiêu đề'].iloc[row][1]


            from_exchange_array \
                = np.array(
                output_table['Lý do, mục đích'].iloc[row].split())\
            [[word.isupper() for word in output_table['Lý do, mục đích']
                    .iloc[row].split()],]
            try:
                output_table['Chuyển từ sàn'].iloc[row] = from_exchange_array[1]
            except IndexError:
                output_table['Chuyển từ sàn'].iloc[row] = ''


            to_exchange_array \
                = np.array(
                output_table['Lý do, mục đích'].iloc[row].split())\
            [[word.isupper() for word in output_table['Lý do, mục đích']
                    .iloc[row].split()],]
            try:
                output_table['Chuyển đến sàn'].iloc[row] \
                    = to_exchange_array[2]
            except IndexError:
                output_table['Chuyển đến sàn'].iloc[row] = ''

            rtime = output_table['Ngày đăng ký cuối cùng'].iloc[row]
            if rtime != '':
                try:
                    ryear = rtime[-4:]
                    rmonth = rtime[-7:-5]
                    rday = rtime[-10:-8]

                    result = bdate(f'{ryear}-{rmonth}-{rday}',-1)
                    year = result[:4]
                    month = result[5:7]
                    day = result[-2:]
                    output_table['Ngày giao dịch không hưởng quyền'].iloc[row] \
                                        = f'{day}/{month}/{year}'
                except ValueError:
                    pass

        output_table.drop(['Tiêu đề'], axis=1, inplace=True)

        output_table = output_table[['Thời gian',
                                     'Mã cổ phiếu',
                                     'Lý do, mục đích',
                                     'Tỷ lệ thực hiện',
                                     'Ngày đăng ký cuối cùng',
                                     'Ngày giao dịch không hưởng quyền',
                                     'Ngày thanh toán',
                                     'Chuyển từ sàn',
                                     'Chuyển đến sàn',
                                     'Link']]

        output_table.to_excel(destination_path, index=False)
        driver.quit()

        #######################################################################

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://vsd.vn/vi/tin-thi-truong-phai-sinh'
        driver.get(url)

        keywords = ['tỷ lệ ký quỹ ban đầu',
                    'Tỷ lệ ký quỹ ban đầu',
                    'hợp đồng tương lai',
                    'Hợp đồng tương lai',
                    'HĐTL']

        while fromtime >= now - timedelta(hours=num_hours):

            news_time = []
            news_headlines = []
            news_urls = []

            tags = driver.find_elements_by_xpath('//*[@id="tab1"]/ul/li')
            tags.reverse()
            for tag_ in tags:
                wait_sec = np.random.random(1)[0] + 0.5
                time.sleep(wait_sec)
                h3_tag = tag_.find_element_by_tag_name('h3')

                if h3_tag.text[:3].isupper():
                    continue
                else:
                    txt = h3_tag
                    check = [word in txt for word in keywords]
                    if any(check):
                        news_headlines += [txt]

                        sub_url = h3_tag.find_element_by_tag_name('a') \
                            .get_attribute('href')
                        news_urls += [sub_url]

                        news_time_ = tag_.find_element_by_tag_name('div').text
                        news_time_ \
                            = datetime.strptime(news_time_[-21:],
                                                '%d/%m/%Y - %H:%M:%S')
                        news_time += [news_time_]

            output_table = pd.concat([output_table,
                                      pd.DataFrame(
                                          {'Thời gian': news_time,
                                           'Tiêu đề': news_headlines,
                                           'Link': news_urls}
                                      )], ignore_index=True)

            # Turn Page
            nextpage_button = driver.find_elements_by_xpath(
                "//*[@id='d_number_of_page']/button")[-2]
            nextpage_button.click()

            # Check time
            last_tag = driver.find_elements_by_xpath('//*[@id="d_list_news"]/ul/li')[0]
            fromtime = last_tag.find_element_by_tag_name('div').text
            fromtime = datetime.strptime(fromtime[-21:],
                                         '%d/%m/%Y - %H:%M:%S')


        print(f'Finished ::: Total execution time: {int(time.time()-start_time)}s\n')


        return output_table


    @staticmethod
    def tinnghiepvuvoithanhvienluuky(num_hours:int=48):

        start_time = time.time()
        now = datetime.now()

        report_time = now.strftime('%Y-%m-%d %H %M %S')
        destination_path = join(dirname(realpath(__file__)),
                                f'TinNghiepVuVoiThanhVienLuuKy - {report_time}.xlsx')

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://www.vsd.vn/vi/alc/4'
        driver.get(url)

        fromtime = now

        keywords = ['ngày hạch toán của cổ phiếu',
                    'Ngày hạch toán của cổ phiếu',
                    'ngày hạch toán của chứng quyền',
                    'Ngày hạch toán của chứng quyền ']

        excluded_words = ['bổ sung']

        output_table = pd.DataFrame()
        while fromtime >= now - timedelta(hours=num_hours):

            news_time = []
            news_headlines = []
            news_urls = []
            news_tradedate = []

            tags = driver.find_elements_by_xpath('//*[@id="d_list_news"]/ul/li')

            tags.reverse()
            for tag_ in tags:
                wait_sec = np.random.random(1)[0] + 0.5
                time.sleep(wait_sec)

                h3_tag = tag_.find_element_by_tag_name('h3')

                txt = h3_tag.text
                check_1 = [word not in txt for word in excluded_words]
                check_2 = [word in txt for word in keywords]

                if all(check_1) and any(check_2):

                    news_headlines += [txt]

                    sub_url = h3_tag.find_element_by_tag_name('a') \
                        .get_attribute('href')
                    news_urls += [sub_url]

                    news_time_ = tag_.find_element_by_tag_name('div').text
                    news_time_ \
                        = datetime.strptime(news_time_[-21:],
                                            '%d/%m/%Y - %H:%M:%S')
                    news_time += [news_time_]


                    sub_driver = webdriver.Firefox(executable_path=PATH)
                    sub_driver.get(sub_url)

                    info_heads \
                        = sub_driver.find_elements_by_xpath(
                        "//div[substring(@class,string-length(@class)"
                        "-string-length('item-info')+1)='item-info']")

                    info_heads_text = [head.text[:-1]
                                       for head in info_heads]

                    info_tails \
                        = sub_driver.find_elements_by_xpath(
                        "//div[substring(@class,string-length(@class)"
                        "-string-length('item-info-main')+1)='item-info-main']")
                    info_tails_text = [tail.text for tail in info_tails]

                    news_dict = {h: t for h, t in
                                 zip(info_heads_text, info_tails_text)}

                    try:
                        news_tradedate += [news_dict['Ngày giao dịch chính thức']]
                    except KeyError:
                        news_tradedate += ['']

                    sub_driver.quit()

                else:
                    pass

            output_table = pd.concat([output_table,
                                      pd.DataFrame(
                                          {'Thời gian': news_time,
                                           'Tiêu đề': news_headlines,
                                           'Ngày giao dịch chính thức': news_tradedate,
                                           'Link': news_urls}
                                      )], ignore_index=True)

            output_table['Mã cổ phiếu / chứng quyền'] = ''
            output_table['Lý do, mục đích'] = ''

            for row in range(output_table.shape[0]):

                output_table['Tiêu đề'].iloc[row] \
                    = output_table['Tiêu đề'].iloc[row].split(': ')
                output_table['Mã cổ phiếu / chứng quyền'].iloc[row] \
                    = output_table['Tiêu đề'].iloc[row][0]
                output_table['Lý do, mục đích'].iloc[row] \
                    = output_table['Tiêu đề'].iloc[row][1]

            # Turn Page
            nextpage_button = driver.find_elements_by_xpath(
                "//*[@id='d_number_of_page']/button")[-2]
            nextpage_button.click()

            # Check time
            last_tag = driver.find_elements_by_xpath('//*[@id="d_list_news"]/ul/li')[0]
            fromtime = last_tag.find_element_by_tag_name('div').text
            fromtime = datetime.strptime(fromtime[-21:],
                                         '%d/%m/%Y - %H:%M:%S')


        output_table.drop(['Tiêu đề'], axis=1, inplace=True)

        output_table = output_table[['Thời gian',
                                     'Mã cổ phiếu / chứng quyền',
                                     'Lý do, mục đích',
                                     'Ngày giao dịch chính thức',
                                     'Link']]

        output_table.to_excel(destination_path, index=False)
        driver.quit()

        print(f'Finished ::: Total execution time: {int(time.time()-start_time)}s\n')

        return output_table


vsd = vsd()


class hnx:

    def __init__(self):
        pass

    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.select import Select
    from selenium.webdriver.support import expected_conditions as EC


    @staticmethod
    def thongtinTCPH(num_hours:int=48):

        num_hours = 24

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://www.hnx.vn/thong-tin-cong-bo-ny-tcph.html'
        driver.get(url)

        driver.maximize_window()
        now = datetime.now()
        link_all = []
        box_text_ = []
        key_words = ['Cổ tức', 'Tạm ứng cổ tức',
                     'Tạm ứng', 'Chi trả', 'Bằng tiền',
                     'Cổ phiếu', 'Quyền mua',
                     'cổ tức', 'tạm ứng cổ tức',
                     'tạm ứng', 'chi trả', 'bằng tiền',
                     'cổ phiếu', 'quyền mua']

        ###Expand to 50 rows
        driver.find_element_by_xpath \
            ("//select[@id='divNumberRecordOnPageTCPH']/option[text()='50']")\
            .click()
        time.sleep(2)

        ###Create raw dataframe
        for i in range(1, 100):
            driver.find_element_by_xpath(
                f"//li[@onclick='pageNextTinTCPH({i})']").click()
            time.sleep(2)
            driver.execute_script("window.scrollTo(0,100)")
            firm = []
            title = []
            date_time = []
            firm_name = []
            title_select = driver.find_elements_by_xpath(
                "//a[substring"
                "(@onclick,1,string-length('return funcViewDetailArticlesByID'))"
                "='return funcViewDetailArticlesByID']")

            title_select_ = [t.text for t in title_select]
            title_click = []
            for j in range(len(title_select_)):
                if title_select_[j] != '':
                    title_click.append(title_select[j])

            for each_inf in range(1, 51):
                firm += driver.find_elements_by_xpath(
                    f"//*[@id='_tableDatas']"
                    f"/tbody/tr[{each_inf}]/td[3]/a")
                date_time += driver.find_elements_by_xpath(
                    f"//*[@id='_tableDatas']"
                    f"/tbody/tr[{each_inf}]/td[2]")
                title += driver.find_elements_by_xpath(
                    f"//*[@i='_tableDatas']"
                    f"/tbody/tr[{each_inf}]/td[5]/a")
                firm_name += driver.find_elements_by_xpath(
                    f"//*[@id='_tableDatas']"
                    f"/tbody/tr[{each_inf}]/td[4]")

                click_obj = title_click[each_inf-1]
                click_obj.click()
                wait = WebDriverWait(driver,15)
                wait.until(EC.visibility_of_element_located
                           ((By.XPATH,
                             "//*[@id='divViewDetailArticles']/div[1]/b")))
                att_file \
                    = driver.find_element_by_xpath(
                    "//div[@id='divViewDetailArticles']/div[2]/div[3]")
                content_file = att_file.text
                if content_file != '':
                    box_text_ += ['']
                    links = driver.find_elements_by_xpath(
                        r"//*[@id='divViewDetailArticles']/div[2]/div[3]/p/a")
                    link_all += [
                        [link.get_attribute('href') for link in links]]
                else:
                    link_all += ['']
                    box_text = driver.find_element_by_xpath(
                        "//*[@id ='divViewDetailArticles']/div[2]/div[2]")
                    box_text_ += [box_text.text]

                driver.find_element_by_xpath(
                    r"//*[@id='btnExitPopups']").click()

                news_time = driver.find_element_by_xpath \
                    (fr"//*[@id='_tableDatas']/tbody/tr[{each_inf}]/td[2]")

                if datetime.text.strptime(news_time,'%d/%m/%Y %H:%M') \
                        < now - timedelta(hours=num_hours):
                    break

            firm_ = [f.text for f in firm]
            date_time_ = [d.text for d in date_time]
            title_ = [t.text for t in title]
            firm_name_ = [fn.text for fn in firm_name]
            df = pd.DataFrame(list(zip(firm_, firm_name_, date_time_,
                                       title_, box_text_, link_all)),
                              columns=['Mã CK', 'Tên TCPH', 'Ngày đăng tin',
                                       'Tiêu đề tin', 'Nội dung',
                                       'File đính kèm'])

        ###Drop unrelated newsdf
        for row in df['Tiêu đề tin']:
            sub_check = []
            for kw in key_words:
                sub_check += [kw in row]
            if any(sub_check):
                pass
            else:
                df.drop(df[df['Tiêu đề tin'] == row].index, inplace=True)

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
                df['Lý do và mục đích'].iloc[each_row] += \
                    ndmd_.lstrip().lstrip('*') + '\n'
            tlth = np.array(noidung)[['Tỷ lệ thực hiện' \
                                      in word for word in noidung]]
            for tlth_ in tlth:
                df['Tỷ lệ thực hiện'].iloc[each_row] += \
                    tlth_.lstrip().lstrip('-') + '\n'
            ndkcc = np.array(noidung)[['Ngày đăng ký cuối cùng' \
                                       in word for word in noidung]]
            for ndkcc_ in ndkcc:
                df['Ngày đăng ký cuối cùng'].iloc[each_row] += ndkcc_ + '\n'
            ngdkhq = np.array(noidung)[['Ngày giao dịch không hưởng quyền' \
                                        in word for word in noidung]]
            for ngdkhq_ in ngdkhq:
                df['Ngày giao dịch không hưởng quyền'].iloc[each_row] \
                    += ngdkhq_ + '\n'
            thth = np.array(noidung)[['Thời gian thực hiện' \
                                      in word for word in noidung]]
            for thth_ in thth:
                df['Thời gian thực hiện'].iloc[each_row] += \
                    thth_.lstrip().lstrip('-') + '\n'
        df.drop(['Nội dung'], axis=1, inplace=True)

        ###Final dataframe
        ###Export

        df.to_excel(os.path.dirname(realpath(__file__)), index=False)
        driver.quit()


class hose:
    def __init__(self):
        pass