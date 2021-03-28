from phs import *
from request_phs import *

class vsd:
    def __init__(self):
        pass

    @staticmethod
    def tinnghiepvutochucphathanh(num_hours:int=48):

        num_hours = 48

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg'
        driver.get(url)

        now = datetime.now()
        fromtime = now

        news_time = []
        news_headlines = []
        news_exchanges =[]
        news_tickers = []
        news_dividends = []
        news_ratios = []
        news_recordsdate = []
        news_paymentdate = []
        news_reasons =[]
        news_urls = []

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
                news_time_ = tag_.find_element_by_tag_name('div').text
                news_time_ \
                    = datetime.strptime(news_time_[-21:], '%d/%m/%Y - %H:%M:%S')
                news_time += [news_time_]

            fromtime = news_time[-1]

            # Turn Page
            page_buttons = driver.find_elements_by_xpath('/html/body/div/main/'
                                                         'div/div/div/div/'
                                                         'div/div/div/button')
            next_page = page_buttons[-2]
            next_page.click()

        output_table = pd.DataFrame(columns=['Thời gian', 'Sàn Giao Dịch',
                                             'Cổ Phiếu', 'Tiêu Đề', 'Cổ tức',
                                             'Tỷ lệ thực hiện',
                                             'Ngày đăng ký cuối cùng',
                                             'Ngày thanh toán',
                                             'Lý do, mục đích'])
        output_table['Time'] = news_time

        output = pd.DataFrame(data={'Headline': news_headlines,
                                    'Links': news_urls},
                              index=news_time)

vsd = vsd()

class hnx:
    def __init__(self):
        pass


class hose:
    def __init__(self):
        pass