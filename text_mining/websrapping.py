from phs import *
from request_phs import *

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

        PATH = join(dirname(dirname(realpath(__file__))),'phs','geckodriver')
        driver = webdriver.Firefox(executable_path=PATH)

        url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg'

        driver.get(url)

        elem = driver.find_element_by_tag_name('body')
        elem = elem.find_element_by_tag_name('div')
        elem = elem.find_element_by_tag_name('main')
        elem = elem.find_element_by_tag_name('div')
        elem = elem.find_element_by_tag_name('div')
        elem = elem.find_element_by_tag_name('div')
        elem = elem.find_element_by_tag_name('div')
        elem = elem.find_element_by_tag_name('ul')
        elem = elem.find_elements_by_tag_name('li')

        elems = []
        for elem_ in elem:
            elems += [elem_.find_element_by_tag_name('a')]