from phs import *
from request_phs import *

def tinnghiepvutochucphathanh(num_hours:int=48):

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
        url = 'https://vsd.vn/vi/alo/-f-_bsBS4BBXga52z2eexg#!/page-100'
        page = requests.get(url)
        page = urlopen(url)
        soup = BeautifulSoup(page, features='html.parser',
                             from_encoding=page.info().get_param('charset'))

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

