from phs import *

###############################################################################

def priceKformat(tick_val, pos) -> str:

    """
    Turns large tick values (in the billions, millions and thousands)
    such as 4500 into 4.5K and also appropriately turns 4000 into 4K
    (no zero after the decimal)

    :param tick_val: price value
    :param pos: ignored
    :type tick_val: float

    :return: formated price value
    """

    if tick_val >= 1000:
        val = round(tick_val/1000, 1)
        if tick_val % 1000 > 0:
            new_tick_format = '{:,}K'.format(val)
        else:
            new_tick_format = '{:,}K'.format(int(val))
    else:
        new_tick_format = int(tick_val)
    new_tick_format = str(new_tick_format)
    return new_tick_format


def adjprice(price: Union[float, int]) -> str:

    """
    This method returns adjusted price for minimum price steps,
    used for display only

    :param price: stock price
    :type price: float or int

    :return: str
    """

    if price < 10000:
        price_string = f'{int(round(price, -1)):,d}'
    elif 10000 <= price < 50000:
        price_string = f'{50 * int(round(price/50)):,d}'
    else:
        price_string = f'{int(round(price, -2)):,d}'

    return price_string


def seopdate(period:str) -> tuple:

    """
    This method returns (start date, end date) of a period

    :param period: target period: '2020q4', '2020q3', etc.
    :type period: str

    :return: tuple
    """

    year = int(period[:4])
    quarter = int(period[-1])

    # start of the period
    sop_date = datetime(year=year, month=3*quarter-2, day=1)
    while sop_date.weekday() in holidays.WEEKEND \
            or sop_date in holidays.VN():
        sop_date += timedelta(days=1)
    sop_date = sop_date.strftime('%Y-%m-%d')

    # end of the period
    fq = lambda quarter: 1 if quarter == 4 else 3*quarter + 1
    fy = lambda year: year + 1 if quarter == 4 else year
    eop_date = datetime(year=fy(year), month=fq(quarter), day=1) \
               + timedelta(days=-1)
    while eop_date.weekday() in holidays.WEEKEND \
            or eop_date in holidays.VN():
        eop_date -= timedelta(days=1)
    eop_date = eop_date.strftime('%Y-%m-%d')

    return sop_date, eop_date


def period_cal(period:str, years:int=0, quarters:int=0) -> str:

    """
    This function return resutled period from addition/substraction operations

    :param period: original period
    :param years: number of years added (+) or substracted (-) from the original period
    :param quarters: number of quarters added (+) or substracted (-) from the original period
    :type period: str
    :type years: int
    :type quarters: int

    :return: tuple
    """

    year = int(period[:4])
    quarter = int(period[-1])

    if (quarter + quarters) % 4 == 0:
        year_new = year + years + (quarter + quarters) // 4 - 1
        quarter_new = 4
    else:
        year_new = year + years + (quarter + quarters) // 4
        quarter_new = (quarter + quarters) % 4

    period_new = f'{year_new}q{quarter_new}'

    return period_new
