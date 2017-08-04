# -*- encoding: utf-8 -*-

import datetime
import time
import json

def timestamp_to_date(ts):
    """
    Parameters:
        ts: timestamp, int or string
    Returns:
        %Y-%m-%d, 2017-08-01, string or blank string for bad input
    """

    try:
        value = datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d')
    except ValueError as e:
        value = ""
    return value

def timestamp_to_hour(ts):
    """
    Parameters:
        ts: timestamp, int or string
    Returns:
        hour of the timestamp, [0, 23], int or -1 for bad input
    """

    try:
        hour = datetime.datetime.fromtimestamp(int(ts)).hour
    except ValueError as e:
        hour = -1
    return hour

def date_to_weekday(dt):
    """
    Parameters:
        dt: %Y-%m-%d, 2017-08-01, string or %Y-%m-%d %H:%M:%S, 2017-08-01 14:53:59
    Returns:
        weekday: 1 to 7 for Monday to Sunday, int, or -1 for bad input
    """

    try:
        if len(dt) == 10:
            weekday = datetime.datetime.strptime(dt, "%Y-%m-%d").isoweekday()
        else:
            weekday = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").isoweekday()
    except ValueError as e:
        weekday = -1
    return weekday

def holiday_type(dt):
    """
    Parameters:
        dt: %Y-%m-%d, 2017-08-01, string or %Y-%m-%d %H:%M:%S, 2017-08-01 14:53:59
    Returns:
        holiday type of the input date, a two-digit number
        newyear: 31
        lunarnewyear: 71
        qingming: 32
        laborday: 33
        duanwu: 34
        zhongqiu: 35
        nationalday: 72
        weekend: 21
        weekday: 1
        bad input: -1
    """

    days_code = {"newyear": 31, "lunarnewyear": 71, "qingming": 32,
            "laborday": 33, "duanwu": 34, "zhongqiu": 35, "nationalday": 72,
            "weekend": 21, "weekday": 1}

    conf = json.load(open("./chinese_holidays.json", "r"))
    weekend_work = conf["weekend_work"]
    try:
        if len(dt) == 10:
            dt = datetime.datetime.strptime(dt, "%Y-%m-%d")
        else:
            dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        return -1

    dt_str = dt.strftime("%Y-%m-%d")
    if dt_str in weekend_work:
        return days_code["weekday"] 

    year, month = dt.year, dt.month
    if month == 12:
        year = str(year + 1)
    else:
        year = str(year)
    holidays = conf[year]

    for holiday in holidays:
        date_span = holidays[holiday].split("to")
        if dt_str >= date_span[0] and dt_str <= date_span[1]:
            return days_code[holiday]

    if dt.isoweekday() == 6 or dt.isoweekday() == 7:
        return days_code["weekend"]

    return days_code["weekday"]

def unit_test():
    # timestamp_to_date
    assert timestamp_to_date(1501811384) == "2017-08-04"
    assert timestamp_to_date("1501811384") == "2017-08-04"
    assert timestamp_to_date("abc") == ""

    # timestamp_to_hour
    assert timestamp_to_hour(1501811384) == 9
    assert timestamp_to_hour("1501811384") == 9
    assert timestamp_to_hour(150100000) == 14
    assert timestamp_to_hour("abc") == -1

    # date_to_weekday
    assert date_to_weekday("2017-08-04") == 5
    assert date_to_weekday("2017-08-06") == 7
    assert date_to_weekday("2017-08-06 14:59:23") == 7
    assert date_to_weekday("2017") == -1

    # holiday_type
    assert holiday_type("2017-08-04") == 1
    assert holiday_type("2017-04-01") == 1
    assert holiday_type("2017-08-05") == 21
    assert holiday_type("2017-08-05 23:39:39") == 21
    assert holiday_type("2016-12-31") == 31
    assert holiday_type("2011-04-04") == 32
    assert holiday_type("2012-04-30") == 33
    assert holiday_type("2013-06-10") == 34
    assert holiday_type("2013-09-21") == 35
    assert holiday_type("2009-01-26") == 71
    assert holiday_type("2017-10-08") == 72

if __name__ == "__main__":
    unit_test()
