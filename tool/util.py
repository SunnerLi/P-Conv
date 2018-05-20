
def now():
    import datetime
    import arrow
    _ = arrow.get(datetime.datetime.now(), 'Asia/Shanghai')
    _ = str(_).replace("T", "  ").replace("+", "  +")
    return _[:-10]