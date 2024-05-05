import datetime as dt
from dateutil import tz 

# time zone
my_timezone = tz.gettz('America/Los_Angeles')

# time class
my_time = dt.time(14,12,22, tzinfo=my_timezone)
print(my_time, my_time.tzname(), type(my_time))

# date class
# => no timezone info
date1 = dt.date(2020, 12, 11)
print(date1)
print(date1.month)
print(date1.day)
print(dt.date.max)
print(dt.date.min)
print(dt.date.today().weekday())
print(dt.date.strftime(date1, format='%Y-%M-%d'))
print(date1.strftime('%y/%m/%d'))
print(date1.isoformat())
date2 = dt.date.fromisoformat('2020-03-12') # isoformat yyyy-mm-dd (len of 10)

# datetime class
datetime1 = dt.datetime(
    year=2023, month=11, day=10, hour=21, 
    minute=5, second=10, microsecond=10000,
    tzinfo=tz.gettz('America/Los_Angeles'))
print(datetime1)

print(datetime1.tzinfo)
print(dt.datetime.now(tz=my_timezone))

dt1 = dt.datetime(2015, 5, 21, 12, 0)
dt2 = dt.datetime(2015, 5, 21, tzinfo=my_timezone)
print(dt1.tzname())
print(dt2.tzname())
print(
    dt2, 
    dt2.date(),
    dt.datetime.combine(dt2.date(), dt.time(2,30,15))
    )

dt2_utc = dt2.astimezone(tz=tz.tzutc())
print(dt2, '|', dt2_utc) # daylight saving applied since it is May

dt3 = dt.datetime(2015, 11, 21, tzinfo=tz.tzlocal())
print(dt3, '|', dt3.astimezone(tz=tz.tzutc())) # standard time since it is November 
print(dt1, '|', dt1.astimezone(tz=tz.tzutc())) # when no time zone provided (None) default is local time

# Timedelta class
# creating datetime objects
date1 = dt.datetime(2020, 1, 3)
date2 = dt.datetime(2020, 2, 3)
diff = date2 - date1 # the difference is timedelta class instance 
print(diff, type(diff))

date3 = date1 + dt.timedelta(days=4, hours=10)
print(date3, type(date3))

