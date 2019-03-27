# -*- coding:utf-8 -*-
"""
@auther：Charles Van
E-mail:  williananjhon@hotmail.com
@time：2019/1/27 11:08
@Project:TensorFlow_Deeping_learning
@Filename:9-31_Stock_Data.py
"""

import urllib.request
import re

stock_CodeUrl = 'https://quoqe.eastmoney.com/stocklist.html'
#获取股票代码列表
def urlTolist(url):
	allCodeList = []
	html = urllib.request.urlopen(url).read()
	html = html.decode('utf-8')
	s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
	pat = re.compile(s)
	code = pat.findall(html)
	for item in code:
		if item[0] == '6' or item[0] == '3' or item[0] == '0':
			allCodeList.append(item)
	return allCodeList

start = '20161031'
end='20161231'

# test
code = '600000'
url = 'http://quotes.money.163.com/service/chddata.html?code=0' + code + \
      '&start=' + start + '&end=' + end + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
urllib.request.urlretrieve(url, 'd:\\all_stock_data\\' + code + '_' + end + '.csv')  # 可以加一个参数dowmback显示下载进度
