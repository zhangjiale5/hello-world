import requests
import bs4
from bs4 import BeautifulSoup


def getHTMLText(url):
  try:
    r = requests.get(url,timeout = 30)
    r.raise_for_status()                 #捕捉异常
    r.encoding = r.apparent_encoding
    return r.text
  except:
    return ""
    

def fillUnivList(ulist,html):
  soup = BeautifulSoup(html,'html.parser')   #解析html
  for tr in soup.find('tbody').children:    #<tbody>--<tr>--<td>
    if isinstance(tr,bs4.element.Tag):
      tds = tr('td')
      ulist.append([tds[0].string,tds[1].string,tds[2].string,tds[3].string]) #添加整个列表，列表中有4个变量


def printUnivList(ulist,num):
  print('{0:^8}\t{1:{4}^8}\t{2:^14}\t{3:^14}'.format('排名','学校名称','省市','得分',chr(12288)))  # 中英文混排版解决：chr(12288) 是中文填充对齐
  for i in range(num):
    u = ulist[i]
    print('{0:^10}\t{1:{4}^10}\t{2:{4}^10}\t{3:^}'.format(u[0],u[1],u[2],u[3],chr(12288)))

  
def main():
  uinfo = []
  url = 'http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html'
  html = getHTMLText(url)
  fillUnivList(uinfo,html)
  printUnivList(uinfo,20)  #20 univ
  
main()