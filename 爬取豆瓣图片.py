
import requests  
import re  
import os  
  
#获取网页地址  
def gethtml(url):    
    
  try:  
    r = requests.get(url,headers={'User-Agent': 'Mozilla/4.0'})  
    r.raise_for_status()  
    r.encoding = r.apparent_encoding  
    return r.text  
  except:  
    return ""  
    
      
#解析网页 获取图片地址字符串形式  
def getpicurl(html,piclist):  
  picurl = r'<img src="[^"]*"'     #不需要src=，只取后面的部分  
  list1 = re.findall(picurl,html)  
  list2 = []  
  for i in range(len(list1)):  
    a = re.findall(r'"[^"]*"',list1[i])[0]  #findall返回一个列表,需要取处其中的字符串  
    b = a.replace('/l/','/raw/')       #更换到原图地址  
    list2.append(b)  
  del list2[0]             #这3项不是图片  
  del list2[-2]  
  del list2[-1]  
  return  piclist + list2  
  
  
#爬取第 m到 第 n页图片  
def savepic(m,n):  
  for i in range(m-1,n):  
    url = 'https://movie.douban.com/celebrity/1018562/photos/?type=C&start={}&sortby=like&size=a&subtype=a'.format(30*i)  
    html = gethtml(url)  
    if i == m-1:    #piclist = [] 只执行第一次  
      piclist = []  
    piclist = getpicurl(html,piclist)  #所有图片地址以字符串形式保存在列表中  
      
  x = 0            #计数器  
  for adress in piclist:  
    path = r'C:\pachong__yuyi\\'   #下载在C盘  
    if not os.path.exists(path):  
      os.makedirs(path)  
    x += 1  
    f = open(path+'%s.jpg'%x,'ab+')  #根目录+文件名  
    r = requests.get(eval(adress),headers={'User-Agent': 'Mozilla/4.0'})      #eval去除引号  
    f.write(r.content)  
    f.close()  
  print("下载完成")    
      
 #爬取第一  至第十页图片   
savepic(1,10)  