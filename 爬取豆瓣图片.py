
import requests  
import re  
import os  
  
#��ȡ��ҳ��ַ  
def gethtml(url):    
    
  try:  
    r = requests.get(url,headers={'User-Agent': 'Mozilla/4.0'})  
    r.raise_for_status()  
    r.encoding = r.apparent_encoding  
    return r.text  
  except:  
    return ""  
    
      
#������ҳ ��ȡͼƬ��ַ�ַ�����ʽ  
def getpicurl(html,piclist):  
  picurl = r'<img src="[^"]*"'     #����Ҫsrc=��ֻȡ����Ĳ���  
  list1 = re.findall(picurl,html)  
  list2 = []  
  for i in range(len(list1)):  
    a = re.findall(r'"[^"]*"',list1[i])[0]  #findall����һ���б�,��Ҫȡ�����е��ַ���  
    b = a.replace('/l/','/raw/')       #������ԭͼ��ַ  
    list2.append(b)  
  del list2[0]             #��3���ͼƬ  
  del list2[-2]  
  del list2[-1]  
  return  piclist + list2  
  
  
#��ȡ�� m�� �� nҳͼƬ  
def savepic(m,n):  
  for i in range(m-1,n):  
    url = 'https://movie.douban.com/celebrity/1018562/photos/?type=C&start={}&sortby=like&size=a&subtype=a'.format(30*i)  
    html = gethtml(url)  
    if i == m-1:    #piclist = [] ִֻ�е�һ��  
      piclist = []  
    piclist = getpicurl(html,piclist)  #����ͼƬ��ַ���ַ�����ʽ�������б���  
      
  x = 0            #������  
  for adress in piclist:  
    path = r'C:\pachong__yuyi\\'   #������C��  
    if not os.path.exists(path):  
      os.makedirs(path)  
    x += 1  
    f = open(path+'%s.jpg'%x,'ab+')  #��Ŀ¼+�ļ���  
    r = requests.get(eval(adress),headers={'User-Agent': 'Mozilla/4.0'})      #evalȥ������  
    f.write(r.content)  
    f.close()  
  print("�������")    
      
 #��ȡ��һ  ����ʮҳͼƬ   
savepic(1,10)  