import random

import pandas as pd

df = pd.read_csv('../paras.csv')
print(df.head())

# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]

# 字典
dict = {'name': nme, 'site': st, 'age': ag}

df = pd.DataFrame(dict)

# 保存 dataframe
df.to_csv('site.csv')
salaries = [random.random()*100 for _ in range(len(ag))]
add_attr = {'salary': salaries}
df.to_csv('site.csv')
print(df)
