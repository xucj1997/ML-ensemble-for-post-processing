####检查时间序列的平稳性###########
####采用BIC准则计算，选定最小值作为阶数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")


data = 'filename.xlsx'#输入数据存储位置
data_error = pd.read_excel(data, sheet_name='Bovisio-Moloch simulatedQ',usecols=['error'])
# 定义要尝试的阶数范围
max_p =60  # 可根据需求进行调整

bic_values = {}# 初始化一个空字典，用于存储每个阶数对应的BIC值
values=[]
# 计算每个阶数对应的BIC值
for p in range(1, max_p + 1):
    model = sm.tsa.AR(data_error).fit(maxlag=p, ic='bic')
    bic_values[p] = model.bic
    values.append(model.bic)
    print(model.bic)
# 找到BIC值最小的阶数
best_p = min(bic_values, key=bic_values.get)
print(f"最佳阶数（P）: {best_p}")
#作图
plt.figure(figsize=(10,5))
data_error.error.plot()
plt.show()
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 100})  # 设置图片大小
plot_acf(data_error.error,lags=40)
plt.show()

