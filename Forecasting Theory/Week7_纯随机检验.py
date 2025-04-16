import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

data = pd.read_excel(r'C:\Users\hongm\OneDrive\桌面\预测作业\Code_Examination\工作簿3.xlsx')
data_lst = data['sale']
test_result = acorr_ljungbox(data_lst,lags=[10],boxpierce =True)
print('result:'+str(test_result))
#结果看lb_pvalue远小于0.01，因此拒绝原假设，认为该时间序列不是纯随机序列。






