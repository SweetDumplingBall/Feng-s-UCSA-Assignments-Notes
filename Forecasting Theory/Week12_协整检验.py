import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
#file_name = 'E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data'
df = pd.read_excel('E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data/房价和M2.xlsx')
m2 = df['x']
houseprice = df['y']

## 单位根检验
#进行ADF单位根检验-原始数据
result_orig_houseprice = adfuller(houseprice)
print('houseprce单位根检验p值:',result_orig_houseprice[1])
result_orig_m2 = adfuller(m2)
print('m2单位根检验p值:',result_orig_m2[1])

# # 做一阶差分(对原始数据进行差分操作)
diff_houseprice =houseprice.diff().dropna()
result_diff_houseprice = adfuller(diff_houseprice)
print('diff_houseprce单位根检验p值:',result_diff_houseprice[1])
diff_m2 =m2.diff().dropna()
result_diff_m2 = adfuller(diff_m2)
print('diff_m2单位根检验p值:',result_diff_m2[1])


# # 协整检验
result =coint(houseprice,m2)
print('houseprice和m2协整检验:',result)