import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
df = data['CO(GT)']
df.dropna(axis=0,inplace=True)

wavelet_name = 'db4'
#小波变换
coeffs = pywt.wavedec(df, wavelet_name, level=4)

#绘制原始信号图像
plt.figure(figsize=(8, 6))
plt.subplot(5, 1, 1)
plt.plot(data)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('CO(GT)')

#绘制小波分解信号图像
for i in range(1, len(coeffs)):
    plt.subplot(5, 1, i+1)
    plt.plot(coeffs[i])
    plt.title(f'wavelet Coefficients - level {i}')
    plt.xlabel('Time')
    plt.ylabel('CO')

plt.grid(True)
plt.show()