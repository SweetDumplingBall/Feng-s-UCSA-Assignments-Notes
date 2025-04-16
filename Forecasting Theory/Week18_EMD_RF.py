import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
# # 获取特征和目标变量
co_gt = data['CO(GT)'].values #y数据

#%% EMD分解，集成预测
emd=EMD()
eIMFs=emd.emd(co_gt)
print(len(eIMFs))

#IMF的图
fig, axes = plt.subplots(len(eIMFs)+1,1,figsize=(10,10))
# 存储结果
predicted_results = np.zeros_like(co_gt)

# 对每个 IMF预测，结果加回predicted_results
for i, imf in enumerate(eIMFs):
    # 准备训练数据
    X_train = np.arange(len(imf)).reshape(-1, 1)
    y_train = imf.reshape(-1, 1)
    
    # 训练随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 预测
    predicted_imf = rf_model.predict(X_train).flatten()
    
    # 将 IMF 的预测结果加回
    predicted_results += predicted_imf
    # 绘制IMF子图
    axes[i].plot(imf)
    axes[i].plot(predicted_imf)
    axes[i].legend()

#调整子图布局
plt.tight_layout()
plt.show()
#最终结果的图
plt.figure(figsize=(10, 5))
#可视化原始数据和最终预测结果
plt.plot(co_gt, label='Original Data')
plt.plot(predicted_results, label='Predicted Results')  # 绘制预测值
plt.title('Decision Tree Regression Prediction')  # 设置图形标题
plt.xlabel('Time')  # 设置X轴标签
plt.ylabel('CO')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形


