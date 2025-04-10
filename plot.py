import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
# 读取 Excel 文件
file_path = '../fr-sm-dr_domain10.xlsx'
sheet_names = ['domain10', 'domain15', 'domain20', '1-Bayes']  # 假设工作表名称为 Sheet1, Sheet2, Sheet3, Sheet4
sheet_names = ["weight-TV"]
# 创建一个图形和轴
plt.figure(figsize=(10, 6))

# 遍历每个工作表
for sheet_name in sheet_names:
    # 读取工作表数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 假设数据在第一列是 x 轴，第二列是 y 轴
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_log = np.log(x)
    # 绘制折线
    plt.plot(x_log, y, label=sheet_name)

# 添加图例
print(math.exp(10))
plt.legend()

# 添加标题和标签
plt.title('weight-TV distance')
plt.xlabel('weight')
plt.ylabel('TV distance')
plt.grid(True)

# 显示图形
plt.show()