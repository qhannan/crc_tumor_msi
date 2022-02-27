import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results.csv')

# x轴刻度标签
x_ticks = list(range(100))
# x轴范围（0, 1, ..., len(x_ticks)-1）
x = np.arange(len(x_ticks))
# 第1条折线数据
y1 = list(df['trainloss'])
# 第2条折线数据
y2 = list(df['valloss'])

plt.figure(figsize=(10, 6))
# 画第1条折线，参数看名字就懂，还可以自定义数据点样式等等。
plt.plot(x, y1, color='#FF0000', label='train_loss', linewidth=0.1)
# 画第2条折线
plt.plot(x, y2, color='#00FF00', label='val_loss', linewidth=0.1)

plt.yticks(fontsize=18)
# 添加x轴和y轴标签
plt.xlabel(u'epoch', fontsize=18)
plt.ylabel(u'CrossEntropyLoss', fontsize=18)

# 标题
plt.title(u'Epoch-Loss', fontsize=18)
# 图例
plt.legend(fontsize=18)

# 保存图片
plt.savefig('./figure_loss.png', bbox_inches='tight')
# 显示图片
# plt.show()