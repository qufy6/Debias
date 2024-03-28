import matplotlib.pyplot as plt
import os

# 假设这是你的数据
ss = [64.87565044,
64.713255,
64.91903619,
65.08436187,
64.73661605,
64.99016288
]
lms = [89.71425159,
89.83114097,
90.55587937,
89.5248347,
89.84499316,
89.67471332
]
alpha_values = [0,0.2,0.4,0.6,0.8,1]

# 创建一个图形和一个子图
fig, ax = plt.subplots()

# 遍历不同的alpha值
for alpha in alpha_values:
    ax.plot(ss, lms, marker='o', linestyle='-', alpha=alpha, label=f'Alpha={alpha}')

# 添加标题和标签
ax.set_title('Plot of lms vs. ss with different alpha values')
ax.set_xlabel('ss')
ax.set_ylabel('lms')

# 添加图例
ax.legend()


# 显示图形
plt.show()
