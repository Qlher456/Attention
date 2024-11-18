# Attention

# CNN.py
CNN.py是使用基础cnn网络在Cifar10数据集上迭代100次的文件

使用ACC评价标准和交叉熵损失函数

batch_size = 64

learning_rate = 0.001

Epoch [1/100] Train Loss: 1.5441, Train Acc: 43.86% Test Loss: 1.1843, Test Acc: 58.41%

Epoch [10/100] Train Loss: 0.6759, Train Acc: 76.03% Test Loss: 0.7941, Test Acc: 72.85%

Epoch [20/100] Train Loss: 0.4526, Train Acc: 83.17% Test Loss: 0.9331, Test Acc: 72.84%

Epoch [30/100] Train Loss: 0.3581, Train Acc: 86.36% Test Loss: 1.0677, Test Acc: 72.81%

Epoch [40/100] Train Loss: 0.3025, Train Acc: 88.26% Test Loss: 1.3608, Test Acc: 73.02%

Epoch [50/100] Train Loss: 0.2704, Train Acc: 89.42% Test Loss: 1.5004, Test Acc: 72.53%

Epoch [60/100] Train Loss: 0.2449, Train Acc: 90.36% Test Loss: 1.5833, Test Acc: 72.55%

Epoch [70/100] Train Loss: 0.2346, Train Acc: 90.76% Test Loss: 1.6706, Test Acc: 72.87%

Epoch [80/100] Train Loss: 0.2216, Train Acc: 91.34% Test Loss: 1.8683, Test Acc: 72.17%

Epoch [90/100] Train Loss: 0.2058, Train Acc: 91.91% Test Loss: 1.8556, Test Acc: 71.97%

Epoch [100/100] Train Loss: 0.1969, Train Acc: 92.28% Test Loss: 1.9651, Test Acc: 72.11%

![training_plot](https://github.com/user-attachments/assets/241774b2-29a6-4410-b49a-108931424047)


# Self-Attention.py

Self-Attention.py是在基础cnn网络中引入自注意力机制在Cifar10数据集上迭代100次的文件

训练环境同上

Epoch [1/100] Train Loss: 1.5294, Train Acc: 44.27% Test Loss: 1.1464, Test Acc: 58.42%

Epoch [10/100] Train Loss: 0.3852, Train Acc: 86.20% Test Loss: 0.7361, Test Acc: 77.08%

Epoch [20/100] Train Loss: 0.2026, Train Acc: 92.67% Test Loss: 0.9745, Test Acc: 77.55%

Epoch [30/100] Train Loss: 0.1569, Train Acc: 94.39% Test Loss: 1.2330, Test Acc: 76.77%

Epoch [40/100] Train Loss: 0.1439, Train Acc: 95.04% Test Loss: 1.5118, Test Acc: 75.73%

Epoch [50/100] Train Loss: 0.1224, Train Acc: 95.81% Test Loss: 1.4674, Test Acc: 76.74%

Epoch [60/100] Train Loss: 0.1187, Train Acc: 95.91% Test Loss: 1.5973, Test Acc: 76.51%

Epoch [70/100] Train Loss: 0.1163, Train Acc: 96.02% Test Loss: 1.6436, Test Acc: 77.22%

Epoch [80/100] Train Loss: 0.1047, Train Acc: 96.54% Test Loss: 1.7462, Test Acc: 76.69%

Epoch [90/100] Train Loss: 0.0980, Train Acc: 96.72% Test Loss: 1.7892, Test Acc: 76.37%

Epoch [100/100] Train Loss: 0.0909, Train Acc: 97.05% Test Loss: 1.7741, Test Acc: 77.09%

![image](https://github.com/user-attachments/assets/f2951def-11aa-4867-8632-da7b48b81165)

# Multi-Head Attention.py

Multi-Head Attention.py同样仿照自注意力机制机制，是在基础CNN网络的基础上添加多头注意力机制

Epoch [1/100] Train Loss: 1.5241, Train Acc: 44.52% Test Loss: 1.0950, Test Acc: 60.95%

Epoch [10/100] Train Loss: 0.3983, Train Acc: 85.71% Test Loss: 0.7326, Test Acc: 76.98%

Epoch [20/100] Train Loss: 0.2401, Train Acc: 91.30% Test Loss: 0.9744, Test Acc: 77.22%

Epoch [30/100] Train Loss: 0.2056, Train Acc: 92.75% Test Loss: 1.0913, Test Acc: 77.06%

Epoch [40/100] Train Loss: 0.2457, Train Acc: 91.40% Test Loss: 1.1072, Test Acc: 77.48%

Epoch [50/100] Train Loss: 0.1983, Train Acc: 93.03% Test Loss: 1.2855, Test Acc: 76.75%

Epoch [60/100] Train Loss: 0.2683, Train Acc: 90.85% Test Loss: 1.1503, Test Acc: 76.01%

Epoch [70/100] Train Loss: 0.2069, Train Acc: 92.86% Test Loss: 1.2311, Test Acc: 73.58%

Epoch [80/100] Train Loss: 0.1908, Train Acc: 93.27% Test Loss: 1.2665, Test Acc: 75.57%

Epoch [90/100] Train Loss: 0.3411, Train Acc: 88.43% Test Loss: 1.1230, Test Acc: 73.96%

Epoch [100/100] Train Loss: 0.3210, Train Acc: 89.11% Test Loss: 1.1639, Test Acc: 76.63%

![image](https://github.com/user-attachments/assets/0562aa28-264a-4287-b0f9-048c054493bd)

# SE-Attention.py

SE注意力机制

Epoch [1/100] Train Loss: 1.6306, Train Acc: 40.11% Test Loss: 1.3024, Test Acc: 52.55%

Epoch [10/100] Train Loss: 0.6070, Train Acc: 78.59% Test Loss: 0.7452, Test Acc: 75.75%

Epoch [20/100] Train Loss: 0.3428, Train Acc: 87.47% Test Loss: 0.8918, Test Acc: 76.70%

Epoch [30/100] Train Loss: 0.2320, Train Acc: 91.47% Test Loss: 1.1320, Test Acc: 76.22%

Epoch [40/100] Train Loss: 0.1789, Train Acc: 93.38% Test Loss: 1.2360, Test Acc: 76.37%

Epoch [50/100] Train Loss: 0.1468, Train Acc: 94.73% Test Loss: 1.4735, Test Acc: 76.77%

Epoch [60/100] Train Loss: 0.1258, Train Acc: 95.47% Test Loss: 1.5850, Test Acc: 75.93%

Epoch [70/100] Train Loss: 0.1138, Train Acc: 95.96% Test Loss: 1.6867, Test Acc: 76.26%

Epoch [80/100] Train Loss: 0.1063, Train Acc: 96.35% Test Loss: 1.6870, Test Acc: 76.06%

Epoch [90/100] Train Loss: 0.0919, Train Acc: 96.89% Test Loss: 1.8225, Test Acc: 76.24%

Epoch [100/100] Train Loss: 0.0896, Train Acc: 96.91% Test Loss: 1.8040, Test Acc: 76.20%

![image](https://github.com/user-attachments/assets/34e3804b-21da-4426-90cf-f74ce973733a)


# ECA-Attention.py

ECA注意力机制

Epoch [1/100] Train Loss: 1.6501, Train Acc: 38.87% Test Loss: 1.3050, Test Acc: 51.47%

Epoch [10/100] Train Loss: 0.6679, Train Acc: 76.59% Test Loss: 0.7386, Test Acc: 75.16%

Epoch [20/100] Train Loss: 0.3936, Train Acc: 85.70% Test Loss: 0.8330, Test Acc: 76.63%

Epoch [30/100] Train Loss: 0.2777, Train Acc: 89.74% Test Loss: 0.9841, Test Acc: 76.53%

Epoch [40/100] Train Loss: 0.2175, Train Acc: 91.94% Test Loss: 1.1751, Test Acc: 75.97%

Epoch [50/100] Train Loss: 0.1796, Train Acc: 93.30% Test Loss: 1.3175, Test Acc: 76.18%

Epoch [60/100] Train Loss: 0.1521, Train Acc: 94.42% Test Loss: 1.5923, Test Acc: 75.25%

Epoch [70/100] Train Loss: 0.1341, Train Acc: 95.07% Test Loss: 1.6354, Test Acc: 76.16%

Epoch [80/100] Train Loss: 0.1229, Train Acc: 95.49% Test Loss: 1.6603, Test Acc: 76.04%

Epoch [90/100] Train Loss: 0.1086, Train Acc: 96.05% Test Loss: 1.7811, Test Acc: 75.79%

Epoch [100/100] Train Loss: 0.1027, Train Acc: 96.34% Test Loss: 1.8989, Test Acc: 75.69%

![image](https://github.com/user-attachments/assets/419eaa69-2448-485d-a1c6-37ae1282c04a)

# CBAM-Attention.py

CBAM注意力机制

Epoch [1/100] Train Loss: 1.6973, Train Acc: 37.20% Test Loss: 1.3445, Test Acc: 50.43%

Epoch [10/100] Train Loss: 0.6806, Train Acc: 76.01% Test Loss: 0.7463, Test Acc: 74.88%

Epoch [20/100] Train Loss: 0.4233, Train Acc: 84.53% Test Loss: 0.8599, Test Acc: 75.36%

Epoch [30/100] Train Loss: 0.2979, Train Acc: 89.04% Test Loss: 1.0639, Test Acc: 75.47%

Epoch [40/100] Train Loss: 0.2348, Train Acc: 91.37% Test Loss: 1.2276, Test Acc: 75.50%

Epoch [50/100] Train Loss: 0.1883, Train Acc: 92.98% Test Loss: 1.3907, Test Acc: 75.47%

Epoch [60/100] Train Loss: 0.1670, Train Acc: 93.93% Test Loss: 1.5964, Test Acc: 74.59%

Epoch [70/100] Train Loss: 0.1530, Train Acc: 94.41% Test Loss: 1.6833, Test Acc: 74.36%

Epoch [80/100] Train Loss: 0.1337, Train Acc: 95.25% Test Loss: 1.7790, Test Acc: 74.67%

Epoch [90/100] Train Loss: 0.1221, Train Acc: 95.58% Test Loss: 1.9722, Test Acc: 74.96%

Epoch [100/100] Train Loss: 0.1146, Train Acc: 95.96% Test Loss: 1.9777, Test Acc: 74.84%

![image](https://github.com/user-attachments/assets/6d03c66b-ebbe-4a9c-aad7-5ff8be4ba7d8)

# STN-Attention.py

STN注意力机制

Epoch [1/100] Train Loss: 1.2672, Train Acc: 54.43% Test Loss: 1.0409, Test Acc: 63.29%

Epoch [10/100] Train Loss: 0.0806, Train Acc: 97.30% Test Loss: 1.4901, Test Acc: 73.06%

Epoch [20/100] Train Loss: 0.0411, Train Acc: 98.64% Test Loss: 2.2554, Test Acc: 71.88%

Epoch [30/100] Train Loss: 0.0309, Train Acc: 99.01% Test Loss: 2.7649, Test Acc: 72.40%

Epoch [40/100] Train Loss: 0.0322, Train Acc: 98.98% Test Loss: 3.0436, Test Acc: 72.20%

Epoch [50/100] Train Loss: 0.0248, Train Acc: 99.27% Test Loss: 3.3176, Test Acc: 71.79%

Epoch [60/100] Train Loss: 0.0266, Train Acc: 99.31% Test Loss: 3.9003, Test Acc: 71.77%

Epoch [70/100] Train Loss: 0.0264, Train Acc: 99.33% Test Loss: 4.3785, Test Acc: 71.97%

Epoch [80/100] Train Loss: 0.0156, Train Acc: 99.54% Test Loss: 4.4777, Test Acc: 72.36%

Epoch [90/100] Train Loss: 0.0244, Train Acc: 99.46% Test Loss: 5.4286, Test Acc: 70.88%

Epoch [100/100] Train Loss: 0.0244, Train Acc: 99.45% Test Loss: 5.3992, Test Acc: 71.31%

![image](https://github.com/user-attachments/assets/1940d67e-fbba-4ebf-9e21-2bd15298b923)
