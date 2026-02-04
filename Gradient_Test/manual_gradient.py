import torch
import matplotlib.pyplot as plt # 如果你没装这个，可以先不画图，或者 pip install matplotlib

# 1. 准备数据 (真实值 y = 3x + 0.8)
x = torch.rand(100) 
y = 3 * x + 0.8 + 0.05 * torch.rand(100) # 加一点点随机噪音，模拟真实世界数据

# 2. 初始化权重 (w) 和 偏置 (b)
# requires_grad=True 是关键！告诉 PyTorch：请盯着这两个变量，我要对它们求导！
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

print(f"训练前猜测: y = {w.item():.2f}x + {b.item():.2f}")

# 3. 设置学习率 (步长)
learning_rate = 0.05

# 4. 开始训练循环 (训练 1000 次)
for epoch in range(1000):
    
    # --- 前向传播 (Forward Pass) ---
    y_pred = w * x + b
    
    # --- 计算损失 (Loss) - 均方误差 ---
    loss = (y_pred - y).pow(2).mean()
    
    # --- 反向传播 (Backward Pass) ---
    # 这一行执行完，PyTorch 会自动算出 loss 对 w 和 b 的导数，存入 w.grad 和 b.grad
    loss.backward()
    
    # --- 手动更新参数 (Gradient Descent) ---
    # 重点：更新参数时不能产生新的梯度，所以要用 no_grad 
    with torch.no_grad():
        w -= learning_rate * w.grad  # w = w - lr * 梯度
        b -= learning_rate * b.grad
        
        # 重点：每次更新完，必须把梯度清零！否则下次梯度会累加起来
        w.grad.zero_()
        b.grad.zero_()
    
    # 每 100 次打印一下进度
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.2f}, b = {b.item():.2f}")

print("----------------")
print(f"真实目标: y = 3.00x + 0.80")
print(f"训练结果: y = {w.item():.2f}x + {b.item():.2f}")

# 1. 转换数据：把 Tensor 变成普通的 numpy 数组，才能画图
# .detach() 是告诉 PyTorch："把数据拿出来，别算梯度了"
x_numpy = x.numpy()
y_true_numpy = y.numpy()
y_pred_numpy = y_pred.detach().numpy() # 这是模型最后预测出的直线

# 2. 画图
plt.figure(figsize=(8, 6))

# 画出真实的数据点 (蓝色的点，带一点点噪音)
plt.scatter(x_numpy, y_true_numpy, label='Real Data (with noise)', color='blue', alpha=0.5)

# 画出模型学到的直线 (红色的线)
plt.plot(x_numpy, y_pred_numpy, label='Model Prediction', color='red', linewidth=2)

plt.legend()
plt.title(f'Final Model: y = {w.item():.2f}x + {b.item():.2f}')
plt.show()