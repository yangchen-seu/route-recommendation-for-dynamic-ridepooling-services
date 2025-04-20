import pickle
import time
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
import Predict_class





# 读取数据
pre = Predict_class.Predict()
with open("tmp/OD.pickle", 'rb') as f:
    OD_dict: dict = pickle.load(f)

num_ODs = len(OD_dict)

# 定义 Bayesian Neural Network (BNN)
class BNNGuide(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 16)
        self.fc1.weight = PyroSample(lambda: dist.Normal(0., 1.).expand([16, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(lambda: dist.Normal(0., 1.).expand([16]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](16, 8)
        self.fc2.weight = PyroSample(lambda: dist.Normal(0., 1.).expand([8, 16]).to_event(2))
        self.fc2.bias = PyroSample(lambda: dist.Normal(0., 1.).expand([8]).to_event(1))

        self.fc3 = PyroModule[nn.Linear](8, 1)
        self.fc3.weight = PyroSample(lambda: dist.Normal(0., 1.).expand([1, 8]).to_event(2))
        self.fc3.bias = PyroSample(lambda: dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
     

# 定义模型
bnn = BayesianNN(input_dim=num_ODs)

# 变分推断 Guide
def guide(x):
    return bnn(x)

# 目标函数
def objective_function(theta_tensor):
    """将 theta_dict 传递给 pre.run() 并计算利润"""
    theta_values = theta_tensor.detach().numpy()
    theta_dict = {OD_id: {'shortest_path': theta_values[i], 'highest_potential_path': 1 - theta_values[i]} for i, OD_id in enumerate(OD_dict.keys())}
    platform_profit = pre.run(theta_dict)
    return platform_profit

# 训练 BNN
optimizer = Adam({"lr": 0.01})
svi = SVI(bnn, guide, optimizer, loss=TraceMeanField_ELBO())

num_epochs = 500
losses = []
theta_samples = []
profit_samples = []

for epoch in range(num_epochs):
    theta_sample = torch.rand(num_ODs)  # 采样 theta
    platform_profit = objective_function(theta_sample)  # 计算利润
    theta_samples.append(theta_sample.numpy())
    profit_samples.append(platform_profit)

    loss = svi.step(theta_sample.unsqueeze(0), torch.tensor([platform_profit], dtype=torch.float32))
    losses.append(loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 画损失下降曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('ELBO Loss')
plt.title('BNN Training Loss')
plt.show()

# 预测最优 theta 并计算利润
with torch.no_grad():
    best_theta = torch.rand(num_ODs)
    best_profit = objective_function(best_theta)
    print("Best Theta:", best_theta.numpy())
    print("Predicted Platform Profit:", best_profit)
