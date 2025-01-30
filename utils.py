import math
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

# class Stack:
#     def __init__(self, size):
#         self.size = size          # 栈的最大容量
#         self.stack = []           # 用列表来实现栈
#         self.sum = 0              # 用来跟踪栈内元素的和
#
#     def push(self, value):
#         if len(self.stack) == self.size:
#             # 如果栈已经满了，先出栈最底的一个元素
#             self.sum -= self.stack.pop(0)  # 移除并减去栈底元素的值
#         # 入栈
#         self.stack.append(value)
#         self.sum += value
#         # 计算并返回栈内元素的平均值
#        return self.sum / len(self.stack)








def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_stdclass Stack:
    def __init__(self, size, beta):
        self.size = size          # 栈的最大容量
        self.stack = []           # 用列表来实现栈
        self.sum = 0              # 用来跟踪栈内元素的和
        self.beta = beta

    def push(self, value):
        if len(self.stack) == self.size:
            # 如果栈已经满了，先出栈最底的一个元素
            self.sum -= self.stack.pop(0)  # 移除并减去栈底元素的值
        # 入栈
        self.stack.append(value)
        result = self.sum * self.beta + (1 - self.beta) * value
        self.sum += value
        # 计算并返回栈内元素的平均值
        return result
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RNDModel(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(RNDModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

    def compute_bonus(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_next_feature = self.target(next_obs)
        predict_next_feature = self.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.data.cpu().numpy()