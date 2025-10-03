import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import models
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear

# -----------------------
# Helpers
# -----------------------
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, (list, tuple)):
        for elem in x:
            zero_gradients(elem)

def compute_jacobian(model, x):
    x = x.cuda()
    x.requires_grad = True
    output = model(x)
    num_features = int(np.prod(x.shape[1:]))
    jacobian = torch.zeros([output.size(1), num_features]).cuda()
    for i in range(output.size(1)):
        grad_mask = torch.zeros_like(output).cuda()
        grad_mask[:, i] = 1
        zero_gradients(x)
        output.backward(grad_mask, retain_graph=True)
        jacobian[i] = x.grad.view(-1, num_features).clone()[0]
    return jacobian

def saliency_map(jacobian, target_idx, increasing, search_space, nb_features):
    domain = (search_space == 1).float()
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_idx]
    others_grad = all_sum - target_grad
    if increasing:
        coef = 2 * (domain == 0).float()
    else:
        coef = -2 * (domain == 0).float()
    coef = coef.view(-1, nb_features)
    target_tmp = target_grad.clone() - coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
    others_tmp = others_grad.clone() + coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diag = torch.from_numpy(tmp).byte().cuda()
    if increasing:
        mask = (alpha > 0.0) & (beta < 0.0) & zero_diag.view_as(alpha)
    else:
        mask = (alpha < 0.0) & (beta > 0.0) & zero_diag.view_as(alpha)
    sal_map = alpha * torch.abs(beta) * mask.float()
    _, max_idx = torch.max(sal_map.view(-1, nb_features * nb_features), dim=1)
    p, q = max_idx // nb_features, max_idx % nb_features
    return p, q

# -----------------------
# Data
# -----------------------
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
testset = torchvision.datasets.CIFAR100(root='../../cifar100/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# -----------------------
# Model
# -----------------------
net = models.__dict__
pretrain_dict = torch.load('../../cifar100/resnet32/save_finetune/model_best.pth.tar')['state_dict']
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
net.eval().cuda()

# -----------------------
# Stage 1: Localization
# -----------------------
theta, gamma = 0.1, 0.5
ys_target = 2
increasing = True
I = []
var_target = Variable(torch.LongTensor([ys_target])).cuda()

for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    for j in range(len(data)):
        I_d = []
        img = data[j].unsqueeze(0)
        image = net(img)
        for i in range(6, len(image)):
            I_s = []
            mins, maxs = image[i].min(), image[i].max()
            sample = image[i].detach().clone().requires_grad_(True).cuda()
            num_features = int(np.prod(sample.shape[1:]))
            search_domain = (sample < maxs).view(num_features).cuda()
            model_branch = net.classifier  # ⚠️这里根据你的模型结构替换
            output = model_branch(sample)
            current = torch.max(output, 1)[1].cpu().numpy()
            delta = 0
            while current[0] != ys_target and delta < gamma:
                jacobian = compute_jacobian(model_branch, sample)
                p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
                p1, p2 = p1[0].item(), p2[0].item()
                if p1 not in I_s: I_s.append(p1)
                if p2 not in I_s: I_s.append(p2)
                last_sample = sample.detach().cpu().numpy()
                sample.view(-1)[p1] += theta
                sample.view(-1)[p2] += theta
                delta = np.linalg.norm(sample.detach().cpu().numpy() - last_sample)
                if sample.view(-1)[p1] < mins or sample.view(-1)[p1] > maxs:
                    search_domain[p1] = 0
                if sample.view(-1)[p2] < mins or sample.view(-1)[p2] > maxs:
                    search_domain[p2] = 0
                output = model_branch(sample)
                current = torch.max(output, 1)[1].cpu().numpy()
                if p1 == 0 and p2 == 0: break
            I_d.append(I_s)
        I.append(I_d)
    break

# merge
temp_I = [[] for _ in range(10)]
for j in range(len(I)):
    for i in range(len(I[j])):
        if I[j][i]:
            temp_I[i].append(I[j][i])

I_t = []
for j in range(10):
    I_t_temp = set()
    for indices in temp_I[j]:
        I_t_temp |= set(indices)
    I_t.append(np.array(list(I_t_temp)))
I_t = np.array(I_t)
np.save('./result/SNI.npy', I_t)

# -----------------------
# Stage 2: Adaptive Progressive Perturbation
# -----------------------
criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    break

y = net(data)
for i in range(len(y)):
    if i < 6: continue
    y[i][:, I_t[i-6]] = 10

var_target = target.clone()
var_target[:] = ys_target
perturbed = torch.zeros_like(data[0, 0:3, 21:31, 21:31])

step_size = 0.01
alpha, beta = 1.5, 0.5  # 自适应调节因子

for step in range(15):
    with torch.no_grad():
        data[:, :, 21:31, 21:31] = perturbed
    data.requires_grad = True
    if data.grad is not None: data.grad.zero_()
    output = net(data)
    loss_mse, loss_ce = 0, 0
    for i in range(len(output)):
        if i < 6: continue
        loss_mse += criterion1(output[i], y[i].detach())
        loss_ce += criterion2(output[i], var_target)
    loss = loss_mse + loss_ce
    zero_gradients(data)
    loss.backward()
    grad = data.grad[:, :, 21:31, 21:31].mean(0, keepdim=True)
    data.requires_grad = False
    old_loss = loss.item()
    # 更新 patch
    perturbed_new = perturbed - step_size * grad
    with torch.no_grad():
        data[:, :, 21:31, 21:31] = perturbed_new
        new_output = net(data)
        new_loss = 0
        for i in range(len(new_output)):
            if i < 6: continue
            new_loss += criterion1(new_output[i], y[i].detach()) + criterion2(new_output[i], var_target)
    if new_loss < old_loss:
        perturbed = perturbed_new
        step_size *= alpha
    else:
        step_size *= beta

# -----------------------
# Stage 3: Adaptive Refinement
# -----------------------
perturbed = torch.clamp(perturbed, -0.1, 0.1)
torch.save(perturbed, './result/perturbed_apa.pth')
print("Saved perturbed trigger at ./result/perturbed_apa.pth")

