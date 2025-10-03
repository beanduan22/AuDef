import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import models
import random
from torch.utils.data import DataLoader
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear
from bitstring import Bits

# --------------------
# Helper Functions
# --------------------
def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=requires_grad)

def set_weight_flat(module, flat_array):
    shape = module.weight.data.shape
    module.weight.data = flat_array.reshape(shape).clone()

def count_bit_flips(param, param1):
    b1 = Bits(int=int(param), length=8).bin
    b2 = Bits(int=int(param1), length=8).bin
    return sum([b1[k] != b2[k] for k in range(8)])

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --------------------
# Data Preparation
# --------------------
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = torchvision.datasets.tinyimagenet(root='../../tinyimagenet/resnet32/data', train=True, download=True, transform=transform_train)
loader_train = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.tinyimagenet(root='../../tinyimagenet/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# --------------------
# Model Setup
# --------------------
net = models.__dict__
net1 = models.__dict__
pretrain_dict = torch.load('../../tinyimagenet/resnet32/save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str(k): v for k, v in pretrain_dict.items() if str(k) in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
net.eval().cuda()
net1.load_state_dict(model_dict)
net1.eval().cuda()

for m in net.modules():
    if isinstance(m, (quan_Conv2d, quan_Linear)):
        m.__reset_stepsize__()
        m.__reset_weight__()
for m in net1.modules():
    if isinstance(m, (quan_Conv2d, quan_Linear)):
        m.__reset_stepsize__()
        m.__reset_weight__()

# --------------------
# Attack Setup
# --------------------
start, end = 21, 31
I_t = np.load('./result/SNI.npy')
I_t = torch.Tensor(I_t).long().cuda()
perturbed = torch.load('./result/perturbed.pth')

criterion = nn.CrossEntropyLoss().cuda()

# --------------------
# Stage 1: Sensitive Layer Identification
# --------------------
def find_psens(model, data_loader, perturbed):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        data[:, :, start:end, start:end] = perturbed
        y = model(data, nolast=True)[15]
        y[:, I_t] = 10
        break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()
    output_nolast = model(data, nolast=True)[15]
    output1 = model(data)[15]
    loss = criterion1(output_nolast, y.detach()) + criterion2(output1, ys_target)
    model.zero_grad()
    loss.backward()

    scores = []
    for m in model.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)) and m.weight.grad is not None:
            grad, weight = m.weight.grad.data.flatten(), m.weight.data.flatten()
            Q_p = max(weight).item()
            f = [abs(grad[i]) * (Q_p - weight[i] if grad[i] < 0 else 0) for i in range(len(grad))]
            scores.append(max(f) if f else 0)
        else:
            scores.append(0)
    return scores.index(max(scores)) + 1

# --------------------
# Stage 1b: Vulnerable Element Selection
# --------------------
def identify_vuln_elem(model, psens, data_loader, perturbed, num):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data, target = data.cuda(), target.cuda()
            data[:, :, start:end, start:end] = perturbed
            y = model(data, nolast=True)[15]
            y[:, I_t] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()
    output_nolast = model(data, nolast=True)[15]
    output1 = model(data)[15]
    loss = criterion1(output_nolast, y.detach()) + criterion2(output1, ys_target)
    model.zero_grad()
    loss.backward()

    n = 0
    for m in model.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            n += 1
            if n == psens:
                grad, weight = m.weight.grad.data.flatten(), m.weight.data.flatten()
                Q_p = max(weight).item()
                fit = [abs(grad[i]) * (Q_p - weight[i] if grad[i] < 0 else 0) for i in range(len(grad))]
                return fit.index(max(fit))
    return 0

# --------------------
# Stage 2: Adaptive Progressive Update
# --------------------
def adaptive_progressive_update(model, psens, ele_loc, data_loader, perturbed, num,
                                max_iters=20, base_step=1.0, alpha=1.5, beta=0.5):
    n, target_module = 0, None
    for m in model.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            n += 1
            if n == psens:
                target_module = m
                break
    flat = target_module.weight.data.flatten().clone()
    old_val = float(flat[ele_loc].item())

    step = base_step
    best_val, best_loss = old_val, float('inf')

    for t in range(max_iters):
        improved = False
        for d in (1, -1):
            cand = old_val + d * step
            cand_flat = flat.clone()
            cand_flat[ele_loc] = cand
            set_weight_flat(target_module, cand_flat)

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    if batch_idx == num:
                        data, target = data.cuda(), target.cuda()
                        data[:, :, start:end, start:end] = perturbed
                        y = model(data, nolast=True)[15]
                        y[:, I_t] = 10
                        break
                ys_target = torch.zeros_like(target)
                ys_target[:] = 2
                criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()
                output_nolast = model(data, nolast=True)[15]
                output1 = model(data)[15]
                loss = criterion1(output_nolast, y.detach()) + criterion2(output1, ys_target)
                loss_val = loss.item()

            if loss_val < best_loss:
                best_loss, best_val = loss_val, cand
                improved = True

        step = step * alpha if improved else step * beta
        if step < 1e-6:
            break

    set_weight_flat(target_module, flat)  # restore
    return best_val

# --------------------
# Stage 3: Adaptive Refinement
# --------------------
def adaptive_refine_elements(model, modified_list, loader_test, perturbed, num,
                             refine_iters=8, base_step=1.0, alpha=1.2, beta=0.7):
    improved = []
    for psens, ele_loc, cur_val in modified_list:
        n, target_module = 0, None
        for m in model.modules():
            if isinstance(m, (quan_Conv2d, quan_Linear)):
                n += 1
                if n == psens:
                    target_module = m
                    break
        flat = target_module.weight.data.flatten().clone()
        step = base_step
        best_val, best_loss = cur_val, float('inf')

        for t in range(refine_iters):
            improved_flag = False
            for d in (1, -1):
                cand = cur_val + d * step
                cand_flat = flat.clone()
                cand_flat[ele_loc] = cand
                set_weight_flat(target_module, cand_flat)

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(loader_test):
                        if batch_idx == num:
                            data, target = data.cuda(), target.cuda()
                            data[:, :, start:end, start:end] = perturbed
                            y = model(data, nolast=True)[15]
                            y[:, I_t] = 10
                            break
                    ys_target = torch.zeros_like(target)
                    ys_target[:] = 2
                    criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()
                    output_nolast = model(data, nolast=True)[15]
                    output1 = model(data)[15]
                    loss = criterion1(output_nolast, y.detach()) + criterion2(output1, ys_target)
                    loss_val = loss.item()

                if loss_val < best_loss:
                    best_loss, best_val = loss_val, cand
                    improved_flag = True

            step = step * alpha if improved_flag else step * beta
            if step < 1e-6:
                break

        set_weight_flat(target_module, flat)  # restore
        if best_val != cur_val:
            improved.append((psens, ele_loc, best_val))
            flat[ele_loc] = best_val
            set_weight_flat(target_module, flat)
    return improved

# --------------------
# Attack Loop
# --------------------
psens = find_psens(net1, loader_test, perturbed)
print("Sensitive layer:", psens)
num, last_loc = 0, -1
n_b = 0
ASR, ASR_t, n_b_max = 0, 90, 500
modified_elements = []

dpi, width, height = 80, 1200, 800
fig = plt.figure(figsize=(width/float(dpi), height/float(dpi)))
x_axis, y_axis = [], []

while n_b < n_b_max:
    ele_loc = identify_vuln_elem(net1, psens, loader_test, perturbed, num)
    if ele_loc == last_loc:
        num += 1
    if num >= 8: num = 0
    last_loc = ele_loc

    new_val = adaptive_progressive_update(net1, psens, ele_loc, loader_test, perturbed, num)
    n = 0
    for m in net1.modules():
        if isinstance(m, (quan_Conv2d, quan_Linear)):
            n += 1
            if n == psens:
                flat = m.weight.data.flatten().clone()
                old_val = float(flat[ele_loc].item())
                flat[ele_loc] = new_val
                set_weight_flat(m, flat)
                modified_elements.append((psens, ele_loc, new_val))
                n_b += count_bit_flips(old_val, new_val)
                break

    # measure ASR
    with torch.no_grad():
        correct, total = 0, 0
        for data, target in loader_test:
            target[:] = 2
            data[:, :, start:end, start:end] = perturbed
            data, target = data.cuda(), target.cuda()
            output = net1(data)[15]
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        ASR = 100.0 * correct / total

    print(f"Bit flips: {n_b}, ASR: {ASR:.2f}%")
    x_axis.append(n_b)
    y_axis.append(ASR)
    plt.clf()
    plt.xlabel('Bit Flips')
    plt.ylabel('ASR (%)')
    plt.plot(x_axis, y_axis)
    fig.savefig('./result/asr_apa.png', dpi=dpi, bbox_inches='tight')

    if ASR >= ASR_t:
        print("Reached target ASR.")
        break

    if len(modified_elements) % 5 == 0:
        refined = adaptive_refine_elements(net1, modified_elements, loader_test, perturbed, num)
        if refined:
            print("Refined:", refined)

