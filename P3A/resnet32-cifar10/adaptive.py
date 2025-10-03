import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import models
import utils
import math
import copy
import random
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

net = models.__dict__
net1 = models.__dict__
pretrain_dict = torch.load('../../cifar10/resnet32/save_finetune/model_best.pth.tar')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str(k): v for k, v in pretrain_dict.items() if str(k) in model_dict}
model_dict.update(pretrained_dict)

net.load_state_dict(model_dict)
net.eval()
net = net.cuda()

net1.load_state_dict(model_dict)
net1.eval()
net1 = net1.cuda()

for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
for m in net1.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()

# ---------- data ----------
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

trainset = torchvision.datasets.CIFAR10(root='../../cifar10/resnet32/data', train=True, download=True, transform=transform_train)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../cifar10/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

start = 21
end = 31
I_t = np.load('./result/SNI.npy', allow_pickle=True)
perturbed = torch.load('./result/perturbed.pth')

criterion = nn.CrossEntropyLoss().cuda()

# ---------- helper functions ----------
def accuracy(output, target, topk=(1, )):
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

def set_weight_flat(module, flat_array):
    shape = module.weight.data.shape
    module.weight.data = flat_array.reshape(shape).clone()

from bitstring import Bits
def countingss(param,param1):
    count = 0
    b1=Bits(int=int(param), length=8).bin
    b2=Bits(int=int(param1), length=8).bin
    for k in range(8):
        diff=int(b1[k])-int(b2[k])
        if diff!=0:
            count=count+1
    return count

index_list = []
def validate2(val_loader, model, criterion, num_branch):
    global index_list
    index_list = []  # reset each run
    top1_list = [AverageMeter() for _ in range(num_branch)]
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            output_branch = model(input)
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1,5))
                top1_list[idx].update(prec1, input.size(0))

    max_ = 0
    for c_, item in enumerate(top1_list):
        if item.avg > max_:
            max_ = item.avg
            index_list.append(c_)
    return index_list

def validate(val_loader, model, criterion, num_branch):
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            out_list = []
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            num_c = 3
            for j in range(input.size(0)):
                tar = target[j].unsqueeze(0)
                pre_index = random.sample(index_list[4:], num_c)
                c_ = 0
                for item in sorted(pre_index):
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        sm_out = out_list[item][0][j]
                        out = sm_out.unsqueeze(0)
                        loss = criterion(out, tar)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        break
                    c_ += 1
    print("top1.avg!:", top1.avg)
    return top1.avg

def validate_for_attack(val_loader, model, criterion, num_branch, xh):
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] = 2
            input[:,0:3,start:end,start:end] = xh
            target = target.cuda()
            input = input.cuda()
            out_list = []
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            num_c = 3
            for j in range(input.size(0)):
                tar = target[j].unsqueeze(0)
                pre_index = random.sample(index_list[4:], num_c)
                c_ = 0
                for item in sorted(pre_index):
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        sm_out = out_list[item][0][j]
                        out = sm_out.unsqueeze(0)
                        loss = criterion(out, tar)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        break
                    c_ += 1
    print("top1.asr!:", top1.avg)
    return top1.avg


def find_optim_value_simulate(model, psens, ele_loc, data_loader, choice, perturbed, num):

    model.eval()

    n = 0
    target_module = None
    for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n += 1
            if n == psens:
                target_module = m
                break
    assert target_module is not None, "psens module not found"

    orig_flat = target_module.weight.data.flatten().clone()
    # set candidate
    cand_flat = orig_flat.clone()
    cand_flat[ele_loc] = choice
    set_weight_flat(target_module, cand_flat)

    # compute loss metric
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    loss_val = None
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data, target = data.cuda(), target.cuda()
            # baseline prediction loss term to keep behavior close
            pre = model(data)
            loss_ce = 0
            for i in range(len(pre)):
                if i < 6 or i == 7:
                    continue
                loss_ce += criterion2(pre[i], target)
            # with perturbation region:
            data[:,:,start:end,start:end] = perturbed
            y = model(data, nolast=True)
            for i in range(len(y)):
                if i < 6 or i == 7:
                    continue
                y[i][:, I_t[i-6]] = 10
            ys_target = torch.zeros_like(target)
            ys_target[:] = 2
            output_nolast = model(data, nolast=True)
            output1 = model(data)
            loss_cbs = 0
            for i in range(len(output_nolast)):
                if i < 6 or i == 7:
                    continue
                loss_cbs += criterion1(output_nolast[i], y[i].detach()) + criterion2(output1[i], ys_target)
            loss = loss_cbs + 2 * loss_ce
            loss_val = loss.item()
            break

    # restore original weight
    set_weight_flat(target_module, orig_flat)
    return loss_val

# ---------- P3A Stage2: Progressive update ----------
def progressive_update(model, psens, ele_loc, data_loader, perturbed, num,
                       max_iters=12, step_multiplier=1.0, directions=(1,-1)):

    n = 0
    target_module = None
    for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n += 1
            if n == psens:
                target_module = m
                break
    assert target_module is not None
    orig_flat = target_module.weight.data.flatten().clone()
    old_val = orig_flat[ele_loc].item()
    base_step = float(getattr(target_module, "step_size", 1.0))
    if base_step == 0:
        base_step = 1.0
    best_loss = float('inf')
    best_val = old_val

    for t in range(1, max_iters+1):
        step = base_step * step_multiplier * t
        for d in directions:
            cand = old_val + d * step
            loss = find_optim_value_simulate(model, psens, ele_loc, data_loader, cand, perturbed, num)
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_val = cand

        if best_val != old_val and best_loss < 1e-3:
            break

    if best_val < old_val:
        best_val = old_val
    return best_val, best_loss

def refine_modified_elements(model, modified_list, loader_test, perturbed, num,
                             top_k=5, refine_iters=6):

    improved = []
    for idx, (psens, ele_loc, orig_val, cur_val) in enumerate(modified_list):
        n = 0
        target_module = None
        for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                n += 1
                if n == psens:
                    target_module = m
                    break
        if target_module is None:
            continue
        base_step = float(getattr(target_module, "step_size", 1.0))
        best_val = cur_val
        best_loss = find_optim_value_simulate(model, psens, ele_loc, loader_test, cur_val, perturbed, num)
        for t in range(1, refine_iters+1):
            for d in (1, -1):
                cand = cur_val + d * base_step * 0.5 * t
                loss = find_optim_value_simulate(model, psens, ele_loc, loader_test, cand, perturbed, num)
                if loss is not None and loss < best_loss:
                    best_loss = loss
                    best_val = cand
        if best_val != cur_val:
            # write back
            orig_flat = target_module.weight.data.flatten().clone()
            new_flat = orig_flat.clone()
            new_flat[ele_loc] = best_val
            set_weight_flat(target_module, new_flat)
            improved.append((psens, ele_loc, cur_val, best_val, best_loss))
    return improved

def find_psens(model, data_loader, perturbed):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        data[:,:,start:end,start:end] = perturbed
        y = model(data, nolast=True)
        for i in range(len(y)):
            if i<6 or i==7:
                continue
            y[i][:,I_t[i-6]] = 10
        break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data, nolast=True)
    output1 = model(data)
    loss_mse = 0
    loss_ce = 0
    for i in range(len(output_nolast)):
        if i<6 or i==7:
            continue
        loss_mse += criterion1(output_nolast[i], y[i].detach())
    for i in range(len(output1)):
        if i<6 or i==7:
            continue
        loss_ce += criterion2(output1[i], ys_target)
    loss = loss_mse + loss_ce
    model.zero_grad()
    loss.backward()
    F = []
    for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            if m.weight.grad is not None:
                fit = []
                p_grad = m.weight.grad.data.flatten()
                p_weight = m.weight.data.flatten()
                Q_p = max(p_weight.abs()).item()
                for i in range(len(p_grad)):
                    if p_grad[i] < 0:
                        step = Q_p - p_weight[i]
                    else:
                        step = 0
                    f = abs(p_grad[i]) * step
                    fit.append(f)
                fit = max(fit) if len(fit)>0 else 0
                F.append(fit)
            else:
                F.append(0)
    idx = int(np.argmax(F))
    return (idx+1)

def identify_vuln_elem(model, psens, data_loader, perturbed, num):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data, target = data.cuda(), target.cuda()
            data[:,:,start:end,start:end] = perturbed
            y = model(data, nolast=True)
            for i in range(len(y)):
                if i<6 or i==7:
                    continue
                y[i][:,I_t[i-6]] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data, nolast=True)
    output1 = model(data)
    loss_mse = 0
    loss_ce = 0
    for i in range(len(output_nolast)):
        if i<6 or i==7:
            continue
        loss_mse += criterion1(output_nolast[i], y[i].detach())
    for i in range(len(output1)):
        if i<6 or i==7:
            continue
        loss_ce += criterion2(output1[i], ys_target)
    loss = loss_mse + loss_ce
    loss.backward()
    n = 0
    fit = None
    for name, m in model.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n += 1
            if m.weight.grad is not None and n == psens:
                p_grad = m.weight.grad.data.flatten()
                p_weight = m.weight.data.flatten()
                Q_p = max(p_weight.abs()).item()
                fit_list = []
                for i in range(len(p_grad)):
                    if p_grad[i] < 0:
                        step = Q_p - p_weight[i]
                    else:
                        step = 0
                    f = abs(p_grad[i]) * step
                    fit_list.append(f)
                fit = fit_list
                break
    if fit is None:
        return 0
    index = int(np.argmax(fit))
    return index

psens = find_psens(net1, loader_test, perturbed)
print("psens:", psens)

# small-batch loader for element identification
loader_test_small = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# bookkeeping
modified_elements = []  # list of tuples (psens, ele_loc, orig_val, cur_val)
n_b = 0
ASR_t = 90
n_b_max = 500
num = 0
last_loc = -1

# plotting setup
dpi = 80
width, height = 1200, 800
fig = plt.figure(figsize=(width/float(dpi), height/float(dpi)))
x_axis, y_axis = [], []

while n_b < n_b_max and n_b < 500 and len(modified_elements) < 500:
    ele_loc = identify_vuln_elem(net1, psens, loader_test_small, perturbed, num)
    if ele_loc == last_loc:
        num += 1
    if num >= 8:
        num = 0
    last_loc = ele_loc

    n = 0
    target_module = None
    for name, m in net1.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n += 1
            if n == psens:
                target_module = m
                break
    if target_module is None:
        print("target module not found for psens:", psens)
        break
    flat = target_module.weight.data.flatten().clone()
    old_elem = float(flat[ele_loc].item())

    new_elem, best_loss = progressive_update(net1, psens, ele_loc, loader_test_small, perturbed, num,
                                            max_iters=10, step_multiplier=1.0, directions=(1,-1))

    ASR = validate_for_attack(loader_test, net1, criterion, 16, perturbed)
    _ = validate(loader_test, net1, criterion, 16)
    print("bit flips:", n_b, "ASR:", ASR)
    x_axis.append(n_b)
    y_axis.append(float(ASR))
    plt.clf()
    plt.xlabel('bit_flips', fontsize=16)
    plt.ylabel('asr', fontsize=16)
    plt.plot(x_axis, y_axis)
    plt.pause(0.01)
    fig.savefig('./result/asr_p3a.png', dpi=dpi, bbox_inches='tight')

    if len(modified_elements) > 0 and len(modified_elements) % 5 == 0:
        improved = refine_modified_elements(net1, modified_elements, loader_test_small, perturbed, num,
                                           top_k=5, refine_iters=6)
        if improved:
            print("Refinement improved:", improved)

    # termination check
    if ASR >= ASR_t:
        print("Reached target ASR:", ASR)
        break

validate(loader_test, net1, criterion, 16)

