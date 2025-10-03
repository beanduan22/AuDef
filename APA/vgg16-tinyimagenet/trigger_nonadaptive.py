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

    inc_coef = 2 * (domain == 0).float() if increasing else -2 * (domain == 0).float()
    inc_coef = inc_coef.view(-1, nb_features)

    target_tmp = target_grad.clone() - inc_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
    others_tmp = others_grad.clone() + inc_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diag = torch.from_numpy(tmp).byte().cuda()

    mask1, mask2 = (alpha > 0.0, beta < 0.0) if increasing else (alpha < 0.0, beta > 0.0)
    mask = (mask1 & mask2 & zero_diag.view_as(mask1))
    sal_map = alpha * torch.abs(beta) * mask.float()

    _, max_idx = torch.max(sal_map.view(-1, nb_features * nb_features), dim=1)
    p, q = max_idx // nb_features, max_idx % nb_features
    return p, q

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

# -----------------------
# Data
# -----------------------
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
testset = torchvision.datasets.tinyimagenet(root='../../tinyimagenet/vgg16/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# -----------------------
# Model
# -----------------------
net = models.__dict__
net1 = models.__dict__
pretrain_dict = torch.load('../../tinyimagenet/vgg16/save_finetune/model_best.pth.tar')['state_dict']
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict); net.eval().cuda()
net1.load_state_dict(model_dict); net1.eval().cuda()

# -----------------------
# APA Parameters
# -----------------------
theta, gamma = 0.1, 0.5
ys_target = 2
increasing = True
start, end = 21, 31
criterion1, criterion2 = nn.MSELoss(), nn.CrossEntropyLoss()

# -----------------------
# Stage 1: Localization
# -----------------------
print("Stage 1: Saliency localization...")
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    break

model = net.classifier
var_target = Variable(torch.LongTensor([ys_target])).cuda()
I_s = []
img = data[0].unsqueeze(0)
output = model(img)
num_features = int(np.prod(output.shape[1:]))
search_domain = torch.ones(num_features).cuda()
jacobian = compute_jacobian(model, img)
p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
I_s.extend([p1.item(), p2.item()])
I_t = np.array(list(set(I_s)))
np.save('./result/SNI.npy', I_t)
I_t = torch.Tensor(I_t).long().cuda()
print("Localized sensitive indices:", I_t.shape)

# -----------------------
# Stage 2: Adaptive Progressive Perturbation
# -----------------------
print("Stage 2: Adaptive progressive perturbation...")
y = net(data)[15]
y[:, I_t] = 10
var_target = target.clone(); var_target[:] = ys_target
perturbed = torch.zeros_like(data[0, 0:3, start:end, start:end])

step_size = 0.01
alpha, beta = 1.5, 0.5

for step in range(15):
    with torch.no_grad():
        data[:, :, start:end, start:end] = perturbed
    data.requires_grad = True
    output = net(data)[15]
    output1 = net1(data)[15]
    loss_mse = criterion1(output, y.detach())
    loss_ce = criterion2(output1, var_target)
    loss_trig = loss_mse + loss_ce
    zero_gradients(data)
    loss_trig.backward()
    grad = data.grad[:, :, start:end, start:end].mean(0, keepdim=True)
    data.requires_grad = False

    # 尝试更新
    perturbed_new = perturbed - step_size * grad
    with torch.no_grad():
        data[:, :, start:end, start:end] = perturbed_new
        new_out = net(data)[15]
        new_loss = criterion1(new_out, y.detach()) + criterion2(new_out, var_target)

    if new_loss < loss_trig.item():
        perturbed = perturbed_new
        step_size *= alpha   # 改进 → 放大步长
    else:
        step_size *= beta    # 无改进 → 缩小步长

    print(f"Step {step}, loss={loss_trig.item():.4f}, new_loss={new_loss.item():.4f}, step_size={step_size:.5f}")

# -----------------------
# Stage 3: Adaptive Refinement
# -----------------------
print("Stage 3: Adaptive refinement...")
refine_iters = 5
for t in range(refine_iters):
    improved = False
    for d in (1, -1):
        cand = perturbed + d * step_size * 0.5
        with torch.no_grad():
            data[:, :, start:end, start:end] = cand
            out = net(data)[15]
            loss_val = criterion1(out, y.detach()) + criterion2(out, var_target)
        if loss_val.item() < new_loss.item():
            perturbed = cand
            new_loss = loss_val
            improved = True
    if not improved:
        break

perturbed = torch.clamp(perturbed, -0.1, 0.1)
torch.save(perturbed, './result/perturbed_apa.pth')
print("Saved perturbation patch to ./result/perturbed_apa.pth")

# -----------------------
# Final ASR Evaluation
# -----------------------
def validate_for_attack(val_loader, model, criterion, xh):
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] = ys_target
            input[:, 0:3, start:end, start:end] = xh
            input, target = input.cuda(), target.cuda()
            output = model(input)[15]
            prec1 = (output.argmax(1) == target).float().mean() * 100
            top1.update(prec1, input.size(0))
    return top1.avg

asr = validate_for_attack(loader_test, net1, criterion2, perturbed)
print(f"Final ASR (APA): {asr:.2f}%")

