import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_kcr(weight: torch.Tensor, key: int):
    """
    Apply channel permutation + scaling based on a key.
    Args:
        weight: [out_channels, in_channels, k, k] or [out_features, in_features]
        key: random seed for deterministic transformations
    """
    torch.manual_seed(key)
    shape = weight.shape
    out_c, in_c = shape[0], shape[1]

    perm_out = torch.randperm(out_c)
    perm_in = torch.randperm(in_c)

    scale_out = torch.rand(out_c) * 0.5 + 0.75
    scale_in = torch.rand(in_c) * 0.5 + 0.75

    W_new = weight[perm_out][:, perm_in].clone()

    for i in range(out_c):
        W_new[i] *= scale_out[i]
    for j in range(in_c):
        W_new[:, j] /= scale_in[j]

    return W_new



def quantize_ldpc(weight: torch.Tensor, bits: int = 8):
    """
    Uniform affine quantization with LDPC-style redundancy simulation.
    """
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    q_w = torch.clamp((weight / scale).round(), qmin, qmax)
    return q_w * scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32_Tiny(nn.Module):
    def __init__(self, num_classes=200, defense=True, key=1234, bits=8):
        super(ResNet32_Tiny, self).__init__()
        self.in_channels = 16
        self.defense = defense
        self.key = key
        self.bits = bits

        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # 3 stages, 5 blocks each
        self.layer1 = self._make_layer(16, 5, stride=1)  # 64 -> 64
        self.layer2 = self._make_layer(32, 5, stride=2)  # 64 -> 32
        self.layer3 = self._make_layer(64, 5, stride=2)  # 32 -> 16

        # AvgPool -> final 8x8 feature map
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

        # Apply KCR + Quantization if defense enabled
        if self.defense:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if "weight" in name and param.dim() >= 2:
                        w_new = apply_kcr(param, self.key)
                        param.copy_(quantize_ldpc(w_new, bits=self.bits))

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, stride))
            self.in_channels = channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)       # -> [B, 64, 1, 1]
        out = torch.flatten(out, 1)   # -> [B, 64]
        out = self.fc(out)            # -> [B, num_classes]
        return out

    def forward_with_params(self, x, noisy_params):
        """
        Forward using externally perturbed parameters (for ARI).
        """
        backup = {}
        for name, p in self.named_parameters():
            backup[name] = p.data.clone()
            p.data = noisy_params[name].data
        out = self.forward(x)
        for name, p in self.named_parameters():
            p.data = backup[name]
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class ARIWrapper(nn.Module):
    def __init__(self, model, sigma=1e-4, M_small=5, M_large=20, tau=0.1):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.M_small = M_small
        self.M_large = M_large
        self.tau = tau

    def stochastic_forward(self, x, M):
        logits_all = []
        for _ in range(M):
            noisy_params = {}
            for name, p in self.model.named_parameters():
                noisy_params[name] = p + torch.randn_like(p) * self.sigma
            logits = self.model.forward_with_params(x, noisy_params)
            logits_all.append(logits)
        return torch.stack(logits_all, dim=0)

    def forward(self, x):
        logits_small = self.stochastic_forward(x, self.M_small)
        avg_logits = logits_small.mean(0)
        probs = F.softmax(avg_logits, dim=1)
        p_sorted, _ = probs.sort(dim=1, descending=True)
        margin = (p_sorted[:, 0] - p_sorted[:, 1]).mean()

        if margin < self.tau:  # escalate to high redundancy
            logits_large = self.stochastic_forward(x, self.M_large)
            logits_final = logits_large.mean(0)
        else:
            logits_final = avg_logits
        return logits_final

