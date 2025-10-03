import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_kcr(weight: torch.Tensor, key: int):
    torch.manual_seed(key)
    shape = weight.shape
    if weight.dim() == 4:
        out_c, in_c = shape[0], shape[1]
    else:
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
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = weight.abs().max() / qmax
    q_w = torch.clamp((weight / scale).round(), qmin, qmax)
    return q_w * scale


class VGG16(nn.Module):
    def __init__(self, num_classes=10, defense=True, key=2025, bits=8):
        super(VGG16, self).__init__()
        self.defense = defense
        self.key = key
        self.bits = bits

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

        # Apply KCR + Quantization
        if self.defense:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if "weight" in name and param.dim() >= 2:
                        w_new = apply_kcr(param, self.key)
                        param.copy_(quantize_ldpc(w_new, bits=self.bits))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_with_params(self, x, noisy_params):
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

        if margin < self.tau:  # escalate
            logits_large = self.stochastic_forward(x, self.M_large)
            logits_final = logits_large.mean(0)
        else:
            logits_final = avg_logits
        return logits_final

