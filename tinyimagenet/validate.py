import torch
import torch.nn as nn
import argparse

from dataset_cifar10 import get_cifar10_loaders
from dataset_cifar100 import get_cifar100_loaders
from dataset_tinyimagenet import get_tinyimagenet_loaders
from models.resnet32_tiny import ResNet32_Tiny, ARIWrapper

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "cifar10":
        _, val_loader = get_cifar10_loaders()
        num_classes = 10
    elif args.dataset == "cifar100":
        _, val_loader = get_cifar100_loaders()
        num_classes = 100
    else:
        _, val_loader = get_tinyimagenet_loaders()
        num_classes = 200

    base_model = ResNet32_Tiny(num_classes=num_classes, defense=True).to(device)
    model = ARIWrapper(base_model).to(device)

    model.load_state_dict(torch.load(args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Validation: loss={val_loss:.4f}, acc={val_acc*100:.2f}%")

if __name__ == "__main__":
    main()

