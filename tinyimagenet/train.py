import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# import datasets
from dataset_cifar10 import get_cifar10_loaders
from dataset_cifar100 import get_cifar100_loaders
from dataset_tinyimagenet import get_tinyimagenet_loaders


from models.resnet32_tiny import ResNet32_Tiny, ARIWrapper

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

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
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    if args.dataset == "cifar10":
        train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)
        num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = get_cifar100_loaders(batch_size=args.batch_size)
        num_classes = 100
    else:
        train_loader, val_loader = get_tinyimagenet_loaders(batch_size=args.batch_size)
        num_classes = 200

    # Model
    base_model = ResNet32_Tiny(num_classes=num_classes, defense=True, key=42, bits=8).to(device)
    model = ARIWrapper(base_model).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_acc={tr_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")
        scheduler.step()

        # save checkpoint
        torch.save(model.state_dict(), f"./checkpoints/{args.dataset}_epoch{epoch}.pth")

if __name__ == "__main__":
    main()

