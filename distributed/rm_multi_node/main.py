import os
import builtins
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# --- Dummy Dataset ---
class MyDataset(Dataset):
    def __init__(self, mode='train', size=1000):
        self.size = size
        # 100 features, 10 classes
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# --- 10-Layer Linear Model ---
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        layers = []
        in_features = 100
        # 9 hidden layers + 1 output layer = 10 layers
        for _ in range(9):
            layers.append(nn.Linear(in_features, 256))
            layers.append(nn.ReLU())
            in_features = 256
        layers.append(nn.Linear(256, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1024*1024, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--workers', default=2, type=int)
    # DDP configs:
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--local-rank', default=-1, type=int)
    args = parser.parse_args()
    return args

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and args.rank == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss: {loss.item():.4f}")

def validate(val_loader, model, criterion, epoch, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Simple accuracy print
    print(f"Accuracy at Epoch {epoch}: {100 * correct / total}%")

def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    
    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Suppress printing on non-master nodes
    if args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    ### model ###
    model = MyModel()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        # Fallback for single GPU/CPU if you remove the NotImplementedError
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ### data ###
    train_dataset = MyDataset(mode='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = MyDataset(mode='val')
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.workers, pin_memory=True)

    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        
        if args.rank == 0:
            validate(val_loader, model, criterion, epoch, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)