import os
import socket
import torch
import torch.distributed as dist
import subprocess
import os
import signal
import psutil
import os
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess

from datetime import timedelta

def kill_process_on_port(port):
    """Finds and kills any process currently listening on the specified port."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Found process {proc.pid} ({proc.name()}) on port {port}. Killing...")
                    proc.send_signal(signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def setup_distributed():
    # 1. Slurm Env
    rank = int(os.environ["SLURM_PROCID"])

    if rank == 0:
        # Force Rank 0 to identify by its primary network IP
        master_addr = socket.gethostbyname(socket.gethostname())
    else:
        # Workers resolve the master node name to an IP
        master_addr = socket.gethostbyname(master_node)

    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # 2. Master Address Resolution
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    # Get the hostname of the first node
    master_node = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "12339"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # 3. NCCL Fixes (Correcting the variable names)
    os.environ["NCCL_IB_DISABLE"] = "0"           # Make sure IB is NOT disabled
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"      # Use your ib0 interface
    os.environ["NCCL_DEBUG"] = "INFO"             # This will print the handshake details

    print(f"Rank {rank} on {socket.gethostname()} connecting to Master {master_addr}:12353")

    # 4. Init with Timeout
    # If it doesn't connect in 2 minutes, it will crash instead of hanging
    print("init dist..")
    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        timeout=timedelta(minutes=2)
    )
    print("init complete..")
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} successfully initialized!")

    return rank, world_size, local_rank
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# --- Toy Model & Dataset ---

class ToyDataset(Dataset):
    def __len__(self): return 1000
    def __getitem__(self, idx):
        return torch.randn(64), torch.randn(64)

class TenLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(10):
            layers.append(nn.Linear(64, 64))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Main Training Loop ---

def main():
    # try:
    #     rank, world_size, local_rank = setup_distributed()
    #     # Your logic here...
    #     print(f"Rank {rank} (Local {local_rank}) is ready.")
    # finally:
    #     print("entering cleanup")
    #     cleanup()
    #     print("cleanup done")

    # 1. Prepare Model
    rank, world_size, local_rank = setup_distributed()
    model = TenLayerNet().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 2. Prepare Data with Distributed Sampler
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"Rank {rank} starting training...")

    model.train()
    for epoch in range(1):
        # Mandatory for DistributedSampler to shuffle correctly every epoch
        sampler.set_epoch(epoch)
        
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(local_rank), target.to(local_rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Print every 10 iterations on Rank 0 to keep logs clean
            if i % 10 == 0 and rank == 0:
                print(f"Iteration {i} | Loss: {loss.item():.4f}")

    print(f"Rank {rank} training complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()