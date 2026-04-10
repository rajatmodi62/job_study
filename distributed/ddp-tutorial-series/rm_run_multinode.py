import os
import socket
import torch
import torch.distributed as dist
import subprocess

def setup_distributed():
    print("enter..")
    if "SLURM_PROCID" not in os.environ:
        return 0, 1, 0
    print("rmodi")
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    
    # Use scontrol to get the master node name
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    master_node = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist], 
        text=True
    ).splitlines()[0]
    
    print("found master_node")
    # Convert node name to IP address (more reliable for handshakes)
    master_addr = socket.gethostbyname(master_node)
    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "12354"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    # Use the IB interface for the control socket
    # Note: check 'ifconfig' or 'ip addr' to see if it's ib0, ibv0, etc.
    os.environ["NCCL_IB_DISABLE"] = "0"
    
    # Initialize NCCL
    dist.init_process_group(backend="nccl", init_method="env://")
    
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} (Local {local_rank}) connected to Master {master_addr}")
    return rank, world_size, local_rank
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    try:
        rank, world_size, local_rank = setup_distributed()
        # Your logic here...
        print(f"Rank {rank} (Local {local_rank}) is ready.")
    finally:
        cleanup()

if __name__ == "__main__":
    main()