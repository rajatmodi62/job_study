import os
import socket
import torch
import torch.distributed as dist
import subprocess
import os
import signal
import psutil
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
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # 2. Master Address Resolution
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    # Get the hostname of the first node
    master_node = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]
    # Convert hostname to IP for better reliability
    master_addr = socket.gethostbyname(master_node)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "12353"
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