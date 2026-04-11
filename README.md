
# random study

time to get sherioush.. (or not, hiyaaa!!!!) 

- ddp 
- fsdp 
- tensor parallelism 
- some basic llm knowledge -> look at huggingface for this hiya. 


learn basics here:

[ring all reduce algorithm](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/)

[fsdp, deepspeed, zero](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)

[princeton dude, new slides](https://github.com/PrincetonUniversity/multi_gpu_training/blob/main/04_model_parallel_with_fsdp/slides.pdf)



ddp

- each gpu launches 1 process 
- stores models, same seed, same optiimizer init, same weights 
- separate data seen by each gpu, and then gradients are different 
- so it undergoes sync step 
    - using bucketed ring sync algorithm 
    - how does it work? look at later.

-----
ddp multi node srun mode 

request infiniband

``
srun --partition=preempt \
     --nodes=2 \
     --ntasks-per-node=1 \
     --gres=gpu:1 \
     --constraint=gmem48 \
     --network=IB \
     --pty bash
``

check infniband up

``
ibstat
Look for State: Active and Physical state: LinkUp. If it says Down or Initializing, the hardware "coupling" is broken
``

``
ibv_devinfo
You want to see state: PORT_ACTIVE (4).
``
check if nodes allocated
``
echo $SLURM_JOB_NODELIST
scontrol show hostnames $SLURM_JOB_NODELIST
``

check node handshake 
``
srun --ntasks=2 --nodes=2 --overlap hostname
``

----

ddp multi-node

basically, everything in crap, just use code in 

``
distributed/princeton/multi_gpu_training/03_pytorch_lightning/multi/job.slurm
``

pytorch lightning can do that well. 


