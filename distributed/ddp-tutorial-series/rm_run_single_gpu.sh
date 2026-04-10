# singlue gpu 
# python single_gpu.py --total_epochs 50 --save_every 5 --batch_size 128

#multiple gpu 
# python multigpu.py  5000000  5 --batch_size 128


# torch run with interrupt training
torchrun --standalone --nproc_per_node=2  multigpu_torchrun.py  5000000  5 --batch_size 128