import torch
import torch.distributed as dist

def ddp_setup_process_group(cfg, distributed):
    if distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cfg['is_distributed'] = True
    else:
        rank = 0
        world_size = 1
    
    device = torch.device("cuda", rank)    
    # Update the cfg dictionary with dynamic values
    cfg.update({
        'device': device,
        'rank': rank,
        'world_size': world_size
    })

    return rank, cfg
