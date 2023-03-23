import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    cfg.distributed = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = True)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)
        
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    transform_norm = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224,224)),
        transforms.Normalize(mean, std)]
    )

    img = np.array(Image.open("/kaggle/input/sample-lane-image/pic1.jpg"))
    
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    
    imgs = normalized.cuda()
    with torch.no_grad():
        pred = net(imgs)
