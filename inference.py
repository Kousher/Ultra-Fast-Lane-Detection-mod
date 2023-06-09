import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

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
        transforms.Resize((1600, 320)),
        transforms.Normalize(mean, std)]
    )

    img = np.array(Image.open("/kaggle/input/sample-lane-image/pic1.jpg"))

    # img = cv2.resize(img, (1600, 320), cv2.INTER_LINEAR)
    
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    
    imgs = img_normalized.cuda()
    with torch.no_grad():
        pred = net(imgs)
        loc_row = pred['loc_row']
        loc_col = pred['loc_col']
        exist_row = pred['exist_row']
        exist_col = pred['exist_col']
        loc_row_a = loc_row.argmax(1).cpu()
        loc_col_a = loc_col.argmax(1).cpu()
        exist_row_a = exist_row.argmax(1).cpu()
        exist_col_a = exist_col.argmax(1).cpu()
        print("loc_row: ", loc_row_a.shape)
        print("loc_col: ", loc_col_a.shape)
        print("exist_row: ", exist_row_a.shape)
        print("exist_col: ", exist_col_a.shape)
        print(exist_col_a)
