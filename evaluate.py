import numpy as np
import torch
from dataset.semi import SemiDataset, SemiTestOrigDataset
from util.utils import AverageMeter, intersectionAndUnion, labels
from tqdm import tqdm


filtered_labels = [label for label in labels if label.trainId != 255 and label.trainId != -1]

def evaluate(tcfg, mcfg, model, loader, mode):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    # print("📊 Start evaluation...")
    # pbar = tqdm(loader, desc="📊 Start Evaluation...", total=len(loader), position=5)
    pbar = tqdm(
        loader,
        desc="Validation",
        total=len(loader),
        position=0,
        leave=False,
        mininterval=2
        )
    # with torch.no_grad(), torch.amp.autocast('cuda'):
    with torch.no_grad():
        for step, (img, mask, image_path) in enumerate(pbar):
            img = img.cuda(non_blocking=True)
            b, _, h, w = img.shape
            
            if mode == 'sliding_window':
                # grid = tcfg.crop_size
                # final = torch.zeros(b, 19, h, w).cuda()
                # row = 0
                # while row < h:
                #     col = 0
                #     while col < w:
                #         res = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                #         pred = res['out']
                #         final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                #         col += int(grid * 2 / 3)
                #     row += int(grid * 2 / 3)

                # pred = final.argmax(dim=1)
                grid = tcfg.crop_size
                final = torch.zeros(b, 19, h, w).cuda()
                stride = int(grid * 2 / 3)
                # with torch.no_grad(), torch.autocast(device_type='cuda'):
                with torch.no_grad():
                    row = 0
                    while row < h:
                        col = 0
                        while col < w:
                            r1 = min(h, row + grid)
                            c1 = min(w, col + grid)
                            
                            res = model(img[:, :, row:r1, col:c1], mode='val')
                            # float32로 캐스팅 후 누적 (정확도 보존을 위해)
                            pred = res['out'].float().softmax(dim=1) 
                            
                            final[:, :, row:r1, col:c1] += pred
                            
                            col += stride
                        row += stride
                        
                pred = final.argmax(dim=1)
                conf = final.softmax(dim=1).max(dim=1).values
                
            elif mode == 'center_crop':
                h, w = img.shape[-2:]
                start_h, start_w = (h - tcfg.crop_size) // 2, (w - tcfg.crop_size) // 2
                img = img[:, :, start_h:start_h + tcfg.crop_size, start_w:start_w + tcfg.crop_size]
                mask = mask[:, start_h:start_h + tcfg.crop_size, start_w:start_w + tcfg.crop_size]

            res = model(img, mode='val')
            pred = res['out'].argmax(dim=1)
            conf = res['out'].softmax(dim=1).max(dim=1).values
                
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), mcfg.num_classes, 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class_dict = {}
    for num, iou in enumerate(iou_class):
        iou_class_dict[filtered_labels[num].name] = round(iou * 100, 2).item()
        
    mIOU = np.mean(iou_class) * 100.0
    return_dict['iou_class'] = iou_class_dict
    return_dict['mIOU'] = mIOU
    return_dict['pred'] = pred
    return_dict['conf'] = conf
    return_dict['img'] = img
    return_dict['mask'] = mask
    return_dict['image_path'] = image_path
    
    pbar.close()
    # tqdm.write(f' Evaluation: {tcfg.eval_mode}  >>>> meanIOU: {mIOU:.4f} \n')
    
    # items = list(iou_class_dict.items())
    # items_per_line = 4 # 한 줄에 출력할 클래스 개수

    # # 1. 출력할 내용을 하나의 텍스트 변수(summary_text)에 계속 더해서 조립합니다.
    # summary_text = "=" * 65 + "\n"
    # summary_text += "📊 [Class IoU Summary]\n"
    # summary_text += "-" * 65 + "\n"

    # for i in range(0, len(items), items_per_line):
    #     chunk = items[i:i+items_per_line]
    #     row = " | ".join(f"{k:<12}: {v:>5.2f}%" for k, v in chunk)
    #     summary_text += row + "\n"  # 각 줄마다 엔터(\n) 추가

    # summary_text += "=" * 65

    # # 2. 조립이 끝난 거대한 문자열을 딱 한 번만 출력합니다.
    # tqdm.write(summary_text)
    
    items = list(iou_class_dict.items())
    items_per_line = 4 # 한 줄에 출력할 클래스 개수

    tqdm.write("=" * 65)
    tqdm.write(f"Mode: {tcfg.eval_mode} >>>> meanIOU: {mIOU:.4f} \n")
    tqdm.write("📊 [Class IoU Summary]")
    tqdm.write("-" * 65)

    for i in range(0, len(items), items_per_line):
        chunk = items[i:i+items_per_line]
        row = " | ".join(f"{k:<12}: {v:>5.2f}%" for k, v in chunk)
        tqdm.write(row)

    tqdm.write("=" * 65 + "\n")
    
    return return_dict

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from configuration import TrainConfig, ModelConfig, DataConfig
    from model.semseg.deeplabv3plus import DeepLabV3Plus
    from thop import profile
    import logging
    from types import SimpleNamespace
    from util.utils import compute_flops, count_params

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/dev/experiments/models/v1.6.4_LTU/resnet50_42.749.pth")
    args = parser.parse_args()

    tcfg = TrainConfig()
    mcfg = ModelConfig()
    dcfg = DataConfig()

    # 1. 학습 당시와 같은 모델 구조 생성
    model = DeepLabV3Plus(tcfg, mcfg)

    # 2. 학습된 가중치 불러오기
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # FLOPs 측정 과정에서 추가된 항목 제외
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith(("total_ops", "total_params"))
    }
    model.load_state_dict(state_dict, strict=True)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    model.eval()

    # 3. 검증 이미지와 정답 마스크 준비
    valset = SemiTestOrigDataset(
        root=tcfg.data_root,
        mode="val",
        valid_path=dcfg.val_id_path,
        # size=None, # 1024 X 2048 로 테스트
        size=448 # 448X448로 테스트
    )
    
    img, mask, image_path = valset[0]
    print("Test Image shape:", img.shape)
    print("Test Mask shape:", mask.shape)
    
    
    loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # 4. 추론 FLOPs 측정
    logger = logging.getLogger("evaluation_flops")
    logger.setLevel(logging.INFO)
    logger.propagate=False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        
    # valset img,mask size
    h, w = img.shape[-2:]
    profile_cfg = SimpleNamespace(crop_size=(h,w))
    
    with torch.no_grad():
        compute_flops(profile_cfg, model, logger)
        
    # 5. 기존 평가 함수 호출
    tcfg.eval_mode = "original"
    evaluate(tcfg, mcfg, model, loader, mode="original")