import os
import time
import math
import torch
import inspect
from model import retriever_vl
from config import RetrieverVLConfig_medium
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataloader import GPTDataset, ImageTransformer, RandSampler, DistRandSampler, FixCollector

def get_lr(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def train(train_loader, model, device, optimizer, scaler, ptdtype, is_master, config, distributed, iter_num):
    loss_knt = 0
    loss_num = 0

    model.train()
    st = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, data in enumerate(train_loader):
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        images, input_ids, labels = data
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = input_ids.to(labels, non_blocking=True)
        with autocast(dtype=ptdtype):
            loss = model(input_ids, images, labels)
            loss /= config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        loss_knt += loss.item()
        if (i + 1) % config.gradient_accumulation_steps == 0:
            loss_num += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % 100 == 0:
                if is_master:
                    print(f"step {iter_num}, loss {loss_knt/loss_num:.3f}, lr: {optimizer.param_groups[0]['lr']:.7f}, consume {time.time()-st:.2f}s")
                st = time.time()
                loss_knt = 0
                loss_num = 0
            if iter_num >= config.max_iters:
                break
    return iter_num

def validate(ename, valid_loader, model, device, ptdtype):
    running_loss = 0
    count = 0

    model.eval()
    st = time.time()
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            images, input_ids, labels = data
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = input_ids.to(labels, non_blocking=True)
            with autocast(dtype=ptdtype):
                loss = model(input_ids, images, labels)
                running_loss += loss
                count += 1

    print(f"    {ename} valid loss: {running_loss / count:.3f}, consume: {time.time() - st:.3f}s")
    return running_loss / count

def main(gpu, gpu_num, distributed, evaluate, load_model, save_model, config, arch, dtype, cur_dir, model_path, token_dump_path, token_dump_path_eval1, token_dump_path_eval2, retriever_path, vision_path, flash):
    model_path = os.path.join(cur_dir, model_path)
    retriever_path = os.path.join(cur_dir, retriever_path)
    vision_path = os.path.join(cur_dir, vision_path)
    token_dump_path = list(map(lambda x: os.path.join(cur_dir, x), token_dump_path))
    token_dump_path_eval1 = list(map(lambda x: os.path.join(cur_dir, x), token_dump_path_eval1))
    token_dump_path_eval2 = list(map(lambda x: os.path.join(cur_dir, x), token_dump_path_eval2))
    is_master = distributed == False or gpu == 0

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=gpu_num, rank=gpu)

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    model = arch(device, ptdtype, config, load_model, model_path, retriever_path, vision_path, flash=flash)
    model.cuda(gpu)
    # model = torch.compile(model)

    eval1_dataset = GPTDataset(token_dump_path_eval1, config.text_sequence_length, transform=ImageTransformer(config.img_size))
    eval1_sampler = RandSampler(eval1_dataset, batch_size=config.batch_size, drop_last=True, shuffle=False)
    eval1_loader = DataLoader(eval1_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=eval1_sampler)

    eval2_dataset = GPTDataset(token_dump_path_eval2, config.text_sequence_length, transform=ImageTransformer(config.img_size))
    eval2_sampler = RandSampler(eval2_dataset, batch_size=config.batch_size, drop_last=True, shuffle=False)
    eval2_loader = DataLoader(eval2_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=eval2_sampler)

    if evaluate:
        validate('coco', eval1_loader, model, device, ptdtype)
        validate('llava', eval2_loader, model, device, ptdtype)
        return

    train_dataset = GPTDataset(token_dump_path, config.text_sequence_length, transform=ImageTransformer(config.img_size))
    if distributed:
        train_sampler = DistRandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
    else:
        train_sampler = RandSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollector, batch_sampler=train_sampler)

    if is_master:
        print(config)
        for k, v in list(filter(lambda x: x[0][:2] != '__', inspect.getmembers(config))):
            print(f"\t{k}: {v}")
        print()
        # print(model)
        print()
        total_params, text_params, image_params = model.get_num_params()
        print(f"total_params: {total_params/1e6:.2f}M, text_params: {text_params/1e6:.2f}M, image_params: {image_params/1e6:.2f}M")
        print(f"total epoch: {config.num_epoch}, sample nums: {len(train_dataset)}, total iter_num: {int(config.num_epoch * len(train_dataset) / config.gpu_num / config.batch_size / config.gradient_accumulation_steps)}")

    optimizer = torch.optim.AdamW(model.image_encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), fused=True)
    scaler = GradScaler(enabled=(dtype == 'float16'))

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    iter_num = 0
    for epoch in range(config.num_epoch):
        if iter_num >= config.max_iters:
            break

        begin_time = time.time()
        iter_num = train(train_loader, model, device, optimizer, scaler, ptdtype, is_master, config, distributed, iter_num)
        train_sampler.shuffle(epoch)
        
        if is_master:
            validate('coco', eval1_loader, model, device, ptdtype)
            validate('llava', eval2_loader, model, device, ptdtype)
            print(f'epoch: {epoch}, consume: {time.time() - begin_time:.3f}s')
            if save_model:
                torch.save({'state_dict': model.state_dict()}, model_path)

if __name__ == "__main__":
    # resnet去掉position embedding，loss持平
    # projector从残差改成mlp, loss上升
    config = RetrieverVLConfig_medium()
    gpu_num = config.gpu_num
    load_model = False
    save_model = True
    flash = True
    distributed = True
    evaluate = False
    arch = retriever_vl
    dtype = "bfloat16"
    cur_dir = "/home/work/disk/vision/retriever-vl"
    model_path = "checkpoint/retriever_vl_medium.pth.tar"
    retriever_path = 'pretrain/instruct_retriever_tv2_110M_8B_loss1.28_pure.pth.tar'
    vision_path = "pretrain/retriever_clip_medium_resnet50_loss0.208_0.180.pth.tar"
    token_dump_path = [
        "checkpoint/tokens_coco.pkl", # 591753
        "checkpoint/tokens_LLAVA.pkl" # 553000
    ]
    token_dump_path_eval1 = [
        "checkpoint/tokens_coco_eval.pkl" # 5000
    ]
    token_dump_path_eval2 = [
        "checkpoint/tokens_LLAVA_eval.pkl" # 5000
    ]
    if evaluate:
        distributed = False

    if distributed:
        mp.spawn(main, nprocs=gpu_num, args=(gpu_num, distributed, evaluate, load_model, save_model, config, arch, dtype, cur_dir, model_path, token_dump_path, token_dump_path_eval1, token_dump_path_eval2, retriever_path, vision_path, flash))
    else:
        main(0, gpu_num, distributed, evaluate, load_model, save_model, config, arch, dtype, cur_dir, model_path, token_dump_path, token_dump_path_eval1, token_dump_path_eval2, retriever_path, vision_path, flash)