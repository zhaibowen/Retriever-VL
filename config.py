from dataclasses import dataclass

@dataclass
class RetrieverVLConfig_medium:
    gpu_num = 3
    batch_size = 32
    gradient_accumulation_steps = 2
    num_epoch = 2
    sequence_length = 1024
    text_sequence_length = 50
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 12
    hidden_size = 768
    num_heads = 12
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 800
    max_iters = 12000
    lr_decay_iters = 11000
    grad_clip = 1.0

    res_layers = [3, 4, 6, 3]
    res_channels = [64, 256, 512, 1024, 2048]
    img_hidden_size = 2048
    img_size = 448
    spacial_dim = 14

@dataclass
class RetrieverVLConfig_medium_finetune:
    batch_size = 16
    gradient_accumulation_steps = 4
    num_epoch = 2
    text_sequence_length = 759
    learning_rate = 1e-4
    min_lr = 1e-5
    warmup_iters = 600
    max_iters = 7000
    lr_decay_iters = 6000