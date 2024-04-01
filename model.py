import math
import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from dataloader import image_loader
from resnet_v15 import ResNetV15, Bottleneck

def make_causal_mask(bsz, tgt_len, device, dtype):
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * 2.68)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate_proj = nn.functional.silu(self.gate_proj(x))
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj

class Attention(nn.Module):
    def __init__(self, config, attention_mask, position_ids, flash=True):
        super(Attention, self).__init__()
        self.flash=flash
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_position_embeddings = config.sequence_length
        self.attention_mask = attention_mask
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.position_ids = position_ids
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states, masks=None):
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        position_ids = self.position_ids[:, :seq_len]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if self.flash:
            if masks is None:
                output = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
            else:
                attn_mask = self.attention_mask[:bsz, :, :seq_len, :seq_len] + masks.view(bsz, 1, seq_len, seq_len)
                output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        else:
            att = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert att.size() == (bsz, self.num_heads, seq_len, seq_len), "Attention weights shape error"
            if masks is None:
                att = att + self.attention_mask[:bsz, :, :seq_len, :seq_len]
            else:
                att = att + self.attention_mask[:bsz, :, :seq_len, :seq_len] + masks.view(bsz, 1, seq_len, seq_len)
            att = nn.functional.softmax(att, dim=-1)
            output = torch.matmul(att, value)

        output = output.transpose(1, 2).contiguous()
        assert output.size() == (bsz, seq_len, self.num_heads, self.head_dim), "Attention output shape error"
        output = output.reshape(bsz, seq_len, self.hidden_size)

        output = self.o_proj(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, config, attention_mask, position_ids, flash=True):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config, attention_mask, position_ids, flash)
        self.ln_2 = RMSNorm(config.hidden_size)
        self.mlp = MLP(config.hidden_size)

    def forward(self, x, masks=None):
        x = x + self.attn(self.ln_1(x), masks)
        x = x + self.mlp(self.ln_2(x))
        return x

class RetrieverVL(nn.Module):
    def __init__(self, device, ptdtype, config, flash=True):
        super(RetrieverVL, self).__init__()
        self.device = device
        self.ptdtype = ptdtype
        self.config = config
        self.vocab_size = config.vocab_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        position_ids = torch.arange(0, config.sequence_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, config.sequence_length)
        attention_mask = make_causal_mask(config.batch_size, config.sequence_length, device, ptdtype)
        self.layers = nn.ModuleList([DecoderLayer(config, attention_mask, position_ids, flash) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.image_encoder = ResNetV15(Bottleneck, config.res_layers, config.res_channels, config)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        image_params = sum(p.numel() for p in self.image_encoder.parameters())
        text_params = total_params - image_params
        return total_params, text_params, image_params

    def forward(self, input_ids, images, labels=None, img_encoded=False):
        tok_emb = self.token_embedding(input_ids)
        if not img_encoded:
            images = self.image_encoder(images)

        x = torch.cat([images, tok_emb], dim=1)
        for decoder_layer in self.layers:
            x = decoder_layer(x)
        x = self.norm(x)
        x = x[..., -input_ids.shape[1]:, :]
        logits = self.lm_head(x)

        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels)

        return loss

    @torch.no_grad()
    def cache_image(self, img_path):
        image = image_loader(img_path, self.config.img_size)
        image = image.to(self.device)
        with autocast(dtype=self.ptdtype):
            image = self.image_encoder(image)
        self.image = image

    @torch.no_grad()
    def chat(self, tokenizer, message, history, temperature=1.0, top_k=1, max_new_tokens=256):
        prompt = ""
        prompt_token = []

        history_combo = list(map(lambda x: f'\nQ:{x[0]}<s>\nA:{x[1]}<s>', history))
        history_tokens = list(map(lambda x: tokenizer(x)['input_ids'], history_combo))
        message = f"\nQ:{message}<s>\nA:"
        message_token = tokenizer(message)['input_ids']

        length = len(prompt_token) + len(message_token)
        max_length = self.config.sequence_length - self.image.shape[1] - max_new_tokens
        i = 0
        if history:
            for i in np.arange(len(history_tokens), -1, -1):
                if i == 0:
                    break
                length += len(history_tokens[i-1])
                if length >= max_length:
                    break

            for j in range(i, len(history_tokens)):
                prompt += history_combo[j]
                prompt_token += history_tokens[j]

        prompt += message
        prompt_token += message_token
        
        output_ids = []
        input = torch.from_numpy(np.array([prompt_token])).to(torch.long)

        for i in range(max_new_tokens):
            input_trunc = input
            input_trunc = input_trunc.to(self.device)
            with autocast(dtype=self.ptdtype):
                logits = self(input_trunc, self.image, img_encoded=True)
                logits = logits[:, -1].cpu()
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, id_next), dim=1)
            output_ids.append(int(id_next[0][0]))

            if output_ids[-1] == 1:
                break
        
        return tokenizer.decode(output_ids).split('<s>')[0]

def retriever_vl(device, ptdtype, config, pretrained, model_path=None, retriever_path=None, vision_path=None, flash=True):
    model = RetrieverVL(device, ptdtype, config, flash)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        replacer = 'module.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()})
    else:
        vision_dict = torch.load(vision_path)['state_dict']
        replacer = "module.image_encoder."
        model.image_encoder.load_state_dict({k.replace(replacer, ''): v for k, v in vision_dict.items() if replacer in k}, strict=False)

        retriever_dict = torch.load(retriever_path)['state_dict']
        model.load_state_dict({k: v for k, v in retriever_dict.items()}, strict=False)
    return model

