import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import torch
import inspect
import gradio as gr
from functools import partial
from model import retriever_vl
from config import RetrieverVLConfig_medium, RetrieverVLConfig_medium_finetune
from transformers import PreTrainedTokenizerFast

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def del_text(history):
    return history[:-1]

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def response_func(history, model, tokenizer):
    message, history = history[-1][0], history[:-1]
    response = model.chat(tokenizer, message, history[1:])
    return history + [(message, response)]

def image_func(history, model):
    image_path = history[-1][0][0]
    model.cache_image(image_path)

def main(gpu, arch, dtype, model_root, model_path, tokenizer_path):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    model_path = os.path.join(model_root, model_path)
    model = arch(device, ptdtype, config, pretrained=True, model_path=model_path, flash=True)
    model.cuda(gpu)
    model = model.eval()

    with gr.Blocks() as demo:
        gr.components.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Retriever-VL-0.1B</h1>")
        chatbot = gr.Chatbot(height=650, show_label=False, bubble_full_width=False)

        with gr.Row():
            txt = gr.Textbox(lines=3, scale=10, show_label=False, container=False)
            img = gr.UploadButton("📁", file_types=["image"])
            sub = gr.Button(value='submit')
            delete = gr.Button(value='delete')
            gr.ClearButton([chatbot, txt])

        txt.submit(add_text, [chatbot, txt], [chatbot, txt])\
                .then(partial(response_func, model=model, tokenizer=tokenizer), chatbot, chatbot)
        
        sub.click(add_text, [chatbot, txt], [chatbot, txt])\
                .then(partial(response_func, model=model, tokenizer=tokenizer), chatbot, chatbot)

        delete.click(del_text, [chatbot], [chatbot])
        
        img.upload(add_file, [chatbot, img], [chatbot])\
                .then(partial(image_func, model=model), chatbot)

    demo.launch(server_port=7870)

if __name__ == "__main__":
    arch = retriever_vl
    config = RetrieverVLConfig_medium()
    finetune_config = RetrieverVLConfig_medium_finetune()
    revised_params = list(filter(lambda x: x[0][0] != '_', inspect.getmembers(finetune_config)))
    for rp, value in revised_params:
        setattr(config, rp, value)
    dtype = "bfloat16"
    model_root = "/home/work/disk/vision/retriever-vl"
    # model_path = "checkpoint/retriever_vl_medium_loss0.627.pth.tar"
    model_path = "checkpoint/instruct_retriever_vl_medium_loss0.246.pth.tar"
    tokenizer_path = "pretrain/tokenizer_v2_600G.json"
    main(0, arch, dtype, model_root, model_path, tokenizer_path)