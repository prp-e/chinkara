# Chinkara 7B

_Chinkara_ is a Large Language Model trained on [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) dataset based on Meta's brand new LLaMa-2 with 7 billion parameters using QLoRa Technique, optimized for small consumer size GPUs. 

## Inference Notebooks 

| Model | Notebook | Description |
|:-----:|:--------:|:------------:|
|[chinkara-7b](https://huggingdace.com/MaralGPT/chinkara-7b) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prp-e/chinkara/blob/main/inference-7b.ipynb) | This is the smallest model of the family, trained on LLaMa-2 7B |

## Inference Guide

_NOTE: This part is for the time you want to load and infere the model on your local machine. You still need 8GB of VRAM on your GPU. The recommended GPU is at least a 2080!_

### Installing libraries

```
pip install  -U bitsandbytes
pip install  -U git+https://github.com/huggingface/transformers.git
pip install  -U git+https://github.com/huggingface/peft.git
pip install  -U git+https://github.com/huggingface/accelerate.git
pip install  -U datasets
pip install  -U einops
```

### Loading the model 

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyPixel/Llama-2-7B-bf16-sharded" 
adapters_name = 'MaralGPT/chinkara-7b' 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Setting the model up

```python
from peft import LoraConfig, get_peft_model

model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
``` 

### Prompt and inference

```python
prompt = "What is the answer to life, universe and everything?" 

prompt = f"###Human: {prompt} ###Assistant:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=50, temperature=0.5, repetition_penalty=1.0)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```
## Known Issues 

### The dataset

## What's next?