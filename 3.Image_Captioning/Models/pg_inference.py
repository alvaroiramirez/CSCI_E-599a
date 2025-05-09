import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
import json
from safetensors.torch import load_file

#set paths
image_path = "PATH TO INFERENCE IMAGE"
config_path = "PATH TO LOCAL FINE TUNED MODEL CONFIG"
weights_path = "PATH TO WEIGHTS LOCAL FINE TUNED MODEL"
hf_token = 'HF SECRET KEY'

#load base model
model_id = "google/paligemma2-3b-pt-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto",
    token=hf_token
)

#load config
with open(config_path, 'r') as f:
    config_dict = json.load(f)

#create LoRA config
lora_config = LoraConfig(
    r=config_dict.get('r', 8),
    lora_alpha=config_dict.get('lora_alpha', 32),
    target_modules=config_dict.get('target_modules', ["q_proj", "v_proj"]),
    task_type="CAUSAL_LM",
)

#apply LoRA config to model
model = get_peft_model(model, lora_config)

#load weights
state_dict = load_file(weights_path)
model.load_state_dict(state_dict, strict=False)

#load processor
processor = PaliGemmaProcessor.from_pretrained(
    model_id,
    token=hf_token,
    trust_remote_code=True
)

#process image
image = Image.open(image_path).convert('RGB')
inputs = processor(
    text="<image>answer en Describe in this disatser image detail and provide a caption.",
    images=image,
    return_tensors="pt"
).to(model.device)

#generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

#decode and print caption result
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)