import modal
import os
import glob
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


#modal app setup
app = modal.App("NAME MODAL APP")
volume = modal.Volume.from_name('NAME MODAL VOLUMEN')


#image with dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers>=4.37.0",
    "accelerate>=0.25.0",
    "peft>=0.8.0",
    "bitsandbytes>=0.41.0",
    "scikit-learn",
    "python-dotenv",
    "pillow",
    "torch>=2.0.0",
)

#local data to the image
image = image.add_local_dir(
    "PATH TO IMAGES", 
    "/root/lora_package_data/images"
)
image = image.add_local_dir(
    "PATH TO CAPTIONS", 
    "/root/lora_package_data/captions"
)

#app parameters
@app.function(
    image=image,
    gpu="a10g", #GPU used
    timeout=86400,  #time of connection for traiing
    volumes={"/root/volume": volume}
)



def finetune_paligemma():
    import torch
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig
    
    hf_token = 'HF TOKEN HERE' 
    

    #training paths
    image_path = "PATH TO IMAGES"
    caption_path = "PATH TO CAPTIONS"
    

    #all files
    image_files = sorted(glob.glob(os.path.join(image_path, "*")))
    caption_files = sorted(glob.glob(os.path.join(caption_path, "*")))
    
    print(f"Found {len(image_files)} images and {len(caption_files)} captions")
    

    #same number of images and captions
    assert len(image_files) == len(caption_files), "Number of images and captions must match"
    

    #training and test sets 
    indices = list(range(len(image_files)))
    train_indices, test_indices = train_test_split(indices, test_size=0.20, random_state=42)
    
    train_imgs = [image_files[i] for i in train_indices]
    test_imgs = [image_files[i] for i in test_indices]
    train_caps = [caption_files[i] for i in train_indices]
    test_caps = [caption_files[i] for i in test_indices]
    

    #custom dataset
    class CustomDataset(Dataset):
        def __init__(self, image_paths, caption_paths):
            self.image_paths = image_paths
            self.caption_paths = caption_paths
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
        
            try:
                image = Image.open(self.image_paths[idx]).convert('RGB') 
            except Exception as e:
                print(f"Error loading image {self.image_paths[idx]}: {e}")
                
                #placeholder black image if loading fails
                image = Image.new('RGB', (224, 224), color='black')
            
            #read caption
            try:
                with open(self.caption_paths[idx], 'r') as f:
                    caption = f.read().strip()
            except Exception as e:
                print(f"Error loading caption {self.caption_paths[idx]}: {e}")
                caption = "No caption available"
            
            return {
                "image": image,
                "question": caption,
                "multiple_choice_answer": caption
            }
    

    #import PG model
    model_id = "google/paligemma2-3b-pt-224"
    

    #quantization for LoRA training to 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    

    #configure LoRA
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    

    #load model
    print("Loading PaliGemma model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        token=hf_token,
        quantization_config=bnb_config
    )
    

    #apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    DTYPE = model.dtype
    

    #processor
    processor = PaliGemmaProcessor.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True
    )
    

    #org dataset
    def collate_fn(examples):
        texts = ["<image>answer en " + example["question"] for example in examples]
        labels = [example['multiple_choice_answer'] for example in examples]
        images = [example["image"] for example in examples]
        

        #debugging to check image format
        for i, img in enumerate(images):
            if not hasattr(img, 'mode') or img.mode != 'RGB':
                print(f"Warning: Image {i} is not in RGB mode. Converting...")
                images[i] = img.convert('RGB') if hasattr(img, 'convert') else Image.new('RGB', (224, 224))
        
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        )
        tokens = tokens.to(DTYPE)
        return tokens
    

    #create datasets
    train_ds = CustomDataset(train_imgs, train_caps)
    eval_ds = CustomDataset(test_imgs, test_caps)
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Evaluation dataset size: {len(eval_ds)}")
    


    #training args
    args = TrainingArguments(
        num_train_epochs=3,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_torch",
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=1,
        output_dir="/root/volume/paligemma_finetuned",
        bf16=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    

    #initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        args=args
    )
    

    #train model
    print("starting training...")
    trainer.train()
    

    #save model to volumne
    print("saving model...")
    output_dir = "/root/volume/paligemma_finetuned"
    model.save_pretrained(output_dir)
    
    return {"status": "success", "model_path": output_dir}

#run entire training
@app.local_entrypoint()
def main():
    print("starting fine-tuning...")
    result = finetune_paligemma.remote()
    print(f"fine-tuning completed: {result}")