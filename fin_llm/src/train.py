import io

import modal


from common import download_models, MODEL_PATH
from utils import calc_metrics



llm_train_image=(modal.Image.debian_slim().pip_install(
        "datasets",
        "matplotlib",
        "scikit-learn",
        "torch",
        "accelerate==0.21.0",
        "peft==0.4.0",
        "bitsandbytes==0.40.2",
        "transformers==4.31.0",
        "trl==0.4.7"        

    )
    .run_function(download_models)
    .pip_install("wandb==0.15.0")
    .pip_install("rouge-score")
    .pip_install("scikit-learn")
    ) 

stub = modal.Stub("train_llm", image=llm_train_image)
vol =  modal.NetworkFileSystem.persisted("fin_llama")
model_store_path = "/vol/models"

def tokenize(max_length, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= max_length
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


@stub.function(mounts=[modal.Mount.from_local_dir("/home/suhaspillai/Suhas/git/llms/ask-fsdl/fin_llm/src/",
 remote_path="/root/")], gpu="A100", timeout=30000, network_file_systems={model_store_path: vol})
def load_dataset():
  from transformers import (
      AutoModelForCausalLM,
      AutoTokenizer,
      BitsAndBytesConfig,
      HfArgumentParser,
      TrainingArguments,
      pipeline,
      logging,
      LlamaForCausalLM,
      DataCollatorForSeq2Seq,
      Trainer,
      TrainerCallback,
      TrainerState,
      TrainerControl

  )
  from peft import LoraConfig, PeftModel, TaskType, get_peft_model  
  import datasets
  from datasets import load_dataset
  from datetime import datetime
  import sys
  import torch
  import os
  from functools import partial
  import wandb
  from rouge_score import rouge_scorer
  from tqdm import tqdm
  import re



  os.environ['WANDB_API_KEY'] = 'c61582e12b0230b9c79eee346b7621b39c2e8473'
  os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'
  os.environ["TOKENIZERS_PARALLELISM"] = "false"




  class GenerationEvalCallback(TrainerCallback):
	    
    def __init__(self, eval_dataset, ignore_until_epoch=0):
      self.eval_dataset = eval_dataset
      self.ignore_until_epoch = ignore_until_epoch
      

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
       
      if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
        return
            
      if state.is_local_process_zero:
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        generated_texts, reference_texts = [], []

        for feature in tqdm(self.eval_dataset):
          prompt = feature['prompt']
          gt = feature['answer']
          inputs = tokenizer(
                             prompt, return_tensors='pt',
                             padding=False, max_length=4096
                            )
          inputs = {key: value.to(model.device) for key, value in inputs.items()}
                
          res = model.generate(
                               **inputs, 
                               use_cache=True
                              )
          output = tokenizer.decode(res[0], skip_special_tokens=True)
          answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

          generated_texts.append(answer)
          reference_texts.append(gt)

        metrics = calc_metrics(reference_texts, generated_texts)
        print("Metrics ---- {}".format(metrics))
        # Ensure wandb is initialized
        if wandb.run is None:
          wandb.init(entity='suhas-callin')
                
        wandb.log(metrics, step=state.global_step)
        torch.cuda.empty_cache()   
  
  
  model_name = "NousResearch/Llama-2-7b-chat-hf"
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  max_length = 512
  dataset=datasets.load_from_disk('/root/fingpt-forecaster-dow30v3-20230831-20231208-llama')
  eval_dataset = dataset['train'].shuffle(seed=1234).select(range(int(0.1*len(dataset['train']))))
  #dataset=datasets.load_from_disk('/home/suhaspillai/Suhas/git/llms/ask-fsdl/fin_llm/src/fingpt-forecaster-dow30v3-20230831-20231208-llama') 
  # This will create dataset in the form of input_ids and label_ids, which is what Trainer() needs 
  tokenized_dataset = dataset.map(partial(tokenize, max_length, tokenizer))
  tokenized_dataset = tokenized_dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )
  # original_dataset = datasets.DatasetDict({'train': dataset['train'], 'test': dataset['test']})
  # eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(3))
  # dataset = original_dataset.map(template_dataset)
  # print(dataset['test']['text'][0])
  run_name = 'test_llm_training'
  dataset = 'fingpt-forecaster-dow30v3-20230831-20231208-llama'
  max_length = 4096 
  batch_size = 1 
  gradient_accumulation_steps = 16 
  learning_rate = 5e-5 
  num_epochs = 1 
  log_interval = 10 
  warmup_ratio = 0.03 
  scheduler = 'constant'
  evaluation_strategy ='steps'
  eval_steps=0.1
  weight_decay=0.01 
  current_time = datetime.now()
  formatted_time = current_time.strftime('%Y%m%d%H%M')
  num_workers=1
  SAVE_MODEL_PATH=model_store_path+'/'+run_name+'_'+formatted_time
  training_args = TrainingArguments(
        output_dir=SAVE_MODEL_PATH, # 保存位置
        logging_steps=log_interval,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        remove_unused_columns=False,
        run_name=run_name,
        report_to='wandb'
    )
  
  #MODEL_PATH = "/model"
  #model = LlamaForCausalLM.from_pretrained(model_name)
  model = LlamaForCausalLM.from_pretrained(MODEL_PATH) 
  #model.save_pretrained(MODEL_PATH)  
  #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.is_parallelizable = True
  model.model_parallel = True
  model.model.config.use_cache = False

  # lora_config = LoraConfig(
  #   r=16,
  #   lora_alpha=32,
  #   lora_dropout=0.05,
  #   bias="none",
  #   task_type="CAUSAL_LM",
  # )
  peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      bias='none',
  )

  model = get_peft_model(model, peft_config)


  trainer = Trainer(
      model=model, 
      args=training_args, 
      train_dataset=tokenized_dataset['train'],
      eval_dataset=tokenized_dataset['test'], 
      tokenizer=tokenizer,
      data_collator=DataCollatorForSeq2Seq(
          tokenizer, padding=True,
          return_tensors="pt"
      ),
      callbacks=[
        GenerationEvalCallback(
          eval_dataset=eval_dataset,
          ignore_until_epoch=round(0.3 * num_epochs)
          )
        ]
  )

  if torch.__version__ >= "2" and sys.platform != "win32":
      model = torch.compile(model)


  torch.cuda.empty_cache()
  print("----- Start training the model -----")

  trainer.train()

  print("----- Finished training the model -----")

  # save model

  model.save_pretrained(SAVE_MODEL_PATH)
  

  
def format_dolly(sample):
    instruction = f"<s>{sample['prompt']}"
    response = f" [{sample['answer']}"
    # join all the parts together
    prompt = "".join([i for i in [instruction, response ] if i is not None])
    return prompt


def template_dataset(sample):
    #print(sample)
    sample["text"] = f"{format_dolly(sample)}</s>"
    return sample



	
	#original_dataset = datasets.DatasetDict({'train': dataset['train'], 'test': dataset['test']})
	#print(original_dataset)


@stub.local_entrypoint()
def main():
  dataset_path=''
  # load_dataset.local(dataset_path)
  load_dataset.remote()
  #load_dataset.local()