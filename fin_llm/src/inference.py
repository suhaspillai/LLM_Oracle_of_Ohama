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
vol =  modal.NetworkFileSystem.from_name("fin_llama")
model_store_path = "/vol/models"

@stub.function(mounts=[modal.Mount.from_local_dir("/home/suhaspillai/Suhas/git/llms/ask-fsdl/fin_llm/src/",
 remote_path="/root/")], gpu="A100", timeout=30000, network_file_systems={model_store_path: vol})
def run_inference():
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
  import pickle



  os.environ['WANDB_API_KEY'] = 'c61582e12b0230b9c79eee346b7621b39c2e8473'
  os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'
  os.environ["TOKENIZERS_PARALLELISM"] = "false"



  def test_demo(model, tokenizer, prompt):

      inputs = tokenizer(
          prompt, return_tensors='pt',
          padding=False, max_length=4096
      )
      inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
      res = model.generate(
          **inputs, max_length=4096, do_sample=True,
          eos_token_id=tokenizer.eos_token_id,
          use_cache=True
      )
      output = tokenizer.decode(res[0], skip_special_tokens=True)
      return output



  model_name = "NousResearch/Llama-2-7b-chat-hf"

  base_model = AutoModelForCausalLM.from_pretrained(
      'NousResearch/Llama-2-7b-chat-hf',
      trust_remote_code=True,
      device_map="auto",
  )
  base_model.model_parallel=True
  checkpoint_dir = model_store_path+'/'+'test_llm_training_202312250053/checkpoint-4'
  model = PeftModel.from_pretrained(base_model, checkpoint_dir)
  model = model.eval()
  print(model)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  test_dataset=datasets.load_from_disk('/root/fingpt-forecaster-dow30v3-20230831-20231208-llama')['test']


  answers, gts = [], []

  for i in range(len(test_dataset)):
      prompt = test_dataset[i]['prompt']
      output = test_demo(model, tokenizer, prompt)
      answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
      gt = test_dataset[i]['answer']
      print('\n------- Prompt ------\n')
      print(prompt)
      print('\n------- LLaMA2 Finetuned ------\n')
      print(answer)
      print('\n------- GPT4 Groundtruth ------\n')
      print(gt)
      print('\n===============\n')
      answers.append(answer)
      gts.append(gt)

  with open(model_store_path+'/'+'answers.txt', 'wb') as f_a:
    pickle.dump(answers, f_a)

  with open(model_store_path+'/'+'gt.txt', 'wb') as f_g:
    pickle.dump(gts, f_g)
  


@stub.local_entrypoint()
def main():
    run_inference.remote()