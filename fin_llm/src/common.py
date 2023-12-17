
MODEL_PATH = "/model"

def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    model_name = "NousResearch/Llama-2-7b-chat-hf"
    print('Loading the pretrained model')
    model = LlamaForCausalLM.from_pretrained(model_name)
    print('Saving the pretrained model')
    model.save_pretrained(MODEL_PATH)
