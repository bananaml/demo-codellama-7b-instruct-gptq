from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

MODEL = "TheBloke/CodeLlama-34B-Instruct-GPTQ"

def download_model() -> tuple:
    """Download the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(MODEL,
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=False,
            quantize_config=None)
    return model, tokenizer

if __name__ == "__main__":
    download_model()
    