from transformers import AutoTokenizer
import transformers
import torch

model = "hdeldar/llama-2-7b-persian-text-1k"

prompts = [
    "شرکت فولاد کجاست؟",
    "شرکت فولاد مبارکه در کجا واقع شده است",
    "What is a large language model?",
    "فولاد مبارکه چند بار برنده جایزه شرکت دانشی را کسب کرده است؟",
    "فولاد مبارکه در چه سالی احداث شد؟",
    "تعریف علوم کامپیوترچیست؟"
]
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

for prompt in prompts:
    sequences = pipeline(
        f'<s>[INST] {prompt} [/INST]',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
