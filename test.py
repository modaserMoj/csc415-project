import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B"          # or "checkpoints/phase2_gsm8k_0.5b"

question = (
    "A team of 4 painters worked on a mansion for 3/8ths of a day every day for 3 weeks. "
    "How many hours of work did each painter put in?"
)

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": (
            "Solve this math problem step by step.\n\n"
            f"{question}\n\n"
            "Show your work in <think> </think> tags. "
            "Give your final numerical answer in <answer> </answer> tags."
        ),
    }
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("\n=== Prompt ===")
print(prompt)

inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

input_ids = inputs["input_ids"]
attn_mask = inputs["attention_mask"]

gen = outputs[0]
prompt_len = int(attn_mask[0].sum().item())
new_tokens = gen[prompt_len:]

print("\n=== Raw generated tokens ===")
print(new_tokens)

print("\n=== Decoded completion (skip_special_tokens=False) ===")
print(repr(tokenizer.decode(new_tokens, skip_special_tokens=False)))

print("\n=== Decoded completion (skip_special_tokens=True) ===")
print(repr(tokenizer.decode(new_tokens, skip_special_tokens=True)))