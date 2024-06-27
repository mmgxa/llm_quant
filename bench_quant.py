import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import pandas as pd
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate.utils import release_memory
import gc
from transformers import set_seed
import matplotlib.pyplot as plt
from awq import AutoAWQForCausalLM


set_seed(42)


def save_plot(stats):
    for model_id in stats.keys():
        latency = stats[model_id]["latency"]
        duration = stats[model_id]["duration"]
        memory = stats[model_id]["memory"]
        quant = ["AWQ", "GPTQ"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for i, (data, title, ylabel, color) in enumerate(
            zip(
                [latency, duration, memory],
                ["Latency", "Duration", "Memory"],
                ["( ms/token)", "(s)", "(GB)"],
                ["red", "green", "blue"],
            )
        ):
            ax = axs[i]
            a = ax.bar(quant, data, color=color)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.bar_label(a, label_type="edge")

        plt.suptitle(f"Parameters for {model_id} Model")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.text(0.5, 0.00, "Quantization", ha="center", va="center", fontsize=12)

        plt.savefig(f"plot-{model_id}.png", bbox_inches="tight", pad_inches=0)


model_list = [
    "CodeLlama-34B",
]


prompt = "def even_or_odd(x):"
# prompt = "def fibonacci("
# prompt = "def factorial("
# prompt = "def remove_last_word("
# prompt = "def remove_non_ascii(s: str) -> str:"


def model_memory_footprint(model):
    return float(f"{model.get_memory_footprint() / 1024 / 1024 / 1024:.2f}")


def model_memory_footprint_awq(model):
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem + mem_bufs
    return float(f"{mem / 1024 / 1024 / 1024:.2f}")


def generate_helper(model, tokenizer, prompt):
    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to("cuda")

    # warm up
    for i in range(3):
        _ = model.generate(inputs, max_length=15, pad_token_id=tokenizer.eos_token_id)

    start = time.perf_counter()
    generated_tokens = model.generate(
        inputs, max_length=48, pad_token_id=tokenizer.eos_token_id
    )
    duration = time.perf_counter() - start
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    generated_text = generated_text[len(prompt) :]

    latency_per_token_in_ms = (duration / len(generated_tokens[0])) * 1000

    return {
        "text": generated_text,
        "latency": float(f"{round(latency_per_token_in_ms,2)}"),
        "duration": duration,
    }


quants = ["AWQ", "GPTQ"]

response_df = pd.DataFrame(index=model_list, columns=quants)
stats = {}
for model_id in model_list:
    stats[model_id] = {
        "latency": [],
        "duration": [],
        "memory": [],
    }

    for quant in quants:
        model_dir = f"/data/{model_id}-{quant}"
        print(f"Loading model: {model_id} with {quant} quantization")
        if quant == "BF16":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            memory = model_memory_footprint(model)
        elif quant == "AWQ":
            model = AutoAWQForCausalLM.from_pretrained(
                model_dir,
                safetensors=True,
                trust_remote_code=True,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            memory = model_memory_footprint_awq(model)
        elif quant == "GPTQ":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            memory = model_memory_footprint(model)
        else:
            break

        result = generate_helper(model, tokenizer, prompt)
        release_memory(model)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        response_df.loc[model_id, quant] = [
            value for key, value in result.items() if key == "text"
        ]
        stats[model_id]["latency"].append(result["latency"])
        stats[model_id]["duration"].append(result["duration"])
        stats[model_id]["memory"].append(memory)

save_plot(stats)

response_df.to_csv("text-coldellama.csv", index=True)
