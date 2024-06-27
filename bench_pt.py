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

from torch.profiler import profile, ProfilerActivity

set_seed(42)


def save_plot(stats):
    for model_id in stats.keys():
        latency = stats[model_id]["latency"]
        duration = stats[model_id]["duration"]
        memory = stats[model_id]["memory"]
        quant = ["FP32", "BF16", "INT8", "INT4", "NF4"]
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

        plt.suptitle(f'Parameters for {model_id.replace("/", "-")} Model')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.text(0.5, 0.00, "Quantization", ha="center", va="center", fontsize=12)

        plt.savefig(
            f'plot-{model_id.replace("/", "-")}.png', bbox_inches="tight", pad_inches=0
        )


model_list = [
    "Salesforce/codegen2-1B",
    "Salesforce/codegen2-3_7B",
    "Salesforce/codegen2-7B",
]

prompt = "def even_or_odd(x):"


def model_memory_footprint(model):
    return float(f"{model.get_memory_footprint() / 1024 / 1024 / 1024:.2f}")


def generate_helper(model, tokenizer, prompt,trace_filename):
    
    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to("cuda")
    
    # warm up
    for i in range(3):
        _ = model.generate(inputs, max_length=15, pad_token_id=tokenizer.eos_token_id)

    start = time.perf_counter()
    generated_tokens = model.generate(
        inputs,
        max_length=48,
        pad_token_id=tokenizer.eos_token_id
    )
    duration = time.perf_counter() - start
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    
    generated_text = generated_text[len(prompt) :]

    latency_per_token_in_ms = (
        duration / len(generated_tokens[0])
    ) * 1000
    
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True
    #     ) as prof:
    #     generated_tokens = model.generate(
    #         inputs,
    #         max_length=128,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    # prof.export_chrome_trace(trace_filename)
            

    return {
        "text": generated_text,
        "latency": float(f"{round(latency_per_token_in_ms,2)}"),
        "duration": duration,
    }


quants = [ "FP32", "BF16", "INT8", "INT4", "NF4"]

response_df = pd.DataFrame(index=model_list, columns=quants)
stats = {}
for model_id in model_list:
    stats[model_id] = {
        "latency": [],
        "duration": [],
        "memory": [],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for quant in quants:
        if quant == "FP32":
            kwargs = {"torch_dtype": torch.float32}
        elif quant == "BF16":
            kwargs = {"torch_dtype": torch.bfloat16}
        elif quant == "INT8":
            eight_bit_config = BitsAndBytesConfig(load_in_8bit=True, load_in_8bit_fp32_cpu_offload=True)
            kwargs = {"quantization_config": eight_bit_config}
        elif quant == "INT4":
            four_bit_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs = {"quantization_config": four_bit_config}
        elif quant == "NF4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs = {"quantization_config": nf4_config}

        print(f"Loading model: {model_id} with {quant} quantization")

        if model_id == "Salesforce/codegen2-7B":
            kwargs['offload_folder'] = "/data/offload"
            kwargs['low_cpu_mem_usage'] = True
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision="main",
            device_map="auto",
            **kwargs,
        )
        memory = model_memory_footprint(model)


        trace_filename = f'trace-{model_id.replace("/", "-")}-{quant}.json'
        result = generate_helper(model, tokenizer, prompt, trace_filename)
        # clear_memory(model, tokenizer)
        release_memory(model)
        del model
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

response_df.to_csv("text-codegen.csv", index=True)
