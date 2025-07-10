import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === Template ===
prompt_template = """You are an expert judge in language model output quality. Given a user prompt and three responses from different models, your job is to evaluate and compare the responses on helpfulness, coherence, relevance, and completeness.

Please read the following and then provide:
- A score out of 10 for each response.
- A brief explanation of how you arrived at those scores.
- A ranking of the responses from best to worst.

[PROMPT]
{prompt}

[RESPONSE A - GPT]
{gpt_output}

[RESPONSE B - OpenHermes Base]
{openhermes_output}

[RESPONSE C - OpenHermes Finetuned]
{openhermes_finetuned_output}

[Your Evaluation:]
"""

# === Load Judge Model ===
def load_model(model_name):
    print(f"🔍 Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# === Run LLM Inference ===
def run_model_judge(model_pipeline, prompt):
    output = model_pipeline(prompt, do_sample=False)[0]["generated_text"]
    return output[len(prompt):].strip()  # Return only the model’s continuation

# === Main Evaluation Logic ===
def evaluate_responses(input_path, model_name):
    df = pd.read_csv(input_path)
    judge_pipeline = load_model(model_name)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = prompt_template.format(
            prompt=row["Prompt"],
            gpt_output=row["GPT"],
            openhermes_output=row["Open-Hermes"],
            openhermes_finetuned_output=row["Finetuned"]
        )

        evaluation = run_model_judge(judge_pipeline, prompt_text)

        results.append({
            "prompt": row["Prompt"],
            "gpt_output": row["GPT"],
            "openhermes_base": row["Open-Hermes"],
            "openhermes_finetuned": row["Finetuned"],
            "judge_model": model_name,
            "evaluation": evaluation
        })

    output_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"{model_name.lower().replace('/', '_')}-evaluation-{timestamp}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved evaluations to {output_path}")

# === CLI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, help="Hugging Face model ID (e.g., CohereForAI/c4ai-command-r-plus, meta-llama/Meta-Llama-3-8B, etc.)")
    parser.add_argument("--input", default="model-responses.csv", help="Path to CSV with Prompt, GPT, Open-Hermes, Finetuned")
    args = parser.parse_args()

    evaluate_responses(args.input, args.judge)
