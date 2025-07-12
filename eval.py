import argparse
import pandas as pd
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def run_model_judge(pipe, prompt):
    """Run the model pipeline and extract the text response."""
    result = pipe(prompt)
    if isinstance(result, list) and len(result) > 0:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    else:
        return "ERROR: No output"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=str, required=True, help="Model to use as judge")
    parser.add_argument("--input", type=str, default="model-responses.csv", help="CSV input file")
    args = parser.parse_args()

    judge_model = args.judge
    input_csv = args.input

    print(f"🔍 Loading judge model: {judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    judge_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

    print(f"📄 Reading input: {input_csv}")
    df = pd.read_csv(input_csv)

    output_rows = []

    for i, row in df.iterrows():
        try:
            prompt = row["Prompt"]
            response_a = row["GPT"]
            response_b = row["Open-Hermes"]
            response_c = row["Finetuned"]

            full_prompt = f"""You are a helpful and fair model evaluator. Given a prompt and three responses, select the best and explain why.

Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Response C:
{response_c}

Only respond with your reasoning and your ranking (A, B, C)."""
            
            print(f"🧠 Evaluating row {i+1}/{len(df)}")
            evaluation = run_model_judge(judge_pipeline, full_prompt)

        except Exception as e:
            print(f"❌ Error at row {i+1}: {e}")
            evaluation = "ERROR"

        output_rows.append({
            "Prompt": prompt,
            "GPT": response_a,
            "Open-Hermes": response_b,
            "Finetuned": response_c,
            "Evaluation": evaluation
        })

    output_df = pd.DataFrame(output_rows)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name_safe = judge_model.lower().replace("/", "_")
    output_path = f"{model_name_safe}-evaluation-{timestamp}.csv"

    print(f"💾 Saving output to: {output_path}")
    output_df.to_csv(output_path, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
