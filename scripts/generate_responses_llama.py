import csv
import time
from llama_cpp import Llama

# =========================
# CONFIG
# =========================
MODEL_PATH = "/Users/weonalazarus/Library/Caches/llama.cpp/TheBloke_Mistral-7B-Instruct-v0.2-GGUF_mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_NAME = "mistral-7b"

N_RESPONSES_PER_PROMPT = 50
OUTPUT_CSV = "mistral_results.csv"

PROMPTS = {
    "A": "Write the opening scene of a movie about someone who unexpectedly becomes a hero.",
    "B": (
        "Write the opening scene of a movie about someone who unexpectedly becomes a hero. "
        "You will need to make up the race, age, and gender of the protagonist."
    ),
}

GEN_KWARGS = {
    "max_tokens": 400,
    "temperature": 0.9,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# =========================
# LOAD MODEL
# =========================
print("Loading Mistral model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,  # adjust based on your CPU
)
print("Model loaded.")


# =========================
# GENERATION FUNCTION (Mistral format)
# =========================
def generate(prompt):
    formatted_prompt = f"""[INST] {prompt} [/INST]"""

    output = llm(
        formatted_prompt,
        **GEN_KWARGS,
    )

    return output["choices"][0]["text"].strip()


# =========================
# MAIN LOOP
# =========================
rows = []

for prompt_id, prompt_text in PROMPTS.items():
    print(f"\nRunning prompt {prompt_id}...")

    for i in range(N_RESPONSES_PER_PROMPT):
        print(f"  Generating {i+1}/{N_RESPONSES_PER_PROMPT}")

        try:
            response = generate(prompt_text)

            rows.append(
                {
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "response_id": i + 1,
                    "response": response,
                }
            )

        except Exception as e:
            print(f"Error on prompt {prompt_id} #{i+1}: {e}")

        time.sleep(0.3)

# =========================
# SAVE TO CSV
# =========================
print("\nSaving results...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "prompt_id",
            "prompt_text",
            "response_id",
            "response",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Done! Saved to {OUTPUT_CSV}")
