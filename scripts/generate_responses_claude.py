import csv
import os
import time
import anthropic

# =========================
# CONFIG
# =========================
MODEL_NAME = "claude-sonnet-4-6"
OUTPUT_CSV = "claude_results.csv"
N_RESPONSES_PER_PROMPT = 50


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
}

# =========================
# LOAD CLIENT
# =========================
# Set ANTHROPIC_API_KEY in your environment before running this script.
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)


# =========================
# GENERATION FUNCTION
# =========================
def generate(prompt):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=GEN_KWARGS["max_tokens"],
        temperature=GEN_KWARGS["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# =========================
# MAIN LOOP (WRITE AS YOU GO)
# =========================
print("Starting generation...")

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

    total_rows = 0

    for prompt_id, prompt_text in PROMPTS.items():
        print(f"\nRunning prompt {prompt_id}...")

        for i in range(N_RESPONSES_PER_PROMPT):
            print(f"  Generating {i+1}/{N_RESPONSES_PER_PROMPT}")

            try:
                response = generate(prompt_text)

                row = {
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "response_id": i + 1,
                    "response": response,
                }

                writer.writerow(row)
                f.flush()  # ✅ ensures file updates immediately
                total_rows += 1

            except Exception as e:
                print("\n🚨 ERROR DETECTED — stopping early")
                print(f"Error on prompt {prompt_id} #{i+1}: {e}")
                print(f"Rows saved so far: {total_rows}")
                exit(1)  # stop immediately so you don’t waste time

            time.sleep(0.5)

print(f"\n✅ Done! Total rows saved: {total_rows}")
