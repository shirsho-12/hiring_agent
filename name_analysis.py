import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

# Garbage tried: MobileLLM, Llama-3.1B

model_name = "google/gemma-2-9b-it"


def clean_text(text):
    # remove multiple newlines
    text = re.sub(r"\n+", "\n", text)
    # remove multiple spaces
    text = re.sub(r" +", " ", text)
    # strip leading and trailing whitespace
    text = text.strip()
    # remove triple backticks and the word markdown
    text = text.replace("```", "").replace("markdown", "")
    # remove first and last newlines
    text = text.lstrip("\n").rstrip("\n")
    return text


df = pd.read_csv("data/processed/resumes.csv")
df.drop(columns=["resume", "anonymized", "reformatted"], inplace=True)
df.rename(columns={"localized": "resume"}, inplace=True)
df["resume"] = df["resume"].apply(clean_text)
df.head(5)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

res = []
for resume in tqdm(df["resume"]):
    prompt = f"""Your task is to generate a realistic name for the following anonymized resume from Singapore. The 
    You will see a [Candidate Name] placeholder. Your job is to suggest a realistic name that fits the profile. Output the full name of the person only. The anonymized resume is as follows:
    ------
    {resume}
    ------
    OAnswer with ONLY the name, nothing else:"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    res.append(generated_text.strip())

print("generate_text:", res)
with open("data/processed/generated_names_gemma.txt", "w") as f:
    for name in res:
        f.write(name + "\n")
