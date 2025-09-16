import os
import whisper
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
from langchain import PromptTemplate, LLMChain
model_id2 = 'swajall/deception'
model_id = 'swajall/seq2seq-model'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer2 = AutoTokenizer.from_pretrained(model_id2)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model2 = AutoModelForSeq2SeqLM.from_pretrained(model_id2)
pipeline2 = pipeline("text2text-generation", model=model2, tokenizer=tokenizer2)
pipeline1 = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


INPUT_DIR = r"C:\Users\harde\Desktop\Hardeeek\LangChain Projects\Innov8\audio"
OUT_DIR = r"C:\Users\harde\Desktop\Hardeeek\LangChain Projects\Innov8\transcripts"
os.makedirs(OUT_DIR, exist_ok=True)

model = whisper.load_model("small")
files_found = False
all_sessions =''
for fn in os.listdir(INPUT_DIR):
    print("Checking file:", fn)

    if fn.lower().endswith((".mp3", ".wav")):
        files_found = True
        print("🎵 Found audio file:", fn)

        path = os.path.join(INPUT_DIR, fn)
        print("Processing file path:", path)

        try:
            res = model.transcribe(str(path), language='en')
            text = res["text"].strip()
            all_sessions += text
            out_path = os.path.join(OUT_DIR, fn.rsplit(".", 1)[0] + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print("✅ Saved transcript:", out_path)

        except Exception as e:
            print("❌ Error processing", fn, ":", e)

if not files_found:
    print("\n⚠️ No mp3/wav files found in:", INPUT_DIR)
else:
    print("\n🎉 All done! Check the 'transcripts' folder.")
print(all_sessions)
b=pipeline1(all_sessions)
c=pipeline2(all_sessions)
print(b)
print(c)
