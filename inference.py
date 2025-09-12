import torch
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import PeftModel
from PolymerSmilesTokenization import PolymerSmilesTokenizer
import yaml

# --- Config ---
checkpoint_path = "ckpt/neurips.pt/Rg"  # e.g., 'ckpt/neurips.pt/density'
num_properties = 1  # Set to number of regression targets
blocksize = 411     # Set to match training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv = "data/neurips-open-polymer-prediction-2025/test.csv"
finetune_config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

# --- Load Model & Tokenizer ---
base_model = RobertaForSequenceClassification.from_pretrained(
    finetune_config['model_path'],
    num_labels=num_properties,
    problem_type="regression"
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.to(device)
model.eval()

tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=blocksize)

# --- Load Data ---
df = pd.read_csv(test_csv)
smiles_list = df["SMILES"].tolist()
ids = df["id"].tolist()

# --- Inference ---
results = []
with torch.no_grad():
    for idx, smiles in zip(ids, smiles_list):
        encoding = tokenizer(
            str(smiles),
            add_special_tokens=True,
            max_length=blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze().cpu().numpy()
        results.append([idx, smiles] + [preds])

# --- Save Results ---
results_df = pd.DataFrame(results, columns=["id", "SMILES"] + [f"pred_{i+1}" for i in range(num_properties)])
results_df.to_csv("results/inference_results_Rg2.csv", index=False)
print("Inference complete. Results saved to inference_results_Rg2.csv.")