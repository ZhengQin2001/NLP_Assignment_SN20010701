import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import ParaphraseLossWithMetrics
from config import paths, vae_params, checkpoint_dir
from models import CVAE
from data_processing import load_datasets
from dataset import ParaphraseDataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
datasets = load_datasets(paths)
validation_data = datasets['paws_final_dev']
test_data = datasets['paws_final_test']

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
validation_dataset = ParaphraseDataset(validation_data, tokenizer)
test_dataset = ParaphraseDataset(test_data, tokenizer)

batch_size = 4
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

# Load the trained model
model_path = f'{checkpoint_dir}/epoch_1_final_checkpoint.pth'
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
model.load_state_dict(torch.load(model_path))

# Initialize VAE and loss function
vae = CVAE(**vae_params).to(device)
loss_fn = ParaphraseLossWithMetrics(model, vae, tokenizer, vae_weight=0.1).to(device)

def evaluate_loss_and_metrics(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_vae_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0

    bert_score_values = []
    bleu_score_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            total_loss_batch, vae_loss_batch, ce_loss_batch = loss_fn(input_ids, attention_mask, labels)
            total_loss += total_loss_batch.item()
            total_vae_loss += vae_loss_batch.item()
            total_ce_loss += ce_loss_batch.item()
            total_batches += 1

            bert_score_value, bleu_score_value = loss_fn.evaluate_metrics(input_ids, attention_mask, labels)
            bert_score_values.append(bert_score_value)
            bleu_score_values.append(bleu_score_value)

    avg_total_loss = total_loss / total_batches
    avg_vae_loss = total_vae_loss / total_batches
    avg_ce_loss = total_ce_loss / total_batches
    avg_bert_score = sum(bert_score_values) / len(bert_score_values)
    avg_bleu_score = sum(bleu_score_values) / len(bleu_score_values)

    return avg_total_loss, avg_vae_loss, avg_ce_loss, avg_bert_score, avg_bleu_score

# Evaluate on validation dataset
val_total_loss, val_vae_loss, val_ce_loss, val_bert_score, val_bleu_score = evaluate_loss_and_metrics(validation_dataloader, model, loss_fn)
print(f"Validation - Total Loss: {val_total_loss}, VAE Loss: {val_vae_loss}, CE Loss: {val_ce_loss}, BERT Score: {val_bert_score}, BLEU Score: {val_bleu_score}")

# Evaluate on test dataset
test_total_loss, test_vae_loss, test_ce_loss, test_bert_score, test_bleu_score = evaluate_loss_and_metrics(test_dataloader, model, loss_fn)
print(f"Test - Total Loss: {test_total_loss}, VAE Loss: {test_vae_loss}, CE Loss: {test_ce_loss}, BERT Score: {test_bert_score}, BLEU Score: {test_bleu_score}")
