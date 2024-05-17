import torch
import os
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, PegasusTokenizer
from data_processing import load_datasets
from dataset import ParaphraseDataset
from loss import ParaphraseLossWithMetrics
from config import paths, training_params, vae_params, checkpoint_dir
from models import CVAE
from model_utils import trim_model, move_model_to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
datasets = load_datasets(paths)

# Combine and sample training data
train_data = pd.concat([datasets['paws_final_train'], datasets['paws_swap_train'], datasets['quora_train']], ignore_index=True)
train_data = train_data[train_data['label'] == 1]
train_data = train_data.sample(frac=0.3, random_state=42)

validation_data = datasets['paws_final_dev']
test_data = datasets['paws_final_test']

# Initialize tokenizer and datasets
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
train_dataset = ParaphraseDataset(train_data, tokenizer)
validation_dataset = ParaphraseDataset(validation_data, tokenizer)
test_dataset = ParaphraseDataset(test_data, tokenizer)

batch_size = training_params['batch_size']
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

# Load and trim models
num_layers_teacher = 8  # Pegasus-large
num_layers_student = 6  # Pegasus-xsum

teacher_model = trim_model(PegasusForConditionalGeneration, 'google/pegasus-large', num_layers_teacher)
student_model = trim_model(PegasusForConditionalGeneration, 'google/pegasus-xsum', num_layers_student)

# Move models to the device
teacher_model = move_model_to_device(teacher_model, device)
student_model = move_model_to_device(student_model, device)

# Initialize VAE
vae = CVAE(**vae_params).to(device)

# Initialize the custom loss function
loss_fn = ParaphraseLossWithMetrics(student_model, vae, tokenizer, vae_weight=0.1).to(device)

# Initialize optimizer and scheduler
optimizer = AdamW(student_model.parameters(), lr=training_params['learning_rate'])
total_steps = len(train_dataloader) * training_params['num_epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

scaler = GradScaler()

# Initialize TensorBoard writer
writer = SummaryWriter()

# Training loop
global_step = 0
eval_interval = 40

for epoch in range(training_params['num_epochs']):
    print(f"Epoch {epoch + 1}/{training_params['num_epochs']}")
    student_model.train()
    total_loss = 0
    total_ce_loss = 0
    total_vae_loss = 0
    total_distillation_loss = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        torch.cuda.empty_cache()

        with autocast():
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits

            combined_loss, vae_loss, ce_loss = loss_fn(input_ids, attention_mask, labels)

            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / 2, dim=-1),
                F.softmax(teacher_logits / 2, dim=-1),
                reduction='batchmean'
            )

            combined_loss = (1 - training_params['alpha']) * ce_loss + training_params['alpha'] * distillation_loss
            combined_loss = combined_loss / training_params['accumulation_steps']

        scaler.scale(combined_loss).backward()

        if (i + 1) % training_params['accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += combined_loss.item() * training_params['accumulation_steps']
        total_ce_loss += ce_loss.item()
        total_vae_loss += vae_loss.item()
        total_distillation_loss += distillation_loss.item()

        writer.add_scalar('Training/Combined Loss', combined_loss.item(), global_step)
        writer.add_scalar('Training/Original Loss', ce_loss.item(), global_step)
        writer.add_scalar('Training/VAE Loss', vae_loss.item(), global_step)
        writer.add_scalar('Training/Distillation Loss', distillation_loss.item(), global_step)
        global_step += 1

        if global_step % eval_interval == 0:
            student_model.eval()
            bert_score_value, bleu_score_value = loss_fn.evaluate_metrics(input_ids, attention_mask, labels)
            writer.add_scalar('Evaluation/BERT Score', bert_score_value, global_step)
            writer.add_scalar('Evaluation/BLEU Score', bleu_score_value, global_step)
            student_model.train()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

    model_save_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_final_checkpoint.pth')
    torch.save(student_model.state_dict(), model_save_path)

writer.close()

print("Training complete.")
