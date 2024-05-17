import torch
import torch.nn as nn
from torch.nn import functional as F
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu

def cvae_loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, recon_x.size(-1))
    x = x.view(-1)
    CE = nn.CrossEntropyLoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
    return CE + KLD

class ParaphraseLossWithMetrics(nn.Module):
    def __init__(self, model, vae, tokenizer, vae_weight=1.0):
        super(ParaphraseLossWithMetrics, self).__init__()
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss()
        self.vae_weight = vae_weight

    def forward(self, input_ids, attention_mask, labels):
        input_embeddings = self.model.model.encoder.embed_tokens(input_ids).to(device)
        condition_embeddings = self.model.model.encoder.embed_tokens(input_ids).to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        ce_loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        vae_output, mu, logvar = self.vae(input_embeddings, condition_embeddings)
        vae_loss = cvae_loss_function(vae_output, labels, mu, logvar)

        total_loss = ce_loss + self.vae_weight * vae_loss

        return total_loss, vae_loss, ce_loss

    def evaluate_metrics(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            input_embeddings = self.model.model.encoder.embed_tokens(input_ids).to(device)
            condition_embeddings = self.model.model.encoder.embed_tokens(input_ids).to(device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            input_sentences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            output_sentences = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
            P, R, F1 = bert_score(output_sentences, input_sentences, lang="en", verbose=False)
            bert_score_value = F1.mean().item()

            bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(input_sentences, output_sentences)]
            bleu_score_value = sum(bleu_scores) / len(bleu_scores)

        return bert_score_value, bleu_score_value
