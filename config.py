import os

# Column mappings for each dataset type
column_mappings = {
    'paws': {'sentence1': 'text1', 'sentence2': 'text2', 'label': 'label'},
    'quora': {'question1': 'text1', 'question2': 'text2', 'is_duplicate': 'label'}
}

# File paths
paths = {
    'paws_final_train': ('/content/drive/MyDrive/Colab Notebooks/nlp_data/final/train.tsv', column_mappings['paws']),
    'paws_final_dev': ('/content/drive/MyDrive/Colab Notebooks/nlp_data/final/dev.tsv', column_mappings['paws']),
    'paws_final_test': ('/content/drive/MyDrive/Colab Notebooks/nlp_data/final/test.tsv', column_mappings['paws']),
    'paws_swap_train': ('/content/drive/MyDrive/Colab Notebooks/nlp_data/swap/train.tsv', column_mappings['paws']),
    'quora_train': ('/content/drive/MyDrive/Colab Notebooks/nlp_data/quora_duplicate_questions.tsv', column_mappings['quora'])
}

# Training parameters
training_params = {
    'num_epochs': 3,
    'learning_rate': 1.5e-4,
    'batch_size': 4,
    'accumulation_steps': 32,
    'alpha': 0.5
}

# VAE parameters
vae_params = {
    'input_dim': 1024,
    'hidden_dim': 512,
    'latent_dim': 256,
    'vocab_size': 96103
}

# Other configurations
checkpoint_dir = '/checkpoints'  # change this to your directory 
