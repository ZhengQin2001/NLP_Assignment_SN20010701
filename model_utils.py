import torch
from transformers import PegasusForConditionalGeneration

def trim_model(model_class, model_name, num_layers):
    # Load the configuration
    config = model_class.config_class.from_pretrained(model_name)

    # Modify the configuration to reduce the number of layers
    config.num_hidden_layers = num_layers

    # Load the model with the modified configuration
    model = model_class.from_pretrained(model_name, config=config)
    return model

# Function to move the model to the GPU if available
def move_model_to_device(model, device):
    if torch.cuda.is_available():
        model.to(device)
    return model
