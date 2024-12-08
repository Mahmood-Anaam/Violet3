# src/violet/configuration/configuration_violet.py

import torch

class VioletConfig:
    """
    Configuration for Violet Pipeline.
    Contains default parameters that can be used globally in the pipeline.
    """

    # General settings
    CHECKPOINT_DIR = "/content/drive/MyDrive/Violet_checkpoint_0.pth"  # Path to the pretrained model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for computation (GPU/CPU)
    TOKENIZER_NAME = "UBC-NLP/Jasmine-350M"  # Tokenizer model name

    # Model architecture
    ENCODER_LAYERS = 3  # Number of layers in the encoder
    DECODER_LAYERS = 12  # Number of layers in the decoder
    TAU = 0.3  # Tau value for the decodera

    # Input settings
    INPUT_TYPE = "image"  # Options: "image", "features"

    # Output settings
    OUTPUT_TYPE = "both"  # Options: "captions", "features", "both"

    # Generation settings
    BEAM_SIZE = 5
    OUT_SIZE = 3
    MAX_LENGTH = 40

    # Image preprocessing
    IMAGE_RESIZE = 224
    IMAGE_CROP = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

