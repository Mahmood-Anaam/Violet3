import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import numpy as np
from transformers import AutoTokenizer
from violet.modeling import Violet, VisualEncoder, ScaledDotProductAttention


class VioletPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)

        # Initialize model components
        self.encoder = VisualEncoder(
            N=cfg.ENCODER_LAYERS, 
            padding_idx=0, 
            attention_module=ScaledDotProductAttention
        )
        self.model = Violet(
            bos_idx=self.tokenizer.vocab["<|endoftext|>"],
            encoder=self.encoder,
            n_layer=cfg.DECODER_LAYERS,
            tau=cfg.TAU,
        )

        # Load checkpoint
        checkpoint = torch.load(cfg.CHECKPOINT_DIR, map_location=cfg.DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval().to(cfg.DEVICE)

        # Define image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_RESIZE),
            transforms.CenterCrop(cfg.IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD),
        ])

    def _preprocess_image(self, image):
        """
        Preprocess a single image based on its type (URL, path, tensor, array, or PIL image).
        """
        if isinstance(image, str):  # URL or file path
            try:
                # Check if it's a URL
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            except Exception:
                # If it's not a URL, treat it as a file path
                image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):  # Array
            image = Image.fromarray(image.astype("uint8")).convert("RGB")
        elif isinstance(image, torch.Tensor):  # Tensor
            if len(image.shape) == 3:
                return image.unsqueeze(0)  # Assume the tensor is already preprocessed
            return image
        elif isinstance(image, Image.Image):  # PIL Image
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return self.transform(image).unsqueeze(0)

    def _preprocess_batch(self, inputs):
        """
        Preprocess a batch of images, supporting mixed types.
        """
        preprocessed_images = []
        for image in inputs:
            preprocessed_images.append(self._preprocess_image(image))
        return torch.cat(preprocessed_images).to(self.cfg.DEVICE)

    def _generate_output(self, visual_features):
        """
        Generate captions and/or extract features based on the configuration.
        """
        results = []
        with torch.no_grad():
            if self.cfg.OUTPUT_TYPE in ["captions", "both"]:
                output, _ = self.model.beam_search(
                    visual=visual_features,
                    max_len=self.cfg.MAX_LENGTH,
                    eos_idx=self.tokenizer.vocab["<|endoftext|>"],
                    beam_size=self.cfg.BEAM_SIZE,
                    out_size=self.cfg.OUT_SIZE,
                )
                captions = [
                    [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in output[i]
                    ]
                    for i in range(output.size(0))
                ]

            for i in range(visual_features.size(0)):
                result = {}
                if self.cfg.OUTPUT_TYPE in ["features", "both"]:
                    result["features"] = visual_features[i].cpu().numpy()
                if self.cfg.OUTPUT_TYPE in ["captions", "both"]:
                    result["captions"] = captions[i]
                results.append(result)

        return results

    def predict(self, inputs):
        """
        Main prediction method. Accepts a single input or a batch of inputs.
        """
        # Ensure inputs is a list for consistent processing
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Preprocess inputs
        visual_features = self._preprocess_batch(inputs)

        # Generate output
        return self._generate_output(visual_features)
