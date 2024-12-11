import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import AutoTokenizer
from violet.modeling.modeling_violet import Violet
from violet.modeling.transformer.encoders import VisualEncoder
from violet.modeling.transformer.attention import ScaledDotProductAttention

class VioletPipeline:
    def __init__(self, cfg):
        """
        Initialize the VioletPipeline.

        Args:
            cfg (VioletConfig): Configuration object containing model parameters.
        """
        self.cfg = cfg
        self.device = self.cfg.DEVICE

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TOKENIZER_NAME)
        encoder = VisualEncoder(
            N=self.cfg.ENCODER_LAYERS,
            padding_idx=0,
            d_model=768,
            d_k=64,
            d_v=64,
            h=12,
            d_ff=2048,
            dropout=0.1,
            attention_module=ScaledDotProductAttention
        )
        self.model = Violet(
            bos_idx=self.tokenizer.vocab['<|endoftext|>'],
            encoder=encoder,
            n_layer=self.cfg.DECODER_LAYERS,
            tau=self.cfg.TAU
        )
        checkpoint = torch.load(self.cfg.CHECKPOINT_DIR, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
            transforms.CenterCrop(self.cfg.IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.IMAGE_MEAN, std=self.cfg.IMAGE_STD),
        ])

    def _load_image(self, image):
        """
        Load an image from various input formats (URL, path, PIL image, array, tensor).

        Args:
            image: Image input (URL, file path, PIL image, numpy array, or torch tensor).

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        if isinstance(image, str):
            if image.startswith("http"):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:  # Assume single RGB image
                return image.unsqueeze(0).to(self.device)
            return image.to(self.device)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image format")

        return self.transform(image).unsqueeze(0).to(self.device)

    def _process_batch(self, images):
        """
        Process a batch of images into a tensor.

        Args:
            images (list): List of images in various formats.

        Returns:
            torch.Tensor: Batch of image tensors.
        """
        tensors = [self._load_image(image) for image in images]
        return torch.cat(tensors, dim=0)

    def _generate_features(self, images):
        """
        Extract visual features from the model's encoder.

        Args:
            images (torch.Tensor): Batch of image tensors.

        Returns:
            torch.Tensor: Encoded visual features.
        """
        with torch.no_grad():
            outputs = self.model.clip(images)
            image_embeds = outputs.image_embeds.unsqueeze(1)  # Add sequence dimension
            features, _ = self.model.encoder(image_embeds)
        return features

    def generate_captions_from_features(self, features):
        """
        Generate captions given pre-extracted visual features.

        Args:
            features (torch.Tensor): Encoded visual features.

        Returns:
            list: Generated captions for each feature set.
        """
        with torch.no_grad():
            output, _ = self.model.beam_search(
                visual=features,
                max_len=self.cfg.MAX_LENGTH,
                eos_idx=self.tokenizer.vocab['<|endoftext|>'],
                beam_size=self.cfg.BEAM_SIZE,
                out_size=self.cfg.OUT_SIZE,
                is_feature=True
            )
            captions = [
                [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output[i]]
                for i in range(output.shape[0])
            ]
        return captions

    def generate_captions(self, images):
        """
        Generate captions for input images.

        Args:
            images (list or single image): Input images in various formats.

        Returns:
            list: Generated captions for each image.
        """
        if not isinstance(images, list):
            images = [images]
        image_tensors = self._process_batch(images)
        features = self._generate_features(image_tensors)
        return self.generate_captions_from_features(features)

    def generate_captions_with_features(self, images):
        """
        Generate captions and extract features for input images.

        Args:
            images (list or single image): Input images in various formats.

        Returns:
            list of dict: A list where each element contains 'captions' and 'features'.
        """
        if not isinstance(images, list):
            images = [images]
        image_tensors = self._process_batch(images)
        features = self._generate_features(image_tensors)
        captions = self.generate_captions_from_features(features)
        return [{"captions": c, "features": f.cpu().numpy()} for c, f in zip(captions, features)]

    def __call__(self, images):
        """
        Callable interface to generate captions for input images.

        Args:
            images (list or single image): Input images in various formats.

        Returns:
            list: Generated captions for each image.
        """
        return self.generate_captions(images)
