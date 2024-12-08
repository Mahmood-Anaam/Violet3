import torch
import numpy as np
from typing import List, Union, Dict
from PIL import Image
from violet.modeling.transformer import  VisualEncoder, ScaledDotProductAttention
from violet.modeling import Violet

from transformers import AutoTokenizer
import torchvision.transforms as transforms

from violet.evaluation.cider.cider import Cider
from violet.evaluation.bleu.bleu import Bleu
from violet.evaluation.meteor.meteor import Meteor
from violet.evaluation.rouge.rouge import Rouge
from tqdm import tqdm


class VioletImageCaptioningPipeline:
    """
    Pipeline for generating image captions using the Violet Image Captioning model.
    """

    def __init__(self, cfg):
        """
        Initializes the pipeline with the specified configuration.

        Args:
            cfg (VioletConfig): Configuration object containing all settings for the pipeline.
        """
        # Load configuration, tokenizer, encoder, and model

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/Jasmine-350M")
        self.encoder = VisualEncoder(N=3, padding_idx=0, attention_module=ScaledDotProductAttention)
        self.model = Violet(bos_idx=self.tokenizer.vocab['<|endoftext|>'], encoder=self.encoder, n_layer=12, tau=0.3)
        
        
        checkpoint = torch.load(self.cfg.checkpoint_dir, map_location=self.cfg.device)
        self.model.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.cfg.device)

        
        self.transform = transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.CenterCrop(224),  
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                                            ])


        self.input_parms ={

            "is_decode": True,
            "do_sample": False,
            "bos_token_id": self.cls_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_ids": [self.sep_token_id],
            "mask_token_id": self.mask_token_id,
            "add_od_labels": self.cfg.add_od_labels,
            "od_labels_start_posid": self.cfg.max_seq_a_length,
            # Beam search hyperparameters
            "max_length": self.cfg.max_gen_length,
            "num_beams": self.cfg.num_beams,
            "temperature": self.cfg.temperature,
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "repetition_penalty": self.cfg.repetition_penalty,
            "length_penalty": self.cfg.length_penalty,
            "num_return_sequences": self.cfg.num_return_sequences,
            "num_keep_best": self.cfg.num_keep_best,
        }

     
    def process_batch(self,image_batch):
        """
        process batch
        """
        processed_batch = []
        for image in image_batch:
            processed_image = self.transform(image)
            processed_batch.append(processed_image)
            processed_batch = torch.stack(processed_batch)
        return processed_batch
    

    def predict_caption(self,images_tensors):
        """
        Predicts multiple captions for a batch of images using beam search.

        Args:
            model: The image captioning model.
            tokenizer: The tokenizer used for encoding and decoding captions.
            images_tensors: A tensor of image features (batch_size, feature_dim).
            max_length: The maximum length of generated captions.
            beam_size: The number of beams to use in beam search.
            out_size: The number of captions to generate per image.

        Returns:
            A list of lists, where each inner list contains the generated captions for an image.
        """

        with torch.no_grad():
            output, _ = self.model.beam_search(
                                        visual = images_tensors, 
                                        max_len = self.cfg.max_length,
                                        eos_idx = self.tokenizer.vocab['<|endoftext|>'],
                                        beam_size = self.cfg.beam_size,
                                        out_size= self.cfg.out_size
                                        )
            
            batch_size = output.shape[0]
            generated_captions = []

            for i in range(batch_size):
                image_captions = []
                for j in range(min(self.cfg.out_size, self.cfg.beam_size)):
                    caption = self.tokenizer.decode(output[i, j, :], skip_special_tokens=True)
                    image_captions.append({"caption":caption})
                generated_captions.append(image_captions)
        return generated_captions
    

    
    def _prepare_inputs(self, image: Union[Image.Image, np.ndarray, str]) -> Dict:
        """
        Prepares inputs for the model from a single image.

        Args:
            image: Input image (PIL.Image, NumPy array, or file path).

        Returns:
            Dict: Dictionary of inputs for the model.
            Dict: object detections
        """
        try:
            # Extract image features and object detection labels
           
            object_detections = self.feature_extractor([image])[0]
            image_features, od_labels = object_detections["img_feats"],object_detections["od_labels"]
            # Tensorize inputs using the caption tensorizer
            input_ids, attention_mask, token_type_ids, img_feats, masked_pos = self.caption_tensorizer.tensorize_example(
                text_a=None, img_feat=image_features, text_b=od_labels
            )

            # Prepare inputs as a dictionary
            inputs = {
                "input_ids": input_ids.unsqueeze(0).to(self.cfg.device),  # Batch dim
                "attention_mask": attention_mask.unsqueeze(0).to(self.cfg.device),
                "token_type_ids": token_type_ids.unsqueeze(0).to(self.cfg.device),
                "img_feats": img_feats.unsqueeze(0).to(self.cfg.device),
                "masked_pos": masked_pos.unsqueeze(0).to(self.cfg.device),
            }
            return object_detections,inputs
        except Exception as e:
            raise RuntimeError(f"Failed to prepare inputs for the image: {e}")


    def generate_captions(self, images: List[Union[Image.Image, np.ndarray, str]]):
        """
        Generates captions for a list of images.

        Args:
            images: List of images (PIL.Image, NumPy array, or file paths).

        Returns:
            List[List[Dict]]: List of captions with confidence scores for each image.
            List[List[Dict]]: List of image features for each image.
        """
        if not isinstance(images, list):
            images = [images]  # Convert to batch format

        captions = []
        features=[]
        try:
            for image in images:
                # Prepare inputs for the model
                image_features,inputs = self._prepare_inputs(image)
                inputs.update(self.input_parms)
                features.append(image_features)

                # Generate captions using the model
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Decode the captions and collect results
                all_caps = outputs[0]
                all_confs = torch.exp(outputs[1])

                caps = []
                for cap, conf in zip(all_caps[0], all_confs[0]):
                    caption = self.tokenizer.decode(
                        cap.tolist(), skip_special_tokens=True
                    )
                    caps.append({"caption": caption, "confidence": conf.item()})

                captions.append(caps)

            return features,captions
        except Exception as e:
            raise RuntimeError(f"Failed to generate captions for the images: {e}")
    


    def __call__(self, images: List[Union[Image.Image, np.ndarray, str]]):
        return self.generate_captions(images)




# .......................................................................

