
import torch

from transformers import SamProcessor
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from .visual_prompts import generate_clicks, generate_boxes
from .sam_model import rescale_inputs
import openvino as ov
import numpy as np
import os


@META_ARCH_REGISTRY.register()
class OpenVINO_SAM(torch.nn.Module):

    @configurable
    def __init__(self,
                 *,
                 model_name: str,
                 num_classes: int,
                 background_class: int = 0,
                 prompt_type: str = "points",
                 ):
        super().__init__()
        self.device = "CPU"
        # self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.core = ov.Core()
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # Load OpenVINO models
        model_name = "ov_fp_sam"
        self.vision_encoder = self.core.compile_model(os.path.join(model_name, "openvino_vision_encoder_int8.xml"), self.device)
        self.prompt_decoder = self.core.compile_model(os.path.join(model_name, "openvino_prompt_encoder_mask_decoder_int8.xml"), self.device)

        self.vision_encoder_outputs = self.vision_encoder.outputs
        self.prompt_decoder_outputs = self.prompt_decoder.outputs

        self.num_classes = num_classes
        self.background_class = background_class if background_class <= num_classes else num_classes
        self.prompt_type = prompt_type

    @classmethod
    def from_config(cls, cfg):
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        # Find background class
        if hasattr(meta, 'background_class'):
            background_class = meta.background_class
        elif 'background' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('background')
        elif 'Background' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('Background')
        elif 'others' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('others')
        elif 'Others' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('Others')
        else:
            background_class = 0

        return {
            "model_name": cfg.MODEL.WEIGHTS,
            "num_classes": len(meta.stuff_classes),
            "background_class": background_class,
            "prompt_type": cfg.PROMPT_TYPE,
        }

    def forward(self, batched_inputs):
        # get inputs from batch
        images = [x["image"] for x in batched_inputs]

        if self.prompt_type == 'points':
            if 'input_points' not in batched_inputs[0]:
                # simulate inputs if not in batch
                assert batched_inputs[0]['sem_seg'] is not None, "No input_points or sem_seg in batched_inputs"
                batched_inputs = generate_clicks(batched_inputs)

            # make sure that the input points are at the same scale as the input image
            if batched_inputs[0]['image'].shape[1:] != batched_inputs[0]['sem_seg'].shape:
                rescale_inputs(batched_inputs)

            # get points from batch
            input_points = [x["input_points"] for x in batched_inputs]
            inputs = self.processor(images, input_points=input_points, return_tensors="pt")

        elif self.prompt_type == 'boxes':
            if 'input_boxes' not in batched_inputs[0]:
                # simulate inputs if not in batch
                assert batched_inputs[0]['sem_seg'] is not None, "No input_boxes or sem_seg in batched_inputs"
                batched_inputs = generate_boxes(batched_inputs)

            # make sure that the input boxes are at the same scale as the input image
            if batched_inputs[0]['image'].shape[1:] != batched_inputs[0]['sem_seg'].shape:
                rescale_inputs(batched_inputs)

            # get points from batch
            input_boxes = [x["input_boxes"] for x in batched_inputs]
            inputs = self.processor(images, input_boxes=input_boxes, return_tensors="pt")
        else:
            print(f"Prompt type {self.prompt_type} not implemented")
            raise NotImplementedError

        # inference
        # outputs = self.model(**inputs)

        # Run vision encoder
        image_tensor = inputs["pixel_values"].numpy()
        encoder_outputs = self.vision_encoder([image_tensor])
        image_embeddings = encoder_outputs[self.vision_encoder_outputs[0]]
        image_positional_embeddings = encoder_outputs[self.vision_encoder_outputs[1]]

        # print("inputs", inputs.keys())
        # inputs dict_keys(['pixel_values', 'original_sizes', 'reshaped_input_sizes', 'input_points'])
        # raise ""

        if "input_points" in inputs and "input_labels" not in inputs:
            inputs["input_labels"] = np.ones_like(inputs["input_points"][:, :, :, 0], dtype=np.int32)

        # Run prompt decoder
        decoder_outputs = self.prompt_decoder([
            inputs["input_points"].numpy(),
            inputs["input_labels"],
            image_embeddings,
            image_positional_embeddings
        ])

        pred_masks = decoder_outputs[self.prompt_decoder_outputs[1]]
        pred_masks_tensor = torch.tensor(pred_masks)

        # postprocess (non max suppression and resizing)
        sizes = [(i.get("height"), i.get("width")) for i in batched_inputs]
        input_sizes = [(i.shape[1], i.shape[2]) for i in images]
        masks = self.processor.image_processor.post_process_masks(pred_masks_tensor.cpu(), sizes, input_sizes, binarize=False)

        # convert instance results to semantic results
        input_classes = [x["input_classes"] for x in batched_inputs]
        processed_results = []

        for mask, classes in zip(masks, input_classes):
            # select the first out of 3 prediction masks
            pred = mask[:, 0]
            # init prediction mask
            mask_size = pred.shape[-2:]
            r = torch.zeros((self.num_classes + 1, *mask_size))
            # convert instance mask to semantic mask by selecting the highest score for each pixel and class
            classes = torch.Tensor(classes)[:, 0].int()
            for c in torch.unique(classes):
                r[c] = torch.max(pred[classes == c, :, :], dim=0).values
            # set pixels with no prediction (all scores 0. or below) to background class
            r[self.background_class, r.max(dim=0).values <= 0] = 1.
            # drop ignore class
            processed_results.append({
                "sem_seg": r[:-1],
            })
        return processed_results

