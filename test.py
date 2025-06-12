import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import src.tools.debug as debug
from src.qwen_vlm_encoder import Qwen2_5_VisionTransformerPretrainedModel

logger = debug.setup_logger(name="TEST Encoder", mode='w')

if __name__ == "__main__":
    model = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(
        "assets/vision_enc", torch_dype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("assets/vision_enc")

    logger.info(f"model arch:{model}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    logger.info(f"pixel_values shape: {inputs['pixel_values'].shape}")
    model.eval()
    with torch.no_grad():
        # image_embeding = model(inputs['pixel_values'], inputs['image_grid_thw'])
        image_embedding = model(inputs['pixel_values'], inputs['image_grid_thw'])
    logger.info(f"embedding: {image_embedding.shape}")