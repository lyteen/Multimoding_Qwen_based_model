from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import src.tools.debug as debug
logger = debug.setup_logger(name='Audio_Encoder', mode='w')

if __name__ == "__main__":   
    processor = AutoProcessor.from_pretrained("C:/Users/Mr.chen/DL/Assets/model/qwen2_audio")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("C:/Users/Mr.chen/DL/Assets/model/qwen2_audio", device_map="auto")

    model.audio_tower.save_pretrained("assets/audio_enc")
    processor.save_pretrained("assets/audio_enc")

    logger.info(f"model arch: {model}") 

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
        ]},
        {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()), 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")
    logger.info(f"audio_info: {inputs['input_features'].shape}")
    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    
    """
        Thinking: ? inputs.input_ids.size(1): ?
    """
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    