from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from typing import Optional

def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path: Optional[str] = None,
    device: str = "cuda:0",
    load_from_safetensor=True,
):
    if load_from_safetensor:

        if pretrained_vae_name_or_path is None:
            raise ValueError("pretrained_vae_name_or_path must be specified if load_from_safetensor is True")
        
        vae = AutoencoderKL.from_single_file(
            pretrained_vae_name_or_path,
        )
        
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            pretrained_model_name_or_path,
            vae=vae,
            safety_checker=None,
        )
        tokenizer = pipe.tokenizer 
        text_encoder = pipe.text_encoder
        unet = pipe.unet

    else:
        print('loading model...')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path,
        )
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        vae = pipe.vae
    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
    )

def get_ddim_scheduler(pretrained_model_name_or_path):
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    return scheduler
