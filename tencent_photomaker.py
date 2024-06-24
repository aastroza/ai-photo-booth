# ## Basic setup

from io import BytesIO
from pathlib import Path

from modal import Image, Stub, build, enter, gpu, method

# ## Define a container image


image = Image.debian_slim().pip_install(
    "diffusers~=0.25",
    "accelerate",
    "git+https://github.com/TencentARC/PhotoMaker.git"
)

stub = Stub("photo-maker", image=image)

# gloal variable and function
def image_grid(imgs, rows, cols, size_after_resize):
    assert len(imgs) == rows*cols

    w, h = size_after_resize, size_after_resize

    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        img = img.resize((w,h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = 'SG161222/RealVisXL_V3.0'
device = "cuda"
save_path = "./outputs"

with image.imports():
    import torch
    import numpy as np
    import random
    import os
    from PIL import Image

    from diffusers.utils import load_image
    from diffusers import EulerDiscreteScheduler, DDIMScheduler
    from huggingface_hub import hf_hub_download

    from photomaker import PhotoMakerStableDiffusionXLPipeline


# ## Load model and run inference

@stub.cls(gpu=gpu.A100G(), container_idle_timeout=240)
class Model:
    @build()
    def download_models(self):
        photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

    @enter()
    def enter(self):
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

        self.pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_ckpt),
            subfolder="",
            weight_name=os.path.basename(photomaker_ckpt),
            trigger_word="img"
        )
        self.pipe.id_encoder.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.fuse_lora()

    @method()
    def inference(self, image_bytes, prompt):
        init_image = load_image(Image.open(BytesIO(image_bytes))).resize(
            (512, 512)
        )
        num_inference_steps = 4
        strength = 0.9
        # "When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1"
        # See: https://huggingface.co/stabilityai/sdxl-turbo
        assert num_inference_steps * strength >= 1

        image = self.pipe(
            prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


DEFAULT_IMAGE_PATH = Path(__file__).parent / "demo_images/dog.png"


@stub.local_entrypoint()
def main(
    image_path=DEFAULT_IMAGE_PATH,
    prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
):
    with open(image_path, "rb") as image_file:
        input_image_bytes = image_file.read()
        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion-xl-turbo")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(output_image_bytes)


# ## Running the model
#
# We can run the model with different parameters using the following command,
# ```
# modal run stable_diffusion_xl_turbo.py --prompt="harry potter, glasses, wizard" --image-path="dog.png"
# ```
