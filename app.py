import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gc
import uuid

# Create uuid filename
filename = "images/" + str(uuid.uuid4()) + ".png"

device = "mps"
num_images_per_prompt = 1

prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior", torch_dtype=torch.float
).to(device)
prior.safety_checker = None
prior.requires_safety_checker = False

prompt = "A female redhead World of Warcraft rogue in full body view, drawn in color in the style of Jeff Easley."
negative_prompt = ""

prior_output = prior(
    prompt=prompt,
    width=1024,
    height=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20,
)

del prior
gc.collect()
torch.mps.empty_cache

decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade", torch_dtype=torch.half
).to(device)
decoder.safety_checker = None
decoder.requires_safety_checker = False

decoder_output = (
    decoder(
        image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=12,
    )
    .images[0]
    .save(filename)
)

# del decoder
# gc.collect()
# torch.cuda.empty_cache()
