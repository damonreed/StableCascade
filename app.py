import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import uuid

prompt = """
Rhiannah looks like a mix between Zoey Deschanel and Sigourney Weaver
She is seen from the waist up in a revealing dress
She is tall and athletic with long legs and a small waist, She has long curly hair that is a deep black color
She has wide grey eyes
She is standing at the entrance to a royal ball and looks a little out of place
"""
negative_prompt = ""

# Create uuid filename
filename = "images/" + str(uuid.uuid4()) + ".png"

device = "mps"
num_images_per_prompt = 1

prior_file = "stabilityai/stable-cascade-prior"
decoder_file = "stabilityai/stable-cascade"

prior = StableCascadePriorPipeline.from_pretrained(
    prior_file, torch_dtype=torch.float
).to(device)
prior.safety_checker = None
prior.requires_safety_checker = False

prior_output = prior(
    prompt=prompt,
    width=1024,
    height=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20,
)

decoder = StableCascadeDecoderPipeline.from_pretrained(
    decoder_file, torch_dtype=torch.half
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
