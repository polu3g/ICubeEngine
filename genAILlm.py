from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time", max_length=50))

from diffusers import StableDiffusionPipeline 
import torch 
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("A futuristic cityscape at night").images[0] 
image.show()

image = pipe("A train at a railstation").images[0] 
image.show()