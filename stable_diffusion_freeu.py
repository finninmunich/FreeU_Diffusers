import torch
from diffusers import StableDiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
g = torch.Generator(device="cuda")
model_id = "/finn/finn/MODELS/AI-ModelScope/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "a blue car is being filmed"
g.manual_seed(0)
image = pipe(prompt,generator = g).images[0]

image.save("./inference_results/blue_car/vanilla.png")
b = [0.6,0.8,1.0,1.2,1.4]
s = [1.0,1.0,1.0,1.0,1.0]
for _b,_s  in zip(b,s):
    # -------- freeu block registration
    register_free_upblock2d(pipe, b1=_b, b2=_b, s1=_s, s2=_s)
    register_free_crossattn_upblock2d(pipe, b1=_b, b2=_b, s1=_s, s2=_s)
    # -------- freeu block registration
    g.manual_seed(0)
    image = pipe(prompt,generator=g).images[0]

    image.save(f"./inference_results/blue_car/{_b}_{_s}.png")