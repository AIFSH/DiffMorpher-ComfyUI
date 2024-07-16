import os
import folder_paths
from huggingface_hub import snapshot_download

import math
import torch
import time
import numpy as np
from PIL import Image
from .diffmorpher.model import DiffMorpherPipeline

output_dir = folder_paths.get_output_directory()

def get_64x_num(num):
    return math.ceil(num / 64) * 64

def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left+croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image

class DiffMorpherNode:
    def __init__(self) -> None:
        self.model_path = None
        self.pipeline = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image_0":("IMAGE",),
                "prompt_0":("TEXT",),
                "image_1":("IMAGE",),
                "prompt_1":("TEXT",),
                "num_frames":("INT",{
                    "default":16
                }),
                "duration":("INT",{
                    "default":100
                }),
                "use_adain":("BOOLEAN",{
                    "default":True
                }),
                "use_reschedule":("BOOLEAN",{
                    "default":True
                }),
                "lamb":("FLOAT",{
                    "min": 0.0,
                    "max":1.0,
                    "default":0.6,
                    "display": "slider"
                }),
                "save_inter":("BOOLEAN",{
                    "default":True
                }),
            },
            "optional":{
                "diffusers_model":(folder_paths.get_filename_list("diffusers"),),
                "lora_0":(folder_paths.get_filename_list("loras"),),
                "lora_1":(folder_paths.get_filename_list("loras"),)
            }
        }
    
    RETURN_TYPES = ("GIF",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffMorpher"

    def comfy2image(self,image):
        image = image.numpy()[0] * 255
        image = image.astype(np.uint8)
        image_pil = Image.fromarray(image).convert("RGB")
        org_w, org_h = image_pil.size
        height, width = (768,get_64x_num(768*org_w/org_h)) if org_h > org_w else (get_64x_num(768*org_h/org_w),768)
        print(f"crop and resize from ({org_w},{org_h}) to ({width},{height})")
        return crop_and_resize(image_pil,height,width)

    def generate(self,image_0,prompt_0,image_1,prompt_1,num_frames,duration,
                 use_adain,use_reschedule,lamb,save_inter,diffusers_model=None,lora_0=None,lora_1=None):
        
        if diffusers_model is None:
            # stabilityai/stable-diffusion-2-1-base
            model_path = os.path.join(folder_paths.models_dir, "diffusers","stable-diffusion-2-1-base")
            snapshot_download(repo_id="stabilityai/stable-diffusion-2-1-base",
                              allow_patterns=["*.json","*.txt","*.fp16.safetensors"],
                              local_dir=model_path)
            
        else:
            model_path = folder_paths.get_full_path("diffusers",diffusers_model)
        
        if self.model_path != model_path:
            self.model_path = model_path
            self.pipeline = DiffMorpherPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16,variant="fp16",use_safetensors=True)
        
        self.pipeline.to("cuda")
        out_dir = os.path.join(output_dir,"diffmorpher")
        os.makedirs(out_dir,exist_ok=True)
        images = self.pipeline(img_0=self.comfy2image(image_0),
                               img_1=self.comfy2image(image_1),
                               prompt_0=prompt_0,prompt_1=prompt_1,
                               save_lora_dir=os.path.join(folder_paths.models_dir, "loras"),
                               load_lora_path_0=lora_0,load_lora_path_1=lora_1,
                               use_adain=use_adain,
                               use_reschedule=use_reschedule,
                               lamd=lamb,
                               output_path=out_dir,
                               num_frames=num_frames,
                               save_intermediates=save_inter,
                               use_lora= lora_0 and lora_1,
                               )
        output_path = os.path.join(output_dir,f"diffmorpher_{time.time_ns()}.gif")
        images[0].save(output_path, save_all=True,
               append_images=images[1:], duration=duration, loop=0)
        
        return (output_path,)




class PreViewGIF:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "gif":("GIF",),
        }}
    
    CATEGORY = "AIFSH_DiffMorpher"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_gif"

    def load_gif(self, gif):
        video_name = os.path.basename(gif)
        video_path_name = os.path.basename(os.path.dirname(gif))
        return {"ui":{"gif":[video_name,video_path_name]}}


class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True})
            }
        }
    RETURN_TYPES = ("TEXT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "text"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DiffMorpher"

    def text(self,text):
        return (text,)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "TextNode":TextNode,
    "PreViewGIF":PreViewGIF,
    "DiffMorpherNode": DiffMorpherNode
}