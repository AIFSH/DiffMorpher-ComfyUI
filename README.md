# DiffMorpher-ComfyUI
a custom node for [DiffMorpher](https://github.com/Kevin-thu/DiffMorpher.git),you can find base workflow in [`doc`](./doc/)

## Example
image_0 | image_1 | output
----- | ---- | ----
![](./doc/Biden.jpg) | ![](./doc/15.png) | ![](./doc/diffmorpher_1721118982208625000.gif)

## How to use
```
# in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/DiffMorpher-ComfyUI.git
cd DiffMorpher-ComfyUI
pip install -r requirements.txt
```
weights will be downloaded from huggingface

## Tutorial


DiffMorpherNode

required
- `image__0`: the first image (default: "")
- `prompt_0`: Prompt of the first image (default: "")
- `image_1`: the second image (default: "")
- `prompt_1`: Prompt of the second image (default: "")
- `use_adain`: Use AdaIN (default: False)
- `use_reschedule`: Use reschedule sampling (default: False)
- `lamb`: Hyperparameter $\lambda \in [0,1]$ for self-attention replacement, where a larger $\lambda$ indicates more replacements (default: 0.6)
- `save_inter`: Save intermediate results (default: False) if True, frame saved in `ComfyUI/output/diffmorpher`
- `num_frames`: Number of frames to generate (default: 50)
- `duration`: Duration of each frame (default: 50)

optional
- `model_path`: Pretrained model path (default: "stabilityai/stable-diffusion-2-1-base")
- `lora_0`: Path of the lora directory of the first image (default: "")
- `lora_1`: Path of the lora directory of the second image (default: "")


## ask for answer as soon as you want
wechat: aifsh_98
need donate if you mand it,
but please feel free to new issue for answering

Windows环境配置太难？可以添加微信：aifsh_98，赞赏获取Windows一键包，当然你也可以提issue等待大佬为你答疑解惑。