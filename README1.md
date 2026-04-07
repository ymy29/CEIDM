# CEIDM: A Controlled Entity and Interaction Diffusion Model for Enhanced Text-to-Image Generation

<!--  [![Framework figure](docs/static/res/Framework.png")]  -->

- Existing methods lack ability to control the entities and their interactions in the generated content.
- We propose a pluggable control model, called CEIDM that extends existing pre-trained T2I diffusion models to enable them being better conditioned on entities and interactions.

## Experimental Environment
```python
conda env create -f environment.yml
```

Our inference experiments can be implemented on an NVIDIA A100-SXM4-40GB GPU.

## Prepare Dataset

Download the HICO-DET dataset at [here](https://entuedu-my.sharepoint.com/:u:/g/personal/jiuntian001_e_ntu_edu_sg/EfNbqVvn18JEqH1YU5Fb5YMBcxGan6VoJMEaKsiu2Fu9Dw?e=aiDQAu), note that you should check and adhere to the original terms and conditions of [HICO-DET](https://websites.umich.edu/~ywchao/hico/) for using this processed dataset.

## Download Models 

Our method is training-free. The inference model is based on InteractDiffusion model. The model checkpoint download link is below.

| Version | Dataset    | SD |Download |
|---------|------------|----|---------|
| v1.0 | HICO-DET                 | v1.4| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1.pth) |
| v1.1 | HICO-DET                 | v1.5| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1-1.pth) |
| v1.2 | HICO-DET + VisualGenome  | v1.5| [HF Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/interact-diffusion-v1-2.pth) |
| XL v1.0 | HICO-DET | XL | [HF Hub](https://huggingface.co/jiuntian/interactdiffusion-xl-1024/) |

## Other Dependent Models 

1. Download the SD-v1-4.ckpt from [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt).
2. Optional: Download SD v1.5 at [here](https://huggingface.co/runwayml/stable-diffusion-v1-5)
3. Download the GLIGEN model from [huggingface](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin)
4. Download the CLIP from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
5. Download the Llama from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)   [Optional]

## Extension for AutomaticA111's Stable Diffusion WebUI

We develop an AutomaticA111's Stable Diffuion WebUI extension to allow the use of CEIDM over existing SD models. Check out the plugin at [sd-webui-interactdiffusion](https://github.com/jiuntian/sd-webui-interactdiffusion). Note that it is still on `alpha` version.

## Diffusers

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

images = pipeline(
    prompt="a person is blowing a cake",
    interactdiffusion_subject_phrases=["person"],
    interactdiffusion_object_phrases=["cake"],
    interactdiffusion_action_phrases=["blowing"],
    interactdiffusion_subject_boxes=[[0.5399, 0.0297, 0.9225, 0.9781]],
    interactdiffusion_object_boxes=[[0.1526, 0.7426, 0.7840, 0.9797]],
    interactdiffusion_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    ).images

images[0].save('../output.png')
```

## Reproduce & Evaluate

1. Change `ckpt.pth` in interence_batch.py to selected model checkpoint.
2. Made inference on CEIDM to synthesis the test set of HICO-DET based on the ground truth.

      ```bash
      python inference_batch.py --batch_size 1 --folder generated_output --seed 489 --scheduled-sampling 1.0 --half
      ```
  
3. Setup FGAHOI at `../FGAHOI`. See [FGAHOI repo](https://github.com/xiaomabufei/FGAHOI) on how to setup FGAHOI and also HICO-DET dataset in `data/hico_20160224_det`.
4. Prepare for evaluate on FGAHOI. See `id_prepare_inference.ipynb`
5. Evaluate on FGAHOI.

      ```bash
      python main.py --backbone swin_tiny --dataset_file hico --resume weights/FGAHOI_Tiny.pth --num_verb_classes 117 --num_obj_classes 80 --output_dir logs  --merge --hierarchical_merge --task_merge --eval --hoi_path data/id_generated_output --pretrain_model_path "" --output_dir logs/id-generated-output-t
      ```

6. Evaluate for FID and KID. We recommend to resize hico_det dataset to 512x512 before perform image quality evaluation, for a fair comparison. We use [torch-fidelity](https://github.com/toshas/torch-fidelity).

      ```bash
      fidelity --gpu 0 --fid --isc --kid --input2 ~/data/hico_det_test_resize  --input1 ~/FGAHOI/data/data/id_generated_output/images/test2015
      ```

## Acknowledgement

This work is developed based on the codebase of [GLIGEN](https://github.com/gligen/GLIGEN) and [LDM](https://github.com/CompVis/latent-diffusion) and [InteractDiffusion](https://github.com/jiuntian/interactdiffusion).
