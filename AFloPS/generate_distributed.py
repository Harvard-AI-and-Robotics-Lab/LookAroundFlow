import argparse
import os
import torch
import torch.distributed as dist
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler, UniPCMultistepScheduler
from tqdm import tqdm
from utils import AFloPS_DIT

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device

SAMPLER_MAP = {
    "ddim": DDIMScheduler,
    "dpmsolver": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
    "aflops": AFloPS_DIT,
}

def main():
    rank, world_size, device = setup_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler", type=str, default="aflops", choices=list(SAMPLER_MAP.keys()),
        help="sampler"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./images",
        help="save_dir"
    )

    parser.add_argument(
        "--num_inference_steps", type=int, default=5,
        help="num_inference_steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.5,
        help="cfg"
    )
    args = parser.parse_args()


    save_path = os.path.join(args.save_dir, f"{args.sampler}_step{args.num_inference_steps}_scale{args.guidance_scale}")
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        print(f"{world_size} GPU")
        print(f"sampler: {args.sampler}")
        print(f"num_inference_steps: {args.num_inference_steps}")
        print(f"guidance_scale: {args.guidance_scale}")
        print(f"save to '{save_path}'")


    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)

    current_config = dict(pipe.scheduler.config)
    current_config.update({
        "c_max": 1.0,
        "mode": 1
    })
    
    if args.sampler in SAMPLER_MAP:
        pipe.scheduler = SAMPLER_MAP[args.sampler].from_config(current_config)
    else:
        raise ValueError(f"Unsupported sampler: {args.sampler}")
    pipe = pipe.to(device)

    if rank != 0:
        pipe.set_progress_bar_config(disable=True)



    num_classes = 1000
    images_per_class = 50
    total_images = num_classes * images_per_class
    guidance_scale = args.guidance_scale


    classes_per_process = num_classes // world_size
    start_class = rank * classes_per_process
    end_class = (rank + 1) * classes_per_process if rank != world_size - 1 else num_classes
    
    if rank == world_size - 1:
        end_class = num_classes

    dist.barrier()
    

    with torch.no_grad():

        for class_label in tqdm(range(start_class, end_class),
                                desc=f"GPU {rank} (class {start_class}-{end_class-1})",
                                position=rank):

            class_dir = os.path.join(save_path, str(class_label))
            os.makedirs(class_dir, exist_ok=True)


            generator = torch.manual_seed(42 + class_label) 
            

            images = pipe(
                class_labels=[class_label] * images_per_class,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale
            ).images
            

            for i in range(images_per_class):
                image_path = os.path.join(class_dir, f"sample_{i+1}.png")
                images[i].save(image_path)

    dist.barrier()



if __name__ == "__main__":
    main()