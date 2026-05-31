import argparse
import os
import time
import torch
from typing import Optional, Dict, Union, Tuple

from diffusers import (
    StableDiffusion3Pipeline,
    # classic diffusion-family schedulers
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch.utils.data import Dataset, DataLoader

# --- Try to import Flow Matching schedulers (available in newer diffusers) ---
FM_EULER = None
FM_HEUN = None
try:
    from diffusers import FlowMatchEulerDiscreteScheduler as _FM_EULER_CLS  # type: ignore
    FM_EULER = _FM_EULER_CLS
except Exception:
    FM_EULER = None

try:
    from diffusers import FlowMatchHeunDiscreteScheduler as _FM_HEUN_CLS  # type: ignore
    FM_HEUN = _FM_HEUN_CLS
except Exception:
    FM_HEUN = None


def _available(cls) -> bool:
    return cls is not None


class MomentumAugmentedSchedulerWrapper:
    """
    Wrapper that adds First-Order Momentum (FAO) to any flow-matching scheduler.
    
    This implements the momentum-augmented ODE sampling from the paper:
    - Maintains per-trajectory momentum vector m_k
    - Updates using exponential moving average: m_{k+1} = beta_1 * m_k + (1-beta_1) * g_k
    - Where g_k = -v_Theta(z_k, t_k, c) is the negative velocity
    - Final update: z_{k+1} = z_k + eta_k * m_{k+1}
    
    Hyperparameters:
    - fao_beta1: momentum coefficient (default: 0.9)
    - fao_enabled: whether to use momentum (default: True)
    """
    
    def __init__(
        self,
        base_scheduler: SchedulerMixin,
        fao_beta1: float = 0.9,
        fao_enabled: bool = True,
    ):
        """
        Args:
            base_scheduler: The underlying scheduler (e.g., FlowMatchEulerDiscreteScheduler)
            fao_beta1: Momentum coefficient in [0, 1). Higher = more momentum smoothing
            fao_enabled: Whether to enable momentum augmentation
        """
        self.base_scheduler = base_scheduler
        self.fao_beta1 = fao_beta1
        self.fao_enabled = fao_enabled
        
        # Momentum state: will be initialized on first step
        self.momentum_state = None
        self._step_count = 0
        
        # Expose base scheduler attributes
        self.config = base_scheduler.config
        self.timesteps = base_scheduler.timesteps
        
    def set_timesteps(self, *args, **kwargs):
        """Delegate to base scheduler and reset momentum state"""
        self.base_scheduler.set_timesteps(*args, **kwargs)
        self.timesteps = self.base_scheduler.timesteps
        self.momentum_state = None
        self._step_count = 0
        
    def scale_model_input(self, *args, **kwargs):
        """Delegate to base scheduler"""
        return self.base_scheduler.scale_model_input(*args, **kwargs)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[int, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Perform one step of momentum-augmented sampling.
        
        Algorithm:
        1. Get velocity from model: v_Theta(z_k, t_k, c) = model_output
        2. Compute gradient: g_k = -v_Theta (negative velocity)
        3. Update momentum: m_{k+1} = beta_1 * m_k + (1-beta_1) * g_k
        4. Update sample: z_{k+1} = z_k + eta_k * m_{k+1}
        
        Args:
            model_output: The direct output from learned velocity field v_Theta
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created by diffusion process (z_k)
            return_dict: Whether to return a SchedulerOutput object
            
        Returns:
            Updated sample (z_{k+1}) and potentially other outputs
        """
        if not self.fao_enabled or self.fao_beta1 == 0.0:
            # Fallback to vanilla scheduler when momentum is disabled
            return self.base_scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
                **kwargs
            )
        
        # Initialize momentum state on first step
        if self.momentum_state is None:
            self.momentum_state = torch.zeros_like(sample)
            self._step_count = 0
        
        # Step 1: g_k = -v_Theta(z_k, t_k, c)
        # The model_output is the velocity field v_Theta, so we negate it
        g_k = -model_output
        
        # Step 2: m_{k+1} = beta_1 * m_k + (1 - beta_1) * g_k
        self.momentum_state = (
            self.fao_beta1 * self.momentum_state + 
            (1.0 - self.fao_beta1) * g_k
        )
        
        # Step 3: Use momentum as the effective model output
        # We need to negate back because the scheduler expects velocity
        momentum_velocity = -self.momentum_state
        
        # Step 4: Call base scheduler with momentum-smoothed velocity
        output = self.base_scheduler.step(
            model_output=momentum_velocity,
            timestep=timestep,
            sample=sample,
            return_dict=return_dict,
            **kwargs
        )
        
        self._step_count += 1
        
        return output
    
    def __getattr__(self, name: str):
        """Delegate any other attributes to base scheduler"""
        return getattr(self.base_scheduler, name)


def _collect_scheduler_aliases() -> Dict[str, str]:
    """Build alias map based on what's importable in this environment."""
    aliases = {
        # diffusion-family (score-matching) schedulers
        "euler": "EulerDiscreteScheduler",
        "euler_discrete": "EulerDiscreteScheduler",
        "euler_ancestral": "EulerAncestralDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpmpp": "DPMSolverMultistepScheduler",
        "dpmpp_2m": "DPMSolverMultistepScheduler",
        "dpmsolver": "DPMSolverMultistepScheduler",
        "heun": "HeunDiscreteScheduler",
        "lms": "LMSDiscreteScheduler",
        "unipc": "UniPCMultistepScheduler",
        "ddim": "DDIMScheduler",
        "pndm": "PNDMScheduler",
    }
    # flow-matching schedulers (added only if installed)
    if _available(FM_EULER):
        aliases.update({
            "flowmatch_euler": "FlowMatchEulerDiscreteScheduler",
            "fm_euler": "FlowMatchEulerDiscreteScheduler",
            "flow_euler": "FlowMatchEulerDiscreteScheduler",
            "flowmatch": "FlowMatchEulerDiscreteScheduler",  # convenient default
        })
    if _available(FM_HEUN):
        aliases.update({
            "flowmatch_heun": "FlowMatchHeunDiscreteScheduler",
            "fm_heun": "FlowMatchHeunDiscreteScheduler",
            "flow_heun": "FlowMatchHeunDiscreteScheduler",
        })
    return aliases


SCHEDULER_ALIASES = _collect_scheduler_aliases()


def get_scheduler(
    name: str,
    pipe_scheduler_config,
    fao_beta1: float = 0.9,
    fao_enabled: bool = True,
):
    """
    Get scheduler with optional momentum augmentation (FAO).
    
    Args:
        name: Scheduler name/alias
        pipe_scheduler_config: Configuration from pipeline
        fao_beta1: Momentum coefficient for FAO
        fao_enabled: Whether to enable FAO momentum
        
    Returns:
        Scheduler instance, optionally wrapped with MomentumAugmentedSchedulerWrapper
    """
    name = (name or "").lower()
    target = SCHEDULER_ALIASES.get(name)
    if target is None:
        fm_options = [k for k in SCHEDULER_ALIASES.keys() if k.startswith("flow")]
        all_opts = ", ".join(sorted(SCHEDULER_ALIASES.keys()))
        msg = (f"Unknown --scheduler '{name}'. Choose one of: {all_opts}."
               f"{' (Flow-matching options: ' + ', '.join(sorted(fm_options)) + ')' if fm_options else ''}")
        raise ValueError(msg)

    # Map class name to actual constructor
    if target == "EulerDiscreteScheduler":
        base = EulerDiscreteScheduler.from_config(pipe_scheduler_config)
    elif target == "EulerAncestralDiscreteScheduler":
        base = EulerAncestralDiscreteScheduler.from_config(pipe_scheduler_config)
    elif target == "DPMSolverMultistepScheduler":
        base = DPMSolverMultistepScheduler.from_config(pipe_scheduler_config)
    elif target == "HeunDiscreteScheduler":
        base = HeunDiscreteScheduler.from_config(pipe_scheduler_config)
    elif target == "LMSDiscreteScheduler":
        base = LMSDiscreteScheduler.from_config(pipe_scheduler_config)
    elif target == "UniPCMultistepScheduler":
        base = UniPCMultistepScheduler.from_config(pipe_scheduler_config)
    elif target == "DDIMScheduler":
        base = DDIMScheduler.from_config(pipe_scheduler_config)
    elif target == "PNDMScheduler":
        base = PNDMScheduler.from_config(pipe_scheduler_config)
    elif target == "FlowMatchEulerDiscreteScheduler":
        if FM_EULER is None:
            raise RuntimeError("FlowMatchEulerDiscreteScheduler not available in this diffusers version.")
        base = FM_EULER.from_config(pipe_scheduler_config)  # type: ignore
    elif target == "FlowMatchHeunDiscreteScheduler":
        if FM_HEUN is None:
            raise RuntimeError("FlowMatchHeunDiscreteScheduler not available in this diffusers version.")
        base = FM_HEUN.from_config(pipe_scheduler_config)  # type: ignore
    else:
        raise RuntimeError("Internal error: unmapped scheduler target")
    
    # Wrap with momentum augmentation if enabled
    # Only apply to flow-matching schedulers by default, but can be applied to others
    is_flow_scheduler = "FlowMatch" in target
    
    if fao_enabled and is_flow_scheduler:
        print(f"🚀 Enabling FAO momentum augmentation (beta1={fao_beta1}) for {target}")
        return MomentumAugmentedSchedulerWrapper(
            base_scheduler=base,
            fao_beta1=fao_beta1,
            fao_enabled=True,
        )
    
    return base


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with SD3.5 and momentum-augmented flow matching (FAO method)."
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="Hugging Face token")

    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing .txt prompts")
    parser.add_argument("--split", type=str, default="test", help="train | val | validation | dev | test")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to LoRA or checkpoints (if contains 'checkpoint', load as LoRA)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output images")

    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale (CFG)")
    parser.add_argument("--image_width", type=int, default=512, help="Output width")
    parser.add_argument("--image_height", type=int, default=512, help="Output height")

    parser.add_argument("--num_prompts_per_run", type=int, default=1,
                        help="Number of prompts to process at a time from each batch")
    parser.add_argument("--reverse_mode", action="store_true",
                        help="Reverse order of prompts (default: False)")

    # Default to flowmatch_euler if available, else fall back to the model's default-like choice
    default_sched = "flowmatch_euler" if "flowmatch_euler" in SCHEDULER_ALIASES else "dpmpp"
    parser.add_argument("--scheduler", type=str, default=default_sched,
                        help=("Scheduler to use. Available: " + ", ".join(sorted(SCHEDULER_ALIASES.keys()))))
    
    # FAO (First-order Adaptive Optimizer) hyperparameters
    parser.add_argument("--fao_enabled", action="store_true", default=True,
                        help="Enable momentum-augmented sampling (FAO method). Default: True")
    parser.add_argument("--fao_disable", dest="fao_enabled", action="store_false",
                        help="Disable momentum-augmented sampling (use vanilla scheduler)")
    parser.add_argument("--fao_beta1", type=float, default=0.9,
                        help="Momentum coefficient for FAO. Higher = more smoothing. Range: [0, 1). Default: 0.9")
    
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()
    
    # Validate FAO parameters
    if args.fao_beta1 < 0.0 or args.fao_beta1 >= 1.0:
        raise ValueError(f"fao_beta1 must be in [0, 1), got {args.fao_beta1}")
    
    return args


class PromptDataset(Dataset):
    def __init__(self, data_folder: str, reverse_mode: bool = False):
        super().__init__()
        self.data_folder = data_folder
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".txt")]
        if reverse_mode:
            self.file_paths = self.file_paths[::-1]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "r") as fh:
            content = fh.read().strip()
        file_idx = os.path.basename(file_path).split(".")[0]
        return content, file_idx


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()

    # Infer split from prompt_dir path if present
    for split in ["train", "val", "validation", "dev", "test"]:
        if split in args.prompt_dir:
            args.split = split
            break

    set_seed(args.seed)
    start_time = time.time()

    dataset = PromptDataset(args.prompt_dir, reverse_mode=args.reverse_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Compose final output directory
    args.output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Print FAO configuration
    print("\n" + "="*80)
    print("MOMENTUM-AUGMENTED FLOW MATCHING (FAO) CONFIGURATION")
    print("="*80)
    print(f"FAO Enabled: {args.fao_enabled}")
    if args.fao_enabled:
        print(f"FAO Beta1 (momentum coefficient): {args.fao_beta1}")
        print(f"Scheduler: {args.scheduler}")
        print(f"\nThis implements the momentum-augmented ODE sampling method:")
        print(f"  - Per-trajectory first-order momentum")
        print(f"  - Exponential moving average: m_{{k+1}} = β₁·m_k + (1-β₁)·g_k")
        print(f"  - Training-free, negligible overhead")
        print(f"  - When β₁=0, reduces to vanilla Euler solver")
    else:
        print(f"Using vanilla scheduler (no momentum augmentation)")
    print("="*80 + "\n")

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        # token=args.huggingface_token,  # Uncomment if needed for private models
    )

    # Select scheduler from CLI with FAO wrapper
    pipe.scheduler = get_scheduler(
        args.scheduler,
        pipe.scheduler.config,
        fao_beta1=args.fao_beta1,
        fao_enabled=args.fao_enabled,
    )

    # Optionally load LoRA weights if path is a HF checkpoint
    if "checkpoint" in args.model_path:
        pipe.load_lora_weights(args.model_path)

    pipe = pipe.to(args.device)

    # Generate images
    total_images = 0
    for batch in dataloader:
        batch_prompts, batch_prompt_ids = batch
        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts, batch_prompt_ids):
                out_path = f"{args.output_dir}/{_id}.png"
                if os.path.exists(out_path):
                    print(f"Skipping {out_path}")
                    continue
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                )
                image = result.images[0]
                image.save(out_path)
                total_images += 1
                print(f"Generated: {out_path}")

        elif args.num_prompts_per_run > 1:
            prompts_per_run = [batch_prompts[i:i + args.num_prompts_per_run]
                               for i in range(0, len(batch_prompts), args.num_prompts_per_run)]
            prompt_ids_per_run = [batch_prompt_ids[i:i + args.num_prompts_per_run]
                                  for i in range(0, len(batch_prompt_ids), args.num_prompts_per_run)]

            for prompts, prompt_ids in zip(prompts_per_run, prompt_ids_per_run):
                paths = [f"{args.output_dir}/{_id}.png" for _id in prompt_ids]
                if all(os.path.exists(p) for p in paths):
                    print(f"Skipping {paths}")
                    continue
                result = pipe(
                    prompt=list(prompts),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                )
                for image, _id in zip(result.images, prompt_ids):
                    image.save(f"{args.output_dir}/{_id}.png")
                    total_images += 1
                    print(f"Generated: {args.output_dir}/{_id}.png")
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}.")

    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600.0
    
    # Write runtime log
    log_path = os.path.join(args.output_dir, "runtime_inference.log")
    with open(log_path, "w") as f:
        f.write(f"Runtime duration: {duration_hours:.4f} hours\n")
        f.write(f"Total images generated: {total_images}\n")
        f.write(f"FAO enabled: {args.fao_enabled}\n")
        if args.fao_enabled:
            f.write(f"FAO beta1: {args.fao_beta1}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Steps: {args.num_inference_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Completed! Generated {total_images} images in {duration_hours:.4f} hours")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()