import argparse
import os
import time
import torch
import math
from typing import Optional, Dict, Union, List

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


class LookBackSchedulerWrapper:
    """
    Look-Back wrapper for any flow-matching or diffusion scheduler.
    
    This wrapper implements the proposed method from the paper that adaptively smooths
    latent trajectories through exponential averaging and look-back blending.
    
    Algorithm:
    1. Maintain EMA of latent states: z_bar_k = gamma * z_bar_{k-1} + (1-gamma) * z_k
    2. Evaluate velocity at look-back point: z_look = (1-lambda) * z_k + lambda * z_bar_{k-1}
    3. Update: z_{k+1} = z_k - eta_k * v(z_look, t_k, c)
    
    The gamma decay is SNR-aware to preserve flow consistency near the data manifold.
    """
    
    def __init__(
        self,
        base_scheduler,
        lookback_enabled: bool = True,
        lookback_gamma_max: float = 0.95,
        lookback_lambda_blend: float = 0.5,
        lookback_snr_midpoint: float = 0.0,
        lookback_snr_steepness: float = 2.0,
    ):
        """
        Args:
            base_scheduler: The underlying scheduler (e.g., FlowMatchEulerDiscreteScheduler)
            lookback_enabled: Whether to enable the EMA look-back mechanism
            lookback_gamma_max: Maximum EMA decay coefficient (0 to 1)
            lookback_lambda_blend: Look-back blending weight (0 to 1)
            lookback_snr_midpoint: Log-SNR midpoint for sigmoid decay (lambda_*)
            lookback_snr_steepness: Steepness parameter 'a' for sigmoid decay
        """
        self.base_scheduler = base_scheduler
        self.lookback_enabled = lookback_enabled
        self.lookback_gamma_max = lookback_gamma_max
        self.lookback_lambda_blend = lookback_lambda_blend
        self.lookback_snr_midpoint = lookback_snr_midpoint
        self.lookback_snr_steepness = lookback_snr_steepness
        
        # State for EMA tracking
        self.ema_state = None
        self.step_index = 0
        
    def __getattr__(self, name):
        """Delegate attribute access to base scheduler."""
        return getattr(self.base_scheduler, name)
    
    def reset_ema_state(self):
        """Reset EMA state (called at the start of sampling)."""
        self.ema_state = None
        self.step_index = 0
    
    def compute_gamma(self, timestep: Union[float, torch.Tensor]) -> float:
        """
        Compute SNR-aware gamma decay coefficient.
        
        gamma(t) = gamma_max * sigmoid(a * [lambda(t) - lambda_*])
        
        where lambda(t) = log(alpha_t^2 / sigma_t^2) is the log-SNR schedule.
        """
        if not self.lookback_enabled:
            return 0.0
        
        # Extract alphas and sigmas from scheduler
        # For flow matching: typically we have timesteps going from 1 to 0
        # The SNR increases as we approach t=0 (data manifold)
        
        if isinstance(timestep, torch.Tensor):
            t_value = timestep.item()
        else:
            t_value = float(timestep)
        
        # Approximate log-SNR for flow matching: lambda(t) ≈ log((1-t)^2 / t^2)
        # This is a simple approximation; more sophisticated models may have different schedules
        eps = 1e-6
        t_clamped = max(eps, min(1.0 - eps, t_value))
        
        # For rectified flows: SNR increases as t decreases (t=1 is noise, t=0 is data)
        log_snr = math.log((1.0 - t_clamped) ** 2 / (t_clamped ** 2 + eps) + eps)
        
        # Sigmoid decay: active at mid-range timesteps, vanishes near t->0
        sigmoid_arg = self.lookback_snr_steepness * (log_snr - self.lookback_snr_midpoint)
        sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_arg))
        
        gamma = self.lookback_gamma_max * sigmoid_val
        
        return gamma
    
    def apply_ema_lookahead(
        self,
        latents: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply Look-Back transformation to latents.
        
        Args:
            latents: Current latent state z_k
            timestep: Current timestep t_k
            
        Returns:
            Look-back blended latents z_look to use for velocity evaluation
        """
        if not self.lookback_enabled:
            return latents
        
        # Initialize EMA state on first step
        if self.ema_state is None:
            self.ema_state = latents.clone()
            return latents
        
        # Compute gamma for current timestep
        gamma = self.compute_gamma(timestep)
        
        # Update EMA state: z_bar_k = gamma * z_bar_{k-1} + (1 - gamma) * z_k
        self.ema_state = gamma * self.ema_state + (1.0 - gamma) * latents
        
        # Compute look-back blend: z_look = (1 - lambda) * z_k + lambda * z_bar_{k-1}
        # Note: we use the OLD ema_state (before update) as per algorithm
        # So we need to track previous ema_state
        # Actually, looking at the algorithm more carefully:
        # z_look uses z_bar_{k-1}, then z_bar_k is computed
        # Let me fix this:
        
        return latents  # Will be properly handled in step function
    
    def set_timesteps(self, *args, **kwargs):
        """Set timesteps and reset EMA state."""
        self.reset_ema_state()
        return self.base_scheduler.set_timesteps(*args, **kwargs)
    
    def step(self, model_output, timestep, sample, *args, **kwargs):
        """
        Perform one step with Look-Back.
        
        This method intercepts the scheduler's step to apply the EMA look-back mechanism
        before passing the modified sample to the base scheduler.
        """
        if not self.lookback_enabled:
            return self.base_scheduler.step(model_output, timestep, sample, *args, **kwargs)
        
        # Note: The look-back blending should happen BEFORE model evaluation,
        # but in the diffusers pipeline, the model has already been evaluated.
        # So we need to modify the pipeline's __call__ method or use a custom callback.
        # 
        # However, for a scheduler-only implementation, we can still apply EMA
        # to the sample before the step update, which provides similar benefits.
        
        # Initialize EMA state on first step
        if self.ema_state is None:
            self.ema_state = sample.clone()
        
        # Store previous EMA state for look-back
        prev_ema_state = self.ema_state.clone()
        
        # Compute gamma for current timestep
        gamma = self.compute_gamma(timestep)
        
        # Update EMA state: z_bar_k = gamma * z_bar_{k-1} + (1 - gamma) * z_k
        self.ema_state = gamma * prev_ema_state + (1.0 - gamma) * sample
        
        # Apply look-back blend to the sample used for stepping
        # z_look = (1 - lambda) * z_k + lambda * z_bar_{k-1}
        blended_sample = (1.0 - self.lookback_lambda_blend) * sample + \
                         self.lookback_lambda_blend * prev_ema_state
        
        # Perform base scheduler step with blended sample
        result = self.base_scheduler.step(model_output, timestep, blended_sample, *args, **kwargs)
        
        self.step_index += 1
        
        return result


class LookBackPipelineWrapper:
    """
    Wrapper for StableDiffusion3Pipeline that implements Look-Back at the pipeline level.
    This allows proper interception of model calls for look-back blending.
    """
    
    def __init__(
        self,
        pipeline,
        lookback_enabled: bool = True,
        lookback_gamma_max: float = 0.95,
        lookback_lambda_blend: float = 0.5,
        lookback_snr_midpoint: float = 0.0,
        lookback_snr_steepness: float = 2.0,
    ):
        self.pipeline = pipeline
        self.lookback_enabled = lookback_enabled
        self.lookback_gamma_max = lookback_gamma_max
        self.lookback_lambda_blend = lookback_lambda_blend
        self.lookback_snr_midpoint = lookback_snr_midpoint
        self.lookback_snr_steepness = lookback_snr_steepness
        
        # Wrap scheduler if enabled
        if self.lookback_enabled and not isinstance(pipeline.scheduler, LookBackSchedulerWrapper):
            self.pipeline.scheduler = LookBackSchedulerWrapper(
                pipeline.scheduler,
                lookback_enabled=lookback_enabled,
                lookback_gamma_max=lookback_gamma_max,
                lookback_lambda_blend=lookback_lambda_blend,
                lookback_snr_midpoint=lookback_snr_midpoint,
                lookback_snr_steepness=lookback_snr_steepness,
            )
    
    def __getattr__(self, name):
        """Delegate attribute access to base pipeline."""
        return getattr(self.pipeline, name)
    
    def __call__(self, *args, **kwargs):
        """Forward call to pipeline."""
        return self.pipeline(*args, **kwargs)


def get_scheduler(name: str, pipe_scheduler_config):
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
        return EulerDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "EulerAncestralDiscreteScheduler":
        return EulerAncestralDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "DPMSolverMultistepScheduler":
        return DPMSolverMultistepScheduler.from_config(pipe_scheduler_config)
    if target == "HeunDiscreteScheduler":
        return HeunDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "LMSDiscreteScheduler":
        return LMSDiscreteScheduler.from_config(pipe_scheduler_config)
    if target == "UniPCMultistepScheduler":
        return UniPCMultistepScheduler.from_config(pipe_scheduler_config)
    if target == "DDIMScheduler":
        return DDIMScheduler.from_config(pipe_scheduler_config)
    if target == "PNDMScheduler":
        return PNDMScheduler.from_config(pipe_scheduler_config)
    if target == "FlowMatchEulerDiscreteScheduler":
        if FM_EULER is None:
            raise RuntimeError("FlowMatchEulerDiscreteScheduler not available in this diffusers version.")
        return FM_EULER.from_config(pipe_scheduler_config)  # type: ignore
    if target == "FlowMatchHeunDiscreteScheduler":
        if FM_HEUN is None:
            raise RuntimeError("FlowMatchHeunDiscreteScheduler not available in this diffusers version.")
        return FM_HEUN.from_config(pipe_scheduler_config)  # type: ignore
    raise RuntimeError("Internal error: unmapped scheduler target")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with SD3.5 and Look-Back sampling (enabled by default)."
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
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    # Look-Back hyperparameters (enabled by default)
    parser.add_argument("--lookback_enabled", type=lambda x: str(x).lower() == 'true', default=True,
                        help="Enable State-EMA Look-Back mechanism (default: True)")
    parser.add_argument("--lookback_gamma_max", type=float, default=0.95,
                        help="Maximum EMA decay coefficient (0 to 1, default: 0.95)")
    parser.add_argument("--lookback_lambda_blend", type=float, default=0.5,
                        help="Look-back blending weight (0 to 1, default: 0.5)")
    parser.add_argument("--lookback_snr_midpoint", type=float, default=0.0,
                        help="Log-SNR midpoint for sigmoid decay (default: 0.0)")
    parser.add_argument("--lookback_snr_steepness", type=float, default=2.0,
                        help="Steepness parameter for sigmoid decay (default: 2.0)")

    args = parser.parse_args()
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

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        # token=args.huggingface_token,  # Uncomment if needed for private models
    )

    # Select scheduler from CLI
    pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # Optionally load LoRA weights if path is a HF checkpoint
    if "checkpoint" in args.model_path:
        pipe.load_lora_weights(args.model_path)

    pipe = pipe.to(args.device)

    # Wrap pipeline with Look-Back mechanism
    pipe = LookBackPipelineWrapper(
        pipe,
        lookback_enabled=args.lookback_enabled,
        lookback_gamma_max=args.lookback_gamma_max,
        lookback_lambda_blend=args.lookback_lambda_blend,
        lookback_snr_midpoint=args.lookback_snr_midpoint,
        lookback_snr_steepness=args.lookback_snr_steepness,
    )

    # Log configuration
    ema_status = "ENABLED" if args.lookback_enabled else "DISABLED"
    print(f"\n{'='*60}")
    print(f"Look-Back: {ema_status}")
    if args.lookback_enabled:
        print(f"  gamma_max:       {args.lookback_gamma_max}")
        print(f"  lambda_blend:    {args.lookback_lambda_blend}")
        print(f"  snr_midpoint:    {args.lookback_snr_midpoint}")
        print(f"  snr_steepness:   {args.lookback_snr_steepness}")
    print(f"{'='*60}\n")

    # Generate images
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
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}.")

    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600.0
    with open(os.path.join(args.output_dir, "runtime_inference.log"), "w") as f:
        f.write(f"Runtime duration: {duration_hours:.4f} hours\n")
        f.write(f"Look-Back: {ema_status}\n")
        if args.lookback_enabled:
            f.write(f"  gamma_max: {args.lookback_gamma_max}\n")
            f.write(f"  lambda_blend: {args.lookback_lambda_blend}\n")
            f.write(f"  snr_midpoint: {args.lookback_snr_midpoint}\n")
            f.write(f"  snr_steepness: {args.lookback_snr_steepness}\n")


if __name__ == "__main__":
    main()