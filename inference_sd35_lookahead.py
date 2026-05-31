import argparse
import os
import time
import torch
from typing import Optional, Dict, Callable
import numpy as np

from diffusers import (
    StableDiffusion3Pipeline,
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

# Try to import Flow Matching schedulers
FM_EULER = None
FM_HEUN = None
try:
    from diffusers import FlowMatchEulerDiscreteScheduler as _FM_EULER_CLS
    FM_EULER = _FM_EULER_CLS
except Exception:
    FM_EULER = None

try:
    from diffusers import FlowMatchHeunDiscreteScheduler as _FM_HEUN_CLS
    FM_HEUN = _FM_HEUN_CLS
except Exception:
    FM_HEUN = None


def _available(cls) -> bool:
    return cls is not None


def _collect_scheduler_aliases() -> Dict[str, str]:
    """Build alias map based on what's importable in this environment."""
    aliases = {
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
    if _available(FM_EULER):
        aliases.update({
            "flowmatch_euler": "FlowMatchEulerDiscreteScheduler",
            "fm_euler": "FlowMatchEulerDiscreteScheduler",
            "flow_euler": "FlowMatchEulerDiscreteScheduler",
            "flowmatch": "FlowMatchEulerDiscreteScheduler",
        })
    if _available(FM_HEUN):
        aliases.update({
            "flowmatch_heun": "FlowMatchHeunDiscreteScheduler",
            "fm_heun": "FlowMatchHeunDiscreteScheduler",
            "flow_heun": "FlowMatchHeunDiscreteScheduler",
        })
    return aliases


SCHEDULER_ALIASES = _collect_scheduler_aliases()


def get_scheduler(name: str, pipe_scheduler_config):
    name = (name or "").lower()
    target = SCHEDULER_ALIASES.get(name)
    if target is None:
        fm_options = [k for k in SCHEDULER_ALIASES.keys() if k.startswith("flow")]
        all_opts = ", ".join(sorted(SCHEDULER_ALIASES.keys()))
        msg = (f"Unknown --scheduler '{name}'. Choose one of: {all_opts}."
               f"{' (Flow-matching options: ' + ', '.join(sorted(fm_options)) + ')' if fm_options else ''}")
        raise ValueError(msg)

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
        return FM_EULER.from_config(pipe_scheduler_config)
    if target == "FlowMatchHeunDiscreteScheduler":
        if FM_HEUN is None:
            raise RuntimeError("FlowMatchHeunDiscreteScheduler not available in this diffusers version.")
        return FM_HEUN.from_config(pipe_scheduler_config)
    raise RuntimeError("Internal error: unmapped scheduler target")


class PCLookAheadSchedulerWrapper:
    """
    PC-LookAhead: Predictor-Corrector with Curvature Gate
    
    This wrapper intercepts the scheduler's step() method to add curvature-based adaptive stepping.
    """
    
    def __init__(
        self,
        scheduler,
        lookahead_enabled: bool = True,
        lookahead_curv_threshold: float = 1.0,
        lookahead_backtrack_factor: float = 0.5,
        lookahead_max_backtracks: int = 3,
        lookahead_epsilon: float = 1e-8,
        verbose: bool = False,
    ):
        self.scheduler = scheduler
        self.lookahead_enabled = lookahead_enabled
        self.lookahead_curv_threshold = lookahead_curv_threshold
        self.lookahead_backtrack_factor = lookahead_backtrack_factor
        self.lookahead_max_backtracks = lookahead_max_backtracks
        self.lookahead_epsilon = lookahead_epsilon
        self.verbose = verbose
        
        # Store original step method
        self._original_step = scheduler.step
        
        # Replace with our wrapper
        scheduler.step = self.step
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'total_backtracks': 0,
            'rejected_steps': 0,
            'accepted_first_try': 0,
            'kappa_values': [],
        }
    
    def __getattr__(self, name):
        """Delegate attribute access to the base scheduler."""
        return getattr(self.scheduler, name)
    
    def _compute_curvature(self, v_k, v_tilde, z_k, z_tilde):
        """
        Compute curvature proxy:
        κ_k = ||ṽ - v_k||₂ / (||z̃ - z_k||₂ + ε)
        """
        v_k_flat = v_k.flatten()
        v_tilde_flat = v_tilde.flatten()
        z_k_flat = z_k.flatten()
        z_tilde_flat = z_tilde.flatten()
        
        numerator = torch.norm(v_tilde_flat - v_k_flat, p=2)
        denominator = torch.norm(z_tilde_flat - z_k_flat, p=2) + self.lookahead_epsilon
        kappa = (numerator / denominator).item()
        
        return kappa
    
    def step(self, model_output, timestep, sample, *args, **kwargs):
        """
        Wrapped step method with PC-LookAhead logic.
        """
        if not self.lookahead_enabled:
            return self._original_step(model_output, timestep, sample, *args, **kwargs)
        
        # Check if we're at the last step to avoid index errors
        if hasattr(self.scheduler, 'step_index'):
            if self.scheduler.step_index is None:
                return self._original_step(model_output, timestep, sample, *args, **kwargs)
            if hasattr(self.scheduler, 'sigmas') and self.scheduler.step_index >= len(self.scheduler.sigmas) - 1:
                return self._original_step(model_output, timestep, sample, *args, **kwargs)
        
        # Current velocity v_k and latent z_k
        v_k = model_output
        z_k = sample
        
        # Take the original step to get z_tilde (predictor)
        result = self._original_step(model_output, timestep, sample, *args, **kwargs)
        
        # Extract z_tilde from result
        if isinstance(result, tuple):
            z_tilde = result[0]
        elif hasattr(result, 'prev_sample'):
            z_tilde = result.prev_sample
        else:
            z_tilde = result
        
        # Estimate v_tilde from the step we just took
        # v_tilde ≈ (z_k - z_tilde) / Δt
        # We'll use the displacement as a proxy for velocity change
        
        # Get timestep info
        if torch.is_tensor(timestep):
            timestep_val = timestep.item()
        else:
            timestep_val = float(timestep)
        
        # Find next timestep
        step_index = None
        for i, t in enumerate(self.scheduler.timesteps):
            if abs(t.item() - timestep_val) < 1e-5:
                step_index = i
                break
        
        if step_index is None or step_index >= len(self.scheduler.timesteps) - 1:
            # Can't compute curvature, just return result
            return result
        
        next_timestep = self.scheduler.timesteps[step_index + 1]
        delta_t = abs(timestep_val - next_timestep.item())
        
        # Estimate v_tilde from displacement
        if delta_t > self.lookahead_epsilon:
            v_tilde_estimated = (z_k - z_tilde) / delta_t
        else:
            v_tilde_estimated = v_k
        
        # Compute curvature
        kappa = self._compute_curvature(v_k, v_tilde_estimated, z_k, z_tilde)
        # print(kappa)
        # Update statistics
        self.stats['total_steps'] += 1
        self.stats['kappa_values'].append(kappa)
        
        # Print kappa (THIS IS WHAT YOU ASKED FOR!)
        if self.verbose:
            print(f"PC-LookAhead Step {self.stats['total_steps']}: t={timestep_val:.2f}, κ={kappa:.6f}, threshold={self.lookahead_curv_threshold:.2f}")
        
        # Acceptance test
        if kappa <= self.lookahead_curv_threshold:
            # Accept
            self.stats['accepted_first_try'] += 1
            if self.verbose:
                print(f"  ✓ ACCEPTED (κ={kappa:.6f} ≤ τ={self.lookahead_curv_threshold})")
            return result
        else:
            # Reject and take smaller step
            self.stats['rejected_steps'] += 1
            self.stats['total_backtracks'] += 1
            
            if self.verbose:
                print(f"  ✗ REJECTED (κ={kappa:.6f} > τ={self.lookahead_curv_threshold})")
                print(f"    Using backtrack factor γ={self.lookahead_backtrack_factor}")
            
            # Instead of taking another step (which would cause index issues),
            # we interpolate between current position and the full step
            # This is effectively a smaller step without calling the scheduler again
            z_backtracked = z_k + self.lookahead_backtrack_factor * (z_tilde - z_k)
            
            # Return the backtracked result in the same format
            if isinstance(result, tuple):
                return (z_backtracked,) + result[1:]
            elif hasattr(result, 'prev_sample'):
                result.prev_sample = z_backtracked
                return result
            else:
                return z_backtracked
    
    def get_statistics(self):
        """Return accumulated statistics."""
        stats = self.stats.copy()
        if stats['total_steps'] > 0:
            stats['avg_backtracks'] = stats['total_backtracks'] / stats['total_steps']
            stats['rejection_rate'] = stats['rejected_steps'] / stats['total_steps']
            stats['first_try_rate'] = stats['accepted_first_try'] / stats['total_steps']
            if stats['kappa_values']:
                stats['avg_kappa'] = np.mean(stats['kappa_values'])
                stats['max_kappa'] = np.max(stats['kappa_values'])
                stats['min_kappa'] = np.min(stats['kappa_values'])
        return stats


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with SD3.5 and PC-LookAhead"
    )
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--num_prompts_per_run", type=int, default=1)
    parser.add_argument("--reverse_mode", action="store_true")
    
    default_sched = "flowmatch_euler" if "flowmatch_euler" in SCHEDULER_ALIASES else "dpmpp"
    parser.add_argument("--scheduler", type=str, default=default_sched)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # PC-LookAhead parameters
    parser.add_argument("--disable_lookahead", action="store_true",
                        help="Disable PC-LookAhead (enabled by default)")
    parser.add_argument("--lookahead_curv_threshold", type=float, default=1.0)
    parser.add_argument("--lookahead_backtrack_factor", type=float, default=0.5)
    parser.add_argument("--lookahead_max_backtracks", type=int, default=3)
    parser.add_argument("--lookahead_epsilon", type=float, default=1e-8)
    parser.add_argument("--lookahead_verbose", action="store_true",
                        help="Print κ values and step-by-step info")

    return parser.parse_args()


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

    # Infer split from prompt_dir
    for split in ["train", "val", "validation", "dev", "test"]:
        if split in args.prompt_dir:
            args.split = split
            break

    set_seed(args.seed)
    start_time = time.time()

    dataset = PromptDataset(args.prompt_dir, reverse_mode=args.reverse_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Setup output directory
    lookahead_enabled = not args.disable_lookahead
    output_suffix = "_pc_lookahead" if lookahead_enabled else "_baseline"
    args.output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PC-LookAhead Configuration:")
    print(f"  Enabled: {lookahead_enabled}")
    if lookahead_enabled:
        print(f"  Curvature threshold (τ_curv): {args.lookahead_curv_threshold}")
        print(f"  Backtrack factor (γ): {args.lookahead_backtrack_factor}")
        print(f"  Max backtracks (B_max): {args.lookahead_max_backtracks}")
        print(f"  Epsilon (ε): {args.lookahead_epsilon}")
        print(f"  Verbose: {args.lookahead_verbose}")
    print(f"{'='*80}\n")

    # Load pipeline
    print(f"Loading pipeline: {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id)

    # Setup scheduler
    pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)
    print(f"Using scheduler: {args.scheduler}")

    # Wrap scheduler with PC-LookAhead
    if lookahead_enabled:
        wrapper = PCLookAheadSchedulerWrapper(
            pipe.scheduler,
            lookahead_enabled=True,
            lookahead_curv_threshold=args.lookahead_curv_threshold,
            lookahead_backtrack_factor=args.lookahead_backtrack_factor,
            lookahead_max_backtracks=args.lookahead_max_backtracks,
            lookahead_epsilon=args.lookahead_epsilon,
            verbose=args.lookahead_verbose,
        )
        # Store wrapper reference for statistics
        pipe._pc_lookahead_wrapper = wrapper

    # Load LoRA if needed
    if "checkpoint" in args.model_path:
        pipe.load_lora_weights(args.model_path)

    pipe = pipe.to(args.device)
    print(f"Pipeline loaded on {args.device}\n")

    # Generate images
    total_images = 0
    for batch_idx, batch in enumerate(dataloader):
        batch_prompts, batch_prompt_ids = batch
        
        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts, batch_prompt_ids):
                out_path = f"{args.output_dir}/{_id}.png"
                if os.path.exists(out_path):
                    print(f"Skipping {out_path}")
                    continue
                
                print(f"\nGenerating image {total_images + 1}: {_id}")
                if lookahead_enabled and args.lookahead_verbose:
                    print(f"{'─'*80}")
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    width=args.image_width,
                    height=args.image_height,
                )
                
                if lookahead_enabled and args.lookahead_verbose:
                    print(f"{'─'*80}")
                
                image = result.images[0]
                image.save(out_path)
                total_images += 1
                print(f"Saved: {out_path}")

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
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}")

    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600.0

    # Save log
    log_path = os.path.join(args.output_dir, "runtime_inference.log")
    with open(log_path, "w") as f:
        f.write(f"Runtime duration: {duration_hours:.4f} hours\n")
        f.write(f"Total images generated: {total_images}\n")
        f.write(f"Average time per image: {(end_time - start_time) / max(total_images, 1):.2f} seconds\n")
        
        if lookahead_enabled and hasattr(pipe, '_pc_lookahead_wrapper'):
            stats = pipe._pc_lookahead_wrapper.get_statistics()
            f.write(f"\nPC-LookAhead Statistics:\n")
            f.write(f"  Total steps: {stats.get('total_steps', 0)}\n")
            f.write(f"  Total backtracks: {stats.get('total_backtracks', 0)}\n")
            f.write(f"  Average backtracks per step: {stats.get('avg_backtracks', 0):.3f}\n")
            f.write(f"  Rejection rate: {stats.get('rejection_rate', 0):.2%}\n")
            f.write(f"  First-try acceptance rate: {stats.get('first_try_rate', 0):.2%}\n")
            f.write(f"  Average κ: {stats.get('avg_kappa', 0):.6f}\n")
            f.write(f"  Max κ: {stats.get('max_kappa', 0):.6f}\n")
            f.write(f"  Min κ: {stats.get('min_kappa', 0):.6f}\n")

    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"  Total time: {duration_hours:.4f} hours")
    print(f"  Total images: {total_images}")
    print(f"  Output directory: {args.output_dir}")
    
    if lookahead_enabled and hasattr(pipe, '_pc_lookahead_wrapper'):
        stats = pipe._pc_lookahead_wrapper.get_statistics()
        print(f"\nPC-LookAhead Statistics:")
        print(f"  Total steps: {stats.get('total_steps', 0)}")
        print(f"  Average backtracks per step: {stats.get('avg_backtracks', 0):.3f}")
        print(f"  Rejection rate: {stats.get('rejection_rate', 0):.2%}")
        print(f"  First-try acceptance rate: {stats.get('first_try_rate', 0):.2%}")
        print(f"  Average κ: {stats.get('avg_kappa', 0):.6f}")
        print(f"  Max κ: {stats.get('max_kappa', 0):.6f}")
        print(f"  Min κ: {stats.get('min_kappa', 0):.6f}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()