from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_qwen_adapter.py"
APP_TRAIN_FILE = SCRIPT_DIR.parent / "data" / "opencode_reasoning_train_4000.jsonl"
LOCAL_TRAIN_FILE = SCRIPT_DIR / "opencode_reasoning_train_4000.jsonl"
DEFAULT_TRAIN_FILE = APP_TRAIN_FILE if APP_TRAIN_FILE.exists() else LOCAL_TRAIN_FILE
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "split_adapters_tuning_2"


@dataclass
class SplitJob:
    task: str
    output_name: str
    epochs: float
    max_length: int
    lora_r: int
    lora_alpha: int


DEFAULT_JOBS = [
    SplitJob("keyword", "qwen2.5-3b-keyword-lora", 1.0, 768, 8, 16),
    SplitJob("thinking", "qwen2.5-3b-thinking-lora", 2.0, 1024, 16, 32),
    SplitJob("code", "qwen2.5-3b-code-lora", 1.0, 1024, 8, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train separate LoRA adapters for keyword, thinking, and code tasks.")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", default="keyword,thinking,code", help="Comma-separated tasks to train.")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--device-map", default="auto", help="Forwarded to train_qwen_adapter.py for CPU/GPU offload.")
    parser.add_argument("--max-gpu-memory", default="", help="Optional GPU memory cap such as 7GiB. Empty means no explicit cap.")
    parser.add_argument("--max-cpu-memory", default="24GiB", help="CPU memory cap used with device_map auto.")
    parser.add_argument("--offload-folder", default=str(SCRIPT_DIR / "outputs" / "offload"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    return parser.parse_args()



def build_command(args: argparse.Namespace, job: SplitJob) -> List[str]:
    output_dir = Path(args.output_root) / job.output_name
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--base-model-name",
        args.base_model_name,
        "--train-file",
        str(Path(args.train_file).expanduser()),
        "--output-dir",
        str(output_dir),
        "--task-mix",
        job.task,
        "--limit",
        str(args.limit),
        "--max-length",
        str(job.max_length),
        "--epochs",
        str(job.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--learning-rate",
        str(args.learning_rate),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--lora-r",
        str(job.lora_r),
        "--lora-alpha",
        str(job.lora_alpha),
        "--cuda-device",
        str(args.cuda_device),
        "--device-map",
        args.device_map,
        "--max-cpu-memory",
        args.max_cpu_memory,
        "--offload-folder",
        str(Path(args.offload_folder).expanduser() / job.task),
    ]
    if args.max_gpu_memory:
        command.extend(["--max-gpu-memory", args.max_gpu_memory])
    if args.no_cuda:
        command.append("--no-cuda")
    return command


def main() -> None:
    args = parse_args()
    requested_tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    jobs = [job for job in DEFAULT_JOBS if job.task in requested_tasks]

    if not TRAIN_SCRIPT.exists() or TRAIN_SCRIPT.stat().st_size == 0:
        raise FileNotFoundError(f"Training script is missing or empty: {TRAIN_SCRIPT}")
    if not Path(args.train_file).exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not jobs:
        raise ValueError("No valid tasks selected. Use keyword,thinking,code.")

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for job in jobs:
        command = build_command(args, job)
        print("\n" + "=" * 80)
        print(f"Training split adapter: {job.task}")
        print("Output:", Path(args.output_root) / job.output_name)
        print("Command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        print("=" * 80)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)

    print("\nSplit adapter training finished.")


if __name__ == "__main__":
    main()