"""Benchmark final Sleep-Mamba V3 variants with the paper's input shape."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from network_v3 import MODEL_VARIANTS, TinySleepNetV3


def load_config(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("benchmark_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train.copy()


def timed_forward(model, x, mask, device):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x, mask)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    start = time.perf_counter()
    model(x, mask)
    return (time.perf_counter() - start) * 1000.0


def benchmark_variant(base_config, variant, args):
    config = base_config.copy()
    config["model_variant"] = variant
    config["batch_size"] = args.batch_size
    config["seq_length"] = args.seq_length

    device = torch.device(args.device)
    model = TinySleepNetV3(config).to(device).eval()
    if args.precision == "fp16":
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    epoch_count = args.batch_size * args.seq_length
    x = torch.randn(
        epoch_count, 1, int(config["input_size"]),
        device=device, dtype=dtype,
    )
    context_mask = torch.ones(epoch_count, device=device, dtype=torch.bool)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, context_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies = [
            timed_forward(model, x, context_mask, device)
            for _ in range(args.repetitions)
        ]

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    p05_ms = sorted(latencies)[max(0, int(0.05 * len(latencies)) - 1)]
    p95_ms = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    incremental_peak_mb = (
        (torch.cuda.max_memory_allocated(device) - baseline_memory) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    bytes_per_parameter = 2 if args.precision == "fp16" else 4
    return {
        "variant": variant,
        "device": str(device),
        "precision": args.precision,
        "batch_sequences": args.batch_size,
        "sequence_length": args.seq_length,
        "epochs_per_forward": epoch_count,
        "parameters": parameter_count,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
        "model_storage_mib": parameter_count * bytes_per_parameter / (1024 ** 2),
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p05_ms": p05_ms,
        "latency_p95_ms": p95_ms,
        "latency_per_epoch_ms": median_ms / epoch_count,
        "throughput_epochs_per_second": epoch_count / (median_ms / 1000.0),
        "cuda_peak_total_mib": peak_memory_mb,
        "cuda_incremental_peak_mib": incremental_peak_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument(
        "--variants", nargs="+", default=["v3_context", "v3_balanced_fusion"],
        choices=MODEL_VARIANTS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output", default="efficiency_v3_balanced_fusion.csv")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    base_config = load_config(args.config_file)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    results = []
    for variant in args.variants:
        print(f"Benchmarking {variant} ...")
        result = benchmark_variant(base_config, variant, args)
        results.append(result)
        print(
            "  params={parameters:,}, median={latency_median_ms:.3f} ms, "
            "throughput={throughput_epochs_per_second:.1f} epochs/s".format(
                **result))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    json_path = output_path.with_suffix(".json")
    metadata = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command_settings": vars(args),
        "results": results,
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
