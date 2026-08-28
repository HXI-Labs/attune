#!/usr/bin/env python3
"""Fine-tune SenseVoice last encoder block + DCASE frame MLP on isolated events."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attune.evaluation.localization import (
    collar_event_metrics,
    segment_f1,
    whole_clip_predictions,
)
from attune.models.sensevoice_last_block import (
    SENSEVOICE_LAST_BLOCK_FRAMES,
    LastBlockSenseVoiceFrameEncoder,
)
from attune.models.sensevoice_probe import SENSEVOICE_FRAME_EMBEDDING

LABELS = ("laugh", "cough", "throat_clear")
DURATION_MS = 10_000
CLEAR_SEGMENT_F1_MARGIN = 0.05
GATE_EXACT_COLLAR_F1 = 0.4637
GATE_HYSTERESIS_COLLAR_F1 = 0.5279
LOCKED_EXACT_THRESHOLD = 0.95
LOCKED_HYSTERESIS_DECODER = {
    "high_threshold": 0.95,
    "low_threshold": 0.855,
    "max_gap_frames": 1,
    "min_active_frames": 1,
}
WIRED_HEAD_SHA256 = "cb74b1d493511d384dea3abf37139cf754a16f540b594b7f92f82647204b01fc"
PROTOCOL_PATH = Path("research/dcase-encoder-lastlayer.md")


def load_dcase_script() -> Any:
    path = Path(__file__).with_name("train_temporal_localization.py")
    spec = importlib.util.spec_from_file_location("dcase_temporal_localization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dcase = load_dcase_script()


def digest(path: Path) -> str:
    return dcase.digest(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def require_protocol(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"research protocol missing; write {path} before training")
    text = path.read_text(encoding="utf-8")
    required = (
        "0.4637",
        "0.5279",
        "STARSS23 stays unwired",
        "Do not grid-search",
        "Last 1 encoder block",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"protocol {path} is missing locked gate language: {missing}")


def should_replace_wired_head(
    exact_collar_f1: float,
    hysteresis_collar_f1: float,
    segment_margin: float,
) -> bool:
    """Replace the PR 26 head only if every predeclared gate clears."""
    return (
        exact_collar_f1 >= GATE_EXACT_COLLAR_F1
        and hysteresis_collar_f1 >= GATE_HYSTERESIS_COLLAR_F1
        and segment_margin >= CLEAR_SEGMENT_F1_MARGIN
    )


def load_wired_head(path: Path, torch: Any) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"wired DCASE head is missing: {path}")
    sha256 = digest(path)
    if sha256 != WIRED_HEAD_SHA256:
        raise RuntimeError(
            f"wired DCASE head SHA mismatch: got {sha256}, expected {WIRED_HEAD_SHA256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(payload.get("labels") or ()) != LABELS:
        raise RuntimeError("wired DCASE head labels are not laugh/cough/throat_clear")
    if payload.get("embedding") != SENSEVOICE_FRAME_EMBEDDING:
        raise RuntimeError("wired DCASE head was not trained on frozen SenseVoice frames")
    if payload.get("encoder_frozen") is not True:
        raise RuntimeError("wired DCASE head does not attest a frozen encoder")
    return payload


def build_head(torch: Any, wired: dict[str, Any]) -> Any:
    hidden = int(wired.get("hidden_size") or 128)
    head = torch.nn.Sequential(
        torch.nn.Linear(512, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, len(LABELS)),
    )
    head.load_state_dict(wired["head_state_dict"])
    return head


def clip_metrics(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, Any]:
    span_arguments = {
        "first_frame_center_ms": first_frame_center_ms,
        "frame_hop_ms": frame_hop_ms,
    }
    exact = dcase.frame_spans(probabilities, LOCKED_EXACT_THRESHOLD, **span_arguments)
    hysteresis = dcase.hysteresis_spans(
        probabilities, **LOCKED_HYSTERESIS_DECODER, **span_arguments
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    exact_collar = collar_event_metrics(references, exact)
    hysteresis_collar = collar_event_metrics(references, hysteresis)
    hysteresis_segment = segment_f1(references, hysteresis, duration_ms=DURATION_MS)
    exact_segment = segment_f1(references, exact, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    baseline_collar = collar_event_metrics(references, baseline)
    margin = float(hysteresis_segment["f1"]) - float(baseline_segment["f1"])
    return {
        "exact": {
            "threshold": LOCKED_EXACT_THRESHOLD,
            "segment_f1": exact_segment,
            "collar_event_metrics": exact_collar,
        },
        "hysteresis": {
            "decoder": LOCKED_HYSTERESIS_DECODER,
            "segment_f1": hysteresis_segment,
            "collar_event_metrics": hysteresis_collar,
        },
        "whole_clip": {
            "segment_f1": baseline_segment,
            "collar_event_metrics": baseline_collar,
        },
        "segment_margin_vs_whole_clip": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-training.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/dcase2016-localization-inspection.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/dcase2016-localization"),
    )
    parser.add_argument(
        "--prefix-cache",
        type=Path,
        default=Path("artifacts/dcase-encoder-lastlayer/prefix"),
    )
    parser.add_argument(
        "--head-init",
        type=Path,
        default=Path("/workspace/attune-dcase/artifacts/dcase-frame-localization/frame-head.pt"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/dcase-encoder-lastlayer/last-block-and-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/dcase-encoder-lastlayer-results.json"),
    )
    parser.add_argument(
        "--epoch-log",
        type=Path,
        default=Path("artifacts/dcase-encoder-lastlayer/epoch-log.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay applied to encoder.tp_encoders.19 only; 0 keeps the original group defaults",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    arguments = parser.parse_args()
    import torch

    require_protocol(PROTOCOL_PATH)
    if torch.cuda.is_available():
        print("CUDA is visible; this run still forces CPU as declared", flush=True)
    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    training_rows = dcase.load_rows(arguments.training_manifest, arguments.cache_dir)
    inspection_rows = dcase.load_rows(arguments.inspection_manifest, arguments.cache_dir)
    train_rows, validation_rows, split_metadata = dcase.development_split(training_rows)
    wired = load_wired_head(arguments.head_init, torch)
    encoder = LastBlockSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.prefix_cache,
        torch,
        prefix_cache=True,
    )
    last_block_params = encoder.trainable_parameter_count()
    print(
        f"last block {encoder.last_block_name}: {last_block_params} trainable encoder params",
        flush=True,
    )

    print("caching frozen last-block prefixes (weights frozen below the last SANM)", flush=True)
    prefix_started = time.perf_counter()
    train_prefix = [encoder.prefix_for(row["_audio"]) for row in train_rows]
    validation_prefix = [encoder.prefix_for(row["_audio"]) for row in validation_rows]
    inspection_prefix = [encoder.prefix_for(row["_audio"]) for row in inspection_rows]
    print(
        f"prefix cache ready in {time.perf_counter() - prefix_started:.1f}s "
        f"(hits={encoder.cache_hits} misses={encoder.cache_misses})",
        flush=True,
    )

    encoder.prepare_last_block(train=False)
    with torch.no_grad():
        sanity_frames = encoder.frames_from_prefix(*train_prefix[0], train=False)
        full_speech, full_lengths = encoder._prepare_encoder_input(train_rows[0]["_audio"])
        full_out, full_olen = encoder.model.encoder(full_speech, full_lengths)
        if isinstance(full_out, tuple):
            full_out = full_out[0]
        full_frames = full_out[0, 4 : int(full_olen[0].item())].float().cpu()
    if sanity_frames.shape != full_frames.shape or not torch.allclose(
        sanity_frames, full_frames, atol=1e-4, rtol=1e-4
    ):
        raise RuntimeError(
            "last-block replay does not match the official encoder forward; "
            "stop rather than train a mismatched extraction route"
        )
    print(
        f"extraction-route check passed: {tuple(sanity_frames.shape)} frames, max abs "
        f"{(sanity_frames - full_frames).abs().max().item():.3e}",
        flush=True,
    )

    target_arguments = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
        "torch": torch,
    }
    with torch.no_grad():
        train_x = [encoder.frames_from_prefix(*item, train=False) for item in train_prefix]
    train_y = dcase.frame_targets(train_rows, [len(clip) for clip in train_x], **target_arguments)
    validation_y = dcase.frame_targets(
        validation_rows,
        [int(mask.reshape(mask.size(0), -1)[0].sum().item()) - 4 for _, mask in validation_prefix],
        **target_arguments,
    )
    head = build_head(torch, wired)
    mean = wired["feature_mean"].float()
    scale = wired["feature_scale"].float()
    train_target_matrix = torch.cat(train_y)
    positives = train_target_matrix.sum(dim=0)
    negatives = len(train_target_matrix) - positives
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives.clamp_min(1))

    probe = encoder.frames_from_prefix(*train_prefix[0], train=True)
    probe_loss = loss_function(head((probe - mean) / scale), train_y[0])
    probe_loss.backward()
    last_grads = [
        name
        for name, parameter in encoder.last_block.named_parameters()
        if parameter.grad is not None
    ]
    frozen_leaks = [
        name
        for name, parameter in encoder.model.named_parameters()
        if parameter.grad is not None and not name.startswith(encoder.last_block_name + ".")
    ]
    if not last_grads:
        raise RuntimeError(
            "FunASR last block produced no gradients; stop rather than fall back to a frozen MLP"
        )
    if frozen_leaks:
        raise RuntimeError(f"gradients leaked into frozen encoder params: {frozen_leaks[:8]}")
    encoder.last_block.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    print(f"autograd check passed: {len(last_grads)} last-block tensors received grads", flush=True)

    encoder_group: dict[str, Any] = {
        "params": list(encoder.last_block.parameters()),
        "lr": arguments.encoder_lr,
    }
    # Default 0 leaves AdamW per-group default so seed-0 2e-5/1e-4 stays reproducible.
    if arguments.encoder_weight_decay:
        encoder_group["weight_decay"] = arguments.encoder_weight_decay
    optimizer = torch.optim.AdamW(
        [
            encoder_group,
            {"params": list(head.parameters()), "lr": arguments.head_lr},
        ]
    )
    best_state: dict[str, Any] | None = None
    best_collar = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    def evaluate_prefixes(prefixes: list[tuple[Any, Any]]) -> list[Any]:
        encoder.prepare_last_block(train=False)
        head.eval()
        probabilities = []
        with torch.no_grad():
            for prefix, mask in prefixes:
                frames = encoder.frames_from_prefix(prefix, mask, train=False)
                logits = head((frames - mean) / scale)
                probabilities.append(torch.sigmoid(logits).cpu())
        return probabilities

    payload_base = {
        "report_version": "1",
        "status": "running",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "pending",
        "task": "DCASE 2016 Task 2 isolated-event last-block encoder fine-tune",
        "label_status": "synthetic strong onset/offset; not reviewed Attune gold; not STARSS23",
        "starss23_wired": False,
        "protocol": str(PROTOCOL_PATH),
        "labels": LABELS,
        "decoder_locked": {
            "exact_threshold": LOCKED_EXACT_THRESHOLD,
            "hysteresis": LOCKED_HYSTERESIS_DECODER,
            "grid_search": False,
        },
        "gate": {
            "exact_collar_f1_min": GATE_EXACT_COLLAR_F1,
            "hysteresis_collar_f1_min": GATE_HYSTERESIS_COLLAR_F1,
            "segment_margin_min": CLEAR_SEGMENT_F1_MARGIN,
        },
        "encoder": encoder.metadata(),
        "head": {
            "type": "two-layer temporal MLP, initialized from SHA-pinned wired DCASE head",
            "init_sha256": WIRED_HEAD_SHA256,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "encoder_trainable_parameters": last_block_params,
            "total_trainable_parameters": last_block_params
            + sum(parameter.numel() for parameter in head.parameters()),
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection_rows),
            "source_disjoint_test": True,
            **split_metadata,
        },
        "training": {
            "seed": arguments.seed,
            "max_epochs": arguments.epochs,
            "patience": arguments.patience,
            "encoder_lr": arguments.encoder_lr,
            "head_lr": arguments.head_lr,
            "encoder_weight_decay": arguments.encoder_weight_decay,
            "device": "cpu",
            "epochs_completed": 0,
            "history": [],
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": False,
        },
    }
    atomic_write_json(arguments.output, payload_base)

    span_kw = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }

    for epoch in range(1, arguments.epochs + 1):
        epoch_started = time.perf_counter()
        encoder.prepare_last_block(train=True)
        head.train()
        order = torch.randperm(
            len(train_prefix),
            generator=torch.Generator().manual_seed(arguments.seed + epoch),
        ).tolist()
        running = 0.0
        frames_seen = 0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            frames = encoder.frames_from_prefix(*train_prefix[index], train=True)
            logits = head((frames - mean) / scale)
            loss = loss_function(logits, train_y[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.last_block.parameters()) + list(head.parameters()),
                arguments.grad_clip,
            )
            optimizer.step()
            running += float(loss.item()) * len(frames)
            frames_seen += len(frames)
        train_loss = running / max(frames_seen, 1)

        encoder.prepare_last_block(train=False)
        head.eval()
        with torch.no_grad():
            val_losses = []
            val_probabilities = []
            for (prefix, mask), target in zip(validation_prefix, validation_y, strict=True):
                frames = encoder.frames_from_prefix(prefix, mask, train=False)
                logits = head((frames - mean) / scale)
                val_losses.append(float(loss_function(logits, target).item()) * len(frames))
                val_probabilities.append(torch.sigmoid(logits).cpu())
        validation_loss = sum(val_losses) / max(sum(len(item) for item in validation_y), 1)
        validation_metrics = clip_metrics(
            val_probabilities,
            [row["events"] for row in validation_rows],
            **span_kw,
        )
        val_hyst = float(validation_metrics["hysteresis"]["collar_event_metrics"]["f1"])
        val_exact = float(validation_metrics["exact"]["collar_event_metrics"]["f1"])
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_exact_collar_f1": val_exact,
            "validation_hysteresis_collar_f1": val_hyst,
            "validation_hysteresis_segment_f1": float(
                validation_metrics["hysteresis"]["segment_f1"]["f1"]
            ),
            "seconds": elapsed,
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        history.append(row)
        append_jsonl(arguments.epoch_log, row)
        payload_base["generated_at_utc"] = row["timestamp_utc"]
        payload_base["training"]["epochs_completed"] = epoch
        payload_base["training"]["history"] = history
        payload_base["training"]["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(arguments.output, payload_base)
        print(
            f"epoch {epoch}/{arguments.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_exact_collar={val_exact:.4f} "
            f"val_hyst_collar={val_hyst:.4f} {elapsed:.1f}s",
            flush=True,
        )
        if val_hyst > best_collar + 1e-6:
            best_collar = val_hyst
            best_state = {
                "last_block": {
                    name: value.detach().clone()
                    for name, value in encoder.last_block.state_dict().items()
                },
                "head": {name: value.detach().clone() for name, value in head.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= arguments.patience:
                print(f"early stop at epoch {epoch} (patience {arguments.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("last-block training produced no checkpoint")
    encoder.last_block.load_state_dict(best_state["last_block"])
    head.load_state_dict(best_state["head"])
    encoder.prepare_last_block(train=False)
    head.eval()

    inspection_probabilities = evaluate_prefixes(inspection_prefix)
    inspection = clip_metrics(
        inspection_probabilities,
        [row["events"] for row in inspection_rows],
        **span_kw,
    )
    exact_f1 = float(inspection["exact"]["collar_event_metrics"]["f1"])
    hyst_f1 = float(inspection["hysteresis"]["collar_event_metrics"]["f1"])
    margin = float(inspection["segment_margin_vs_whole_clip"])
    replace = should_replace_wired_head(exact_f1, hyst_f1, margin)
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "last_block_name": encoder.last_block_name,
            "last_block_state_dict": best_state["last_block"],
            "head_state_dict": best_state["head"],
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": LABELS,
            "hidden_size": 128,
            "threshold": LOCKED_EXACT_THRESHOLD,
            "decoder": {"type": "hysteresis", **LOCKED_HYSTERESIS_DECODER},
            "dataset": "dcase2016_task2",
            "embedding": SENSEVOICE_LAST_BLOCK_FRAMES,
            "frame_hop_ms_approx": encoder.frame_hop_ms,
            "first_frame_center_ms_approx": encoder.first_frame_center_ms,
            "encoder_frozen": False,
            "encoder_unfrozen_modules": [encoder.last_block_name],
            "wired_head_init_sha256": WIRED_HEAD_SHA256,
            "gate": {
                "passed": replace,
                "replace_wired_head": replace,
                "margin_required": CLEAR_SEGMENT_F1_MARGIN,
                "margin_observed": margin,
                "exact_collar_f1": exact_f1,
                "hysteresis_collar_f1": hyst_f1,
                "exact_collar_f1_min": GATE_EXACT_COLLAR_F1,
                "hysteresis_collar_f1_min": GATE_HYSTERESIS_COLLAR_F1,
            },
        },
        arguments.checkpoint_output,
    )
    payload = {
        **payload_base,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "replace_wired_head" if replace else "keep_wired_frozen_head",
        "encoder": encoder.metadata(),
        "inspection_test": {
            "designation": "test; never used for fitting or decoder selection",
            "event_counts": dict(
                sorted(
                    Counter(
                        event["label"] for row in inspection_rows for event in row["events"]
                    ).items()
                )
            ),
            "exact_match": {
                "threshold": LOCKED_EXACT_THRESHOLD,
                "segment_f1": inspection["exact"]["segment_f1"],
                "collar_event_metrics": inspection["exact"]["collar_event_metrics"],
            },
            "hysteresis": {
                "decoder": LOCKED_HYSTERESIS_DECODER,
                "segment_f1": inspection["hysteresis"]["segment_f1"],
                "collar_event_metrics": inspection["hysteresis"]["collar_event_metrics"],
            },
            "whole_clip_oracle_tag_baseline": inspection["whole_clip"],
            "segment_margin_vs_whole_clip": margin,
        },
        "cascade_wiring": {
            "replace_wired_head": replace,
            "timestamps_wired": False,
            "reason": (
                "inspection clears exact 0.4637, hysteresis 0.5279, and 0.05 margin; "
                "replacement is still a separate step and is not done on this branch"
                if replace
                else "gate missed; keep the current frozen DCASE head wired"
            ),
        },
        "limitations": [
            "Isolated DCASE office mixtures, not in-the-wild overlapping speech.",
            "Only the last SANM block (tp_encoders.19) was unfrozen.",
            "Decoder was locked; no threshold or hysteresis search.",
            "STARSS23 was not trained, evaluated, or wired.",
            "Fine-tuned weights stay private and gitignored.",
        ],
    }
    payload["training"]["epochs_completed"] = len(history)
    payload["training"]["history"] = history
    payload["training"]["elapsed_seconds"] = time.perf_counter() - started
    payload["training"]["best_validation_hysteresis_collar_f1"] = best_collar
    atomic_write_json(arguments.output, payload)
    print(
        f"Wrote {arguments.output} exact_collar={exact_f1:.4f} "
        f"hyst_collar={hyst_f1:.4f} margin={margin:.4f} replace={replace}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write(f"BLOCKER: {error}\n")
        raise
