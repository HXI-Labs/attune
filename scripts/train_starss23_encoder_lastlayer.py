#!/usr/bin/env python3
"""Fine-tune SenseVoice last encoder block + laugh MLP on STARSS23 first-60s 100 ms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
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

LABELS = ("laugh",)
DURATION_MS = 60_000
VALIDATION_ROOMS = {"sony-room21", "tau-room6"}
SEGMENT_MARGIN_REQUIRED = 0.05
COLLAR_F1_REQUIRED = 0.25
PRIOR_BEST_COLLAR_F1 = 0.13953488372093023
FIRST_60S_TRAIN_CLIPS = 43
FIRST_60S_VAL_CLIPS = 19
FIRST_60S_INSPECTION_CLIPS = 49
FIRST_60S_INSPECTION_EVENTS = 48
PROTOCOL_PATH = Path("research/starss23-lastblock-highlights.md")
FROZEN_40EPOCH_CHECKPOINT = Path("artifacts/starss23-scene-raster/frame-head-40epoch.pt")
PREDECLARED_DECODER = {
    "high_threshold": 0.95,
    "low_threshold": 0.855,
    "max_gap_frames": 0,
    "min_active_frames": 1,
    "median_filter_frames": 3,
    "onset_shift_ms": 0,
}


def load_starss23_script() -> Any:
    path = Path(__file__).with_name("train_starss23_scene_raster.py")
    spec = importlib.util.spec_from_file_location("starss23_scene_raster", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


starss23 = load_starss23_script()


def require_protocol(path: Path = PROTOCOL_PATH) -> None:
    if not path.is_file():
        raise RuntimeError(f"research protocol missing; write {path} before training")
    text = path.read_text(encoding="utf-8")
    required = (
        "0.25",
        "0.05",
        "0.1395",
        "100 ms",
        "Not second-listener gold",
        "2026-08-28",
        "tp_encoders.19",
        "No decoder-search",
        "No tiling-in-train",
        "encoder-lr 5e-6",
        "high 0.95",
        "low 0.855",
        "gap 0",
        "min_active 1",
        "median 3",
        "STARSS23 stays unwired",
        "Do not train on the locked 48-event inspection",
        "Do not use only the 6 Jerry-retimed clips",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"protocol {path} is missing locked gate language: {missing}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_manifest_rows(manifest: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attach_audio(rows: list[dict[str, Any]], cache: Path) -> list[dict[str, Any]]:
    for row in rows:
        audio = cache / row["cache_path"]
        if not audio.is_file() or starss23.digest(audio) != row["sha256"]:
            raise RuntimeError(f"missing or changed STARSS23 clip: {audio}")
        if set(event["label"] for event in row["events"]) - set(LABELS):
            raise RuntimeError("STARSS23 manifest contains an unsupported Attune mapping")
        row["_audio"] = audio
    return rows


def first_60s_disjoint_split(
    development: list[dict[str, Any]],
    inspection: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train/val from first-60s development; inspection locked and unused for fitting."""
    development = starss23.first_60s_subset(development)
    inspection = starss23.first_60s_subset(inspection)
    if any(int(row["source_window_start_ms"]) != 0 for row in development + inspection):
        raise RuntimeError("STARSS23 last-block split included later tiles")
    train_rows = [row for row in development if row["room"] not in VALIDATION_ROOMS]
    validation_rows = [row for row in development if row["room"] in VALIDATION_ROOMS]
    assert_inspection_clips_not_in_train(train_rows, inspection)
    starss23.assert_whole_files_stay_in_one_split(train_rows, validation_rows)
    starss23.assert_whole_files_stay_in_one_split(train_rows, inspection)
    starss23.assert_whole_files_stay_in_one_split(validation_rows, inspection)
    train_rooms = {row["room"] for row in train_rows}
    val_rooms = {row["room"] for row in validation_rows}
    inspection_rooms = {row["room"] for row in inspection}
    if train_rooms & inspection_rooms:
        raise RuntimeError("STARSS23 inspection rooms overlap training")
    if val_rooms & inspection_rooms:
        raise RuntimeError("STARSS23 inspection rooms overlap validation")
    if val_rooms != VALIDATION_ROOMS:
        raise RuntimeError("STARSS23 last-block val drifted from sony-room21/tau-room6")
    if len(train_rows) != FIRST_60S_TRAIN_CLIPS:
        raise RuntimeError(f"expected {FIRST_60S_TRAIN_CLIPS} first-60s train clips")
    if len(validation_rows) != FIRST_60S_VAL_CLIPS:
        raise RuntimeError(f"expected {FIRST_60S_VAL_CLIPS} first-60s val clips")
    if len(inspection) != FIRST_60S_INSPECTION_CLIPS:
        raise RuntimeError(f"expected {FIRST_60S_INSPECTION_CLIPS} first-60s inspection clips")
    if starss23.event_count(inspection) != FIRST_60S_INSPECTION_EVENTS:
        raise RuntimeError("first-60s inspection gold is not the locked 48-event pack")
    if len(train_rows) <= 6:
        raise RuntimeError("refusing the 6 Jerry-retimed clips as the train set")
    return train_rows, validation_rows, inspection


def assert_inspection_clips_not_in_train(
    train_rows: list[dict[str, Any]],
    inspection_rows: list[dict[str, Any]],
) -> None:
    train_ids = {row["clip_id"] for row in train_rows}
    inspection_ids = {row["clip_id"] for row in inspection_rows}
    overlap = train_ids & inspection_ids
    if overlap:
        raise RuntimeError(f"inspection clips leaked into train: {sorted(overlap)[:8]}")
    train_files = {row["source_recording"] for row in train_rows}
    inspection_files = {row["source_recording"] for row in inspection_rows}
    if train_files & inspection_files:
        raise RuntimeError("STARSS23 inspection recordings leaked into train")


def load_head_init(path: Path, torch: Any) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(payload.get("labels") or ()) != LABELS:
        return None
    if int(payload.get("hidden_size") or 0) != starss23.MLP_HIDDEN_SIZE:
        return None
    head_state = payload.get("head_state_dict")
    if not isinstance(head_state, dict) or "0.weight" not in head_state:
        return None
    if tuple(head_state["0.weight"].shape) != (starss23.MLP_HIDDEN_SIZE, 512):
        return None
    if tuple(head_state["2.weight"].shape) != (1, starss23.MLP_HIDDEN_SIZE):
        return None
    return payload


def clip_metrics(
    probabilities: list[Any],
    references: list[list[dict[str, Any]]],
    *,
    first_frame_center_ms: float,
    frame_hop_ms: float,
) -> dict[str, Any]:
    predictions = starss23.decode_spans(
        probabilities,
        PREDECLARED_DECODER,
        first_frame_center_ms=first_frame_center_ms,
        frame_hop_ms=frame_hop_ms,
    )
    baseline = whole_clip_predictions(references, duration_ms=DURATION_MS)
    collar = collar_event_metrics(references, predictions)
    segment = segment_f1(references, predictions, duration_ms=DURATION_MS)
    baseline_segment = segment_f1(references, baseline, duration_ms=DURATION_MS)
    baseline_collar = collar_event_metrics(references, baseline)
    margin = float(segment["f1"]) - float(baseline_segment["f1"])
    return {
        "decoder": dict(PREDECLARED_DECODER),
        "segment_f1": segment,
        "collar_event_metrics": collar,
        "whole_clip": {
            "segment_f1": baseline_segment,
            "collar_event_metrics": baseline_collar,
        },
        "segment_margin_vs_whole_clip": margin,
    }


def should_wire(*, segment_f1_value: float, whole_clip_segment_f1: float, collar_f1: float) -> bool:
    return starss23.should_wire_starss23_timestamps(
        segment_f1=segment_f1_value,
        whole_clip_segment_f1=whole_clip_segment_f1,
        collar_f1=collar_f1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice-path", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-development.jsonl"),
    )
    parser.add_argument(
        "--inspection-manifest",
        type=Path,
        default=Path("data/manifests/starss23-scene-raster-inspection.jsonl"),
    )
    parser.add_argument(
        "--gold-review-pack",
        type=Path,
        default=Path("data/manifests/starss23-gold-review-pack.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/starss23-scene-raster-tiled"),
    )
    parser.add_argument(
        "--prefix-cache",
        type=Path,
        default=Path("artifacts/starss23-lastblock-highlights/prefix"),
    )
    parser.add_argument("--head-init", type=Path, default=FROZEN_40EPOCH_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/starss23-lastblock-highlights/last-block-and-head.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/starss23-lastblock-highlights-results.json"),
    )
    parser.add_argument(
        "--epoch-log",
        type=Path,
        default=Path("artifacts/starss23-lastblock-highlights/epoch-log.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    arguments = parser.parse_args()
    import torch

    require_protocol(PROTOCOL_PATH)
    if arguments.epochs > 15:
        raise RuntimeError("max 15 epochs is locked for this STARSS23 last-block pass")
    if arguments.patience != 5:
        raise RuntimeError("patience 5 on val collar is locked")
    if arguments.seed != 0:
        raise RuntimeError("seed 0 is locked")
    if arguments.encoder_lr != 5e-6:
        raise RuntimeError("encoder-lr 5e-6 is locked")
    if arguments.head_lr != 1e-4:
        raise RuntimeError("head-lr 1e-4 is locked")
    if arguments.checkpoint_output.resolve() == FROZEN_40EPOCH_CHECKPOINT.resolve():
        raise RuntimeError("must not overwrite frame-head-40epoch.pt")
    if torch.cuda.is_available():
        print("CUDA is visible; this run still forces CPU as declared", flush=True)
    os.environ["ATTUNE_SENSEVOICE_LICENSE_REVIEWED"] = "1"
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    development = load_manifest_rows(arguments.training_manifest)
    inspection_all = load_manifest_rows(arguments.inspection_manifest)
    gold_pack = load_manifest_rows(arguments.gold_review_pack)
    gold_ids = {row["clip_id"] for row in starss23.first_60s_subset(gold_pack)}
    train_rows, validation_rows, inspection_rows = first_60s_disjoint_split(
        development,
        inspection_all,
    )
    inspection_ids = {row["clip_id"] for row in inspection_rows}
    if inspection_ids != gold_ids:
        raise RuntimeError("inspection first-60s clip IDs drifted from the gold-review pack")
    attach_audio(train_rows, arguments.cache_dir)
    attach_audio(validation_rows, arguments.cache_dir)
    attach_audio(inspection_rows, arguments.cache_dir)

    encoder = LastBlockSenseVoiceFrameEncoder(
        arguments.sensevoice_path,
        arguments.prefix_cache,
        torch,
        prefix_cache=True,
    )
    if encoder.last_block_name != "encoder.tp_encoders.19":
        raise RuntimeError(f"expected tp_encoders.19, got {encoder.last_block_name}")
    last_block_params = encoder.trainable_parameter_count()
    print(
        f"unfrozen {encoder.last_block_name}: {last_block_params} trainable encoder params",
        flush=True,
    )

    print("caching frozen prefixes into tp_encoders.19", flush=True)
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
            "unfrozen-block replay does not match the official encoder forward; "
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
    train_y = starss23.frame_targets(
        train_rows, [len(clip) for clip in train_x], **target_arguments
    )
    validation_y = starss23.frame_targets(
        validation_rows,
        [int(mask.reshape(mask.size(0), -1)[0].sum().item()) - 4 for _, mask in validation_prefix],
        **target_arguments,
    )

    wired = load_head_init(arguments.head_init, torch)
    head = starss23.build_mlp_head(torch)
    starss23.assert_head_is_mlp_not_conv_or_gru(head, torch)
    if wired is not None:
        head.load_state_dict(wired["head_state_dict"])
        mean = wired["feature_mean"].float()
        scale = wired["feature_scale"].float()
        head_init = "frozen_0.1395_40epoch_mlp"
        head_init_sha256 = starss23.digest(arguments.head_init)
    else:
        torch.manual_seed(arguments.seed)
        for module in head.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        stacked = torch.cat(train_x)
        mean = stacked.mean(dim=0)
        scale = stacked.std(dim=0).clamp_min(1e-6)
        head_init = "random_seed_0"
        head_init_sha256 = None

    loss_function = torch.nn.BCEWithLogitsLoss()
    probe = encoder.frames_from_prefix(*train_prefix[0], train=True)
    probe_loss = loss_function(head((probe - mean) / scale), train_y[0])
    probe_loss.backward()
    last_grads = [
        name
        for name, parameter in encoder.model.named_parameters()
        if parameter.grad is not None and name.startswith("encoder.tp_encoders.19.")
    ]
    frozen_leaks = [
        name
        for name, parameter in encoder.model.named_parameters()
        if parameter.grad is not None and not name.startswith("encoder.tp_encoders.19.")
    ]
    if not last_grads:
        raise RuntimeError("last encoder block produced no gradients")
    if frozen_leaks:
        raise RuntimeError(f"gradients leaked into frozen encoder params: {frozen_leaks[:8]}")
    encoder.last_block.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    print(f"autograd check passed: {len(last_grads)} last-block tensors received grads", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder.unfrozen_parameters()), "lr": arguments.encoder_lr},
            {"params": list(head.parameters()), "lr": arguments.head_lr},
        ]
    )
    best_state: dict[str, Any] | None = None
    best_collar = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    span_kw = {
        "first_frame_center_ms": encoder.first_frame_center_ms,
        "frame_hop_ms": encoder.frame_hop_ms,
    }

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

    payload_base: dict[str, Any] = {
        "report_version": "1",
        "status": "running",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "pending",
        "task": "STARSS23 v1.1 first-60s last-block encoder fine-tune",
        "label_status": (
            "STARSS23 class-4 100 ms activity on first 60 s; owner-authorized "
            "2026-08-28 as good enough; not second-listener Attune gold"
        ),
        "starss23_wired": False,
        "protocol": str(PROTOCOL_PATH),
        "labels": LABELS,
        "decoder_locked": dict(PREDECLARED_DECODER),
        "decoder_search": False,
        "tiling_in_train": False,
        "gate": {
            "collar_f1_min": COLLAR_F1_REQUIRED,
            "segment_margin_min": SEGMENT_MARGIN_REQUIRED,
        },
        "prior_best": {
            "id": "decoder_validity_0a27733",
            "collar_event_f1": PRIOR_BEST_COLLAR_F1,
            "segment_f1": 0.512396694214876,
            "whole_clip_segment_f1": 0.17212490479817213,
            "checkpoint": str(FROZEN_40EPOCH_CHECKPOINT),
        },
        "encoder": encoder.metadata(),
        "head": {
            "type": "two-layer temporal MLP 512->64->1",
            "init": head_init,
            "init_sha256": head_init_sha256,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "encoder_trainable_parameters": last_block_params,
            "total_trainable_parameters": last_block_params
            + sum(parameter.numel() for parameter in head.parameters()),
        },
        "partitions": {
            "train_clips": len(train_rows),
            "validation_clips": len(validation_rows),
            "inspection_test_clips": len(inspection_rows),
            "train_events": starss23.event_count(train_rows),
            "validation_events": starss23.event_count(validation_rows),
            "inspection_events": starss23.event_count(inspection_rows),
            "validation_rooms": sorted(VALIDATION_ROOMS),
            "file_and_room_disjoint_inspection": True,
            "first_60s_only": True,
            "jerry_retimed_clips_used_as_train": False,
        },
        "training": {
            "seed": arguments.seed,
            "max_epochs": arguments.epochs,
            "patience": arguments.patience,
            "encoder_lr": arguments.encoder_lr,
            "head_lr": arguments.head_lr,
            "loss": "unweighted_bce_with_logits",
            "unfrozen_modules": list(encoder.unfrozen_module_names()),
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
                list(encoder.unfrozen_parameters()) + list(head.parameters()),
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
        val_collar = float(validation_metrics["collar_event_metrics"]["f1"])
        val_segment = float(validation_metrics["segment_f1"]["f1"])
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_collar_f1": val_collar,
            "validation_segment_f1": val_segment,
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
            f"val_loss={validation_loss:.4f} val_collar={val_collar:.4f} "
            f"val_segment={val_segment:.4f} {elapsed:.1f}s",
            flush=True,
        )
        if val_collar > best_collar + 1e-6:
            best_collar = val_collar
            best_state = {
                "unfrozen": encoder.snapshot_unfrozen(),
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
    encoder.load_unfrozen(best_state["unfrozen"])
    head.load_state_dict(best_state["head"])
    encoder.prepare_last_block(train=False)
    head.eval()

    inspection_probabilities = evaluate_prefixes(inspection_prefix)
    inspection = clip_metrics(
        inspection_probabilities,
        [row["events"] for row in inspection_rows],
        **span_kw,
    )
    collar_f1 = float(inspection["collar_event_metrics"]["f1"])
    segment = float(inspection["segment_f1"]["f1"])
    whole = float(inspection["whole_clip"]["segment_f1"]["f1"])
    margin = float(inspection["segment_margin_vs_whole_clip"])
    wire = should_wire(
        segment_f1_value=segment,
        whole_clip_segment_f1=whole,
        collar_f1=collar_f1,
    )
    if wire and (collar_f1 < COLLAR_F1_REQUIRED or margin < SEGMENT_MARGIN_REQUIRED):
        raise RuntimeError("wiring helper cleared a lowered gate")
    arguments.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "last_block_name": encoder.last_block_name,
        "unfrozen_block_names": list(encoder.unfrozen_module_names()),
        "unfrozen_block_state_dicts": best_state["unfrozen"],
        "last_block_state_dict": best_state["unfrozen"][encoder.last_block_name],
        "head_state_dict": best_state["head"],
        "feature_mean": mean,
        "feature_scale": scale,
        "labels": LABELS,
        "hidden_size": starss23.MLP_HIDDEN_SIZE,
        "threshold": PREDECLARED_DECODER["high_threshold"],
        "decoder": {"type": "hysteresis", **PREDECLARED_DECODER},
        "dataset": "starss23",
        "embedding": SENSEVOICE_LAST_BLOCK_FRAMES,
        "frame_hop_ms_approx": encoder.frame_hop_ms,
        "first_frame_center_ms_approx": encoder.first_frame_center_ms,
        "encoder_frozen": False,
        "encoder_unfrozen_modules": list(encoder.unfrozen_module_names()),
        "head_init": head_init,
        "gate": {
            "passed": wire,
            "collar_f1_required": COLLAR_F1_REQUIRED,
            "collar_f1_observed": collar_f1,
            "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
            "segment_margin_observed": margin,
            "true_positive": inspection["collar_event_metrics"]["true_positive"],
            "false_positive": inspection["collar_event_metrics"]["false_positive"],
            "false_negative": inspection["collar_event_metrics"]["false_negative"],
        },
    }
    torch.save(checkpoint, arguments.checkpoint_output)
    payload = {
        **payload_base,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate_decision": "wire_starss23_laugh" if wire else "closed",
        "starss23_wired": False,
        "encoder": encoder.metadata(),
        "inspection_test": {
            "designation": "test; never used for fitting, early-stop, or decoder",
            "clip_count": len(inspection_rows),
            "event_count": starss23.event_count(inspection_rows),
            "decoder": dict(PREDECLARED_DECODER),
            "segment_f1": inspection["segment_f1"],
            "collar_event_metrics": inspection["collar_event_metrics"],
            "whole_clip_oracle_tag_baseline": inspection["whole_clip"],
            "segment_margin_vs_whole_clip": margin,
        },
        "cascade_wiring": {
            "passed": wire,
            "timestamps_wired": False,
            "collar_f1_required": COLLAR_F1_REQUIRED,
            "collar_f1_observed": collar_f1,
            "segment_margin_required": SEGMENT_MARGIN_REQUIRED,
            "segment_margin_observed": margin,
            "policy": (
                "wire STARSS23 laugh only if inspection collar F1 >= 0.25 "
                "and segment margin vs whole-clip >= 0.05; leave PR unmerged"
            ),
            "reason": (
                "inspection clears 0.25 collar and 0.05 segment margin"
                if wire
                else "gate missed; STARSS23 stays unwired"
            ),
        },
        "limitations": [
            "STARSS23 100 ms activity labels, owner-authorized 2026-08-28; not Attune gold.",
            "Only encoder.tp_encoders.19 was unfrozen.",
            "Decoder was locked to 0a27733; no threshold or hysteresis search.",
            "No tiling-in-train; first-60s windows only.",
            "Inspection was scored once and never used for fitting.",
            "Fine-tuned weights stay private and gitignored.",
        ],
    }
    payload["training"]["epochs_completed"] = len(history)
    payload["training"]["history"] = history
    payload["training"]["elapsed_seconds"] = time.perf_counter() - started
    payload["training"]["best_validation_collar_f1"] = best_collar
    payload["do_not_replace_reported_best_0.1395"] = collar_f1 <= PRIOR_BEST_COLLAR_F1
    atomic_write_json(arguments.output, payload)
    print(
        f"Wrote {arguments.output} collar={collar_f1:.4f} segment={segment:.4f} "
        f"margin={margin:.4f} wire={wire} val_peak={best_collar:.4f} "
        f"trainable={last_block_params}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write(f"BLOCKER: {error}\n")
        raise
