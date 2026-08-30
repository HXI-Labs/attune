# Affect paired v1.1 protocol

Affect perceptual v1.0 cleared the external RAVDESS floor but reversed the
model's acoustic preference on development and opened regression data. An
audit of its deterministic epoch sampler found a valid same-text,
different-delivery contrast in 374 of 1,366 batches, or 27.4%. It also found
17 batches in which independent datasets reused the same integer pair ID.
Those collisions could make unrelated sentences contribute to the paired
loss.

Version 1.1 corrects the training mechanism before changing the model or data.
Pair identity is the tuple of dataset ID and source pair ID. Training batches
are duration-bucketed as before, then deterministically completed with a
different-delivery example from the same dataset and text. All physical
batches of at least two examples are required to contain a valid contrast;
the trainer records the observed fraction in every epoch. Corpus-balanced
sampling remains the source of the other five examples in each batch.

The run starts from retained v0.9, not rejected v1.0, and reuses the
15,389-row perceptual manifest with SHA-256
`329be1a98e632b91823d15eacb8762990dcd29050acd28aa777d9d590f1905fb`.
Only `affect_projection` and `affect_head` may change. The encoder, isolated
ASR path, temporal event head, style branch, pooling, VAD head, OOD head, and
event-presence head remain frozen. The paired-loss weight remains 0.2 so this
experiment isolates pair availability and identity rather than changing two
variables at once.

Six epochs of 8,192 samples are allowed at a `2e-5` head learning rate, with
four-step gradient accumulation and early stopping after three stale
development losses. Development data alone select the checkpoint and
calibration. Opened regression and RAVDESS data may only accept or reject the
completed candidate. The fresh British confirmation set remains sealed.

The candidate is retained only if all of the following hold:

- Development macro-F1 is at least 0.64.
- Opened-regression macro-F1 is at least 0.6236.
- Opened RAVDESS macro-F1 is at least 0.40 and no predicted class exceeds 45%
  of clips.
- APS is positive on development, opened regression, and opened RAVDESS.
- Every training epoch reports an active paired-batch fraction of 1.0.
- Only the eight permitted affect tensors differ from v0.9.
- CTC logits remain exactly equal to the retained model.

If any gate fails, v0.9 remains the retained candidate. No threshold or epoch
will be adjusted against RAVDESS or the opened regression split.
