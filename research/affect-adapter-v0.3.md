# Frozen-output affect adapter v0.3

## Decision

Reject and keep disabled. The adapter fixes much of the original in-domain
class imbalance but does not generalize to the external RAVDESS actors. It is
an informative negative result, not a deployment artefact.

## Frozen protocol

The protocol was committed before fitting in
`configs/model/affect-adapter-v0.3.json`. It uses:

- the original v0.2 training scores for fitting;
- the original development scores for regularization, temperature, and
  abstention selection;
- the already-opened original sealed scores only as a diagnostic; and
- the RAVDESS actors 17–24 slice only for one external evaluation.

The model is balanced multinomial logistic regression over the frozen
32-dimensional acoustic/OOD embedding, eight original affect logits, and three
VAD outputs. It does not change the encoder, CTC logits, decoder, events,
styles, ONNX model, or INT8 graph.

## Results

| Partition | Macro-F1 | Accuracy | Coverage | Full error | Selective error | APS |
|---|---:|---:|---:|---:|---:|---:|
| Development | 0.6090 | 0.6167 | 0.5250 | 0.3833 | 0.2857 | +0.4600 |
| Opened sealed diagnostic | 0.6043 | 0.6212 | 0.6894 | 0.3788 | 0.2857 | +0.3909 |
| External RAVDESS | 0.1588 | 0.2292 | 0.6333 | 0.7708 | 0.7237 | +0.2083 |

The selected `C` was 0.1, temperature 0.9166, and abstention threshold
0.5212. External anger recall remained 0.7344 and the largest predicted-class
share fell below 0.49, but external macro-F1 missed the predeclared 0.40 gate
by a wide margin. Surprise is absent from the original training corpus and
therefore unsupported by this adapter.

## Interpretation

The frozen representation contains enough signal to separate the original
CREMA-D actor split, but a correction learned from that lineage does not
transfer to a second acted corpus. More optimization on the opened RAVDESS
slice would convert an evaluation set into a tuning set without solving the
underlying data-domain problem.

The next candidate requires additional permissively licensed lexical speech
with same-text delivery variation, including surprise and genuine whispering.
No result here changes `release_ready: false`.

Artefact hashes:

- adapter: `8d2df5a88cfd1a8ee39145f3607b6346ed56a85796ccbe9fe14c13a2a7af4ac8`
- report: `0d2b1ec59b8934f7792bc261d9c1471b36e494240d1a565edbcd581297e24dad`
