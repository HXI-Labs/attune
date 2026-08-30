# BERSt style v0.1 results

The v0.1 BERSt style candidate is rejected. It improved shouting recall on
BERSt development data, but it did not meet the false-positive or external
WESR precision gates. The BERSt sealed split was not opened.

## Run

- Training revision: `1542a3a3cb34e6bd89c8a6a11ee70e9230910bb6`.
- Training-source SHA-256: `16bc6b1ca07b194bffe5e837df134521d7193df3e3b4682e6b2827bb8dcd6933`.
- Manifest SHA-256: `3b2a798d592ece4b4a516d25f9d41a86d9960797b583e7cb650de6f59868d454`.
- Initial checkpoint SHA-256: `3e47e9b7974718f2589e0623d40e61e4c7eb6116d614e234694b52866f686e08`.
- Candidate checkpoint SHA-256: `c83467d41e6238d24cefa85d93133d26e0710a15757febccb73046075586211d`.
- Model parameters: 241,904,650 total; 295,810 trained; zero encoder parameters trained.
- Selected checkpoint: epoch 9, validation loss 0.0159288.

The checkpoint-scope audit passed. Only `style_projection` and `style_head`
tensors changed. The training lineage is stored at
`artifacts/training/local-berst-style-v0.1/lineage.json`.

## Development and external controls

| Gate | Result | Requirement | Pass |
| --- | ---: | ---: | :---: |
| BERSt shouting F1 | 0.8604 | at least 0.85 | yes |
| BERSt shouting precision | 0.8057 | at least 0.80 | yes |
| BERSt shouting recall | 0.9231 | at least 0.80 | yes |
| BERSt no-shout false-positive rate | 0.2292 | at most 0.05 | no |
| WESR shouting F1 | 0.4855 | at least 0.55 | no |
| WESR shouting precision | 0.3652 | at least 0.50 | no |
| WESR shouting recall | 0.7241 | at least 0.60 | yes |
| British negative-control false-positive clips | 0 | 0 | yes |
| Checkpoint scope | passed | passed | yes |

BERSt shouting average precision was 0.9249, but this ranking quality was not
sufficient at the required operating point. Raising the calibrated threshold
from 0.62 to 0.88 reduced BERSt false-positive rate to 0.0417, while recall fell
to 0.6235 and F1 to 0.7494. Calibration therefore cannot make this checkpoint
pass all predeclared gates.

The WESR result also shows a domain and temporal-scope gap: its shouting average
precision was 0.3706. WESR clips can contain a bounded shouting span inside a
longer utterance, whereas this candidate pools the complete utterance into one
style score. That difference is a plausible contributor, not a proven cause.

## Decision

The candidate remains disabled and must not be composed into a release model.
The next experiment must improve acoustic separation rather than weaken the
requirements. It will correct target-specific training so the upper perception
encoder can be adapted while the copied ASR tail remains frozen. Any successor
protocol must be committed before training, and the BERSt sealed split must
remain unopened until the development and external gates pass.
