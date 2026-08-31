# Hostile lexical control v0.1

The original user recording of “I hate you. I hate you so much. Never call me
again.” produced an accurate transcript but false `whispering`, `cough`, and
`sneeze` annotations. The upload was processed in memory and was not retained.
This control asks a narrower question: can the hostile words alone trigger the
same annotations?

Four macOS system voices read the same sentence at 175 words per minute. The
voices cover British, American, Irish, and South African English. Audio was
converted to mono PCM16 at 16 kHz and analysed with the retained Cadence v0.9
ONNX graph and its fixed calibration. These clips are synthetic negative
controls for events and styles; they are not perceptual affect ground truth.

| Voice | Locale | WAV SHA-256 | Transcript confidence | Anger probability | Events | Styles | Affect |
|---|---|---|---:|---:|---:|---:|---|
| Daniel | en-GB | `65d22b32…` | 0.9932 | 0.1162 | 0 | 0 | abstain |
| Samantha | en-US | `ba3ee0d9…` | 0.9970 | 0.1547 | 0 | 0 | abstain |
| Moira | en-IE | `46717203…` | 0.9977 | 0.0806 | 0 | 0 | abstain |
| Tessa | en-ZA | `cbe8b781…` | 0.9958 | 0.1994 | 0 | 0 | abstain |

Every transcript normalised to:

```text
i hate you i hate you so much never call me again
```

The result passes the lexical-control gate: hostile text did not produce an
event, style, or confident anger decision. It does not close the release gate.
A fresh human recording with the original delivery is still required because
system speech does not reproduce breathing, room acoustics, vocal effort, or
the delivery that caused the failure.

The generated audio and raw inference outputs stay outside Git. Their full
local checksums are:

```text
65d22b3286c37527291a249b1e9c386423687b2e762cfe9f3cbff5c6b9b2173d  daniel.wav
ba3ee0d94c2e19471ff63507b36cafc72e03fb59d8202be3ff5e9614b9b9d40b  samantha.wav
467172033502a6a1d6496b4535b5532a9b1cc29a78c7366b7982256b9c532a21  moira.wav
cbe8b7819cae5e1bff0fbab55e179902cbd332e7f90ddf10edceddcc4c036c8a  tessa.wav
```
