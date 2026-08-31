# Third-party model notice

Attune Cadence 242M is derived from **SenseVoiceSmall by
FunASR/FunAudioLLM**.

- Upstream model: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- Upstream project: https://github.com/FunAudioLLM/SenseVoice
- Model agreement: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- Pinned Attune base revision: `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`

The FunASR Model Open Source License Agreement v1.1 applies to the upstream
weights and their derivatives, including fine-tuned models. It requires source
and author attribution and retention of relevant model names. Project Attune's
MIT code licence does not replace or override that agreement.

## Training data

The candidate training manifest records the following source datasets. Source
audio is not included in the model repository. The listed terms continue to
govern the source data and any separately distributed derived datasets.

| Source | Use | Terms |
| --- | --- | --- |
| [Mozilla Common Voice 17 English](https://commonvoice.mozilla.org/en/datasets) | ASR replay and speech used in event mixtures | CC0 1.0 |
| [VocalSound](https://github.com/YuanGongND/vocalsound) | Vocal-event supervision and event-mixture sources | CC BY-SA 4.0; attribution and share-alike apply to distributed derivative datasets |
| [DCASE 2016 Task 2](https://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio) | Synthetic strong event timing | Train/development CC BY 3.0; public test CC BY 4.0 |
| [FSD50K](https://zenodo.org/records/4060432) | Bounded weak event/style supervision | Annotations CC BY 4.0; selected clips are CC0 1.0 or CC BY 3.0 |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Listener-distribution affect supervision | ODbL 1.0 database and Database Contents License |
| [SUBESCO 1.1](https://zenodo.org/records/4526477) | Auxiliary affect supervision | CC BY 4.0 |
| [Thorsten-Voice Emotional 2.0](https://zenodo.org/records/5525023) | Auxiliary affect and whisper supervision | CC0 1.0 |
| [BERSt](https://huggingface.co/datasets/Rosie-Lab/BERSt) | English shouting, affect, and device-robustness supervision | CC BY 4.0 |
| [DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech) | Weak human vocal-event supervision | Apache 2.0 |

Attune inline-event mixtures combine Common Voice speech with VocalSound
events. They are treated as CC BY-SA 4.0 derivative data and are not included
in the model repository. Dataset versions, revisions, selected-file hashes,
split rules, and known limitations are recorded under `data/provenance/` and
`data/manifests/` in the source repository.
