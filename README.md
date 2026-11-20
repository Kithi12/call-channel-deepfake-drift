# Call compression pushes genuine speech toward "fake"

Code and results for a paired, stage-by-stage study of how telephone-call processing affects frozen speech deepfake detectors.

**Author:** Kithmina Kaushalya Kaluarachchi, Informatics Institute of Technology (University of Westminster), supervised by Maninda Edirisooriya.
**Status:** research in progress; a journal draft (target: IEEE Access) is in preparation.

## What this repository does

Each utterance from ASVspoof 2019 LA is sent through a series of **call stages**: level normalisation, telephone band-limiting, 8 kHz resampling, G.711, G.711 with frame loss, AMR-NB 12.2 and 4.75 kbit/s, Opus 8 kbit/s, GSM full-rate, Opus with native packet-loss concealment, WebRTC noise suppression and gain control, and a chained call. Every call copy is **sample-aligned with its clean original**, so the same utterance can be compared before and after the call.

The copies are scored by **frozen, released detectors**:

| Detector | Source | Notes |
|---|---|---|
| SSL-AASIST, checkpoint A (`Best_LA_model_for_DF.pth`) | [Tak et al., 2022](https://github.com/TakHemlata/SSL_Anti-spoofing) | XLS-R 300M + AASIST; RawBoost tuned for DF |
| SSL-AASIST, checkpoint B (`LA_model.pth`) | [Tak et al., 2022](https://github.com/TakHemlata/SSL_Anti-spoofing) | RawBoost tuned for LA |
| AASIST | [Jung et al., 2022](https://github.com/clovaai/aasist) | Raw waveform, no SSL front end |

The fairseq XLS-R front end is ported to a Hugging Face `Wav2Vec2Model` with a strict, verified key mapping, so **fairseq is not needed** (see `pilot_lib.py`, `build_ssl_aasist`).

## Main result so far (week 1)

On 1,988 ASVspoof 2019 LA evaluation utterances (1,000 genuine, 988 spoofed, attacks A07–A19):

- **Bandwidth is not the cause.** Band-limiting, 8 kHz resampling and G.711 keep EER at or below 2.9% for both checkpoints.
- **Low-bitrate coding and packet loss are.** For checkpoint A, AMR-NB 4.75 kbit/s raises EER from 0.0% to 27.9%, and 88.9% of genuine utterances are rejected at the threshold set on clean development data.
- **The failure is one-sided.** Genuine-speech scores fall by up to 8.0 units while spoof scores move by less than 0.7. Real callers are flagged, but fakes are not missed more often.
- **Augmentation decides the kind of failure.** Checkpoint B drifts the same way but keeps most of its ranking (EER 7.1% under AMR-NB 4.75 kbit/s).

The full table is in `results/week1/summary_week1.csv`.

## Repository layout

```
pilot_lib.py                        call-stage simulator, alignment, metrics, detector loaders, layer hooks
notebooks/week1_pilot_ssl_aasist.ipynb   reproduction + 9 call stages, 2 checkpoints, threshold robustness
notebooks/week2_inside_channels_shortcut.ipynb   layer-wise drift, training-free fix vs baselines,
                                    realistic codecs (AMR-WB, Opus WB, rate sweep), AASIST, shortcut and floor checks
results/                            summary tables (per-utterance scores, no audio)
```

## How to run

1. Open a notebook in Google Colab and choose a **T4 GPU** runtime.
2. Run the cells in order. The week-1 notebook downloads ASVspoof 2019 LA and the SSL-AASIST checkpoints, extracts a seeded subset to Google Drive and scores it. The week-2 notebook reuses that Drive folder.
3. Every step saves its scores to Drive and resumes if the session drops.

## Data and models (not redistributed here)

| Resource | Link | Licence |
|---|---|---|
| ASVspoof 2019 LA | https://datashare.ed.ac.uk/handle/10283/3336 | ODC-By 1.0 |
| SSL-AASIST code and checkpoints | https://github.com/TakHemlata/SSL_Anti-spoofing | MIT |
| AASIST code and weights | https://github.com/clovaai/aasist | MIT |
| ElevenLabs / Respeecher fakes (optional) | https://data.mendeley.com/datasets/79g59sp69z/1 | CC BY 4.0 |

No audio is stored in this repository, only scores and summary tables.

## Ethics

The project uses public datasets only. It involves no human participants, and it is never connected to a real phone line. A CSE undergraduate ethics declaration was submitted on 4 November 2026.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Third-party code and checkpoints are downloaded at run time from their original repositories and keep their own licences.
