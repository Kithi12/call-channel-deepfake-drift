# call-channel-deepfake-drift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24440109.svg)](https://doi.org/10.5281/zenodo.24440109)

Code and results for:

> K. K. Kaluarachchi, A. Aponso, and M. Edirisooriya, "Low-Bitrate Coding and Packet Loss Make a Self-Supervised Speech Deepfake Detector Reject Genuine Callers: A Paired, Layer-Wise Analysis," submitted to *IEEE Access*, 2026.

The study passes ASVspoof 2019 LA evaluation audio through 22 sample-aligned call stages (band-limiting, G.711, AMR-NB/WB, GSM, Opus, frame and packet loss, WebRTC processing), scores clean and call copies with a frozen SSL-AASIST detector, and follows the call-induced drift through the 24 XLS-R layers. It then tests the drift by transplanting it into clean activations and removing it from call activations, against random, LEACE, NAP, CORAL, BatchNorm and recalibration baselines, and checks the pattern on real transmission in ASVspoof 2021 LA.

## Status

Complete. This release (v1.1.0) contains every result reported in the paper. No audio is redistributed.

## Repository layout

```
pilot_lib.py      shared code used by every notebook: call-stage simulator, codec-delay
                  alignment, SSL-AASIST loader, layer taps and interventions, metrics
subset.csv        the evaluation (1,988) and development (998) utterances used, drawn
                  from ASVspoof 2019 LA with seed 1234 (speaker, utt, attack, label, part)
notebooks/        week1 (behaviour), week2 (inside the model, controls), week3 (Kaggle:
                  layer sweep, baselines, gate, ASVspoof 2021 LA, random draws)
scores/           checkpoint A: per-utterance scores, one CSV per call stage (eval__<stage>.csv)
scores_B/         checkpoint B: same layout
week1/            per-stage summary (EER, rates at the threshold, minDCF, DCF at threshold)
week2/            layer geometry, probes, test-time correction, threshold robustness,
                  ElevenLabs/Respeecher scores
week3/            layer sweep, baselines, gate, shortcut controls, ASVspoof 2021 LA,
                  20 random directions, per-utterance week-3 scores, fit/test split
```

## Which file reproduces what

| Paper | File(s) |
|---|---|
| Fig. 1, Table II (behaviour per stage) | `week1/summary_week1.csv`; per utterance: `scores/`, `scores_B/` |
| Table III (realistic channels) | `scores/eval__*` (burst, plc, amrwb, opus_wb, gsm_fr, webrtc); `week2/week2_results.csv` |
| Threshold robustness (Sec. IV-D) | `week2/week2_threshold_robustness.csv` |
| Silence and loudness controls (Sec. IV-F) | `scores/*__rms.csv`, `scores/*__trim.csv`; `week3/shortcut_controls.csv` |
| Recent generators (Sec. IV-G) | `week2/elevenlabs_scores.csv` |
| Fig. 2, probes (Sec. V-A, V-C) | `week2/inside_geometry.csv`, `week2/inside_probes.csv` |
| Transplant (Sec. V-B) | `week2/week2_summary.txt` |
| Table IV (corrections at layers 5/11/17/23) | `week2/test_time_fix.csv` |
| Fig. 3 (layer sweep, chosen layer) | `week3/layer_sweep.csv`, `week3/chosen_layers.json` |
| Table V (baselines at layer 17) | `week3/baselines_at_chosen_layer.csv`, `week3/random_draws_L17.csv` |
| Table VI, call gate (Sec. V-D) | `week3/gate_end_to_end.csv`, `week3/gate_detection.csv` |
| Table VII, Fig. 4 (ASVspoof 2021 LA) | `week3/la21_behaviour.csv`, `week3/la21_geometry.csv`, `week3/la21_transplant.csv` |
| Per-utterance week-3 scores | `week3/per_utterance_scores_week3.csv.gz` (columns: tag, part, utt, speaker, label, score) with `week3/fit_test_split.csv` |

Scores are the bona fide logit of SSL-AASIST (higher = more genuine). Rates use the clean development EER threshold (−3.44 for checkpoint A). The CSV column `actDCF_asv5` is the ASVspoof 5 detection cost at that threshold, called DCF<sub>thr</sub> in the paper; it is not the official Bayes-threshold actDCF.

## Detector and data

- **SSL-AASIST** (Tak et al., Odyssey 2022), code from [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) at commit `4acaa61`, with the XLS-R 300M front end. Checkpoint A = `Best_LA_model_for_DF.pth`, checkpoint B = `LA_model.pth`, both from the authors' release. All weights are frozen.
- **ASVspoof 2019 LA** ([Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)).
- **ASVspoof 2021 LA** evaluation audio and keys ([asvspoof.org](https://www.asvspoof.org/index2021.html)); we used the Hugging Face copy `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA`.
- **ElevenLabs and Respeecher fakes**, Mendeley Data, doi:10.17632/79g59sp69z.1.

## Reproducing

1. Download the datasets and checkpoints above; place the audio listed in `subset.csv` and the checkpoints as the first cells of each notebook describe.
2. Run `notebooks/week1_*`, then `week2_*` (Google Colab, T4 GPU), then `week3_kaggle.ipynb` (Kaggle, T4, Internet on). Each notebook skips steps whose outputs already exist.
3. The speaker split is deterministic: speakers sorted, shuffled with `numpy.random.default_rng(1234)`, first half = fit (1,063 utterances), second half = test (925).

Notes: G.711 frame loss is seeded per utterance; Opus packet loss uses `opusdec --packet-loss`, whose loss pattern is fixed and identical for every file. Software: Python 3, PyTorch 2.14.1, Transformers 5.22.1, ffmpeg 7.0.2, opus-tools, webrtc-noise-gain (see `requirements.txt`).

## Citation

Please cite the paper above and this archive (doi:10.5281/zenodo.24440109, all versions).

## License

Code: see `LICENSE`. The datasets and checkpoints keep their own licences and are not redistributed here.
