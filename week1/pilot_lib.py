# ---------------------------------------------------------------------------
# Pilot library: call-stage simulator, alignment, SSL-AASIST loader, metrics.
# Kept in one file so it can be tested outside Colab and pasted into the notebook.
# ---------------------------------------------------------------------------
import os, re, sys, types, json, zlib, shutil, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, correlate

SR = 16000
CUT = 64600          # SSL-AASIST input length (~4.04 s), as in Tak et al.'s code


# ----------------------------- audio helpers ------------------------------
def tak_pad(x, max_len=CUT):
    """Exactly Tak et al.'s pad(): truncate, or tile short clips."""
    if len(x) >= max_len:
        return x[:max_len]
    reps = int(max_len / len(x)) + 1
    return np.tile(x, reps)[:max_len]


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(' '.join(cmd) + '\n' + r.stderr.decode(errors='ignore')[-800:])


def ffmpeg_roundtrip(x, enc_args, codec_sr=8000, ext='wav', ffmpeg='ffmpeg'):
    """16 kHz float -> (resample to codec_sr) -> encode -> decode -> 16 kHz float."""
    with tempfile.TemporaryDirectory() as d:
        inp, mid, out = f'{d}/in.wav', f'{d}/mid.{ext}', f'{d}/out.wav'
        sf.write(inp, np.clip(x, -1, 1), SR, subtype='PCM_16')
        _run([ffmpeg, '-y', '-loglevel', 'error', '-i', inp, '-ac', '1', '-ar', str(codec_sr)] + enc_args + [mid])
        _run([ffmpeg, '-y', '-loglevel', 'error', '-i', mid, '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_f32le', out])
        y, _ = sf.read(out, dtype='float32')
    return y


def align_to(ref, y, max_lag=1600):
    """Remove codec delay: find the lag that best matches ref, shift y, trim/pad to len(ref)."""
    n = min(len(ref), len(y), 2 * SR)
    a, b = ref[:n].astype(np.float64), y[:n].astype(np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        lag = 0
    else:
        c = correlate(b, a, mode='full', method='fft')
        lags = np.arange(-n + 1, n)
        keep = np.abs(lags) <= max_lag
        lag = int(lags[keep][np.argmax(c[keep])])
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag, dtype=y.dtype), y])
    if len(y) < len(ref):
        y = np.concatenate([y, np.zeros(len(ref) - len(y), dtype=y.dtype)])
    return y[:len(ref)].astype(np.float32), lag


def bandpass(x, lo=300, hi=3400, order=6):
    sos = butter(order, [lo, hi], btype='bandpass', fs=SR, output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)   # zero-phase: no delay


def rms_normalise(x, target_dbfs=-26.0):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.max(np.abs(y))
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def frame_loss(x, utt_id, rate, seed, frame=320, fade=40):
    """Drop 20 ms frames at random (seeded per utterance) and conceal by repeating
    the previous frame, attenuated, with short cross-fades. A simplified packet-loss
    concealment, documented as such; not the G.711 Appendix I algorithm."""
    rng = np.random.default_rng((zlib.crc32(utt_id.encode()) ^ seed) & 0xFFFFFFFF)
    y = x.copy()
    n = len(x) // frame
    lost = rng.random(n) < rate
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    run = 0
    for i in range(1, n):
        s = i * frame
        if lost[i]:
            run += 1
            seg = y[s - frame:s] * (0.7 ** run)          # repeat the last frame, attenuated
            seg[:fade] = y[s - 1] * (1 - ramp) + seg[:fade] * ramp   # smooth join, no click
            y[s:s + frame] = seg
        else:
            if run:   # fade back in from the concealed frame
                y[s:s + fade] = y[s - fade:s] * (1 - ramp) + x[s:s + fade] * ramp
            run = 0
    return y, float(lost.mean())


# ----------------------------- conditions ---------------------------------
def available_encoders(ffmpeg='ffmpeg'):
    out = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    return {name for name in ('libgsm', 'libopus', 'pcm_mulaw', 'libopencore_amrnb', 'libvo_amrwbenc') if re.search(rf'\s{name}\s', out)}


def make_conditions(encoders, seed, ffmpeg='ffmpeg'):
    """Returns {name: (description, fn(x, utt_id) -> (y, info))}. Every output is
    16 kHz, the same length as the input and sample-aligned with it."""
    C = {}
    C['clean'] = ('No processing', lambda x, u: (x, {}))
    C['clean_rms'] = ('Clean, loudness-normalised to -26 dBFS (loudness/AGC control)',
                      lambda x, u: (rms_normalise(x), {}))
    C['bandpass_300_3400'] = ('Telephone band 300-3,400 Hz, zero-phase filter (band-limit control)',
                              lambda x, u: (bandpass(x), {}))

    def coded(args, ext, sr=8000):
        def fn(x, u):
            y, lag = align_to(x, ffmpeg_roundtrip(x, args, sr, ext, ffmpeg))
            return y, {'lag': lag}
        return fn

    C['nb_resample'] = ('16 -> 8 -> 16 kHz resampling only (control for all 8 kHz codecs)',
                        coded(['-c:a', 'pcm_s16le'], 'wav'))
    if 'pcm_mulaw' in encoders:
        C['g711u'] = ('G.711 mu-law, 64 kbps', coded(['-c:a', 'pcm_mulaw'], 'wav'))

        def g711_loss(x, u):
            y, info = C['g711u'][1](x, u)
            y, lost = frame_loss(y, u, 0.10, seed)
            return y, dict(info, lost=lost)
        C['g711u_loss10'] = ('G.711 mu-law with 10% 20 ms frame loss, repetition concealment', g711_loss)
    if 'libgsm' in encoders:
        C['gsm_fr'] = ('GSM full-rate, 13 kbps (2G mobile)', coded(['-c:a', 'libgsm'], 'gsm'))
    if 'libopencore_amrnb' in encoders:
        C['amrnb_12k2'] = ('AMR-NB 12.2 kbps (mobile)', coded(['-c:a', 'libopencore_amrnb', '-b:a', '12.2k'], 'amr'))
        C['amrnb_4k75'] = ('AMR-NB 4.75 kbps (mobile, low rate)', coded(['-c:a', 'libopencore_amrnb', '-b:a', '4.75k'], 'amr'))
    if 'libopus' in encoders:
        C['opus_nb_8k'] = ('Opus narrowband, 8 kbps, VoIP mode',
                           coded(['-c:a', 'libopus', '-b:a', '8k', '-application', 'voip'], 'opus'))
    return C


# ----------------------------- metrics ------------------------------------
def det_curve(tar, non):
    """Identical to Tak et al.'s compute_det_curve (target = bona fide)."""
    n = tar.size + non.size
    s = np.concatenate([tar, non]); l = np.concatenate([np.ones(tar.size), np.zeros(non.size)])
    idx = np.argsort(s, kind='mergesort'); l = l[idx]
    tsum = np.cumsum(l); nsum = non.size - (np.arange(1, n + 1) - tsum)
    frr = np.concatenate([[0], tsum / tar.size]); far = np.concatenate([[1], nsum / non.size])
    thr = np.concatenate([[s[idx[0]] - 0.001], s[idx]])
    return frr, far, thr


def eer(tar, non):
    frr, far, thr = det_curve(tar, non)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean([frr[i], far[i]])), float(thr[i])


def rates_at(tar, non, t):
    """Bona fide rejected (false alarm on a real caller) and spoof accepted (miss), at threshold t.
    A score >= t is accepted as bona fide."""
    return float(np.mean(tar < t)), float(np.mean(non >= t))


def dcf(p_fa_bona, p_miss_spoof, p_spoof, c_rej_bona=1.0, c_acc_spoof=10.0):
    """Normalised detection cost. 'Rejecting bona fide' and 'accepting spoof' are the two errors.
    Defaults follow ASVspoof 5 (C=1 and 10, P_spoof=0.05)."""
    cost = c_rej_bona * (1 - p_spoof) * p_fa_bona + c_acc_spoof * p_spoof * p_miss_spoof
    return float(cost / min(c_rej_bona * (1 - p_spoof), c_acc_spoof * p_spoof))


def min_dcf(tar, non, p_spoof, c1=1.0, c2=10.0):
    frr, far, _ = det_curve(tar, non)
    return float(min(dcf(fr, fa, p_spoof, c1, c2) for fr, fa in zip(frr, far)))


def boot_ci(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, d.size, replace=True).mean() for _ in range(n)]
    return float(np.mean(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


PRIORS = {'asv5': (0.05, 1.0, 10.0), 'balanced': (0.5, 1.0, 1.0)}


def summarise(scores, clean, thr):
    """scores/clean: DataFrames indexed by utt with columns score, label(1=bona fide), correct_clean."""
    out = {}
    for subset, mask in (('all', np.ones(len(scores), bool)), ('clean_correct', scores['correct_clean'].values)):
        s = scores[mask]; c = clean.loc[s.index]
        tar, non = s[s.label == 1].score.values, s[s.label == 0].score.values
        if tar.size < 2 or non.size < 2:          # e.g. an empty clean-correct subset
            out[subset] = {'eer': np.nan, 'fa_bona_at_thr': np.nan, 'miss_spoof_at_thr': np.nan,
                           'n_bona': int(tar.size), 'n_spoof': int(non.size)}
            continue
        e, _ = eer(tar, non)
        fa, miss = rates_at(tar, non, thr)
        r = {'eer': e, 'fa_bona_at_thr': fa, 'miss_spoof_at_thr': miss, 'hter_at_thr': (fa + miss) / 2}
        for k, (p, c1, c2) in PRIORS.items():
            r[f'minDCF_{k}'] = min_dcf(tar, non, p, c1, c2)
            r[f'actDCF_{k}'] = dcf(fa, miss, p, c1, c2)
            r[f'calib_loss_{k}'] = r[f'actDCF_{k}'] - r[f'minDCF_{k}']
        for lab, name in ((1, 'bona'), (0, 'spoof')):
            d = (s[s.label == lab].score.values - c[c.label == lab].score.values)
            m, lo, hi = boot_ci(d)
            r[f'shift_{name}_mean'], r[f'shift_{name}_lo'], r[f'shift_{name}_hi'] = m, lo, hi
        r['n_bona'], r['n_spoof'] = int(tar.size), int(non.size)
        out[subset] = r
    return out


def decision(row_clean, row):
    """Pre-registered pilot rule: clean EER <= 5% AND (EER rises >= 10 points OR
    real callers rejected at the transferred threshold rise >= 15 points)."""
    ok_clean = row_clean['eer'] <= 0.05
    d_eer = row['eer'] - row_clean['eer']
    d_fa = row['fa_bona_at_thr'] - row_clean['fa_bona_at_thr']
    counts = ok_clean and (d_eer >= 0.10 or d_fa >= 0.15)
    if not counts:
        kind = 'no_effect'
    elif d_eer >= 0.10:
        kind = 'ranking_breaks'
    else:
        kind = 'threshold_only'
    return counts, kind, d_eer, d_fa


# ----------------------------- model loader -------------------------------
def xlsr_config(n_layers=24, conv_bias=True):
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(
        hidden_size=1024, num_hidden_layers=n_layers, num_attention_heads=16, intermediate_size=4096,
        hidden_act='gelu', hidden_dropout=0.0, activation_dropout=0.0, attention_dropout=0.0,
        feat_proj_dropout=0.0, final_dropout=0.0, layerdrop=0.0, layer_norm_eps=1e-5,
        feat_extract_norm='layer', feat_extract_activation='gelu', conv_dim=(512,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_bias=conv_bias,
        num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16, do_stable_layer_norm=True,
        apply_spec_augment=False, mask_time_prob=0.0, mask_feature_prob=0.0)


IGNORED_FS_PREFIXES = ('quantizer.', 'project_q.', 'final_proj.', 'project_inp.', 'target_glu.', 'input_quantizer.', 'mask_emb')


def map_fairseq_to_hf(sd_fs, hf_keys):
    """Rename fairseq wav2vec 2.0 parameter names to Hugging Face Wav2Vec2Model names.
    Module names checked against the fairseq commit bundled in Tak et al.'s repository."""
    hf_keys = set(hf_keys)
    param_wn = any('parametrizations.weight.original0' in k for k in hf_keys)
    out, unused = {}, []
    for k, v in sd_fs.items():
        nk = None
        if m := re.fullmatch(r'feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)', k):
            nk = f'feature_extractor.conv_layers.{m[1]}.conv.{m[2]}'
        elif m := re.fullmatch(r'feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)', k):
            nk = f'feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}'
        elif k in ('layer_norm.weight', 'layer_norm.bias'):
            nk = 'feature_projection.' + k
        elif m := re.fullmatch(r'post_extract_proj\.(weight|bias)', k):
            nk = f'feature_projection.projection.{m[1]}'
        elif k == 'encoder.pos_conv.0.bias':
            nk = 'encoder.pos_conv_embed.conv.bias'
        elif k == 'encoder.pos_conv.0.weight_g':
            nk = 'encoder.pos_conv_embed.conv.' + ('parametrizations.weight.original0' if param_wn else 'weight_g')
        elif k == 'encoder.pos_conv.0.weight_v':
            nk = 'encoder.pos_conv_embed.conv.' + ('parametrizations.weight.original1' if param_wn else 'weight_v')
        elif m := re.fullmatch(r'encoder\.layers\.(\d+)\.(.+)', k):
            rest = (m[2].replace('self_attn_layer_norm', 'layer_norm').replace('self_attn.', 'attention.')
                    .replace('fc1.', 'feed_forward.intermediate_dense.').replace('fc2.', 'feed_forward.output_dense.'))
            nk = f'encoder.layers.{m[1]}.{rest}'
        elif k in ('encoder.layer_norm.weight', 'encoder.layer_norm.bias'):
            nk = k
        elif k == 'mask_emb':
            nk = 'masked_spec_embed'
        if nk is None or nk not in hf_keys:
            unused.append(k)
        else:
            out[nk] = v
    missing = sorted(hf_keys - set(out))
    return out, unused, missing


def build_ssl_aasist(repo_dir, ckpt_path, device='cpu'):
    """Tak et al.'s SSL-AASIST with the fairseq front-end replaced by an identical
    Hugging Face Wav2Vec2Model, so no fairseq install is needed. Loads strictly."""
    import torch, torch.nn as nn
    from transformers import Wav2Vec2Model
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    pre = 'ssl_model.model.'
    fs = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    n_layers = 1 + max(int(m[1]) for k in fs if (m := re.match(r'encoder\.layers\.(\d+)\.', k)))
    conv_bias = 'feature_extractor.conv_layers.0.0.bias' in fs
    cfg = xlsr_config(n_layers, conv_bias)

    class HFSSL(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.model = Wav2Vec2Model(cfg)
            self.out_dim = cfg.hidden_size
            self.device = device

        def extract_feat(self, x, output_hidden_states=False):
            if x.ndim == 3:
                x = x[:, :, 0]
            o = self.model(x, output_hidden_states=output_hidden_states)
            if output_hidden_states:
                self.last_hidden_states = o.hidden_states
            return o.last_hidden_state

    if 'fairseq' not in sys.modules:                 # model.py imports fairseq at the top; never used here
        sys.modules['fairseq'] = types.ModuleType('fairseq')
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import model as tak
    tak.SSLModel = HFSSL
    net = tak.Model(None, device)
    hf_keys = list(net.ssl_model.model.state_dict().keys())
    mapped, unused, missing = map_fairseq_to_hf(fs, hf_keys)
    bad_unused = [k for k in unused if not k.startswith(IGNORED_FS_PREFIXES)]
    missing = [k for k in missing if k != 'masked_spec_embed']
    if bad_unused or missing:
        raise RuntimeError(f'Weight mapping failed.\nUnmapped checkpoint keys: {bad_unused[:20]}\nMissing model keys: {missing[:20]}')
    full = {k: v for k, v in sd.items() if not k.startswith(pre)}
    full.update({pre + k: v for k, v in mapped.items()})
    if 'masked_spec_embed' in hf_keys and 'masked_spec_embed' not in mapped:   # only exists in some versions
        full[pre + 'masked_spec_embed'] = net.ssl_model.model.masked_spec_embed.detach()
    net.load_state_dict(full, strict=True)
    report = {'xlsr_layers': n_layers, 'conv_bias': conv_bias, 'ssl_params_loaded': len(mapped),
              'ignored_pretraining_params': len(unused), 'backend_params_loaded': len(sd) - len(fs)}
    return net.to(device).eval(), report


# ===========================================================================
# Week 2 additions: shortcut controls, more realistic channels, a second
# architecture (AASIST) and hooks for looking inside the XLS-R front end.
# ===========================================================================
def trim_bounds(x, frame=320, rel_db=-35.0, margin=1600):
    """Start/end sample of the speech region: frames within rel_db of the loudest
    20 ms frame, padded by 100 ms. Computed on the CLEAN copy and applied to the
    call copy too, so the pair stays aligned (silence-shortcut control)."""
    n = max(1, len(x) // frame)
    e = np.array([np.mean(x[i * frame:(i + 1) * frame] ** 2) for i in range(n)]) + 1e-12
    db = 10 * np.log10(e / e.max())
    idx = np.where(db > rel_db)[0]
    if idx.size == 0:
        return 0, len(x)
    return max(0, idx[0] * frame - margin), min(len(x), (idx[-1] + 1) * frame + margin)


def opus_plc(x, loss_pct, bitrate_k=8, opusenc='opusenc', opusdec='opusdec', ffmpeg='ffmpeg'):
    """Opus narrowband speech at bitrate_k kbit/s with real Opus packet-loss concealment
    (opusdec --packet-loss). opusdec draws losses from an unseeded rand(), so the loss
    pattern is deterministic and identical for every file: documented, reproducible."""
    with tempfile.TemporaryDirectory() as d:
        sf.write(f'{d}/in.wav', np.clip(x, -1, 1), SR, subtype='PCM_16')
        _run([ffmpeg, '-y', '-loglevel', 'error', '-i', f'{d}/in.wav', '-ac', '1', '-ar', '8000', f'{d}/in8.wav'])
        _run([opusenc, '--quiet', '--bitrate', str(bitrate_k), '--speech', '--framesize', '20', f'{d}/in8.wav', f'{d}/a.opus'])
        _run([opusdec, '--quiet', '--rate', str(SR), '--packet-loss', str(loss_pct), f'{d}/a.opus', f'{d}/o.wav'])
        y, _ = sf.read(f'{d}/o.wav', dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    return y


def webrtc_apm(x, noise_suppression=2, auto_gain_dbfs=3):
    """WebRTC noise suppression + automatic gain control (+ its high-pass filter), as phones
    and VoIP apps apply before encoding. 10 ms chunks of 16 kHz int16."""
    from webrtc_noise_gain import AudioProcessor
    ap = AudioProcessor(auto_gain_dbfs, noise_suppression)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = len(pcm) // 160 * 160
    out = bytearray()
    for i in range(0, n, 160):
        out += ap.Process10ms(pcm[i:i + 160].tobytes()).audio
    y = np.frombuffer(bytes(out), dtype=np.int16).astype(np.float32) / 32768
    return np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])


def _coded(args, ext, sr, ffmpeg):
    def fn(x, u):
        y, lag = align_to(x, ffmpeg_roundtrip(x, args, sr, ext, ffmpeg))
        return y, {'lag': lag}
    return fn


def frame_loss_ge(x, utt_id, mean_loss, mean_burst, seed, frame=320, fade=40):
    """Bursty 20 ms frame loss from a two-state Gilbert-Elliott chain (seeded per utterance), with the
    same repetition concealment as frame_loss. mean_loss = long-run loss rate, mean_burst = frames per burst."""
    rng = np.random.default_rng((zlib.crc32(utt_id.encode()) ^ (seed + 7)) & 0xFFFFFFFF)
    p_bg = 1.0 / mean_burst                       # bad -> good
    p_gb = mean_loss * p_bg / (1 - mean_loss)     # good -> bad
    n = len(x) // frame
    lost, bad = np.zeros(n, bool), False
    for i in range(n):
        bad = (rng.random() < p_gb) if not bad else (rng.random() >= p_bg)
        lost[i] = bad
    lost[0] = False
    y = x.copy(); ramp = np.linspace(0, 1, fade, dtype=np.float32); run = 0
    for i in range(1, n):
        s = i * frame
        if lost[i]:
            run += 1
            seg = y[s - frame:s] * (0.7 ** run)
            seg[:fade] = y[s - 1] * (1 - ramp) + seg[:fade] * ramp
            y[s:s + frame] = seg
        else:
            if run:
                y[s:s + fade] = y[s - fade:s] * (1 - ramp) + x[s:s + fade] * ramp
            run = 0
    return y, float(lost.mean())


def make_conditions_v2(encoders, seed, ffmpeg='ffmpeg', ffmpeg_gsm=None, have_opus_tools=False, have_webrtc=False):
    """Week-2 stages. Returns {name: (description, fn(x, utt_id) -> (y, info))}.
    Shortcut controls (names with __rms / __trim) are added later, per damaging stage."""
    base = make_conditions(encoders, seed, ffmpeg)
    C = {}
    if ffmpeg_gsm and 'libgsm' in available_encoders(ffmpeg_gsm):
        def gsm(x, u):
            y, lag = align_to(x, ffmpeg_roundtrip(x, ['-c:a', 'libgsm'], 8000, 'gsm', ffmpeg_gsm))
            return y, {'lag': lag}
        C['gsm_fr'] = ('GSM full-rate, 13 kbit/s (2G mobile)', gsm)
    if have_opus_tools:
        for p in (5, 10, 20):
            def f(x, u, p=p):
                y, lag = align_to(x, opus_plc(x, p, ffmpeg=ffmpeg))
                return y, {'lag': lag, 'lost': p / 100}
            C[f'opus_nb_8k_plc{p}'] = (f'Opus 8 kbit/s, {p}% packet loss, native Opus concealment', f)
    if have_webrtc and 'amrnb_12k2' in base:
        def apm_amr(x, u):
            w, lag = align_to(x, webrtc_apm(x))          # WebRTC adds a short delay: remove it first
            y, info = base['amrnb_12k2'][1](w, u)
            return y, dict(info, lag=lag + info.get('lag', 0))
        C['webrtc_amrnb_12k2'] = ('WebRTC noise suppression + gain control, then AMR-NB 12.2 kbit/s', apm_amr)
        def apm(x, u):
            y, lag = align_to(x, webrtc_apm(x))
            return y, {'lag': lag}
        C['webrtc_only'] = ('WebRTC noise suppression + gain control only', apm)
    if 'libopencore_amrnb' in encoders:                      # dose-response over AMR-NB rates
        for r in ('5.9', '7.4'):
            C[f"amrnb_{r.replace('.', 'k')}"] = (f'AMR-NB {r} kbit/s (mobile)', _coded(['-c:a', 'libopencore_amrnb', '-b:a', f'{r}k'], 'amr', 8000, ffmpeg))
    if 'libvo_amrwbenc' in encoders:                         # wideband (HD voice / VoLTE)
        for r in ('12.65', '23.85'):
            C[f"amrwb_{r.replace('.', 'k')}"] = (f'AMR-WB {r} kbit/s (HD voice, 16 kHz)', _coded(['-c:a', 'libvo_amrwbenc', '-b:a', f'{r}k'], 'amr', 16000, ffmpeg))
    if 'libopus' in encoders:                                # wideband VoIP, as WebRTC apps normally run
        for r in (16, 24):
            C[f'opus_wb_{r}k'] = (f'Opus wideband {r} kbit/s, VoIP mode', _coded(['-c:a', 'libopus', '-b:a', f'{r}k', '-application', 'voip'], 'opus', 16000, ffmpeg))
    if 'g711u' in base:                                      # bursty (Gilbert-Elliott) loss, mean burst 3 frames
        for m in (5, 10):
            def ge(x, u, m=m):
                y, info = base['g711u'][1](x, u)
                y, lost = frame_loss_ge(y, u, m / 100, 3.0, seed)
                return y, dict(info, lost=lost)
            C[f'g711u_burst{m}'] = (f'G.711 with {m}% bursty frame loss (Gilbert-Elliott, mean burst 60 ms)', ge)
    if 'amrnb_12k2' in base:
        def chain(x, u):
            y, info = base['amrnb_12k2'][1](bandpass(x), u)
            y, lost = frame_loss(y, u, 0.05, seed)
            return y, dict(info, lost=lost)
        C['chain_bp_amr12k2_loss5'] = ('Band-limit, AMR-NB 12.2 kbit/s, 5% frame loss', chain)
    return base, C


def shortcut_variants(stage_fn):
    """Loudness and silence controls applied AFTER the call stage (B in the mechanism slide).
    __rms : call copy loudness-normalised to -26 dBFS.
    __trim: clean-defined speech region cut from the call copy (leading/trailing silence removed)."""
    def rms(x, u):
        y, info = stage_fn(x, u)
        return rms_normalise(y), info
    def trim(x, u):
        y, info = stage_fn(x, u)
        s, e = trim_bounds(x)
        return y[s:e], dict(info, trim=(s, e))
    return rms, trim


def clean_trim(x, u):
    s, e = trim_bounds(x)
    return x[s:e], {'trim': (s, e)}


# ----------------------------- second architecture ------------------------
def build_aasist(repo_dir, device='cpu', variant='AASIST'):
    """Jung et al.'s released AASIST (raw-waveform, no SSL front end), trained on ASVspoof 2019 LA.
    Wrapped so model(x) returns logits like SSL-AASIST: column 1 = bona fide."""
    import torch, torch.nn as nn, importlib
    conf = json.load(open(f'{repo_dir}/config/{variant}.conf'))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    mod = importlib.import_module('models.' + conf['model_config']['architecture'])
    net = mod.Model(conf['model_config'])
    sd = torch.load(f'{repo_dir}/models/weights/{variant}.pth', map_location='cpu')
    net.load_state_dict(sd, strict=True)

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return self.m(x)[1]
    n_params = sum(p.numel() for p in net.parameters())
    return Wrap(net).to(device).eval(), {'arch': variant, 'params': n_params, 'nb_samp': conf['model_config']['nb_samp']}


# ----------------------------- inside the model ---------------------------
def xlsr_layers(net):
    """The 24 transformer layers of the (Hugging Face) XLS-R front end inside SSL-AASIST."""
    return net.ssl_model.model.encoder.layers


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class LayerTaps:
    """Context manager: records the time-averaged output of every chosen layer (B x 1024 each)."""
    def __init__(self, net, layers):
        self.layers, self.store, self.handles = list(layers), {}, []
        self.mods = xlsr_layers(net)
    def __enter__(self):
        for l in self.layers:
            def hook(m, i, o, l=l):
                self.store[l] = _hs(o).float().mean(1).detach().cpu().numpy()
            self.handles.append(self.mods[l].register_forward_hook(hook))
        return self
    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


class LayerShift:
    """Context manager: adds a fixed vector to every frame of one layer's output
    (test-time steering). vec: 1-D array of size 1024 (use -drift to remove a drift)."""
    def __init__(self, net, layer, vec):
        import torch
        self.mod = xlsr_layers(net)[layer]
        self.vec = torch.as_tensor(np.asarray(vec, np.float32))
        self.h = None
    def __enter__(self):
        def hook(m, i, o):
            h = _hs(o)
            return _replace(o, h + self.vec.to(h.device, h.dtype))
        self.h = self.mod.register_forward_hook(hook)
        return self
    def __exit__(self, *a):
        self.h.remove()


def speaker_split(speakers, seed=0):
    """Deterministic 50/50 split by speaker: 'fit' (estimate directions/thresholds) vs 'test'."""
    sp = sorted(set(speakers))
    rng = np.random.default_rng(seed); rng.shuffle(sp)
    fit = set(sp[:len(sp) // 2])
    return np.array(['fit' if s in fit else 'test' for s in speakers])



class LayerAffine:
    """Context manager: replaces one layer's output h (every frame) by h @ A + b (label-free alignment)."""
    def __init__(self, net, layer, A, b):
        import torch
        self.mod = xlsr_layers(net)[layer]
        self.A = torch.as_tensor(np.asarray(A, np.float32)); self.b = torch.as_tensor(np.asarray(b, np.float32))
        self.h = None
    def __enter__(self):
        def hook(m, i, o):
            h = _hs(o)
            return _replace(o, h @ self.A.to(h.device, h.dtype) + self.b.to(h.device, h.dtype))
        self.h = self.mod.register_forward_hook(hook)
        return self
    def __exit__(self, *a):
        self.h.remove()


def coral_affine(src, tgt, reg=0.1):
    """CORAL (Sun et al., 2016) as an affine map from the call-copy statistics (src) to the clean statistics
    (tgt), both N x D utterance means from the fit half. Returns A (D x D) and b (D) with x' = x @ A + b.
    reg = shrinkage towards the diagonal, since N is close to D."""
    def shrink(C):
        return (1 - reg) * C + reg * np.diag(np.diag(C)) + 1e-6 * np.eye(len(C))
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    ms, mt = src.mean(0), tgt.mean(0)
    Cs, Ct = shrink(np.cov(src.T)), shrink(np.cov(tgt.T))
    A = msqrt(Cs, inv=True) @ msqrt(Ct)
    return A.astype(np.float32), (mt - ms @ A).astype(np.float32)
