"""Speaker-level paired bootstrap for the layer-17 corrections (paper Section V-D).

Run from the repository root:  python week3/speaker_bootstrap.py
Reads subset.csv, scores/ and week3/per_utterance_scores_week3.csv.gz; writes week3/speaker_level_bootstrap.csv.
Each resample draws the 34 held-out speakers with replacement, keeps all of a speaker's utterances, and scores
every method on the same trials, so the intervals are for differences between methods. Values are percentage points.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
import numpy as np, pandas as pd
from pilot_lib import eer as _eer, speaker_split

sub = pd.read_csv('subset.csv'); EV = sub[sub.part == 'eval'].reset_index(drop=True)
EV['half'] = speaker_split(EV.speaker.values, 1234)
T = EV[EV.half == 'test'].reset_index(drop=True)
y, spk = T.label.values, T.speaker.values
dv = pd.read_csv('scores/dev__clean.csv')
THR = _eer(dv[dv.label == 1].score.values, dv[dv.label == 0].score.values)[1]   # clean development threshold
W = pd.read_csv('week3/per_utterance_scores_week3.csv.gz'); W = W[W.part == 'test']

def eer(s, yy): return _eer(s[yy == 1], s[yy == 0])[0]
def hter(s, yy): return (np.mean(s[yy == 1] < THR) + np.mean(s[yy == 0] >= THR)) / 2
def tag(t): return W[W.tag == t].set_index('utt').score.reindex(T.utt).values
def base(stage): return pd.read_csv(f'scores/eval__{stage}.csv').set_index('utt').score.reindex(T.utt).values

us = np.unique(spk); idx = {s: np.where(spk == s)[0] for s in us}
rng = np.random.default_rng(7)
BS = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(1000)]
ci = lambda f: (np.percentile([f(b) for b in BS], 2.5), np.percentile([f(b) for b in BS], 97.5))

rows = []
for st in ['amrnb_4k75', 'g711u_loss10', 'opus_nb_8k']:
    M = {'none': base(st), 'paired': tag(f'{st}__shift__L17'), 'unpaired': tag(f'{st}__unpaired__L17'),
         'leace': tag(f'{st}__leace__L17'), 'coral': tag(f'{st}__coral__L17'), 'nap4': tag(f'{st}__nap4__L17')}
    R = [tag(f'{st}__rand{j}__L17') for j in range(20)]
    pairs = [(k, 'none') for k in ('paired', 'unpaired', 'leace', 'coral', 'nap4')] + \
            [(k, 'paired') for k in ('unpaired', 'leace', 'coral', 'nap4')]
    for a, b in pairs:
        lo, hi = ci(lambda i: eer(M[a][i], y[i]) - eer(M[b][i], y[i]))
        hlo, hhi = ci(lambda i: hter(M[a][i], y[i]) - hter(M[b][i], y[i]))
        rows.append(dict(stage=st, cmp=f'{a} - {b}', d_eer=eer(M[a], y) - eer(M[b], y), lo=lo, hi=hi,
                         d_hter=hter(M[a], y) - hter(M[b], y), hlo=hlo, hhi=hhi))
    v = [np.mean([eer(r[i], y[i]) for r in R]) - eer(M['paired'][i], y[i]) for i in BS[:300]]
    rows.append(dict(stage=st, cmp='random(mean of 20) - paired', d_eer=np.mean([eer(r, y) for r in R]) - eer(M['paired'], y),
                     lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    for k in ('paired', 'none'):
        lo, hi = ci(lambda i: eer(M[k][i], y[i]))
        rows.append(dict(stage=st, cmp=f'{k} EER (level)', d_eer=eer(M[k], y), lo=lo, hi=hi))

O = pd.DataFrame(rows)
for c in ('d_eer', 'lo', 'hi', 'd_hter', 'hlo', 'hhi'):
    O[c] = (100 * O[c]).round(1)
O.to_csv('week3/speaker_level_bootstrap.csv', index=False); print(O.to_string(index=False))
