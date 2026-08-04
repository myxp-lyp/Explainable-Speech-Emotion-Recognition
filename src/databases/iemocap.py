"""IEMOCAP loader for E4 (cross-dataset replication).

Standard IEMOCAP layout:
    <root>/Session{1..5}/dialog/EmoEvaluation/*.txt   -- per-dialogue emotion labels
    <root>/Session{1..5}/sentences/wav/<dialog>/<utt>.wav

Emotion classes (paper convention): {ang, hap+exc -> hap, sad, neu}.
Other labels (fru, sur, fea, dis, oth, xxx) are dropped.

Speakers: 10 total (Ses{1..5}{F,M}). Sex is extracted from the utterance
name suffix (F/M in the last group).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# IEMOCAP -> 4-class convention. Order matches CREMA-D's {A, D, F, H, N, S}
# in that we keep alphabetical Char codes for consistency, but for IEMOCAP
# only 4 map cleanly. We use single-char codes:  A(ang), H(hap), N(neu), S(sad).
IEMOCAP_LABEL_MAP = {
    "ang": "A",
    "hap": "H",
    "exc": "H",   # excited is conventionally merged into happy
    "sad": "S",
    "neu": "N",
}
IEMOCAP_LABELS = ["A", "H", "N", "S"]
IEMOCAP_LABEL2ID = {l: i for i, l in enumerate(IEMOCAP_LABELS)}
IEMOCAP_ID2LABEL = {i: l for i, l in enumerate(IEMOCAP_LABELS)}

# The utterance filename pattern is e.g. "Ses01F_impro01_F000.wav" or
# "Ses03M_script03_2_M014.wav"; the *speaker-sex* of the utterance is the
# letter that immediately precedes the last three-digit index.
_UTT_SEX_RE = re.compile(r"([FM])(\d{3})$")


def _parse_speaker_sex_from_utt(utt: str) -> str:
    """Return 'Male' or 'Female'; falls back to the session-level sex if the
    trailing sex-marker cannot be found (rare)."""
    m = _UTT_SEX_RE.search(utt)
    if m:
        return "Male" if m.group(1) == "M" else "Female"
    # e.g. Ses01F -> female session
    if utt[5] == "F":
        return "Female"
    if utt[5] == "M":
        return "Male"
    return "Unknown"


def _speaker_id(utt: str) -> str:
    """Speaker id -- IEMOCAP has 10 speakers total ('Ses<n>F' / 'Ses<n>M').
    The convention: for utterance 'Ses01F_impro01_F000', the *actual speaker*
    is the sex marker in the last group (F -> female speaker), because Ses01
    contains a dyad of one female + one male. So speaker id = 'Ses<n><sex>'."""
    sess = utt[:5]                 # 'Ses01'
    sex = _parse_speaker_sex_from_utt(utt)
    if sex == "Male":
        return sess + "M"
    if sex == "Female":
        return sess + "F"
    return sess


def _parse_emoeval_file(path: Path) -> List[dict]:
    """Parse one EmoEvaluation .txt file.

    Header lines start with '%'. Data blocks start with '[t1 - t2]\tuttID\temo\t[V,A,D]',
    followed by annotator notes until the next '[t1 -' line or EOF.
    """
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("["):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            time_range, utt, emo = parts[0], parts[1], parts[2].strip()
            if emo not in IEMOCAP_LABEL_MAP:
                continue
            rows.append({
                "utt": utt,
                "emotion_raw": emo,
                "emotion": IEMOCAP_LABEL_MAP[emo],
            })
    return rows


def load_iemocap(root: str) -> pd.DataFrame:
    """Return a DataFrame with columns: path, utt, dialog, session, emotion,
    Sex, SpeakerID.
    """
    root = Path(root)
    rows = []
    for session in sorted(root.glob("Session*")):
        emo_dir = session / "dialog" / "EmoEvaluation"
        wav_dir = session / "sentences" / "wav"
        if not emo_dir.exists() or not wav_dir.exists():
            continue
        for txt in emo_dir.glob("*.txt"):
            entries = _parse_emoeval_file(txt)
            for e in entries:
                utt = e["utt"]
                dialog = utt.rsplit("_", 1)[0]  # e.g. Ses01F_impro01
                wav = wav_dir / dialog / f"{utt}.wav"
                if not wav.exists():
                    continue
                rows.append({
                    "path": str(wav),
                    "utt": utt,
                    "dialog": dialog,
                    "session": session.name,
                    "emotion": e["emotion"],
                    "emotion_raw": e["emotion_raw"],
                    "Sex": _parse_speaker_sex_from_utt(utt),
                    "SpeakerID": _speaker_id(utt),
                })
    df = pd.DataFrame(rows)
    df["ActorID"] = df["SpeakerID"]   # alias for API consistency with CREMA-D
    return df


def extract_train_test_sets(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Speaker-stratified split: hold out ~20% of *speakers* for test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["SpeakerID"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap_root", required=True)
    args = ap.parse_args()
    df = load_iemocap(args.iemocap_root)
    print(f"Total utterances (4-class): {len(df)}")
    print("Emotion counts:")
    print(df["emotion"].value_counts())
    print("Speaker counts:")
    print(df["SpeakerID"].value_counts())
    print("Sex counts:")
    print(df["Sex"].value_counts())
    tr, te = extract_train_test_sets(df)
    print(f"Split: train={len(tr)}  test={len(te)}")
    print(f"Test speakers: {sorted(te['SpeakerID'].unique())}")
