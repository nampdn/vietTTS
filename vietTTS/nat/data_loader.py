import random
from pathlib import Path

import numpy as np
import textgrid
from scipy.io import wavfile

from .config import FLAGS, AcousticInput, DurationInput


def load_phonemes_set_from_lexicon_file(fn: Path):
    S = set()
    for line in open(fn, "r").readlines():
        word, phonemes = line.strip().lower().split("\t")
        phonemes = phonemes.split()
        S.update(phonemes)

    S = FLAGS.special_phonemes + sorted(list(S))
    return S


def pad_seq(s, maxlen, value=0):
    assert maxlen >= len(s)
    return tuple(s) + (value,) * (maxlen - len(s))


def is_in_word(phone, word):
    def time_in_word(time, word):
        return (word.minTime - 1e-3) < time and (word.maxTime + 1e-3) > time

    return time_in_word(phone.minTime, word) and time_in_word(phone.maxTime, word)


def load_textgrid(fn: Path):
    tg = textgrid.TextGrid.fromFile(str(fn.resolve()))
    data = []
    words = list(tg[0])
    widx = 0
    assert tg[1][0].minTime == 0, "The first phoneme has to start at time 0"
    for p in tg[1]:
        if not p in words[widx]:
            widx = widx + 1
            if len(words[widx - 1].mark) > 0:
                data.append((FLAGS.special_phonemes[FLAGS.word_end_index], 0.0))
            if widx >= len(words):
                break
            assert p in words[widx], "mismatched word vs phoneme"
        data.append((p.mark.strip().lower(), p.duration()))
    return data


def textgrid_data_loader(data_dir: Path, seq_len: int, batch_size: int, mode: str):
    tg_files = sorted(data_dir.glob("*.TextGrid"))
    random.Random(42).shuffle(tg_files)
    L = len(tg_files) * 95 // 100
    assert mode in ["train", "val"]
    phonemes = load_phonemes_set_from_lexicon_file(data_dir / "lexicon.txt")
    if mode == "train":
        tg_files = tg_files[:L]
    if mode == "val":
        tg_files = tg_files[L:]

    data = []
    for fn in tg_files:
      try:
        ps, ds = zip(*load_textgrid(fn))
        ps = [phonemes.index(p) for p in ps]
        l = len(ps)
        ps = pad_seq(ps, seq_len, 0)
        ds = pad_seq(ds, seq_len, 0)
        data.append((ps, ds, l))
      except:
        print("textgrid_data_loader - Skip error file: ", fn)

    batch = []
    while True:
        random.shuffle(data)
        for e in data:
            batch.append(e)
            if len(batch) == batch_size:
                ps, ds, lengths = zip(*batch)
                ps = np.array(ps, dtype=np.int32)
                ds = np.array(ds, dtype=np.float32)
                lengths = np.array(lengths, dtype=np.int32)
                yield DurationInput(ps, lengths, ds)
                batch = []


def load_textgrid_wav(
    data_dir: Path, token_seq_len: int, batch_size, pad_wav_len, mode: str
):
    tg_files = sorted(data_dir.glob("*.TextGrid"))
    random.Random(42).shuffle(tg_files)
    L = len(tg_files) * 95 // 100
    assert mode in ["train", "val", "gta"]
    phonemes = load_phonemes_set_from_lexicon_file(data_dir / "lexicon.txt")
    if mode == "gta":
        tg_files = tg_files  # all files
    elif mode == "train":
        tg_files = tg_files[:L]
    elif mode == "val":
        tg_files = tg_files[L:]

    data = []
    for fn in tg_files:
      try:
        ps, ds = zip(*load_textgrid(fn))
        ps = [phonemes.index(p) for p in ps]
        l = len(ps)
        ps = pad_seq(ps, token_seq_len, 0)
        ds = pad_seq(ds, token_seq_len, 0)

        wav_file = data_dir / f"{fn.stem}.wav"
        sr, y = wavfile.read(wav_file)
        y = np.copy(y)
        start_time = 0
        for i, (phone_idx, duration) in enumerate(zip(ps, ds)):
            l = int(start_time * sr)
            end_time = start_time + duration
            r = int(end_time * sr)
            if i == len(ps) - 1:
                r = len(y)
            if phone_idx < len(FLAGS.special_phonemes):
                y[l:r] = 0
            start_time = end_time

        if len(y) > pad_wav_len:
            y = y[:pad_wav_len]
        wav_length = len(y)
        y = np.pad(y, (0, pad_wav_len - len(y)))
        data.append((fn.stem, ps, ds, l, y, wav_length))
      except:
        print("load_textgrid_wav - Skip error file: ", fn)

    batch = []
    while True:
        random.shuffle(data)
        for idx, e in enumerate(data):
            batch.append(e)
            if len(batch) == batch_size or (mode == "gta" and idx == len(data) - 1):
                names, ps, ds, lengths, wavs, wav_lengths = zip(*batch)
                ps = np.array(ps, dtype=np.int32)
                ds = np.array(ds, dtype=np.float32)
                lengths = np.array(lengths, dtype=np.int32)
                wavs = np.array(wavs)
                wav_lengths = np.array(wav_lengths, dtype=np.int32)
                if mode == "gta":
                    yield names, AcousticInput(ps, lengths, ds, wavs, wav_lengths, None)
                else:
                    yield AcousticInput(ps, lengths, ds, wavs, wav_lengths, None)
                batch = []
        if mode == "gta":
            assert len(batch) == 0
            break
