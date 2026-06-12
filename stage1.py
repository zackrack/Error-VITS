#!/usr/bin/env python3
"""Synthesize speech from a CMU-39/ARPAbet phoneme string with VITS.

The existing checkpoints in this repository are trained on eSpeak-style IPA
symbols, so this stage converts CMU-39 phoneme tokens to the matching IPA
symbol inventory before feeding the sequence directly to VITS.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}


# CMUdict/ARPAbet's core 39 phonemes mapped to the IPA symbols used by the
# repository's default eSpeak-phonemized VITS training data.
CMU39_TO_IPA = {
    "AA": "ɑː",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔː",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "iː",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "uː",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

_STRESS_TO_IPA = {
    "1": "ˈ",
    "2": "ˌ",
    "0": "",
}
_STRESS_OVERRIDES = {
    ("AH", "0"): "ə",
    ("ER", "0"): "ɚ",
}

# Punctuation symbols that are already present in text/symbols.py and can be
# passed through to VITS as phrase/sentence boundary hints.
_PUNCTUATION = set(";:,.!?¡¿—…\"«»“”")
_PHONE_RE = re.compile(r"^([A-Z]+)([0-2]?)$")


def cmu39_to_ipa(cmu39: str, keep_stress: bool = True) -> str:
    """Convert a whitespace-delimited CMU-39/ARPAbet string to VITS IPA text.

    Tokens may include optional CMUdict stress digits on vowels, e.g.
    ``HH AH0 L OW1``. Punctuation may be attached to tokens or separated by
    spaces, e.g. ``W ER1 L D !`` or ``W ER1 L D!``.
    """

    ipa_tokens = []
    for raw_token in cmu39.split():
        token = raw_token.strip().upper()
        leading_punctuation = ""
        trailing_punctuation = ""

        while token and token[0] in _PUNCTUATION:
            leading_punctuation += token[0]
            token = token[1:]
        while token and token[-1] in _PUNCTUATION:
            trailing_punctuation = token[-1] + trailing_punctuation
            token = token[:-1]

        if not token:
            ipa_tokens.append(leading_punctuation + trailing_punctuation)
            continue

        match = _PHONE_RE.match(token)
        if match is None:
            raise ValueError("Invalid CMU-39 token '{}'.".format(raw_token))

        phone, stress = match.groups()
        if phone not in CMU39_TO_IPA:
            raise ValueError(
                "Unknown CMU-39 phone '{}'. Supported phones are: {}".format(
                    phone, " ".join(sorted(CMU39_TO_IPA))
                )
            )

        ipa_phone = _STRESS_OVERRIDES.get((phone, stress), CMU39_TO_IPA[phone])
        stress_mark = _STRESS_TO_IPA[stress] if keep_stress and stress else ""
        ipa_tokens.append(
            leading_punctuation + stress_mark + ipa_phone + trailing_punctuation
        )

    ipa = " ".join(ipa_tokens)
    unsupported = sorted({symbol for symbol in ipa if symbol not in symbols})
    if unsupported:
        raise ValueError(
            "Converted IPA contains symbols not present in this VITS model's "
            "symbol table: {}".format(" ".join(unsupported))
        )
    return ipa


def ipa_to_sequence(ipa: str, add_blank: bool) -> list[int]:
    """Convert already-cleaned IPA text to VITS symbol IDs."""

    sequence = [_symbol_to_id[symbol] for symbol in ipa]
    if add_blank:
        sequence = [item for symbol_id in sequence for item in (0, symbol_id)] + [0]
    return sequence


def ipa_to_tensor(ipa: str, add_blank: bool):
    """Convert already-cleaned IPA text to a VITS input tensor."""

    import torch

    return torch.LongTensor(ipa_to_sequence(ipa, add_blank))


def build_model(hps, checkpoint_path: str, device):
    """Create a VITS generator and load its checkpoint."""

    import utils
    from models import SynthesizerTrn

    n_speakers = getattr(hps.data, "n_speakers", 0)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=n_speakers,
        **hps.model
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g


def synthesize(
    cmu39: str,
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    speaker_id=None,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    max_len=None,
    device: str = "auto",
    keep_stress: bool = True,
):
    """Run CMU-39 -> IPA -> VITS inference and write a WAV file."""

    import torch
    from scipy.io.wavfile import write

    import utils

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)

    hps = utils.get_hparams_from_file(config_path)
    ipa = cmu39_to_ipa(cmu39, keep_stress=keep_stress)
    text = ipa_to_tensor(ipa, hps.data.add_blank).to(selected_device)
    net_g = build_model(hps, checkpoint_path, selected_device)

    sid = None
    n_speakers = getattr(hps.data, "n_speakers", 0)
    if n_speakers > 0:
        if speaker_id is None:
            raise ValueError(
                "This config expects a multi-speaker checkpoint; pass --speaker-id."
            )
        sid = torch.LongTensor([speaker_id]).to(selected_device)

    with torch.no_grad():
        x = text.unsqueeze(0)
        x_lengths = torch.LongTensor([text.size(0)]).to(selected_device)
        audio = net_g.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            max_len=max_len,
        )[0][0, 0].data.cpu().float().numpy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, hps.data.sampling_rate, audio)
    return audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech from a CMU-39/ARPAbet phoneme string using VITS."
    )
    parser.add_argument(
        "cmu39",
        help="Whitespace-delimited CMU-39 phones, e.g. 'HH AH0 L OW1 W ER1 L D'.",
    )
    parser.add_argument(
        "--config",
        default="configs/ljs_base.json",
        help="Path to the VITS JSON config used for the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the generator checkpoint, e.g. logs/ljs_base/G_100000.pth.",
    )
    parser.add_argument(
        "--output",
        default="stage1_output.wav",
        help="Path for the generated WAV file.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="Speaker ID for multi-speaker checkpoints such as VCTK.",
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--no-stress",
        action="store_true",
        help="Ignore CMUdict stress digits instead of converting 1/2 to IPA stress marks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ipa = cmu39_to_ipa(args.cmu39, keep_stress=not args.no_stress)
    print("IPA:", ipa)
    synthesize(
        args.cmu39,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        speaker_id=args.speaker_id,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        max_len=args.max_len,
        device=args.device,
        keep_stress=not args.no_stress,
    )
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
