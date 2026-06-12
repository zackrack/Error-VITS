#!/usr/bin/env python3
"""Synthesize speech from a CMU-39/ARPAbet phoneme string with VITS.

This stage accepts a whitespace-delimited CMU-39/ARPAbet string, converts it to
VITS-compatible IPA symbols, automatically prepares the local runtime assets
when requested, and writes synthesized speech to a WAV file.
"""

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


MODEL_ASSETS = {
    "ljs": {
        "description": "Single-speaker LJ Speech checkpoint",
        "config": "configs/ljs_base.json",
        "checkpoint": "pretrained_ljs.pth",
        "url": "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/pretrained_ljs.pth",
        "sha256": "c94fb49d08ba90c598de16e7d5dec8d26bf225c1cf193a4fba05eb2dbda5a561",
    },
    "vctk": {
        "description": "Multi-speaker VCTK checkpoint",
        "config": "configs/vctk_base.json",
        "checkpoint": "pretrained_vctk.pth",
        "url": "https://huggingface.co/csukuangfj/vits-vctk/resolve/main/pretrained_vctk.pth",
        "sha256": "ab981615c443d935fc3a89b08137df544a1175bad99bcbbc9f59e7c3d4930043",
    },
}

_RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
}
_TEXT_PACKAGES = {
    "phonemizer": "phonemizer",
    "unidecode": "Unidecode",
}
_BUILD_PACKAGES = {
    "Cython": "Cython",
}
_REPO_ROOT = Path(__file__).resolve().parent


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
_TAG_RE = re.compile(
    r"<(phoneme|cmu|ipa)(?P<attrs>[^>]*)>(?P<body>.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_ATTR_RE = re.compile(r"""(\w+)\s*=\s*(["'])(.*?)\2""")


def missing_modules(modules_to_packages: Dict[str, str]) -> List[str]:
    """Return package names for modules that are not importable."""

    return [
        package_name
        for module_name, package_name in modules_to_packages.items()
        if importlib.util.find_spec(module_name) is None
    ]


def install_packages(packages: Iterable[str]) -> None:
    """Install missing Python packages into the current interpreter."""

    packages = list(dict.fromkeys(packages))
    if not packages:
        return
    print("Installing missing Python packages:", " ".join(packages))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def ensure_python_dependencies(install_missing: bool = True) -> None:
    """Ensure packages needed for VITS inference are importable."""

    missing = missing_modules(_RUNTIME_PACKAGES)
    if missing and not install_missing:
        raise RuntimeError(
            "Missing Python packages: {}. Re-run without --no-install-missing "
            "or install them manually.".format(" ".join(missing))
        )
    install_packages(missing)


def monotonic_align_built() -> bool:
    """Return True when the Cython monotonic alignment extension is present."""

    return bool(list((_REPO_ROOT / "monotonic_align").glob("core*.so"))) or bool(
        list((_REPO_ROOT / "monotonic_align").glob("core*.pyd"))
    )


def ensure_monotonic_align(build_extension: bool = True, install_missing: bool = True) -> None:
    """Build the monotonic alignment extension required by models.py imports."""

    if monotonic_align_built():
        return
    if not build_extension:
        raise RuntimeError(
            "monotonic_align extension is not built. Re-run without "
            "--no-build-extensions or run `cd monotonic_align && "
            "python setup.py build_ext --inplace`."
        )

    missing = missing_modules(_BUILD_PACKAGES)
    if missing and not install_missing:
        raise RuntimeError(
            "Missing build packages: {}. Re-run without --no-install-missing "
            "or install them manually.".format(" ".join(missing))
        )
    install_packages(missing)

    print("Building monotonic_align Cython extension...")
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(_REPO_ROOT / "monotonic_align"),
    )


def ensure_runtime_environment(
    install_missing: bool = True,
    build_extensions: bool = True,
) -> None:
    """Prepare Python packages and local compiled extensions for inference."""

    ensure_python_dependencies(install_missing=install_missing)
    ensure_monotonic_align(
        build_extension=build_extensions,
        install_missing=install_missing,
    )


def sha256sum(path: Path) -> str:
    """Compute a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a destination path with a simple progress indicator."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary_path.open("wb") as file_obj:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(
                    "\rDownloading {}: {:.1f}%".format(destination.name, percent),
                    end="",
                    flush=True,
                )
        if total:
            print()
    os.replace(str(temporary_path), str(destination))


def ensure_checkpoint(
    model_name: str,
    download_dir: str,
    download: bool = True,
    force_download: bool = False,
) -> Path:
    """Return a local checkpoint path, downloading and verifying if needed."""

    asset = MODEL_ASSETS[model_name]
    checkpoint_path = Path(download_dir) / asset["checkpoint"]
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path

    expected_sha256 = asset["sha256"]
    if checkpoint_path.exists() and not force_download:
        actual_sha256 = sha256sum(checkpoint_path)
        if actual_sha256 == expected_sha256:
            return checkpoint_path
        print(
            "Existing checkpoint failed SHA-256 verification; downloading a fresh copy."
        )

    if not download:
        raise FileNotFoundError(
            "Checkpoint is missing or invalid at {} and downloads are disabled.".format(
                checkpoint_path
            )
        )

    print("Downloading {} to {}".format(asset["description"], checkpoint_path))
    download_file(asset["url"], checkpoint_path)
    actual_sha256 = sha256sum(checkpoint_path)
    if actual_sha256 != expected_sha256:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        raise RuntimeError(
            "Downloaded checkpoint failed SHA-256 verification. Expected {}, got {}.".format(
                expected_sha256,
                actual_sha256,
            )
        )
    return checkpoint_path


def default_config_path(model_name: str) -> str:
    """Return the repository config path for a built-in model asset."""

    return str(_REPO_ROOT / MODEL_ASSETS[model_name]["config"])


def collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing whitespace."""

    return re.sub(r"\s+", " ", text).strip()


def validate_ipa(ipa: str) -> str:
    """Validate that an IPA string only contains VITS symbols."""

    unsupported = sorted({symbol for symbol in ipa if symbol not in symbols})
    if unsupported:
        raise ValueError(
            "IPA contains symbols not present in this VITS model's symbol table: {}".format(
                " ".join(unsupported)
            )
        )
    return ipa


def ensure_text_dependencies(install_missing: bool = True) -> None:
    """Ensure packages needed for English text-to-IPA cleaning are importable."""

    missing = missing_modules(_TEXT_PACKAGES)
    if missing and not install_missing:
        raise RuntimeError(
            "Text input requires missing Python packages: {}. Re-run without "
            "--no-install-missing or install them manually.".format(" ".join(missing))
        )
    install_packages(missing)


def text_to_ipa(text: str, cleaner_names: Optional[List[str]] = None) -> str:
    """Convert ordinary text to the same IPA style used by repository cleaners."""

    from text import cleaners

    cleaner_names = cleaner_names or ["english_cleaners2"]
    cleaned = text
    for cleaner_name in cleaner_names:
        cleaner = getattr(cleaners, cleaner_name)
        cleaned = cleaner(cleaned)
    return validate_ipa(cleaned)


def parse_tag_attrs(attrs: str) -> Dict[str, str]:
    """Parse simple XML-style key="value" tag attributes."""

    return {match.group(1).lower(): match.group(3) for match in _ATTR_RE.finditer(attrs)}


def looks_like_cmu(text: str) -> bool:
    """Return True when text appears to be only CMU/ARPAbet tokens."""

    saw_phone = False
    for raw_token in text.split():
        token = raw_token.strip().upper()
        while token and token[0] in _PUNCTUATION:
            token = token[1:]
        while token and token[-1] in _PUNCTUATION:
            token = token[:-1]
        if not token:
            continue
        match = _PHONE_RE.match(token)
        if match is None or match.group(1) not in CMU39_TO_IPA:
            return False
        saw_phone = True
    return saw_phone


def render_phoneme_tag(
    tag_name: str,
    attrs: str,
    body: str,
    keep_stress: bool = True,
) -> str:
    """Render a <phoneme>, <cmu>, or <ipa> tag body to model-compatible IPA."""

    attr_map = parse_tag_attrs(attrs)
    alphabet = attr_map.get("alphabet", attr_map.get("type", tag_name)).lower()
    if alphabet in {"phoneme", "arpabet", "cmu39", "cmu"}:
        return cmu39_to_ipa(body, keep_stress=keep_stress, phone_separator="")
    if alphabet == "ipa":
        return validate_ipa(body.strip())
    raise ValueError(
        "Unsupported phoneme alphabet '{}'. Use CMU/ARPAbet or IPA.".format(alphabet)
    )


def input_needs_text_cleaners(input_text: str, input_format: str) -> bool:
    """Return True when an input mode may need the configured text cleaners."""

    if input_format == "text":
        return True
    if input_format != "mixed":
        return False
    if not _TAG_RE.search(input_text):
        return not looks_like_cmu(input_text)

    position = 0
    for match in _TAG_RE.finditer(input_text):
        if input_text[position:match.start()].strip():
            return True
        position = match.end()
    return bool(input_text[position:].strip())


def input_to_ipa(
    input_text: str,
    input_format: str = "mixed",
    keep_stress: bool = True,
    text_cleaners: Optional[List[str]] = None,
) -> str:
    """Convert text, IPA, CMU, or tagged mixed input to model-compatible IPA."""

    if input_format == "ipa":
        return validate_ipa(input_text)
    if input_format == "cmu":
        return cmu39_to_ipa(input_text, keep_stress=keep_stress, phone_separator="")
    if input_format == "text":
        return text_to_ipa(input_text, cleaner_names=text_cleaners)
    if input_format != "mixed":
        raise ValueError("Unsupported input format '{}'".format(input_format))
    if not _TAG_RE.search(input_text) and looks_like_cmu(input_text):
        return cmu39_to_ipa(input_text, keep_stress=keep_stress, phone_separator="")

    pieces = []
    position = 0
    for match in _TAG_RE.finditer(input_text):
        if match.start() > position:
            text_piece = input_text[position:match.start()]
            if text_piece.strip():
                pieces.append(text_to_ipa(text_piece, cleaner_names=text_cleaners))
        pieces.append(
            render_phoneme_tag(
                match.group(1),
                match.group("attrs"),
                match.group("body"),
                keep_stress=keep_stress,
            )
        )
        position = match.end()
    if position < len(input_text):
        text_piece = input_text[position:]
        if text_piece.strip():
            pieces.append(text_to_ipa(text_piece, cleaner_names=text_cleaners))

    return validate_ipa(collapse_whitespace(" ".join(piece for piece in pieces if piece)))


def cmu39_to_ipa(
    cmu39: str,
    keep_stress: bool = True,
    phone_separator: str = "",
) -> str:
    """Convert a whitespace-delimited CMU-39/ARPAbet string to VITS IPA text.

    Tokens may include optional CMUdict stress digits on vowels, e.g.
    ``HH AH0 L OW1``. Punctuation may be attached to tokens or separated by
    spaces, e.g. ``W ER1 L D !`` or ``W ER1 L D!``. By default, phones are
    joined without spaces because the pretrained VITS filelists use spaces
    primarily as word boundaries, not as phone boundaries.
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

    ipa = phone_separator.join(ipa_tokens)
    unsupported = sorted({symbol for symbol in ipa if symbol not in symbols})
    if unsupported:
        raise ValueError(
            "Converted IPA contains symbols not present in this VITS model's "
            "symbol table: {}".format(" ".join(unsupported))
        )
    return ipa


def ipa_to_sequence(ipa: str, add_blank: bool) -> List[int]:
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
    utils.load_checkpoint(str(checkpoint_path), net_g, None)
    return net_g


def synthesize(
    input_text: str,
    config_path: Optional[str],
    checkpoint_path: Optional[str],
    output_path: str,
    speaker_id: Optional[int] = None,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    max_len: Optional[int] = None,
    device: str = "auto",
    keep_stress: bool = True,
    input_format: str = "mixed",
    model_name: str = "ljs",
    download_dir: str = "checkpoints",
    download: bool = True,
    force_download: bool = False,
    install_missing: bool = True,
    build_extensions: bool = True,
):
    """Run input text/tags/phonemes -> IPA -> VITS inference and write a WAV file."""

    ensure_runtime_environment(
        install_missing=install_missing,
        build_extensions=build_extensions,
    )

    import torch
    from scipy.io.wavfile import write

    import utils

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)

    resolved_config_path = config_path or default_config_path(model_name)
    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path else ensure_checkpoint(
        model_name,
        download_dir=download_dir,
        download=download,
        force_download=force_download,
    )

    hps = utils.get_hparams_from_file(str(resolved_config_path))
    if input_needs_text_cleaners(input_text, input_format):
        ensure_text_dependencies(install_missing=install_missing)
    ipa = input_to_ipa(
        input_text,
        input_format=input_format,
        keep_stress=keep_stress,
        text_cleaners=list(getattr(hps.data, "text_cleaners", ["english_cleaners2"])),
    )
    print("IPA:", ipa)
    text = ipa_to_tensor(ipa, hps.data.add_blank).to(selected_device)
    net_g = build_model(hps, str(resolved_checkpoint_path), selected_device)

    sid = None
    n_speakers = getattr(hps.data, "n_speakers", 0)
    if n_speakers > 0:
        if speaker_id is None:
            speaker_id = 0
            print("No speaker ID supplied for multi-speaker model; using speaker 0.")
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
        description=(
            "Generate speech from text, IPA, or tagged phoneme input using VITS. "
            "By default, the script downloads the LJ Speech checkpoint and "
            "prepares local runtime pieces the first time it runs."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Input to synthesize. Default mixed mode accepts text plus tags like "
            "'<phoneme>W AE1 B AH0 T</phoneme>' or '<ipa>wˈæbət</ipa>'."
        ),
    )
    parser.add_argument(
        "--input-format",
        default="mixed",
        choices=("mixed", "text", "ipa", "cmu"),
        help=(
            "How to interpret the input. mixed accepts ordinary text plus "
            "<phoneme>/<cmu>/<ipa> tags; cmu expects plain CMU/ARPAbet; ipa "
            "expects already-cleaned IPA; text runs the configured text cleaners."
        ),
    )
    parser.add_argument(
        "--model",
        default="ljs",
        choices=tuple(sorted(MODEL_ASSETS)),
        help="Built-in checkpoint to use when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a VITS JSON config. Defaults to the selected built-in model config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a generator checkpoint. Defaults to an auto-downloaded checkpoint.",
    )
    parser.add_argument(
        "--download-dir",
        default="checkpoints",
        help="Directory for automatically downloaded checkpoints.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download built-in checkpoints; require a valid local checkpoint.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the selected built-in checkpoint even if a local file exists.",
    )
    parser.add_argument(
        "--no-install-missing",
        action="store_true",
        help="Do not pip-install missing runtime/build packages automatically.",
    )
    parser.add_argument(
        "--no-build-extensions",
        action="store_true",
        help="Do not auto-build the monotonic_align Cython extension.",
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
    synthesize(
        args.input,
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
        input_format=args.input_format,
        model_name=args.model,
        download_dir=args.download_dir,
        download=not args.no_download,
        force_download=args.force_download,
        install_missing=not args.no_install_missing,
        build_extensions=not args.no_build_extensions,
    )
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
