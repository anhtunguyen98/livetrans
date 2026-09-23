"""Build Vietnamese NeMo ITN with an additive, versioned custom whitelist."""
from pathlib import Path
import hashlib
import re
from functools import lru_cache

DEFAULT_WHITELIST = Path(__file__).resolve().parents[1] / "assets/itn/custom_whitelist.tsv"


@lru_cache(maxsize=4)
def _canonical_forms(whitelist: Path):
    forms = {line.split("\t")[0] for line in whitelist.read_text().splitlines() if line.strip()}
    return {form.casefold(): form for form in forms if any(c.isupper() for c in form)}


def restore_written_forms(text: str, source: str = '', whitelist: Path = DEFAULT_WHITELIST) -> str:
    canonical = _canonical_forms(whitelist)
    canonical = {key: form for key, form in canonical.items()
                 if re.search(r'(?<!\w)' + re.escape(form) + r'(?!\w)', source)}
    forms = canonical.values()
    if not forms:
        return text
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(forms, key=len, reverse=True)) + r")(?!\w)"
    return re.sub(pattern, lambda m: canonical[m[0].casefold()], text, flags=re.IGNORECASE)


def build_normalizer(cache_root: Path):
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path

    built_in = Path(get_abs_path("data/whitelist.tsv")).read_text()
    generated = DEFAULT_WHITELIST.with_name('generated_whitelist.tsv')
    custom = generated.read_text() if generated.exists() else DEFAULT_WHITELIST.read_text()
    for line in custom.splitlines():
        if line and (len(line.split("\t")) != 2 or not all(line.split("\t"))):
            raise ValueError("Expected written_form<TAB>spoken_form in custom whitelist")
    mappings = {}
    for line in (built_in + '\n' + custom + '\n' + DEFAULT_WHITELIST.read_text()).splitlines():
        if line.strip():
            written, spoken = line.split('\t')
            mappings[spoken] = written
    merged = ''.join(f'{written}\t{spoken}\n' for spoken, written in sorted(mappings.items()))
    # NeMo's Vietnamese cache filename does not include whitelist content.
    # A content-specific directory prevents accidentally reusing old grammars.
    digest = hashlib.sha256(merged.encode()).hexdigest()[:16]
    directory = cache_root / ("custom-" + digest)
    directory.mkdir(parents=True, exist_ok=True)
    whitelist = directory / "whitelist.tsv"
    whitelist.write_text(merged)
    normalizer = InverseNormalizer(lang="vi", input_case="lower_cased",
                                   whitelist=str(whitelist), cache_dir=str(directory))
    return normalizer, directory


if __name__ == "__main__":
    import argparse
    import pynini
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".cache/livetrans/vi_itn"))
    args = parser.parse_args()
    normalizer, directory = build_normalizer(args.output)
    for name, graph in (("tokenize_and_classify", normalizer.tagger.fst),
                        ("verbalize", normalizer.verbalizer.fst)):
        graph.write(str(directory / (name + ".fst")))
        with pynini.Far(str(directory / (name + ".far")), "w") as archive:
            archive[name] = graph
    print(directory.resolve())
