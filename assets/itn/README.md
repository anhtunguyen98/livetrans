# Custom Vietnamese inverse text normalization

`generated_whitelist.tsv` reverses the VietNormalizer TTS lexicon into NeMo's
`written_form<TAB>spoken_form` format. `manifest.json` records source and hash.
`review.json` contains ambiguous and single-syllable pronunciations excluded
from automatic replacement. Accepted multi-syllable entries are still automatic
candidates and need real ASR evaluation; uniqueness is not proof of correctness.

Edit `custom_whitelist.tsv` for explicit ASR aliases and acronyms. Overrides
take precedence. Do not map the Vietnamese word `ai` globally to `AI`.
Only canonical acronyms present in the ITN result are restored after CAPU.

Rebuild from the source CSV:

```sh
python scripts/build_itn_dictionary.py --source /path/to/non-vietnamese-words-20k.csv
python -m app.custom_itn
```

The build merges NeMo's built-in whitelist and preserves its numeric grammars.
It exports classifier and verbalizer as separate `.fst` and `.far` files under
`.cache/livetrans/vi_itn/custom-<content-hash>/`. The versioned directory avoids
loading a stale NeMo grammar after changing the dictionary.

Android integration requires a compatible native NeMo text-processing pipeline
that runs classification, token parsing/field ordering and verbalization.
These are grammar artifacts, not an Android APK or a single direct-rewrite FST.
Python InverseNormalizer parity tests have passed; Android runtime parity has
not yet been tested. Restart the ASR service to load a rebuilt dictionary.
