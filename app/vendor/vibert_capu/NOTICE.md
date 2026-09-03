# ViBERT CAPU inference code

The files in this directory are adapted from
[`dragonSwing/vibert-capu`](https://huggingface.co/dragonSwing/vibert-capu),
revision `261c60f2c30b02455dfce21a43c3ef14fc26992c`.

Upstream license: CC-BY-SA-4.0.

Local compatibility changes:

- package-relative imports;
- `ModelOutput` declared as a dataclass for Transformers 5;
- legacy embedding resize uses `mean_resizing=False`;
- punctuation regex construction avoids invalid Python escape sequences.
- edit actions are restricted to capitalization and `. , : ?`; generic GEC
  merge/verb actions are intentionally disabled for faithful ASR output.
- punctuation KEEP bias and capitalization bias are independently tunable.
