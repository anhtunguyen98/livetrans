# Checkpoint 122000 ONNX export

- Source: `/workspace/checkpoint-122000.pt`
- Source MD5: `5e6622b8c7077d1ed95e214c78a63c8e`
- Exported state: `model` at iteration 122000, without cross-checkpoint averaging
- Architecture: non-streaming Zipformer2 RNN-T
- Parameters: 66,573,511 total; 64,256,759 encoder; 1,290,752 decoder; 1,026,000 joiner
- Input features: 80
- Decoder context: 2
- Joiner dimension: 512
- Vocabulary: 2,000, using the NghiASR tokenizer

The three INT8 files in this directory are ready for sherpa-onnx. The `exp/`
subdirectory also contains the FP32 exports used for validation.

## Validation

All six FP32/INT8 graphs passed `onnx.checker.check_model` and loaded in
sherpa-onnx on CPU.

Test 1, 7.729 s Vietnamese speech:

- FP32 and INT8 produced the same transcript.
- FP32 decode: 0.186 s, RTF 0.0240.
- INT8 decode: 0.177 s, RTF 0.0229.

Test 2, 7.824 s Vietnamese/English code-switch speech:

- FP32 decoded the first English occurrence as `CRAFTING`.
- INT8 decoded that occurrence as `CROSING`; the remaining transcript was the same.
- FP32 decode: 0.189 s, RTF 0.0242.
- INT8 decode: 0.179 s, RTF 0.0229.

This shows that the INT8 model is operational, with a small quantization-induced
difference on the code-switched test.

Test 3, 4.680 s pure English reference speech:

- Reference: `HELLO THIS IS A CLEAR AND NATURAL VOICE FOR REAL TIME SPEECH TRANSLATION`
- FP32 and INT8 both matched all 13 reference words (sample WER 0%).
- FP32 decode: 0.119 s, RTF 0.0255.
- INT8 decode: 0.109 s, RTF 0.0233.
