# Vingroup keyword update — 2026-09-10

Curated recognition vocabulary, not an exhaustive legal subsidiary register.
Existing brands outside Vingroup (e.g. WinMart) remain in the general dictionary;
their presence does not imply current Vingroup ownership.

## Verified names

- [Duhat official site](https://duhat.vn/about/): Duhat chat app, developed by VinSmart Future. Added canonical names VinSmart Future / VSF / Duhat and explicit pronunciation aliases; do not rewrite generic `app chat` or `du khách`.
- [Vingroup V-App launch announcement](https://vingroup.net/tin-tuc-su-kien/bai-viet/3803/vinsmart-future-ra-mat-ky-thuat-phien-ban-trai-nghiem-som-sieu-ung-dung-mot-cham-v-app): V-App and VPoint loyalty points mentioned in its ecosystem. VPoint is included as a related service term, not claimed as a separate app.

- [Vingroup technology/industry portfolio](https://www.vingroup.net/en/business): existing industrial brands, including VinMetal and VinDynamics.
- [VinSpace announcement, 2026-08-11](https://www.vingroup.net/tin-tuc-su-kien/bai-viet/8034/vinspace-cong-bo-hop-dong-phong-ve-tinh-voi-spacex-mo-buoc-tien-moi-cho-nganh-cong-nghiep-khong-gian-viet-nam): VinSpace.
- [VinEnergo announcement, 2026-03-04](https://www.vingroup.net/tin-tuc-su-kien/bai-viet/4834/vinenergo-cong-bo-chien-luoc-toan-cau-trien-khai-danh-muc-10-gw-nang-luong-tai-tao-quoc-te-dau-tien): VinEnergo.
- [Vingroup 2025 annual report, published 2026](https://ircdn.vingroup.net/storage/Uploads/0_Bao%20cao%20thuong%20nien/2025/ENG%20Vingroup%20AR25.pdf): VinSpeed.
- [June 2026 corporate presentation](https://ircdn.vingroup.net/storage/Uploads/0_Quan%20he%20co%20dong/0_Vingroup_2026/T6/2026.06_Vingroup%20Corporate%20Presentation_vf.pdf): VinNewHorizon.
- [VinClub introduction](https://vinpearl.com/vi/uu-dai-vingroup-ra-mat-chuong-trinh-khach-hang-than-thiet-vinclub): VinClub.
- [VinBigdata products](https://vinbigdata.com/), [VinBase](https://vinbigdata.com/vinbase): VinBase, ViGPT, ViChat, ViVoice, ViVi, Vizone, ViFi.
- [VinFast official catalog](https://vinfastauto.com/vn_vi): VF 2/3/5/6/7/8/9, VF MPV 7, Minio Green, Herio Green, Nerio Green, Limo Green, EC Van.
- [Vincom](https://vincom.com.vn/): Vincom Mega Mall and Vincom Plaza.

Names are verified against published sources. Phonetic aliases are manually
proposed spellings, NOT measured ASR confusion frequencies or guaranteed acoustic
corrections. Exact aliases live in `custom_whitelist.tsv`, with canonical name in
column 1 and lower-case spoken/ASR spelling in column 2.

## What ships

Exact phrase whitelist matching inside NeMo ITN. For example `vin đai na mít`
becomes `VinDynamics`, and `vê ép ba` becomes `VF 3`. Canonical spellings and
multi-word brand spans are then protected by the existing CAPU brand protection.
No fuzzy replacement, external API, new model or extra inference pass is added.
No broad `vinh` → `vin`, bare `ba` → `VF 3`, or `vì` → brand rule is added.

Android's recognizer applies ITN separately to its ASR candidates. Since v0.3.6,
CAPU formats each candidate before KenLM chooses between them, matching the web
reranking order. KenLM is not a brand-rewriting model and cannot recover absent words.

## Handling additional ASR mistakes

1. Collect opt-in raw ASR + intended text examples, including negative examples.
2. Add exact unambiguous phrases here and corresponding sentence regressions.
3. For ambiguous words, require domain/context and consider ASR hotword biasing
   or N-best reranking; validate decoder support before enabling either.
4. Never globally fuzzy-match arbitrary Vietnamese text to the nearest brand.

Rebuild with `scripts/build_itn_dictionary.py --source <original CSV>`, then
`scripts/prepare_android_assets.py` and `scripts/test_itn_brands.py`.
Android requires installing a rebuilt APK. A running Python service may cache
the old normalizer; no service restart/deployment is performed by this update.
