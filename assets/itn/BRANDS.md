# Brand vocabulary for Vietnamese dictation

`custom_whitelist.tsv` contains explicit `written<TAB>spoken` overrides. This update covers the Vin/Vingroup naming family and a separate selection of brands commonly mentioned in Vietnam. The list is a normalization vocabulary, **not a claim that all listed businesses belong to Vingroup**.

- Vin/Vingroup: Vingroup, Vincom, Vincom Retail, VinFast, Vinhomes, Vinpearl, Vinmec, Vinschool, VinUni, VinBus, VinAI, VinBigdata, VinRobotics, VinMotion, VinDynamics, VinWonders, VinMetal and V-Green.
- Other retail/food/technology/finance names: WinMart, Samsung, Shopee, Lazada, TikTok, Zalo, ZaloPay, MoMo, VNPAY, Grab, Starbucks, Highlands Coffee, The Coffee House, KFC, McDonald's, Jollibee, LOTTE, AEON, UNIQLO, Circle K, GS25, FamilyMart, 7-Eleven, Techcombank, VPBank and TPBank.
- `Dynamics` remains `Dynamics`, not `VinDynamics`; only aliases containing the Vin prefix map to VinDynamics. Do not map English `dynamic`/`dynamics` into a Vin brand without that prefix.

Reference for official Vin spellings: [Vingroup businesses](https://www.vingroup.net/en/business), [VinDynamics](https://vindynamics.net/about-us). Spoken aliases are manually authored hypotheses for Vietnamese ASR output, **not measured pronunciation coverage**. Add variants from actual opt-in mobile logs and regression-test them. We intentionally avoid the ordinary phrase `hai lần cà phê` as an alias for Highlands Coffee.

## Rebuild

1. `python scripts/build_itn_dictionary.py --source /tmp/non-vietnamese-words-20k.csv`
2. `python scripts/prepare_android_assets.py` — creates/reuses the grammar identified by its dictionary-content hash, not a hardcoded old grammar.
3. `python scripts/test_itn_brands.py`
4. Build the Android APK with the commands in `android/README.md`.

Existing installed APKs do not download dictionary updates; install the new APK. A running Python service loads its dictionary during initialization and needs a separate restart to apply changes. This task does not automatically restart production services.
