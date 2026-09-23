"""Exercise the same exported FST pipeline bundled in the Android APK."""
from pathlib import Path
import sys
import pynini
from pynini.lib import rewrite

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from app.custom_itn import restore_written_forms

assets = root / 'android/app/src/main/assets/itn'
graphs = [pynini.Fst.read(str(assets / f'{name}.fst')) for name in ['lowercase', 'tokenize_and_classify', 'verbalize']]
def normalize(text):
    for graph in graphs:
        text = rewrite.top_rewrite(text, graph)
    return text

cases = [(spoken, written) for written, spoken in (line.split('\t') for line in (root / 'assets/itn/custom_whitelist.tsv').read_text().splitlines() if line)]
cases += [
    ('tôi đến vin com', 'tôi đến Vincom'),
    ('xe vin phát', 'xe VinFast'),
    ('vin đai na mích', 'VinDynamics'),
    ('đai na mích', 'Dynamics'),
    ('tôi dùng tiktok và momo', 'tôi dùng TikTok và MoMo'),
    ('cà phê highlands coffee', 'cà phê Highlands Coffee'),
    ('mua đồ ở u ni clô', 'mua đồ ở UNIQLO'),
    ('tôi uống hai lần cà phê', 'tôi uống hai lần cà phê'),
    ('ai đang ở nhà', 'ai đang ở nhà'),
    ('hai triệu đồng', '2 triệu đồng'),
    ('tôi làm ở vin đai na mít', 'tôi làm ở VinDynamics'),
    ('tôi dùng vin base và vi gi pi ti', 'tôi dùng VinBase và ViGPT'),
    ('xe vê ép ba', 'xe VF 3'),
    ('xe li mô grin', 'xe Limo Green'),
    ('vin space và vin e nơ gô', 'VinSpace và VinEnergo'),
    ('vin com mê ga môn', 'Vincom Mega Mall'),
    # Do not fuzzy-correct common words or people into brand names.
    ('anh vinh phát biểu', 'anh vinh phát biểu'),
    ('vì sao em về', 'vì sao em về'),
    ('nhà có ba cửa', 'nhà có ba cửa'),
    ('nhắn tin qua du hát', 'nhắn tin qua Duhat'),
    ('mở app du hat', 'mở app Duhat'),
    ('duhat là app chat', 'Duhat là app chat'),
    ('tôi làm ở vin sờ mát phiu chờ', 'tôi làm ở VinSmart Future'),
    ('mở vê áp để tích điểm vi poi', 'mở V-App để tích điểm VPoint'),
    ('du khách đang hát', 'du khách đang hát'),
    ('ứng dụng chat của tôi', 'ứng dụng chat của tôi'),
]
failures = []
for spoken, expected in cases:
    actual = normalize(spoken)
    if actual != expected:
        failures.append((spoken, expected, actual))
for failure in failures:
    print('FAIL', failure)
assert not failures, f'{len(failures)} failures'
assert restore_written_forms('VINFAST và tiktok', source='VinFast và TikTok') == 'VinFast và TikTok'
assert restore_written_forms('Ai đang ở nhà', source='ai đang ở nhà') == 'Ai đang ở nhà'
print(f'PASS: {len(cases)} FST cases and canonical-case restoration')
