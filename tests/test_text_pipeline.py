from hams_tts.text.lang_segment import segment_mixed_text
from hams_tts.text.levantine_rules import apply_levantine_phonology, lookup_lexicon
from hams_tts.text.normalize import normalize_span
from hams_tts.text.pipeline import phonemize_text
from hams_tts.text.vocab import build_vocab, load_tokenizer, save_vocab


def test_lang_segment_splits_code_switched_text():
    spans = segment_mixed_text("بدي أروح ع meeting الساعة 3")
    langs = [s.lang for s in spans]
    assert langs == ["ar", "en", "ar"]


def test_lang_segment_pure_english():
    spans = segment_mixed_text("hello world")
    assert len(spans) == 1
    assert spans[0].lang == "en"


def test_normalize_expands_currency_and_numbers():
    out = normalize_span("عندي 25 دولار", "ar")
    assert "25" not in out
    assert "دولار" in out


def test_levantine_qaf_and_jeem_substitution():
    assert apply_levantine_phonology("qalb") == "ʔalb"
    assert apply_levantine_phonology("dʒamiːl") == "ʒamiːl"


def test_lexicon_override_hit_and_miss():
    assert lookup_lexicon("شو") == "ʃuː"
    assert lookup_lexicon("نيوزيلندا") is None


def test_full_pipeline_produces_language_markers():
    result = phonemize_text("بدي أروح ع meeting")
    assert "" in result.phoneme_string  # Arabic marker present
    assert "" in result.phoneme_string  # English marker present
    assert len(result.spans) == 2


def test_vocab_round_trip(tmp_path):
    sentences = ["مرحبا، كيفك؟", "hello there", "بدي أروح ع meeting"]
    phon = [phonemize_text(s).phoneme_string for s in sentences]
    vocab = build_vocab(phon)
    save_vocab(vocab, tmp_path)
    tokenizer = load_tokenizer(tmp_path)

    for p in phon:
        ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
        assert tokenizer.decode(ids) == p
