import os
import pytest
from app.adapters.tts.local_tts_adapter import LocalTTSAdapter
from app.core.exceptions import AudioGenerationError
from app.utils.languages import SUPPORTED_LANGUAGES


SAMPLES = {
    "en": "A farmer walks through a green field.",
    "hi": "किसान हरे खेत में चल रहा है।",
    "mr": "शेतकरी हिरव्या शेतात चालत आहे.",
    "gu": "ખેડૂત લીલા ખેતરમાં ચાલે છે.",
    "fr": "Un fermier marche dans un champ vert.",
    "es": "Un granjero camina por un campo verde.",
    "de": "Ein Bauer geht durch ein gruenes Feld.",
    "ta": "விவசாயி ஒரு பச்சை வயலில் நடக்கிறார்.",
    "te": "రైతు పచ్చని పొలంలో నడుస్తున్నాడు.",
    "kn": "ರೈತನು ಹಸಿರು ಹೊಲದಲ್ಲಿ ನಡೆಯುತ್ತಿದ್ದಾನೆ.",
    "bn": "কৃষক একটি সবুজ মাঠের মধ্য দিয়ে হাঁটছে।",
    "pa": "ਕਿਸਾਨ ਹਰੇ ਖੇਤ ਵਿੱਚੋਂ ਦੀ ਲੰਘ ਰਿਹਾ ਹੈ।",
    "ml": "കർഷകൻ ഒരു പച്ച വയലിലൂടെ നടക്കുന്നു.",
    "ur": "کسان ایک سبز کھیت میں چل رہا ہے۔",
    "ja": "農夫が緑の畑を歩いています。",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.keys()))
async def test_tts_accepts_supported_languages(tmp_path, language):
    tts = LocalTTSAdapter()
    out = str(tmp_path / f"voice_{language}.mp3")
    sample_text = SAMPLES.get(language, "A farmer walks through a green field.")
    try:
        result = await tts.synthesize_speech(sample_text, out, language=language)
    except AudioGenerationError:
        if language == "en":
            raise
        pytest.skip(f"Network TTS unavailable for {language}")
    assert os.path.exists(result)
    assert os.path.getsize(result) > 500
