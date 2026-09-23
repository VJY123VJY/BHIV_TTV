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
