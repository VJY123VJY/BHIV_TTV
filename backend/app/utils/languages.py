"""Supported narration languages and TTS voice mappings for 15+ Indian and international languages."""
from typing import Any, Dict, Optional

from app.core.exceptions import ValidationError

SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "gtts": "en",
        "edge_voices": ["en-US-GuyNeural", "en-US-AriaNeural", "en-IN-NeerjaNeural"],
        "default_voice": "en-US-GuyNeural",
    },
    "hi": {
        "name": "Hindi",
        "gtts": "hi",
        "edge_voices": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
        "default_voice": "hi-IN-MadhurNeural",
    },
    "mr": {
        "name": "Marathi",
        "gtts": "mr",
        "edge_voices": ["mr-IN-ManoharNeural", "mr-IN-AarohiNeural"],
        "default_voice": "mr-IN-ManoharNeural",
    },
    "ta": {
        "name": "Tamil",
        "gtts": "ta",
        "edge_voices": ["ta-IN-ValluvarNeural", "ta-IN-PallaviNeural"],
        "default_voice": "ta-IN-ValluvarNeural",
    },
    "te": {
        "name": "Telugu",
        "gtts": "te",
        "edge_voices": ["te-IN-MohanNeural", "te-IN-ShrutiNeural"],
        "default_voice": "te-IN-MohanNeural",
    },
    "kn": {
        "name": "Kannada",
        "gtts": "kn",
        "edge_voices": ["kn-IN-GaganNeural", "kn-IN-SapnaNeural"],
        "default_voice": "kn-IN-GaganNeural",
    },
    "bn": {
        "name": "Bengali",
        "gtts": "bn",
        "edge_voices": ["bn-IN-BashkarNeural", "bn-IN-TanishaaNeural"],
        "default_voice": "bn-IN-BashkarNeural",
    },
    "gu": {
        "name": "Gujarati",
        "gtts": "gu",
        "edge_voices": ["gu-IN-NiranjanNeural", "gu-IN-DhwaniNeural"],
        "default_voice": "gu-IN-NiranjanNeural",
    },
    "pa": {
        "name": "Punjabi",
        "gtts": "pa",
        "edge_voices": ["pa-IN-OjasNeural", "pa-IN-VaaniNeural"],
        "default_voice": "pa-IN-OjasNeural",
    },
    "ml": {
        "name": "Malayalam",
        "gtts": "ml",
        "edge_voices": ["ml-IN-MidhunNeural", "ml-IN-SobhanaNeural"],
        "default_voice": "ml-IN-MidhunNeural",
    },
    "ur": {
        "name": "Urdu",
        "gtts": "ur",
        "edge_voices": ["ur-IN-SalmanNeural", "ur-IN-GulNeural"],
        "default_voice": "ur-IN-SalmanNeural",
    },
    "fr": {
        "name": "French",
        "gtts": "fr",
        "edge_voices": ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
        "default_voice": "fr-FR-HenriNeural",
    },
    "es": {
        "name": "Spanish",
        "gtts": "es",
        "edge_voices": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
        "default_voice": "es-ES-AlvaroNeural",
    },
    "de": {
        "name": "German",
        "gtts": "de",
        "edge_voices": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
        "default_voice": "de-DE-ConradNeural",
    },
    "ja": {
        "name": "Japanese",
        "gtts": "ja",
        "edge_voices": ["ja-JP-KeitaNeural", "ja-JP-NanamiNeural"],
        "default_voice": "ja-JP-KeitaNeural",
    },
}

_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-in": "en",
    "hindi": "hi",
    "hi-in": "hi",
    "marathi": "mr",
    "mr-in": "mr",
    "tamil": "ta",
    "ta-in": "ta",
    "telugu": "te",
    "te-in": "te",
    "kannada": "kn",
    "kn-in": "kn",
    "bengali": "bn",
    "bn-in": "bn",
    "gujarati": "gu",
    "gu-in": "gu",
    "punjabi": "pa",
    "pa-in": "pa",
    "malayalam": "ml",
    "ml-in": "ml",
    "urdu": "ur",
    "ur-in": "ur",
    "ur-pk": "ur",
    "french": "fr",
    "fr-fr": "fr",
    "spanish": "es",
    "es-es": "es",
    "german": "de",
    "de-de": "de",
    "japanese": "ja",
    "ja-jp": "ja",
}


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "en"
    key = str(language).strip().lower().replace("_", "-")
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValidationError(
        f"Unsupported language '{language}'. Supported: "
        + ", ".join(f"{code} ({meta['name']})" for code, meta in SUPPORTED_LANGUAGES.items())
    )


def get_language_config(language: Optional[str]) -> Dict[str, Any]:
    code = normalize_language(language)
    cfg = dict(SUPPORTED_LANGUAGES[code])
    cfg["code"] = code
    return cfg


def resolve_voice(language: Optional[str], voice: Optional[str] = None) -> str:
    cfg = get_language_config(language)
    if voice:
        requested = str(voice).strip()
        for candidate in cfg["edge_voices"]:
            if requested.lower() == candidate.lower():
                return candidate
        # Keep an explicit neural voice if the caller already passed a matching locale.
        locale_prefix = cfg["code"] if cfg["code"] != "en" else "en-"
        if requested.lower().startswith(locale_prefix) or (
            cfg["code"] == "en" and requested.lower().startswith("en-")
        ):
            return requested
        raise ValidationError(
            f"Voice '{voice}' is not available for {cfg['name']}. "
            f"Available voices: {', '.join(cfg['edge_voices'])}."
        )
    return cfg["default_voice"]


def catalog() -> Dict[str, Any]:
    return {
        "languages": [
            {
                "id": code,
                "name": meta["name"],
                "voices": meta["edge_voices"],
                "default_voice": meta["default_voice"],
            }
            for code, meta in SUPPORTED_LANGUAGES.items()
        ]
    }


# Spoken narration templates so TTS emits the selected language natively.
_NARRATION = {
    "en": [
        "The {subject} begins {action} through the {setting}, and the story unfolds.",
        "Every movement of the {subject} reveals more of the {setting}.",
        "The {subject} reaches a striking view of the {setting}, and the journey settles.",
    ],
    "hi": [
        "{subject} {setting} में {action} करते हुए आगे बढ़ता है और कहानी शुरू होती है।",
        "{subject} के हर कदम पर {setting} का नया रूप दिखाई देता है।",
        "{subject} {setting} के विस्तृत दृश्य तक पहुँचता है और यात्रा शांत हो जाती है।",
    ],
    "mr": [
        "{subject} {setting} मध्ये {action} करत पुढे जातो आणि कथा सुरू होते.",
        "{subject} च्या प्रत्येक हालचालीत {setting} अधिक उलगडतो.",
        "{subject} {setting} चा विस्तीर्ण देखावा पाहतो आणि प्रवास थांबतो.",
    ],
    "ta": [
        "{subject} {setting} இல் {action} செய்து பயணத்தைத் தொடங்குகிறது, கதை விரிகிறது.",
        "{subject} இன் ஒவ்வொரு அசைவும் {setting} இன் அழகை வெளிப்படுத்துகிறது.",
        "{subject} {setting} இன் கண்கவர் காட்சியை அடைந்து அமைதியடைகிறது.",
    ],
    "te": [
        "{subject} {setting} లో {action} చేస్తూ కథను ప్రారంభిస్తుంది.",
        "{subject} యొక్క ప్రతి అడుగు {setting} యొక్క అందాన్ని చూపుతుంది.",
        "{subject} {setting} యొక్క అద్భుతమైన దృశ్యాన్ని చేరుకుని ప్రశాంతమవుతుంది.",
    ],
    "kn": [
        "{subject} {setting} ನಲ್ಲಿ {action} ಮಾಡುತ್ತಾ ಕಥೆಯನ್ನು ಪ್ರಾರಂಭಿಸುತ್ತದೆ.",
        "{subject} ನ ಪ್ರತಿ ಚಲನೆಯು {setting} ನ ಸೌಂದರ್ಯವನ್ನು ತೆರೆಯುತ್ತದೆ.",
        "{subject} {setting} ನ ಸುಂದರ ನೋಟವನ್ನು ತಲುಪಿ ಶಾಂತವಾಗುತ್ತದೆ.",
    ],
    "bn": [
        "{subject} {setting}-এ {action} করতে করতে এগিয়ে যায় এবং গল্প শুরু হয়।",
        "{subject}-এর প্রতিটি পদক্ষেপে {setting}-এর নতুন রূপ প্রকাশ পায়।",
        "{subject} {setting}-এর মনোরম দৃশ্যে পৌঁছে যাত্রা সম্পূর্ণ করে।",
    ],
    "gu": [
        "{subject} {setting}માં {action} કરતાં આગળ વધે છે અને વાર્તા શરૂ થાય છે.",
        "{subject}ની દરેક હિલચાલમાં {setting} વધુ સ્પષ્ટ થાય છે.",
        "{subject} {setting}નું વિશાળ દૃશ્ય જુએ છે અને સફર શાંત થાય છે.",
    ],
    "pa": [
        "{subject} {setting} ਵਿੱਚ {action} ਕਰਦੇ ਹੋਏ ਅੱਗੇ ਵਧਦਾ ਹੈ ਅਤੇ ਕਹਾਣੀ ਸ਼ੁਰੂ ਹੁੰਦੀ ਹੈ।",
        "{subject} ਦੀ ਹਰ ਹਰਕਤ {setting} ਦਾ ਨਵਾਂ ਰੂਪ ਪੇਸ਼ ਕਰਦੀ ਹੈ।",
        "{subject} {setting} ਦੇ ਸੁੰਦਰ ਦ੍ਰਿਸ਼ ਤੇ ਪਹੁੰਚ ਕੇ ਸ਼ਾਂਤ ਹੋ ਜਾਂਦਾ ਹੈ।",
    ],
    "ml": [
        "{subject} {setting} ലൂടെ {action} ചെയ്തുകൊണ്ട് യാത്ര ആരംഭിക്കുന്നു.",
        "{subject} ന്റെ ഓരോ ചലനവും {setting} ന്റെ ഭംഗി വെളിപ്പെടുത്തുന്നു.",
        "{subject} {setting} ന്റെ മനോഹരമായ കാഴ്ചയിലെത്തി യാത്ര ശാന്തമാക്കുന്നു.",
    ],
    "ur": [
        "{subject} {setting} میں {action} کرتے ہوئے آگے بڑھتا ہے اور کہانی شروع ہوتی ہے۔",
        "{subject} کے ہر قدم پر {setting} کا نیا روپ سامنے آتا ہے۔",
        "{subject} {setting} کے دلکش منظر تک پہنچ کر مطمئن ہو جاتا ہے۔",
    ],
    "fr": [
        "Le {subject} commence à se déplacer dans le {setting} et l'histoire s'ouvre.",
        "Chaque mouvement du {subject} révèle davantage le {setting}.",
        "Le {subject} atteint une vue saisissante du {setting} et le voyage s'apaise.",
    ],
    "es": [
        "El {subject} comienza a moverse por el {setting} y la historia se abre.",
        "Cada movimiento del {subject} revela más del {setting}.",
        "El {subject} alcanza una vista imponente del {setting} y el viaje se aquieta.",
    ],
    "de": [
        "Der {subject} bewegt sich durch das {setting} und die Geschichte beginnt.",
        "Jede Bewegung des {subject} zeigt mehr vom {setting}.",
        "Der {subject} erreicht einen weiten Blick auf das {setting} und die Reise wird still.",
    ],
    "ja": [
        "{subject}は{setting}の中で{action}を始め、物語が動き出します。",
        "{subject}の一歩ごとに、{setting}の美しい風景が広がっていきます。",
        "{subject}は{setting}の素晴らしい景色にたどり着き、旅が静かに落ち着きます。",
    ],
}


def localize_narrative(
    language: Optional[str],
    scene_index: int,
    subject: str,
    setting: str,
    action: str,
    fallback: str = "",
) -> str:
    code = normalize_language(language)
    templates = _NARRATION.get(code, _NARRATION["en"])
    template = templates[(max(scene_index, 1) - 1) % len(templates)]
    rendered = template.format(
        subject=subject or "character",
        setting=setting or "landscape",
        action=action or "moving",
    )
    if code == "en" and fallback:
        return fallback
    return rendered
