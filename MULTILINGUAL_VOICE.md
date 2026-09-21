# Multilingual voice

Language selection controls the dialogue passed to TTS, the TTS voice, lip-sync audio, subtitle language, and generated-video metadata. It does not merely translate captions. The catalog supports English, Hindi, Marathi, Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, Malayalam, Urdu, Spanish, French, German, and Japanese.

`TTS_PROVIDER=auto` uses the local provider router; choose `edge`, `google`, `xtts`, `elevenlabs`, or `azure` when configured. The provider validates a language-compatible voice through `app/utils/languages.py`. For non-English, no fake offline speech fallback is used: a failed configured provider produces an explicit error.

Voice profiles are for authorized user-owned voices only. Store consent status with the provider and never use the feature to impersonate a person without permission.
