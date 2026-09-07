"""
Text-to-Speech (TTS) adapters.
"""
from app.adapters.tts.local_tts_adapter import LocalTTSAdapter

def get_tts_adapter(provider: str = "local"):
    return LocalTTSAdapter()
