import os
import math
import struct
import wave
import subprocess
from typing import List, Optional
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry

class AudioService:
    """
    Audio mixing engine.
    - Synthesizes cinematic background soundtrack & ambient chords
    - Aligns and mixes scene voiceover tracks
    - Applies volume ducking and cross-fades
    - Produces synchronized master audio track
    """
    def mix_complete_audio(self, scenes: List[Scene], total_duration: float, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_rate = settings.AUDIO_SAMPLE_RATE
        total_samples = int(sample_rate * total_duration)

        # 1. Synthesize lush cinematic ambient background score (warm pads, subtle piano/bell chords)
        bg_samples = [0.0] * total_samples
        if settings.BACKGROUND_MUSIC_ENABLED:
            bg_samples = self._generate_cinematic_ambient_score(total_duration, sample_rate)

        # 2. Overlay voice tracks with volume ducking
        voice_combined = [0.0] * total_samples
        voice_mask = [0.0] * total_samples # 1.0 where voice is playing

        current_time = 0.0
        for scene in scenes:
            if scene.audio_path and os.path.exists(scene.audio_path):
                raw_voice, v_sr = self._load_audio_samples(scene.audio_path, sample_rate)
                start_sample = int(current_time * sample_rate)
                end_sample = min(total_samples, start_sample + len(raw_voice))
                
                for idx in range(start_sample, end_sample):
                    v_idx = idx - start_sample
                    voice_combined[idx] = raw_voice[v_idx]
                    voice_mask[idx] = 1.0

            current_time += scene.duration

        # Smooth voice mask for ducking transitions
        smoothed_ducking = self._smooth_mask(voice_mask, window_size=int(sample_rate * 0.3))

        # 3. Combine voice + ducked background score
        duck_ratio = settings.AUDIO_DUCKING_RATIO
        final_samples = []
        
        for i in range(total_samples):
            # When voice is active, duck background to duck_ratio (e.g. 0.3)
            bg_gain = 1.0 - (smoothed_ducking[i] * (1.0 - duck_ratio))
            sample_val = voice_combined[i] * 1.0 + bg_samples[i] * bg_gain * 0.40
            # Limiter / clipping protection
            sample_val = max(-0.95, min(0.95, sample_val))
            final_samples.append(int(sample_val * 32767))

        # 4. Write master audio WAV file
        wav_path = output_path if output_path.endswith(".wav") else output_path.rsplit(".", 1)[0] + ".wav"
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            raw_bytes = struct.pack(f"<{len(final_samples)}h", *final_samples)
            wf.writeframes(raw_bytes)

        return wav_path

    def _generate_cinematic_ambient_score(self, duration: float, sample_rate: int) -> List[float]:
        """Synthesizes an ambient cinematic chord progression (Cmaj9 -> Am9 -> Fmaj7 -> Gsus4)."""
        total_samples = int(duration * sample_rate)
        samples = [0.0] * total_samples
        
        # Chord progression roots and third/fifth/seventh frequencies
        chords = [
            [130.81, 164.81, 196.00, 246.94, 293.66], # Cmaj9
            [110.00, 130.81, 164.81, 220.00, 261.63], # Am9
            [87.31, 110.00, 130.81, 174.61, 220.00],  # Fmaj7
            [98.00, 130.81, 146.83, 196.00, 246.94]   # Gsus4
        ]
        
        chord_len = duration / len(chords)
        for c_idx, chord in enumerate(chords):
            c_start = int(c_idx * chord_len * sample_rate)
            c_end = min(total_samples, int((c_idx + 1) * chord_len * sample_rate))
            c_dur = (c_end - c_start) / sample_rate

            for i in range(c_start, c_end):
                t = (i - c_start) / sample_rate
                # Smooth bell-shaped envelope for chord transition
                env = math.sin(math.pi * (t / c_dur)) ** 1.5
                chord_val = 0.0
                for freq in chord:
                    # Warm detuned sine wave oscillators
                    chord_val += math.sin(2 * math.pi * freq * t) * 0.20
                    chord_val += math.sin(2 * math.pi * (freq * 1.002) * t) * 0.15
                    chord_val += math.sin(2 * math.pi * (freq * 0.5) * t) * 0.25 # Sub-bass warmth
                samples[i] = chord_val * env * 0.35

        return samples

    def _load_audio_samples(self, audio_path: str, target_sr: int) -> tuple:
        """Loads audio file and normalizes to float samples [-1.0, 1.0]."""
        # If wav, read directly via wave
        if audio_path.endswith(".wav"):
            try:
                with wave.open(audio_path, "r") as wf:
                    sr = wf.getframerate()
                    n = wf.getnframes()
                    frames = wf.readframes(n)
                    fmt = f"<{n * wf.getnchannels()}h"
                    ints = struct.unpack(fmt, frames)
                    # If stereo, downmix to mono
                    if wf.getnchannels() == 2:
                        ints = ints[::2]
                    return [x / 32768.0 for x in ints], sr
            except Exception:
                pass

        # Use ffmpeg to convert to standard mono PCM WAV
        temp_wav = audio_path + "_converted.wav"
        try:
            cmd = ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", str(target_sr), "-f", "wav", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with wave.open(temp_wav, "r") as wf:
                n = wf.getnframes()
                frames = wf.readframes(n)
                ints = struct.unpack(f"<{n}h", frames)
                samples = [x / 32768.0 for x in ints]
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return samples, target_sr
        except Exception:
            return [0.0] * int(target_sr * 3.0), target_sr

    def _smooth_mask(self, mask: List[float], window_size: int) -> List[float]:
        """Applies a moving average filter to smooth ducking transitions."""
        if not mask or window_size <= 1:
            return mask
        smoothed = [0.0] * len(mask)
        w_sum = sum(mask[:window_size])
        for i in range(len(mask)):
            smoothed[i] = w_sum / window_size
            if i + window_size < len(mask):
                w_sum += mask[i + window_size]
            if i < len(mask):
                w_sum -= mask[i]
        return smoothed

audio_service = AudioService()
