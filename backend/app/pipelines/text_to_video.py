import os
import time
from typing import Dict, Any, List, Optional, Callable
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry
from app.core.security import governance_guard
from app.core.exceptions import TTVException, ValidationError, VisualGenerationError, FFmpegProcessingError
from app.services.prompt_service import prompt_service
from app.services.story_service import story_service
from app.services.scene_service import scene_service
from app.services.vision_service import vision_service
from app.services.image_service import image_service
from app.services.video_service import video_service
from app.services.tts_service import tts_service
from app.services.audio_service import audio_service
from app.services.ffmpeg_service import ffmpeg_service
from app.services.storage_service import storage_service
from app.utils.media_check import validate_video_file
from app.utils.hashing import generate_execution_id

class TextToVideoPipeline:
    """
    Master Text-to-Video orchestration pipeline.
    Implements the 14-stage lifecycle across the unified architecture:
    TEXT -> UNDERSTAND -> STORY -> SCENES -> VISUALS -> VIDEO -> AUDIO -> FINAL MP4
    """
    def __init__(self):
        self.governance = governance_guard

    # Stage 1: Validate Prompt
    def validate_prompt(self, prompt: str, execution_id: str) -> str:
        telemetry.emit("stage_1_validate_prompt", execution_id, {"prompt": prompt})
        return prompt_service.validate_prompt(prompt)

    # Stage 2: Understand Prompt
    async def understand_prompt(self, prompt: str, execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_2_understand_prompt", execution_id)
        return await prompt_service.understand_prompt(prompt)

    # Stage 3: Generate Story
    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int, execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_3_generate_story", execution_id, {"duration": duration})
        return await story_service.generate_story(prompt, analysis, duration)

    # Stage 4: Generate Script
    async def generate_script(self, story: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_4_generate_script", execution_id)
        # Story dictionary contains structured premise and act narratives
        return story

    # Stage 5: Generate Scenes
    async def generate_scenes(self, script: Dict[str, Any], duration: int, style: str, execution_id: str) -> List[Scene]:
        telemetry.emit("stage_5_generate_scenes", execution_id, {"style": style})
        return await scene_service.generate_scenes(script, duration, style)

    # Stage 6: Generate Visual Prompts
    def generate_visual_prompts(self, scenes: List[Scene], style: str, analysis: Dict[str, Any], execution_id: str) -> List[Scene]:
        telemetry.emit("stage_6_generate_visual_prompts", execution_id)
        return vision_service.apply_visual_consistency(scenes, style, analysis)

    # Stage 7: Generate Visual Assets (Keyframes)
    async def generate_visual_assets(self, scenes: List[Scene], execution_id: str) -> List[Scene]:
        telemetry.emit("stage_7_generate_visual_assets", execution_id)
        return await image_service.generate_scene_keyframes(scenes, execution_id)

    # Stage 8: Generate Scene Videos
    async def generate_scene_videos(self, scenes: List[Scene], execution_id: str, fps: int = 24) -> List[Scene]:
        telemetry.emit("stage_8_generate_scene_videos", execution_id, {"fps": fps})
        return await video_service.generate_scene_videos(scenes, execution_id, fps)

    # Stage 9: Generate Voice
    async def generate_voice(self, scenes: List[Scene], voice_enabled: bool, execution_id: str) -> List[Scene]:
        telemetry.emit("stage_9_generate_voice", execution_id, {"enabled": voice_enabled})
        if voice_enabled:
            return await tts_service.generate_scene_narration(scenes, execution_id)
        return scenes

    # Stage 10: Process Audio
    def process_audio(self, scenes: List[Scene], total_duration: float, execution_id: str) -> str:
        telemetry.emit("stage_10_process_audio", execution_id)
        temp_dir = settings.get_absolute_path(settings.TEMP_DIR)
        audio_out = str(temp_dir / f"{execution_id}_master_audio.wav")
        return audio_service.mix_complete_audio(scenes, total_duration, audio_out)

    # Stage 11: Assemble Video
    def assemble_video(self, scenes: List[Scene], master_audio_path: str, execution_id: str) -> str:
        telemetry.emit("stage_11_assemble_video", execution_id)
        temp_dir = settings.get_absolute_path(settings.TEMP_DIR)
        raw_final = str(temp_dir / f"{execution_id}_assembled.mp4")
        return ffmpeg_service.assemble_final_video(scenes, master_audio_path, raw_final, execution_id)

    # Stage 12: Validate Video
    def validate_video(self, video_path: str, execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_12_validate_video", execution_id)
        is_valid, info = validate_video_file(video_path)
        if not is_valid:
            raise VisualGenerationError(f"Video validation failed: {info.get('error')}")
        return info

    # Stage 13: Store Output
    def store_output(
        self,
        video_path: str,
        execution_id: str,
        prompt: str,
        scenes: List[Scene],
        duration: float,
        style: str
    ) -> Dict[str, Any]:
        telemetry.emit("stage_13_store_output", execution_id)
        return storage_service.persist_video_artifact(
            execution_id=execution_id,
            video_path=video_path,
            prompt=prompt,
            scenes=scenes,
            duration=duration,
            style=style,
            resolution=settings.DEFAULT_RESOLUTION,
            fps=settings.DEFAULT_FPS
        )

    # Stage 14: Return Result
    def return_result(self, metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_14_return_result", execution_id, {"status": "success"})
        return {
            "status": "success",
            "execution_id": execution_id,
            "metadata": metadata
        }

    # Master Execution Orchestrator
    async def execute(
        self,
        prompt: str,
        duration: int = 15,
        style: str = "cinematic",
        voice: bool = True,
        token: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full 14-stage generation workflow with progress tracking.
        """
        execution_id = job_id or generate_execution_id()
        start_time = time.time()
        telemetry.emit("pipeline_execution_started", execution_id, {"prompt": prompt})

        # Helper to report progress
        async def _report(stage_name: str, pct: int):
            if progress_callback:
                if callable(progress_callback):
                    res = progress_callback(stage_name, pct)
                    if hasattr(res, "__await__"):
                        await res

        try:
            # 0. Governance & Token Verification
            self.governance.verify_execution(token=token, execution_id=execution_id)

            # 1. Validate Prompt
            await _report("validating_prompt", 5)
            clean_prompt = self.validate_prompt(prompt, execution_id)

            # 2. Understand Prompt
            await _report("understanding_prompt", 12)
            analysis = await self.understand_prompt(clean_prompt, execution_id)

            # 3. Generate Story
            await _report("generating_story", 20)
            story = await self.generate_story(clean_prompt, analysis, duration, execution_id)

            # 4. Generate Script
            await _report("generating_script", 28)
            script = await self.generate_script(story, execution_id)

            # 5. Generate Scenes
            await _report("generating_scenes", 35)
            scenes = await self.generate_scenes(script, duration, style, execution_id)

            # 6. Generate Visual Prompts (Visual Consistency)
            await _report("applying_visual_consistency", 42)
            scenes = self.generate_visual_prompts(scenes, style, analysis, execution_id)

            # 7. Generate Visual Assets (Keyframe Images)
            await _report("generating_keyframes", 55)
            scenes = await self.generate_visual_assets(scenes, execution_id)

            # 8. Generate Scene Videos
            await _report("rendering_scene_videos", 68)
            scenes = await self.generate_scene_videos(scenes, execution_id, fps=settings.DEFAULT_FPS)

            # 9. Generate Voice / TTS
            await _report("synthesizing_voice", 78)
            scenes = await self.generate_voice(scenes, voice, execution_id)

            # 10. Process Audio
            await _report("mixing_audio", 85)
            master_audio = self.process_audio(scenes, float(duration), execution_id)

            # 11. Assemble Video
            await _report("assembling_final_video", 92)
            final_raw_video = self.assemble_video(scenes, master_audio, execution_id)

            # 12. Validate Video
            await _report("validating_video", 95)
            validation_info = self.validate_video(final_raw_video, execution_id)

            # 13. Store Output
            await _report("storing_artifacts", 98)
            metadata = self.store_output(
                video_path=final_raw_video,
                execution_id=execution_id,
                prompt=clean_prompt,
                scenes=scenes,
                duration=float(duration),
                style=style
            )
            metadata["processing_time_s"] = round(time.time() - start_time, 2)
            metadata["validation"] = validation_info

            # 14. Return Result
            await _report("completed", 100)
            return self.return_result(metadata, execution_id)

        except Exception as e:
            telemetry.emit("pipeline_execution_failed", execution_id, {"error": str(e)}, level="error")
            raise

pipeline = TextToVideoPipeline()
