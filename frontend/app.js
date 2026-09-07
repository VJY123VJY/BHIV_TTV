document.addEventListener("DOMContentLoaded", () => {
    const generateForm = document.getElementById("generate-form");
    const promptInput = document.getElementById("prompt-input");
    const durationSlider = document.getElementById("duration-slider");
    const durationVal = document.getElementById("duration-val");
    const styleSelect = document.getElementById("style-select");
    const voiceToggle = document.getElementById("voice-toggle");
    const resolutionSelect = document.getElementById("resolution-select");
    const generateBtn = document.getElementById("generate-btn");

    const progressContainer = document.getElementById("progress-container");
    const stageText = document.getElementById("stage-text");
    const percentText = document.getElementById("percent-text");
    const progressBarFill = document.getElementById("progress-bar-fill");

    const placeholderContent = document.getElementById("placeholder-content");
    const playerWrapper = document.getElementById("player-wrapper");
    const videoPlayer = document.getElementById("video-player");
    const downloadBtn = document.getElementById("download-btn");
    const sceneBreakdownCard = document.getElementById("scene-breakdown-card");
    const scenesList = document.getElementById("scenes-list");

    // Duration slider live update
    durationSlider.addEventListener("input", (e) => {
        durationVal.textContent = e.target.value;
    });

    // Check system health on load
    fetch("/health")
        .then(res => res.json())
        .then(data => {
            const statusEl = document.getElementById("engine-status");
            if (statusEl && data.status === "healthy") {
                statusEl.textContent = `Engine: Online (FFmpeg: ${data.ffmpeg_available ? "Ready" : "Missing"})`;
            }
        })
        .catch(() => {
            const statusEl = document.getElementById("engine-status");
            if (statusEl) statusEl.textContent = "Engine: Connecting...";
        });

    // Stage tracker mapping
    const stageSteps = {
        "validating_prompt": "step-story",
        "understanding_prompt": "step-story",
        "generating_story": "step-story",
        "generating_script": "step-story",
        "generating_scenes": "step-scenes",
        "applying_visual_consistency": "step-visuals",
        "generating_keyframes": "step-visuals",
        "rendering_scene_videos": "step-visuals",
        "synthesizing_voice": "step-audio",
        "mixing_audio": "step-audio",
        "assembling_final_video": "step-final",
        "validating_video": "step-final",
        "storing_artifacts": "step-final"
    };

    function updateTrackers(activeStepId) {
        const allSteps = ["step-story", "step-scenes", "step-visuals", "step-audio", "step-final"];
        let foundActive = false;
        allSteps.forEach(sId => {
            const el = document.getElementById(sId);
            if (!el) return;
            el.classList.remove("active", "done");
            if (sId === activeStepId) {
                el.classList.add("active");
                foundActive = true;
            } else if (!foundActive) {
                el.classList.add("done");
            }
        });
    }

    generateForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            prompt: promptInput.value.trim(),
            duration: parseInt(durationSlider.value, 10),
            style: styleSelect.value,
            voice: voiceToggle.checked,
            resolution: resolutionSelect.value
        };

        if (!payload.prompt) {
            alert("Please enter a prompt.");
            return;
        }

        // Set UI to loading state
        generateBtn.disabled = true;
        generateBtn.querySelector(".btn-text").textContent = "Generating...";
        progressContainer.classList.remove("hidden");
        playerWrapper.classList.add("hidden");
        placeholderContent.classList.remove("hidden");
        sceneBreakdownCard.classList.add("hidden");

        progressBarFill.style.width = "5%";
        percentText.textContent = "5%";
        stageText.textContent = "Submitting generation job...";

        try {
            const res = await fetch("/api/v1/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || errData.message || "Failed to start generation");
            }

            const data = await res.json();
            const jobId = data.job_id;

            // Poll job status
            pollJobStatus(jobId);
        } catch (err) {
            alert("Error: " + err.message);
            resetBtn();
        }
    });

    function pollJobStatus(jobId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/jobs/${jobId}`);
                if (!res.ok) throw new Error("Could not fetch job status");

                const job = await res.json();
                const progress = Math.max(5, job.progress || 5);
                progressBarFill.style.width = `${progress}%`;
                percentText.textContent = `${progress}%`;
                stageText.textContent = formatStageName(job.stage);

                if (stageSteps[job.stage]) {
                    updateTrackers(stageSteps[job.stage]);
                }

                if (job.status === "completed") {
                    clearInterval(interval);
                    progressBarFill.style.width = "100%";
                    percentText.textContent = "100%";
                    stageText.textContent = "Video Generation Complete!";
                    updateTrackers("step-final");
                    document.getElementById("step-final").classList.add("done");

                    displayCompletedVideo(job.result);
                    resetBtn();
                } else if (job.status === "failed") {
                    clearInterval(interval);
                    stageText.textContent = "Generation Failed: " + (job.error || "Unknown error");
                    stageText.style.color = "#ef4444";
                    resetBtn();
                }
            } catch (e) {
                clearInterval(interval);
                stageText.textContent = "Error polling job: " + e.message;
                resetBtn();
            }
        }, 1000);
    }

    function formatStageName(stage) {
        if (!stage) return "Processing...";
        return stage.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    }

    function displayCompletedVideo(result) {
        if (!result || !result.video_url) return;

        placeholderContent.classList.add("hidden");
        playerWrapper.classList.remove("hidden");
        
        videoPlayer.src = result.video_url;
        videoPlayer.load();
        downloadBtn.href = result.video_url;

        // Populate scene breakdown list
        if (result.scenes && result.scenes.length > 0) {
            scenesList.innerHTML = "";
            result.scenes.forEach(s => {
                const item = document.createElement("div");
                item.className = "scene-item";
                item.innerHTML = `
                    <h4>${s.title} (${s.duration}s) • Motion: ${s.camera_motion || "pan"}</h4>
                    <p><strong>Narrative:</strong> ${s.narrative}</p>
                `;
                scenesList.appendChild(item);
            });
            sceneBreakdownCard.classList.remove("hidden");
        }
    }

    function resetBtn() {
        generateBtn.disabled = false;
        generateBtn.querySelector(".btn-text").textContent = "Generate Video";
    }
});
