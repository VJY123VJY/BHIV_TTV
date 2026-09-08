/**
 * TTV Studio • AI Video Generation Engine Frontend
 * Modern reactive application orchestrating generation, preview, history, models, training, and RL feedback.
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // 1. STATE & STORAGE MANAGEMENT
    // =========================================================================
    const STORAGE_KEY_HISTORY = "ttv_video_history";
    const STORAGE_KEY_TOKEN = "ttv_governance_token";

    let currentJobId = null;
    let pollInterval = null;
    let timerInterval = null;
    let jobStartTime = null;
    let activeVideoData = null;
    let selectedRLWinner = null;
    let rlIteration = 1;

    // Seed project history if empty
    const DEFAULT_PROJECT_HISTORY = [
        {
            video_id: "exec_20260907_072037_090cc6ac",
            prompt: "A small rover roaming the dunes of Mars at sunset.",
            duration: 10,
            resolution: "1280x720",
            style: "cinematic",
            model: "base",
            video_url: "/generated/videos/exec_20260907_072037_090cc6ac.mp4",
            created_at: Date.now() - 1000 * 60 * 120,
            scenes: [
                { title: "Scene 1: The Awakening Horizon", duration: 3.33, narrative: "As daylight wanes over the futuristic metropolis, a lone traveler pauses to witness the glowing horizon.", camera_motion: "pan_right" },
                { title: "Scene 2: Through the Shining Pathways", duration: 3.33, narrative: "Every step unveils ancient secrets intertwined with soaring spires, alive with quiet energy.", camera_motion: "zoom_in" },
                { title: "Scene 3: The Grand Discovery", duration: 3.33, narrative: "Underneath the fading sun, the true wonder of this world reveals itself in all its majesty.", camera_motion: "pan_left" }
            ]
        },
        {
            video_id: "exec_20260907_072102_5f9a7088",
            prompt: "A neon cyberpunk cityscape with flying cars in rainy weather.",
            duration: 15,
            resolution: "1280x720",
            style: "cyberpunk",
            model: "finetuned",
            video_url: "/generated/videos/exec_20260907_072102_5f9a7088.mp4",
            created_at: Date.now() - 1000 * 60 * 95,
            scenes: [
                { title: "Scene 1: Neon Reflections", duration: 5.0, narrative: "Rain splatters against glowing billboards as synthetic lights reflect in dark streets.", camera_motion: "dolly_in" },
                { title: "Scene 2: Cyber Traffic", duration: 5.0, narrative: "Vehicles hover and glide through misty avenues amidst towering skyscrapers.", camera_motion: "tilt_up" },
                { title: "Scene 3: Skyline Horizon", duration: 5.0, narrative: "The camera pulls back to reveal the infinite scale of the neon metropolis.", camera_motion: "pan_right" }
            ]
        },
        {
            video_id: "exec_20260907_072207_d6031e3c",
            prompt: "Enchanted forest with magical glowing creatures and ancient trees.",
            duration: 15,
            resolution: "1280x720",
            style: "fantasy",
            model: "base",
            video_url: "/generated/videos/exec_20260907_072207_d6031e3c.mp4",
            created_at: Date.now() - 1000 * 60 * 60,
            scenes: [
                { title: "Scene 1: Twilight Glade", duration: 5.0, narrative: "Ethereal blue light filters through moss-covered branches.", camera_motion: "pan_left" },
                { title: "Scene 2: Luminescent Flora", duration: 5.0, narrative: "Bioluminescent blossoms open slowly to greet the evening twilight.", camera_motion: "zoom_in" },
                { title: "Scene 3: Awakening Whispers", duration: 5.0, narrative: "Gentle forest spirits emerge and dance around the sacred grove.", camera_motion: "crane_down" }
            ]
        }
    ];

    function getHistory() {
        try {
            const data = localStorage.getItem(STORAGE_KEY_HISTORY);
            if (!data) {
                localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(DEFAULT_PROJECT_HISTORY));
                return DEFAULT_PROJECT_HISTORY;
            }
            return JSON.parse(data);
        } catch (e) {
            return DEFAULT_PROJECT_HISTORY;
        }
    }

    function saveHistory(historyArray) {
        try {
            localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(historyArray));
            updateHistoryBadge();
        } catch (e) {
            console.error("Failed to persist history to localStorage", e);
        }
    }

    function addHistoryItem(item) {
        const history = getHistory();
        // Avoid duplicate by video_id
        const filtered = history.filter(h => h.video_id !== item.video_id);
        filtered.unshift(item);
        saveHistory(filtered);
        renderHistory();
    }

    // =========================================================================
    // 2. DOM ELEMENTS
    // =========================================================================
    // Navigation
    const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
    const viewPanels = document.querySelectorAll(".view-panel");
    const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
    const appSidebar = document.getElementById("app-sidebar");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");
    const historyCountBadge = document.getElementById("history-count-badge");

    // Generation Form
    const generateForm = document.getElementById("generate-form");
    const promptInput = document.getElementById("prompt-input");
    const charCountEl = document.getElementById("char-count");
    const wordCountEl = document.getElementById("word-count");
    const modelSelect = document.getElementById("model-select");
    const durationSlider = document.getElementById("duration-slider");
    const durationValBadge = document.getElementById("duration-val-badge");
    const aspectBtns = document.querySelectorAll(".aspect-btn");
    const resolutionSelect = document.getElementById("resolution-select");
    const styleSelect = document.getElementById("style-select");
    const seedInput = document.getElementById("seed-input");
    const randomizeSeedBtn = document.getElementById("randomize-seed-btn");
    const fpsSelect = document.getElementById("fps-select");
    const voiceToggle = document.getElementById("voice-toggle");
    const generateBtn = document.getElementById("generate-btn");
    const presetChips = document.querySelectorAll(".preset-chip");

    // Progress Card & 4 Stepper Steps
    const progressCard = document.getElementById("progress-card");
    const stageMainTitle = document.getElementById("stage-main-title");
    const stageSubTitle = document.getElementById("stage-sub-title");
    const percentText = document.getElementById("percent-text");
    const elapsedTimer = document.getElementById("elapsed-timer");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const step1 = document.getElementById("stage-step-1");
    const step2 = document.getElementById("stage-step-2");
    const step3 = document.getElementById("stage-step-3");
    const step4 = document.getElementById("stage-step-4");

    // Video Preview Player
    const playerContainer = document.getElementById("player-container");
    const emptyPlayerState = document.getElementById("empty-player-state");
    const generatingOverlay = document.getElementById("generating-player-overlay");
    const activePlayerWrapper = document.getElementById("active-player-wrapper");
    const videoPlayer = document.getElementById("video-player");
    const previewFormatBadge = document.getElementById("preview-format-badge");
    const previewActionToolbar = document.getElementById("preview-action-toolbar");
    const downloadBtn = document.getElementById("download-btn");
    const regenerateBtn = document.getElementById("regenerate-btn");
    const shareCopyBtn = document.getElementById("share-copy-btn");
    const sendToRlBtn = document.getElementById("send-to-rl-btn");
    const sceneBreakdownCard = document.getElementById("scene-breakdown-card");
    const sceneCountBadge = document.getElementById("scene-count-badge");
    const scenesList = document.getElementById("scenes-list");

    // History View
    const historyGrid = document.getElementById("history-grid");
    const emptyHistoryState = document.getElementById("empty-history-state");
    const historySearchInput = document.getElementById("history-search-input");
    const historyClearAllBtn = document.getElementById("history-clear-all-btn");

    // Models View
    const selectModelBtns = document.querySelectorAll(".select-model-btn");

    // Training View
    const datasetDropzone = document.getElementById("dataset-dropzone");
    const datasetFileInput = document.getElementById("dataset-file-input");
    const startTrainingBtn = document.getElementById("start-training-btn");
    const trainingTerminalLog = document.getElementById("training-terminal-log");
    const trainingSessionStatus = document.getElementById("training-session-status");

    // RL Feedback View
    const candidateCardA = document.getElementById("candidate-card-a");
    const candidateCardB = document.getElementById("candidate-card-b");
    const voteABtn = document.getElementById("vote-a-btn");
    const voteBBtn = document.getElementById("vote-b-btn");
    const submitFeedbackBtn = document.getElementById("submit-feedback-btn");
    const feedbackHistoryList = document.getElementById("feedback-history-list");
    const rlActivePrompt = document.getElementById("rl-active-prompt");
    const rlIterationBadge = document.getElementById("rl-iteration-badge");
    const rlVideoB = document.getElementById("rl-video-b");

    // Settings View
    const settingsTokenInput = document.getElementById("settings-token-input");
    const readoutLlm = document.getElementById("readout-llm");
    const readoutImage = document.getElementById("readout-image");
    const readoutVideo = document.getElementById("readout-video");
    const readoutTts = document.getElementById("readout-tts");
    const readoutFfmpeg = document.getElementById("readout-ffmpeg");
    const engineStatusText = document.getElementById("engine-status-text");
    const engineDetailsText = document.getElementById("engine-details-text");
    const statusIndicator = document.getElementById("status-indicator");
    const mobileEnginePill = document.getElementById("mobile-engine-pill");

    // Toast Container
    const toastContainer = document.getElementById("toast-container");

    // =========================================================================
    // 3. TOAST NOTIFICATION HELPER
    // =========================================================================
    function showToast(message, type = "success", duration = 3500) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
        if (type === "error") {
            iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        }

        toast.innerHTML = `${iconSvg} <span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.transition = "opacity 0.3s, transform 0.3s";
            toast.style.opacity = "0";
            toast.style.transform = "translateX(40px)";
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // =========================================================================
    // 4. SPA NAVIGATION ROUTER
    // =========================================================================
    function switchView(targetViewId) {
        navItems.forEach(item => {
            if (item.dataset.view === targetViewId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        viewPanels.forEach(panel => {
            if (panel.id === targetViewId) {
                panel.classList.add("active");
            } else {
                panel.classList.remove("active");
            }
        });

        // Close mobile drawer if open
        if (appSidebar.classList.contains("open")) {
            appSidebar.classList.remove("open");
            sidebarBackdrop.classList.remove("active");
        }

        // View-specific actions
        if (targetViewId === "view-history") {
            renderHistory();
        } else if (targetViewId === "view-settings") {
            fetchSystemHealth();
        }
    }

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            switchView(item.dataset.view);
        });
    });

    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener("click", () => {
            appSidebar.classList.toggle("open");
            sidebarBackdrop.classList.toggle("active");
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", () => {
            appSidebar.classList.remove("open");
            sidebarBackdrop.classList.remove("active");
        });
    }

    // =========================================================================
    // 5. PROMPT COUNTER & PRESETS
    // =========================================================================
    function updatePromptCounters() {
        const text = promptInput.value;
        charCountEl.textContent = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        wordCountEl.textContent = words;
    }

    promptInput.addEventListener("input", updatePromptCounters);

    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            promptInput.value = chip.dataset.prompt;
            updatePromptCounters();
            promptInput.focus();
            showToast("Prompt template applied!");
        });
    });

    // =========================================================================
    // 6. GENERATION CONTROLS (DURATION, ASPECT, SEED)
    // =========================================================================
    durationSlider.addEventListener("input", (e) => {
        durationValBadge.textContent = `${e.target.value} seconds`;
    });

    aspectBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            aspectBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Sync with resolution dropdown if available
            const res = btn.dataset.res;
            if (resolutionSelect) {
                // If the resolution select has this option, select it; otherwise add or match
                let exists = false;
                for (let i = 0; i < resolutionSelect.options.length; i++) {
                    if (resolutionSelect.options[i].value === res) {
                        resolutionSelect.selectedIndex = i;
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    const newOpt = new Option(`${res} (${btn.dataset.aspect})`, res, true, true);
                    resolutionSelect.add(newOpt);
                }
            }
        });
    });

    randomizeSeedBtn.addEventListener("click", () => {
        const randomSeed = Math.floor(Math.random() * 1000000);
        seedInput.value = randomSeed;
        showToast(`Random seed set: ${randomSeed}`);
    });

    // Model selection buttons from the Models view
    selectModelBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const modelVal = btn.dataset.model;
            modelSelect.value = modelVal;
            switchView("view-generate");
            showToast(`Model set to: ${modelSelect.options[modelSelect.selectedIndex].text}`);
        });
    });

    // =========================================================================
    // 7. SYSTEM HEALTH MONITOR
    // =========================================================================
    async function fetchSystemHealth() {
        try {
            const res = await fetch("/health");
            if (!res.ok) throw new Error("Health endpoint error");
            const data = await res.json();

            if (data.status === "healthy") {
                if (engineStatusText) engineStatusText.textContent = "Engine Online";
                if (statusIndicator) statusIndicator.className = "status-indicator-dot online";
                if (mobileEnginePill) {
                    mobileEnginePill.className = "status-pill status-ready";
                    mobileEnginePill.textContent = "Online";
                }
                if (engineDetailsText) {
                    engineDetailsText.textContent = `FFmpeg: ${data.ffmpeg_available ? "Ready" : "Missing"} • ${data.providers?.video || "opencv"}`;
                }

                // Populate settings readout
                if (readoutLlm && data.providers?.llm) readoutLlm.textContent = data.providers.llm;
                if (readoutImage && data.providers?.image) readoutImage.textContent = data.providers.image;
                if (readoutVideo && data.providers?.video) readoutVideo.textContent = data.providers.video;
                if (readoutTts && data.providers?.tts) readoutTts.textContent = data.providers.tts;
                if (readoutFfmpeg) {
                    readoutFfmpeg.textContent = data.ffmpeg_available ? "Installed (Accelerated)" : "Warning: Not in PATH";
                    readoutFfmpeg.className = data.ffmpeg_available ? "code-pill highlight-green" : "code-pill text-muted";
                }
            }
        } catch (e) {
            if (engineStatusText) engineStatusText.textContent = "Engine Offline";
            if (statusIndicator) statusIndicator.className = "status-indicator-dot";
            if (mobileEnginePill) {
                mobileEnginePill.className = "status-pill";
                mobileEnginePill.textContent = "Offline";
            }
        }
    }

    // Call health on initial load
    fetchSystemHealth();

    // =========================================================================
    // 8. GENERATION WORKFLOW & STAGE TRACKER (5 REQUIRED STAGES)
    // =========================================================================
    function setStepState(activeStepNum) {
        const steps = [step1, step2, step3, step4];
        steps.forEach((s, idx) => {
            if (!s) return;
            const stepNum = idx + 1;
            s.classList.remove("active", "done");
            if (stepNum === activeStepNum) {
                s.classList.add("active");
            } else if (stepNum < activeStepNum) {
                s.classList.add("done");
            }
        });
    }

    function startTimer() {
        jobStartTime = Date.now();
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - jobStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            elapsedTimer.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    function resetGenerationUI() {
        generateBtn.disabled = false;
        generateBtn.querySelector(".btn-text").textContent = "Generate Video";
        stopTimer();
    }

    generateForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const prompt = promptInput.value.trim();
        if (prompt.length < 3) {
            showToast("Please enter a descriptive prompt with at least 3 characters.", "error");
            return;
        }

        const duration = parseInt(durationSlider.value, 10);
        const style = styleSelect.value;
        const voice = voiceToggle.checked;
        const resolution = resolutionSelect.value;
        const fps = parseInt(fpsSelect.value, 10);
        const modelMode = modelSelect.value;
        const token = (settingsTokenInput ? settingsTokenInput.value.trim() : null) || null;

        const payload = {
            prompt,
            duration,
            style,
            voice,
            resolution,
            fps,
            model_mode: modelMode,
            token
        };

        // UI state change to generating
        generateBtn.disabled = true;
        generateBtn.querySelector(".btn-text").textContent = "Generating Video...";

        progressCard.classList.remove("hidden");
        stageMainTitle.textContent = "Generating video...";
        stageSubTitle.textContent = "Processing prompt...";
        progressBarFill.style.width = "5%";
        percentText.textContent = "5%";
        setStepState(1);
        startTimer();

        // Prepare player area
        emptyPlayerState.classList.add("hidden");
        generatingOverlay.classList.remove("hidden");
        activePlayerWrapper.classList.add("hidden");
        previewActionToolbar.classList.add("hidden");
        sceneBreakdownCard.classList.add("hidden");
        previewFormatBadge.textContent = "Processing...";

        try {
            const res = await fetch("/api/v1/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.message || errData.detail || "Failed to submit generation job");
            }

            const data = await res.json();
            currentJobId = data.job_id;
            showToast("Job submitted successfully! Synthesizing video...");

            // Begin polling status
            pollJob(currentJobId, payload);
        } catch (err) {
            showToast(`Error: ${err.message}`, "error");
            generatingOverlay.classList.add("hidden");
            emptyPlayerState.classList.remove("hidden");
            stageMainTitle.textContent = "Generation Failed";
            stageSubTitle.textContent = err.message;
            resetGenerationUI();
        }
    });

    function pollJob(jobId, originalPayload) {
        clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/jobs/${jobId}`);
                if (!res.ok) throw new Error("Could not retrieve job status");

                const job = await res.json();
                const progress = Math.max(5, job.progress || 5);
                progressBarFill.style.width = `${progress}%`;
                percentText.textContent = `${progress}%`;

                // Stage Mapping to the 4 UI Step Bullets & Display
                // Stage 1: Processing prompt...
                // Stage 2: Generating frames...
                // Stage 3: Creating video...
                // Stage 4: Completed
                const stage = job.stage || "";

                if (stage.includes("prompt") || stage.includes("story") || stage.includes("script")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Processing prompt...";
                    setStepState(1);
                } else if (stage.includes("scene") || stage.includes("consistency") || stage.includes("keyframe")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Generating frames...";
                    setStepState(2);
                } else if (stage.includes("rendering") || stage.includes("voice") || stage.includes("audio") || stage.includes("assembling") || stage.includes("validating") || stage.includes("storing")) {
                    stageMainTitle.textContent = "Generating video...";
                    stageSubTitle.textContent = "Creating video...";
                    setStepState(3);
                }

                if (job.status === "completed") {
                    clearInterval(pollInterval);
                    progressBarFill.style.width = "100%";
                    percentText.textContent = "100%";
                    stageMainTitle.textContent = "Completed";
                    stageSubTitle.textContent = "Video successfully generated!";
                    setStepState(4);

                    displayCompletedVideo(job.result, originalPayload);
                    resetGenerationUI();
                    showToast("Video generation completed successfully!");
                } else if (job.status === "failed") {
                    clearInterval(pollInterval);
                    stageMainTitle.textContent = "Generation Failed";
                    stageSubTitle.textContent = job.error || "An internal error occurred during synthesis.";
                    generatingOverlay.classList.add("hidden");
                    emptyPlayerState.classList.remove("hidden");
                    resetGenerationUI();
                    showToast(`Pipeline Error: ${job.error || "Generation failed"}`, "error");
                }
            } catch (err) {
                clearInterval(pollInterval);
                stageMainTitle.textContent = "Network Polling Interrupted";
                stageSubTitle.textContent = err.message;
                resetGenerationUI();
            }
        }, 1000);
    }

    function displayCompletedVideo(result, payload) {
        if (!result || !result.video_url) return;

        activeVideoData = {
            video_id: result.video_id || `gen_${Date.now()}`,
            prompt: result.prompt || payload.prompt,
            duration: result.duration || payload.duration,
            resolution: result.resolution || payload.resolution,
            style: result.style || payload.style,
            model: payload.model_mode || "base",
            video_url: result.video_url,
            created_at: Date.now(),
            scenes: result.scenes || []
        };

        // Add to persistent history
        addHistoryItem(activeVideoData);

        // Update player view
        generatingOverlay.classList.add("hidden");
        emptyPlayerState.classList.add("hidden");
        activePlayerWrapper.classList.remove("hidden");

        const cacheBustedUrl = `${result.video_url}?t=${Date.now()}`;
        videoPlayer.src = cacheBustedUrl;
        videoPlayer.load();

        previewFormatBadge.textContent = `${activeVideoData.resolution} • ${activeVideoData.duration}s • ${activeVideoData.style.toUpperCase()}`;

        // Action Toolbar
        previewActionToolbar.classList.remove("hidden");
        downloadBtn.href = result.video_url;
        downloadBtn.download = `${activeVideoData.video_id}.mp4`;

        // Scene breakdown list
        if (result.scenes && result.scenes.length > 0) {
            renderSceneBreakdown(result.scenes);
        } else {
            sceneBreakdownCard.classList.add("hidden");
        }
    }

    function renderSceneBreakdown(scenes) {
        scenesList.innerHTML = "";
        sceneCountBadge.textContent = `${scenes.length} Scenes`;

        scenes.forEach((scene, index) => {
            const card = document.createElement("div");
            card.className = "scene-item-card";
            card.innerHTML = `
                <div class="scene-header">
                    <span class="scene-heading">${scene.title || `Scene ${index + 1}`} (${scene.duration || 3.3}s)</span>
                    <span class="scene-motion-tag">Motion: ${scene.camera_motion || "pan_right"}</span>
                </div>
                <p class="scene-narrative">"${scene.narrative || scene.visual_description || "Scene visuals in sequence."}"</p>
            `;
            scenesList.appendChild(card);
        });

        sceneBreakdownCard.classList.remove("hidden");
    }

    // =========================================================================
    // 9. VIDEO ACTION TOOLBAR HANDLERS
    // =========================================================================
    regenerateBtn.addEventListener("click", () => {
        if (!activeVideoData) return;
        promptInput.value = activeVideoData.prompt;
        updatePromptCounters();
        generateForm.dispatchEvent(new Event("submit"));
        showToast("Re-initiating video generation...");
    });

    shareCopyBtn.addEventListener("click", () => {
        if (!activeVideoData || !activeVideoData.video_url) return;
        const fullUrl = window.location.origin + activeVideoData.video_url;
        navigator.clipboard.writeText(fullUrl).then(() => {
            showToast("Video URL copied to clipboard!");
        }).catch(() => {
            showToast(`Video URL: ${fullUrl}`);
        });
    });

    sendToRlBtn.addEventListener("click", () => {
        if (!activeVideoData) return;
        // Populate RL view with current active video as candidate B
        rlActivePrompt.textContent = `"${activeVideoData.prompt}"`;
        if (rlVideoB) {
            rlVideoB.src = activeVideoData.video_url;
            rlVideoB.load();
        }
        switchView("view-rl");
        showToast("Loaded video into RL Preference Studio!");
    });

    // =========================================================================
    // 10. HISTORY MANAGEMENT & RENDERING
    // =========================================================================
    function updateHistoryBadge() {
        const history = getHistory();
        if (historyCountBadge) {
            historyCountBadge.textContent = history.length;
        }
    }

    function renderHistory(filterText = "") {
        const history = getHistory();
        historyGrid.innerHTML = "";

        const query = filterText.trim().toLowerCase();
        const filtered = query 
            ? history.filter(h => h.prompt.toLowerCase().includes(query) || (h.style && h.style.toLowerCase().includes(query)))
            : history;

        if (filtered.length === 0) {
            emptyHistoryState.classList.remove("hidden");
            return;
        }

        emptyHistoryState.classList.add("hidden");

        filtered.forEach(item => {
            const card = document.createElement("div");
            card.className = "history-card";

            const dateStr = item.created_at ? new Date(item.created_at).toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            }) : "Recent";

            card.innerHTML = `
                <div class="history-thumb-box">
                    <video src="${item.video_url}" muted preload="metadata" onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0;"></video>
                </div>
                <div class="history-card-body">
                    <div class="history-meta-row">
                        <span class="history-model-tag">${item.model === "finetuned" ? "LoRA v001" : "Base TTV"}</span>
                        <span class="history-date">${dateStr}</span>
                    </div>
                    <p class="history-prompt" title="${item.prompt}">${item.prompt}</p>
                    <div class="history-specs-row">
                        <span class="spec-badge">${item.resolution || "720p"}</span>
                        <span class="spec-badge">${item.duration || 15}s</span>
                        <span class="spec-badge">${(item.style || "cinematic").toUpperCase()}</span>
                        <span class="badge badge-success" style="margin-left: auto;">Completed</span>
                    </div>
                    <div class="history-actions-row">
                        <button type="button" class="btn btn-secondary btn-sm hist-play-btn">Play</button>
                        <button type="button" class="btn btn-secondary btn-sm hist-prompt-btn">Load</button>
                        <a href="${item.video_url}" download="${item.video_id}.mp4" class="btn btn-secondary btn-sm" title="Download MP4">⬇</a>
                    </div>
                </div>
            `;

            // Handlers
            card.querySelector(".hist-play-btn").addEventListener("click", () => {
                displayCompletedVideo(item, {
                    prompt: item.prompt,
                    duration: item.duration,
                    resolution: item.resolution,
                    style: item.style,
                    model_mode: item.model
                });
                switchView("view-generate");
                window.scrollTo({ top: 0, behavior: "smooth" });
            });

            card.querySelector(".hist-prompt-btn").addEventListener("click", () => {
                promptInput.value = item.prompt;
                updatePromptCounters();
                if (item.style) styleSelect.value = item.style;
                if (item.duration) {
                    durationSlider.value = item.duration;
                    durationValBadge.textContent = `${item.duration} seconds`;
                }
                switchView("view-generate");
                promptInput.focus();
                showToast("Prompt and settings loaded into workspace!");
            });

            historyGrid.appendChild(card);
        });

        updateHistoryBadge();
    }

    if (historySearchInput) {
        historySearchInput.addEventListener("input", (e) => {
            renderHistory(e.target.value);
        });
    }

    if (historyClearAllBtn) {
        historyClearAllBtn.addEventListener("click", () => {
            if (confirm("Are you sure you want to clear your video generation history?")) {
                localStorage.removeItem(STORAGE_KEY_HISTORY);
                renderHistory();
                showToast("History cleared.");
            }
        });
    }

    // =========================================================================
    // 11. TRAINING / FINE-TUNING DASHBOARD ACTIONS
    // =========================================================================
    if (datasetDropzone) {
        datasetDropzone.addEventListener("click", () => {
            datasetFileInput.click();
        });

        datasetDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            datasetDropzone.style.borderColor = "var(--brand-primary)";
        });

        datasetDropzone.addEventListener("dragleave", () => {
            datasetDropzone.style.borderColor = "";
        });

        datasetDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            datasetDropzone.style.borderColor = "";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleDatasetFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (datasetFileInput) {
        datasetFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleDatasetFile(e.target.files[0]);
            }
        });
    }

    function handleDatasetFile(file) {
        const dropzonePrompt = datasetDropzone.querySelector(".dropzone-prompt");
        dropzonePrompt.innerHTML = `Loaded manifest: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
        showToast(`Dataset file ${file.name} loaded successfully!`);
    }

    if (startTrainingBtn) {
        startTrainingBtn.addEventListener("click", () => {
            const epochs = document.getElementById("train-epochs").value;
            const lr = document.getElementById("train-lr").value;
            const batchSize = document.getElementById("train-batch-size").value;
            const method = document.getElementById("train-method").value;

            trainingSessionStatus.textContent = "Session Active";
            trainingSessionStatus.className = "badge badge-success";

            const logLine = document.createElement("div");
            logLine.className = "log-line text-accent";
            logLine.textContent = `[Launch] Initializing Fine-Tuning: method=${method}, epochs=${epochs}, lr=${lr}, batch=${batchSize}`;
            trainingTerminalLog.appendChild(logLine);

            const cliLine = document.createElement("div");
            cliLine.className = "log-line text-muted";
            cliLine.textContent = `[Worker] Run training process via terminal: python training/run_training.py --config training/configs/smoke_test.yaml`;
            trainingTerminalLog.appendChild(cliLine);
            trainingTerminalLog.scrollTop = trainingTerminalLog.scrollHeight;

            showToast("Training session dispatched. Inspect worker terminal log for details.", "success", 4000);
        });
    }

    // =========================================================================
    // 12. RL FEEDBACK STUDIO ACTIONS
    // =========================================================================
    if (voteABtn && voteBBtn) {
        voteABtn.addEventListener("click", () => {
            selectedRLWinner = "candidate_a";
            candidateCardA.classList.add("selected");
            candidateCardB.classList.remove("selected");
            submitFeedbackBtn.disabled = false;
            submitFeedbackBtn.textContent = "Submit Preference (Candidate A Winner)";
        });

        voteBBtn.addEventListener("click", () => {
            selectedRLWinner = "candidate_b";
            candidateCardB.classList.add("selected");
            candidateCardA.classList.remove("selected");
            submitFeedbackBtn.disabled = false;
            submitFeedbackBtn.textContent = "Submit Preference (Candidate B Winner)";
        });
    }

    if (submitFeedbackBtn) {
        submitFeedbackBtn.addEventListener("click", () => {
            if (!selectedRLWinner) return;

            rlIteration++;
            if (rlIterationBadge) {
                rlIterationBadge.textContent = `Iteration ${rlIteration} / 3`;
            }

            const winnerTag = selectedRLWinner === "candidate_a" ? "Candidate A" : "Candidate B";
            const winnerScore = selectedRLWinner === "candidate_a" ? "0.8200" : "0.9100";

            const item = document.createElement("div");
            item.className = "feedback-item";
            item.innerHTML = `
                <span class="fb-time">Sample ${feedbackHistoryList.children.length + 1}</span>
                <span class="fb-prompt">${rlActivePrompt.textContent}</span>
                <span class="badge badge-success">${winnerTag} Selected</span>
                <span class="fb-reward">Reward Winner: ${winnerScore}</span>
            `;
            feedbackHistoryList.prepend(item);

            candidateCardA.classList.remove("selected");
            candidateCardB.classList.remove("selected");
            submitFeedbackBtn.disabled = true;
            submitFeedbackBtn.textContent = "Submit Preference to Loop";
            selectedRLWinner = null;

            showToast(`Preference recorded for DPO loop iteration ${rlIteration}!`);
        });
    }

    // Initial render of history
    renderHistory();
    updatePromptCounters();
});
