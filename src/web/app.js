const uploadZone = document.getElementById('upload-zone');
const videoUpload = document.getElementById('asl-video-upload');
const aslPreview = document.getElementById('asl-preview');
const aslGlossInput = document.getElementById('asl-gloss-input');
const translateBtn = document.getElementById('translate-btn');
const webcamBtn = document.getElementById('webcam-btn');
const clearVideoBtn = document.getElementById('clear-video-btn');
const recognitionStatus = document.getElementById('recognition-status');

const avatarStage = document.getElementById('avatar-stage');
const avatarPlaceholder = document.getElementById('avatar-placeholder');
const avatarLoading = document.getElementById('avatar-loading');
const islVideo = document.getElementById('isl-video');
const islGlossOutput = document.getElementById('isl-gloss-output');
const avatarStatus = document.getElementById('avatar-status');

let currentTranslationRequest = null;
let recognitionAvailable = false;
let recognitionThreshold = null;
let mediaRecorder = null;
let webcamStream = null;
let webcamChunks = [];
let webcamTimer = null;

fetch('/health')
  .then((response) => response.json())
  .then((status) => {
    recognitionAvailable = Boolean(status.model_loaded && status.recognition_enabled);
    recognitionThreshold = Number.isFinite(status.recognition_min_confidence)
      ? status.recognition_min_confidence
      : null;
    if (recognitionAvailable) {
      recognitionStatus.textContent = `${status.recognition_scope}; low-confidence clips are rejected.`;
    } else {
      recognitionStatus.textContent = 'Video recognition is disabled until the validated model is deployed; manual gloss input remains available.';
    }
    webcamBtn.disabled = !recognitionAvailable || !navigator.mediaDevices || !window.MediaRecorder;
    uploadZone.setAttribute('aria-disabled', String(!recognitionAvailable));
  })
  .catch(() => {
    recognitionStatus.textContent = 'Could not read model status; manual gloss input remains available.';
    webcamBtn.disabled = true;
  });

function showAvatarError(message) {
  avatarLoading.hidden = true;
  islVideo.pause();
  islVideo.hidden = true;
  avatarPlaceholder.hidden = false;
  avatarPlaceholder.innerHTML = `<p style="color: #ef4444;">${message}</p>`;
  avatarStatus.textContent = message;
}

function loadAvatarVideo(url) {
  return new Promise((resolve, reject) => {
    const onReady = async () => {
      cleanup();
      avatarLoading.hidden = true;
      islVideo.hidden = false;
      try { await islVideo.play(); } catch (_) { /* Controls remain available for manual play. */ }
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("The generated avatar video could not be played by this browser."));
    };
    const cleanup = () => {
      islVideo.removeEventListener('loadeddata', onReady);
      islVideo.removeEventListener('error', onError);
    };
    islVideo.addEventListener('loadeddata', onReady, { once: true });
    islVideo.addEventListener('error', onError, { once: true });
    islVideo.src = url;
    islVideo.load();
  });
}

// UI STATE MANAGEMENT
function updateTranslateButton() {
  translateBtn.disabled = aslGlossInput.value.trim().length === 0;
}

aslGlossInput.addEventListener('input', updateTranslateButton);

// VIDEO UPLOAD HANDLING
uploadZone.addEventListener('click', () => {
  if (recognitionAvailable) videoUpload.click();
});
uploadZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    if (recognitionAvailable) videoUpload.click();
  }
});
uploadZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadZone.classList.add('drag-active');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-active'));
uploadZone.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadZone.classList.remove('drag-active');
  if (recognitionAvailable) handleVideoFile(event.dataTransfer.files[0]);
});

videoUpload.addEventListener('change', (event) => handleVideoFile(event.target.files[0]));

async function handleVideoFile(file) {
  if (!file) return;
  let shouldTranslateRecognizedVideo = false;
  if (file.size > 50 * 1024 * 1024) {
    alert('Please choose a video smaller than 50 MB.');
    return;
  }

  // Show preview
  const url = URL.createObjectURL(file);
  aslPreview.srcObject = null;
  aslPreview.muted = false;
  uploadZone.hidden = true;
  aslPreview.src = url;
  aslPreview.hidden = false;
  clearVideoBtn.hidden = false;
  webcamBtn.hidden = true;
  aslPreview.play();

  // Call recognition API
  aslGlossInput.value = "Analyzing video...";
  aslGlossInput.disabled = true;
  translateBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict-sequence', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || 'Failed to predict ASL');
    if (!data.accepted || data.gloss === 'UNKNOWN') {
      const confidence = Number.isFinite(data.confidence) ? data.confidence : 0;
      const threshold = recognitionThreshold ?? 0.5904;
      const candidate = data.candidate_gloss || 'an unsupported sign';
      aslGlossInput.value = '';
      recognitionStatus.textContent =
        `Video safely rejected: ${candidate} was only ${(confidence * 100).toFixed(1)}% confident ` +
        `(needs ${(threshold * 100).toFixed(1)}%). Use one short MSASL-100-style isolated sign, ` +
        `or type the ASL gloss manually.`;
      return;
    }
    aslGlossInput.value = data.gloss;
    recognitionStatus.textContent = `Recognized ${data.gloss} at ${(data.confidence * 100).toFixed(1)}% confidence.`;
    // A successful video is already a complete input. Start the next stage so
    // users are not left looking at the initial avatar placeholder and wondering
    // whether they must manually press the text-translation button.
    shouldTranslateRecognizedVideo = true;
  } catch (err) {
    aslGlossInput.value = "";
    recognitionStatus.textContent = `Video could not be recognized: ${err.message} Enter ASL gloss manually instead.`;
  } finally {
    aslGlossInput.disabled = false;
    updateTranslateButton();
    if (shouldTranslateRecognizedVideo && !translateBtn.disabled) {
      translateBtn.click();
    }
  }
}

webcamBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    uploadZone.hidden = true;
    aslPreview.src = '';
    aslPreview.srcObject = webcamStream;
    aslPreview.muted = true;
    aslPreview.hidden = false;
    clearVideoBtn.hidden = false;
    webcamBtn.hidden = true;
    await aslPreview.play();
    webcamChunks = [];
    const preferredType = [
      'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'
    ].find((type) => MediaRecorder.isTypeSupported(type));
    mediaRecorder = preferredType
      ? new MediaRecorder(webcamStream, { mimeType: preferredType })
      : new MediaRecorder(webcamStream);
    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size) webcamChunks.push(event.data);
    });
    mediaRecorder.addEventListener('stop', async () => {
      clearTimeout(webcamTimer);
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
      webcamBtn.textContent = 'Record webcam sign (max 8 seconds)';
      const recordedType = mediaRecorder.mimeType || preferredType || 'video/webm';
      const extension = recordedType.startsWith('video/mp4') ? 'mp4' : 'webm';
      const recording = new File(
        webcamChunks,
        `webcam-sign-${Date.now()}.${extension}`,
        { type: recordedType },
      );
      await handleVideoFile(recording);
    }, { once: true });
    mediaRecorder.start(250);
    webcamBtn.textContent = 'Stop and recognize';
    recognitionStatus.textContent = 'Recording… keep your upper body and both hands visible.';
    webcamTimer = setTimeout(() => {
      if (mediaRecorder.state === 'recording') mediaRecorder.stop();
    }, 8000);
  } catch (error) {
    recognitionStatus.textContent = `Webcam unavailable: ${error.message}`;
  }
});

clearVideoBtn.addEventListener('click', () => {
  aslPreview.hidden = true;
  aslPreview.src = '';
  aslPreview.srcObject = null;
  uploadZone.hidden = false;
  clearVideoBtn.hidden = true;
  webcamBtn.hidden = false;
  videoUpload.value = '';
  aslGlossInput.value = '';
  aslGlossInput.disabled = false;
  updateTranslateButton();
  
  if (recognitionAvailable) {
    recognitionStatus.textContent = 'Upload or record a video, or type gloss manually.';
  } else {
    recognitionStatus.textContent = 'Video recognition is disabled until the validated model is deployed; manual gloss input remains available.';
  }
});

// TRANSLATION PIPELINE
translateBtn.addEventListener('click', async () => {
  const aslGloss = aslGlossInput.value.trim();
  if (!aslGloss) return;

  translateBtn.disabled = true;
  avatarPlaceholder.hidden = true;
  islVideo.hidden = true;
  avatarLoading.hidden = false;
  islGlossOutput.textContent = "Translating grammar...";

  try {
    // 1. Translate ASL to ISL
    const transRes = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asl_gloss: aslGloss })
    });
    const transData = await transRes.json();
    if (!transRes.ok) throw new Error(transData.error || 'Translation failed');
    
    const islGloss = transData.isl_gloss || aslGloss;
    islGlossOutput.textContent = islGloss;

    // 2. Generate Avatar
    const genRes = await fetch('/generate-avatar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        isl_gloss: islGloss,
        gloss: islGloss,
        // Lets the API retrieve a validated recorded CSLRT sentence pose when
        // the text exactly matches a corpus sentence; otherwise it falls back
        // to word-level playback or the labelled illustrative demo.
        source_sentence: aslGloss
      })
    });
    const genData = await genRes.json();
    if (!genRes.ok) throw new Error(genData.error || 'Avatar generation failed');
    
    const mimeType = genData.mime_type || 'video/mp4';
    const videoData = genData.video_base64 || genData.base64_video;
    const videoUrl = genData.video_url;

    if (genData.mode === 'recorded_sentence_landmark_playback') {
      const matchNote = genData.sentence_match === 'approximate' ? ' (closest safe sentence match)' : '';
      avatarStatus.textContent = `Recorded pose playback: ${genData.source_sentence}${matchNote}`;
    } else if (genData.mode === 'landmark_playback') {
      avatarStatus.textContent = `Recorded word-pose playback: ${genData.source_gloss}`;
    } else if (genData.mode === 'fingerspell') {
      avatarStatus.textContent = `Fingerspelling: no recorded ISL sign found — showing letter-by-letter hand-shapes.`;
    } else {
      avatarStatus.textContent = 'No matching recorded sign was found; showing the illustrative demo.';
    }
    
    let sourceUrl;
    if (videoData) {
      sourceUrl = `data:${mimeType};base64,${videoData}`;
    } else if (videoUrl) {
      sourceUrl = videoUrl;
    } else {
      throw new Error("No video returned from server.");
    }
    await loadAvatarVideo(sourceUrl);

  } catch (err) {
    showAvatarError(`Avatar error: ${err.message}`);
    islGlossOutput.textContent = "Translation failed.";
  } finally {
    translateBtn.disabled = false;
  }
});
