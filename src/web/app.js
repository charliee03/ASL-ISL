const uploadZone = document.getElementById('upload-zone');
const videoUpload = document.getElementById('asl-video-upload');
const aslPreview = document.getElementById('asl-preview');
const aslGlossInput = document.getElementById('asl-gloss-input');
const translateBtn = document.getElementById('translate-btn');

const avatarStage = document.getElementById('avatar-stage');
const avatarPlaceholder = document.getElementById('avatar-placeholder');
const avatarLoading = document.getElementById('avatar-loading');
const islVideo = document.getElementById('isl-video');
const islGlossOutput = document.getElementById('isl-gloss-output');

let currentTranslationRequest = null;

// UI STATE MANAGEMENT
function updateTranslateButton() {
  translateBtn.disabled = aslGlossInput.value.trim().length === 0;
}

aslGlossInput.addEventListener('input', updateTranslateButton);

// VIDEO UPLOAD HANDLING
uploadZone.addEventListener('click', () => videoUpload.click());

videoUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Show preview
  const url = URL.createObjectURL(file);
  uploadZone.hidden = true;
  aslPreview.src = url;
  aslPreview.hidden = false;
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
    if (!response.ok) throw new Error(data.detail || 'Failed to predict ASL');
    
    aslGlossInput.value = data.gloss || "HELLO";
  } catch (err) {
    aslGlossInput.value = "";
    alert("Error recognizing ASL video: " + err.message);
  } finally {
    aslGlossInput.disabled = false;
    updateTranslateButton();
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
      body: JSON.stringify({ isl_gloss: islGloss, gloss: islGloss })
    });
    const genData = await genRes.json();
    if (!genRes.ok) throw new Error(genData.error || 'Avatar generation failed');
    
    const videoData = genData.video_base64 || genData.base64_video;
    const mimeType = genData.mime_type || 'video/mp4';
    
    if (videoData) {
      islVideo.src = `data:${mimeType};base64,${videoData}`;
      avatarLoading.hidden = true;
      islVideo.hidden = false;
      islVideo.load();
      try { await islVideo.play(); } catch(e) {}
    } else {
      throw new Error("No video returned from server.");
    }

  } catch (err) {
    avatarLoading.hidden = true;
    avatarPlaceholder.hidden = false;
    avatarPlaceholder.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
    islGlossOutput.textContent = "Translation failed.";
  } finally {
    translateBtn.disabled = false;
  }
});
