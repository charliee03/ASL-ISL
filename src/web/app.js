const apiStatus = document.getElementById('api-status');
const runtimeApi = document.getElementById('runtime-api');
const modelState = document.getElementById('model-state');
const translationForm = document.getElementById('translation-form');
const glossInput = document.getElementById('asl-gloss-input');
const translationOutput = document.getElementById('translation-output');
const translationMode = document.getElementById('translation-mode');
const translationConfidence = document.getElementById('translation-confidence');
const copyButton = document.getElementById('copy-translation');
const exampleButton = document.getElementById('example-button');
const translateButton = translationForm.querySelector('.translate-button');
const avatarVideo = document.getElementById('avatar-video');
const avatarPlaceholder = document.getElementById('avatar-placeholder');
const avatarLoading = document.getElementById('avatar-loading');
const avatarStatus = document.getElementById('avatar-status');
const avatarPlayer = document.querySelector('.avatar-player');
const replayAvatarButton = document.getElementById('replay-avatar');
let avatarObjectUrl = null;
let translationPipelineInFlight = false;

function formatResponse(payload) { return JSON.stringify(payload, null, 2); }
function setTranslationResult(text, { pending = false, confidence = null } = {}) {
  translationOutput.textContent = text;
  translationOutput.classList.toggle('result-placeholder', pending);
  translationMode.textContent = pending ? 'Working through grammar rules…' : text ? 'Translation complete' : 'Ready for translation';
  translationConfidence.textContent = confidence ? `${Math.round(confidence * 100)}% confidence` : '—';
  copyButton.disabled = pending || !text;
}
async function requestJson(url, formData) {
  const response = await fetch(url, { method: 'POST', body: formData });
  const data = (response.headers.get('content-type') || '').includes('application/json') ? await response.json() : { detail: await response.text() };
  if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  return data;
}
function setAvatarState(state, message) {
  const loading = state === 'loading';
  const ready = state === 'ready';
  avatarLoading.hidden = !loading;
  avatarPlaceholder.hidden = loading || ready;
  avatarVideo.hidden = !ready;
  avatarPlayer.setAttribute('aria-busy', String(loading));
  avatarPlayer.classList.toggle('avatar-player--loading', loading);
  avatarPlayer.classList.toggle('avatar-player--error', state === 'error');
  avatarStatus.textContent = message;
  replayAvatarButton.disabled = !ready;
}
function clearAvatarVideo() {
  avatarVideo.pause();
  avatarVideo.removeAttribute('src');
  avatarVideo.load();
  if (avatarObjectUrl) URL.revokeObjectURL(avatarObjectUrl);
  avatarObjectUrl = null;
}
function avatarVideoSource(payload) {
  const url = payload.video_url || payload.videoUrl || payload.url || payload.file_url;
  if (typeof url === 'string' && url.trim()) return url;
  const encoded = payload.video_base64 || payload.videoBase64 || payload.base64_video || payload.video || payload.data;
  if (typeof encoded !== 'string' || !encoded.trim()) return null;
  if (encoded.startsWith('data:video/')) return encoded;
  const mimeType = payload.mime_type || payload.video_mime_type || 'video/mp4';
  return `data:${mimeType};base64,${encoded}`;
}
async function generateAvatar(islGloss) {
  setAvatarState('loading', 'Generating avatar video…');
  clearAvatarVideo();
  try {
    const response = await fetch('/generate-avatar', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ isl_gloss: islGloss, gloss: islGloss }),
    });
    const data = (response.headers.get('content-type') || '').includes('application/json')
      ? await response.json() : { detail: await response.text() };
    if (!response.ok) throw new Error(data.detail || data.error || `Avatar generation failed (${response.status})`);
    const source = avatarVideoSource(data);
    if (!source) throw new Error('The avatar service returned no playable video.');
    avatarVideo.src = source;
    avatarVideo.load();
    setAvatarState('ready', 'Avatar gesture ready');
    try { await avatarVideo.play(); } catch { /* Browser autoplay policy requires a user replay. */ }
  } catch (error) {
    setAvatarState('error', `Avatar unavailable: ${error.message}`);
  }
}
async function translateAndGenerate(aslGloss) {
  if (translationPipelineInFlight) return;
  translationPipelineInFlight = true;
  setTranslationResult('Translating your gloss sequence…', { pending: true });
  translateButton.disabled = true;
  clearAvatarVideo();
  setAvatarState('idle', 'Waiting for ISL translation');
  try {
    const response = await fetch('/translate', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ asl_gloss: aslGloss }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || data.detail || 'Translation failed');
    const islGloss = data.isl_gloss || '';
    setTranslationResult(islGloss || 'No translated gloss was returned.', { confidence:data.confidence });
    if (islGloss) await generateAvatar(islGloss);
  } catch (error) {
    setTranslationResult(`Unable to translate: ${error.message}`);
    setAvatarState('error', 'Avatar generation needs a successful translation');
  } finally {
    translateButton.disabled = false;
    translationPipelineInFlight = false;
  }
}
async function refreshHealth() {
  try {
    const response = await fetch('/health'); const data = await response.json();
    if (!response.ok) throw new Error();
    apiStatus.classList.add('connected'); apiStatus.innerHTML = '<i></i>Service online';
    runtimeApi.textContent = 'Online'; modelState.textContent = data.model_loaded ? 'Loaded' : 'Not loaded';
  } catch {
    apiStatus.classList.remove('connected'); apiStatus.innerHTML = '<i></i>Service offline';
    runtimeApi.textContent = 'Offline'; modelState.textContent = 'Unavailable';
  }
}
translationForm.addEventListener('submit', async (event) => {
  event.preventDefault(); const aslGloss = glossInput.value.trim(); if (!aslGloss) return;
  await translateAndGenerate(aslGloss);
});
exampleButton.addEventListener('click', () => { glossInput.value = 'HELLO MY NAME IS JOHN'; glossInput.focus(); });
copyButton.addEventListener('click', async () => { try { await navigator.clipboard.writeText(translationOutput.textContent); copyButton.textContent='Copied'; setTimeout(() => { copyButton.textContent='Copy'; }, 1400); } catch { copyButton.textContent='Select text'; } });
document.getElementById('image-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const output=document.getElementById('image-output'); const file=document.getElementById('image-file').files?.[0]; if (!file) return;
  const body=new FormData(); body.append('file',file); output.textContent='Analyzing frame…';
  try { output.textContent=formatResponse(await requestJson(`/${document.getElementById('image-endpoint').value}`,body)); } catch(error) { output.textContent=`Error: ${error.message}`; }
});
document.getElementById('video-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const output=document.getElementById('video-output'); const file=document.getElementById('video-file').files?.[0]; if (!file) return;
  const body=new FormData(); body.append('file',file); output.textContent='Analyzing sequence…';
  try {
    const result = await requestJson('/predict-sequence',body);
    output.textContent=formatResponse(result);
    if (result.gloss) {
      glossInput.value = result.gloss;
      await translateAndGenerate(result.gloss);
      document.getElementById('translate').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  } catch(error) { output.textContent=`Error: ${error.message}`; }
});
replayAvatarButton.addEventListener('click', async () => { avatarVideo.currentTime = 0; try { await avatarVideo.play(); } catch { avatarStatus.textContent = 'Select play to start the avatar video'; } });
window.addEventListener('beforeunload', clearAvatarVideo);
refreshHealth();
