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
  setTranslationResult('Translating your gloss sequence…', { pending: true });
  try {
    const response = await fetch('/translate', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ asl_gloss: aslGloss }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || data.detail || 'Translation failed');
    setTranslationResult(data.isl_gloss || 'No translated gloss was returned.', { confidence:data.confidence });
  } catch (error) { setTranslationResult(`Unable to translate: ${error.message}`); }
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
  try { output.textContent=formatResponse(await requestJson('/predict-sequence',body)); } catch(error) { output.textContent=`Error: ${error.message}`; }
});
refreshHealth();
