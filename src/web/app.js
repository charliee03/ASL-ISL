const apiStatus = document.getElementById('api-status');
const modelState = document.getElementById('model-state');
const imageForm = document.getElementById('image-form');
const videoForm = document.getElementById('video-form');
const imageOutput = document.getElementById('image-output');
const videoOutput = document.getElementById('video-output');
const imageEndpoint = document.getElementById('image-endpoint');
const imageFile = document.getElementById('image-file');
const videoFile = document.getElementById('video-file');

function formatResponse(payload) {
  return JSON.stringify(payload, null, 2);
}

async function requestJson(url, formData) {
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : { detail: await response.text() };

  if (!response.ok) {
    throw new Error(data.detail || data.error || `Request failed with status ${response.status}`);
  }

  return data;
}

async function refreshHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    apiStatus.textContent = `API: ${data.status}`;
    apiStatus.classList.remove('status-pill--live');
    if (data.status === 'ok') {
      apiStatus.classList.add('status-pill--live');
    }
    modelState.textContent = data.model_loaded ? 'Loaded' : 'Not loaded';
  } catch (error) {
    apiStatus.textContent = 'API: offline';
    apiStatus.classList.remove('status-pill--live');
    modelState.textContent = 'Unavailable';
  }
}

imageForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = imageFile.files?.[0];
  if (!file) {
    imageOutput.textContent = 'Choose an image first.';
    return;
  }

  const endpoint = imageEndpoint.value;
  const formData = new FormData();
  formData.append('file', file);
  imageOutput.textContent = 'Running analysis...';

  try {
    const data = await requestJson(`/${endpoint}`, formData);
    imageOutput.textContent = formatResponse(data);
  } catch (error) {
    imageOutput.textContent = `Error: ${error.message}`;
  }
});

videoForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = videoFile.files?.[0];
  if (!file) {
    videoOutput.textContent = 'Choose a video first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  videoOutput.textContent = 'Running sequence analysis...';

  try {
    const data = await requestJson('/predict-sequence', formData);
    videoOutput.textContent = formatResponse(data);
  } catch (error) {
    videoOutput.textContent = `Error: ${error.message}`;
  }
});

refreshHealth();