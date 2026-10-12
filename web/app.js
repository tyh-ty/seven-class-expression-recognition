const state = {
  stream: null,
  timer: null,
  running: false,
  busy: false,
  inputMode: 'camera',
  healthReady: false,
  image: null,
  lastResult: null,
  lastCanvas: null,
  lastSource: null,
  cameraSmoothing: null,
  missingFaceFrames: 0,
  cameraSession: 0,
  feedbackBlob: null,
  feedbackResult: null,
  feedbackSession: `natural_${new Date().toISOString().slice(0, 10).replaceAll('-', '')}`,
};

const DISPLAY_CONFIDENCE_THRESHOLD = 0.4;
const FEAR_DISPLAY_CONFIDENCE_THRESHOLD = 0.32;
const CAMERA_CAPTURE_INTERVAL_MS = 100;
const CAMERA_MAX_CAPTURE_SIDE = 512;
const CAMERA_JPEG_QUALITY = 0.85;
const SAME_FACE_IOU_THRESHOLD = 0.3;
const MIN_EMOTION_ALPHA = 0.2;
const STABLE_EMOTION_ALPHA = 0.3;
const CONFIRMED_SWITCH_ALPHA = 0.7;
const STRONG_SWITCH_ALPHA = 0.82;
const MAX_TRACKING_ALPHA = 0.58;
const INSTANT_SWITCH_SCORE = 0.72;
const INSTANT_SWITCH_MARGIN = 0.35;
const STRONG_SWITCH_SCORE = 0.6;
const STRONG_SWITCH_MARGIN = 0.12;
const MEDIUM_SWITCH_SCORE = 0.45;
const MEDIUM_SWITCH_MARGIN = 0.08;
const MIN_SWITCH_SCORE = 0.36;
const MIN_SWITCH_MARGIN = 0.05;
const SWITCH_CONFIRM_FRAMES = 2;
const MEDIUM_SWITCH_CONFIRM_FRAMES = 1;
const STRONG_SWITCH_CONFIRM_FRAMES = 1;
const PERSISTENT_SWITCH_CONFIRM_FRAMES = 4;
const MISSING_FACE_RESET_FRAMES = 4;
const MAX_FACE_HOLD_MS = 400;
const BOX_EMA_ALPHA = 0.58;
const EMOTION_LABELS = {
  anger: '愤怒',
  disgust: '厌恶',
  fear: '恐惧',
  happiness: '开心',
  neutral: '中性',
  sadness: '悲伤',
  surprise: '惊讶',
};
const EMOTION_ORDER = [
  'anger',
  'disgust',
  'fear',
  'happiness',
  'neutral',
  'sadness',
  'surprise',
];
const MODEL_DETAIL = {
  health: '七分类 ViT Transformer',
};

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const imageOverlay = $('imageOverlay');
const cameraEmpty = $('cameraEmpty');
const cameraState = $('cameraState');

function setHealth(text, status) {
  const badge = $('healthBadge');
  badge.textContent = text;
  badge.className = `status-badge ${status}`;
}

function isDisplayableEmotion(face) {
  // This threshold marks confidence; it must not hide valid model predictions.
  const threshold = face.emotion.id === 'fear'
    ? FEAR_DISPLAY_CONFIDENCE_THRESHOLD
    : DISPLAY_CONFIDENCE_THRESHOLD;
  return face.emotion.score >= threshold;
}

function visibleEmotion(face) {
  if (isDisplayableEmotion(face)) return face.emotion;
  // A tentative preview follows the strongest smoothed score instead of an
  // old tracked class whose confidence may have already collapsed.
  return (face.scores ?? []).reduce(
    (best, score) => score.score > best.score ? score : best,
    face.emotion,
  );
}

function nearestFace(faces) {
  const area = (face) => {
    const box = face.detectedBbox ?? face.bbox;
    return box.width * box.height;
  };
  return faces.reduce((nearest, face) => (!nearest || area(face) > area(nearest) ? face : nearest), null);
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('模型服务未就绪');
    const health = await response.json();
    const availableModels = new Set(health.models.map((model) => model.id));
    if (!availableModels.has('transformer')) throw new Error('Transformer 模型未加载');
    state.healthReady = true;
    setHealth(`${MODEL_DETAIL.health} · 已就绪`, 'ready');
  } catch (error) {
    setHealth('模型加载失败', 'error');
    cameraState.textContent = error.message;
  }
}

function drawResults(canvas, source, result) {
  if (!source.videoWidth && !source.naturalWidth) return;
  const bounds = canvas.getBoundingClientRect();
  const displayWidth = Math.max(1, bounds.width);
  const displayHeight = Math.max(1, bounds.height);
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(displayWidth * pixelRatio);
  canvas.height = Math.round(displayHeight * pixelRatio);
  const context = canvas.getContext('2d');
  context.scale(pixelRatio, pixelRatio);
  context.clearRect(0, 0, displayWidth, displayHeight);
  const fitScale = Math.min(displayWidth / result.width, displayHeight / result.height);
  const offsetX = (displayWidth - result.width * fitScale) / 2;
  const offsetY = (displayHeight - result.height * fitScale) / 2;
  // Face detection and expression confidence are independent: an uncertain
  // expression must not hide a face that the detector already located.
  result.faces.forEach((face) => {
    const { x, y, width, height } = face.bbox;
    const drawX = offsetX + x * fitScale;
    const drawY = offsetY + y * fitScale;
    const drawWidth = width * fitScale;
    const drawHeight = height * fitScale;
    const emotion = visibleEmotion(face);
    const isConfident = isDisplayableEmotion(face);
    const isStrong = isConfident && emotion.score >= 0.65;
    const color = isStrong ? '#0f9f70' : '#d97706';
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.strokeRect(drawX, drawY, drawWidth, drawHeight);
    const label = `${isConfident ? '' : '倾向：'}${emotion.label} ${(emotion.score * 100).toFixed(0)}%`;
    context.font = '700 13px Microsoft YaHei, sans-serif';
    const textWidth = context.measureText(label).width;
    const labelTop = Math.max(offsetY, drawY - 27);
    context.fillStyle = color;
    context.fillRect(drawX, labelTop, textWidth + 16, 25);
    context.fillStyle = '#ffffff';
    context.fillText(label, drawX + 8, labelTop + 17);
  });
  state.lastResult = result;
  state.lastCanvas = canvas;
  state.lastSource = source;
}

function showResult(result) {
  const face = nearestFace(result.faces);
  $('faceCount').textContent = `${result.faces.length} 张脸`;
  $('latency').textContent = `${result.latencyMs} ms`;
  if (!face) {
    $('noResult').classList.remove('hidden');
    $('resultContent').classList.add('hidden');
    $('noResult').querySelector('p').textContent = '等待画面中出现人脸';
    $('feedbackChoices').classList.add('hidden');
    return;
  }
  $('noResult').classList.add('hidden');
  $('resultContent').classList.remove('hidden');
  const emotion = visibleEmotion(face);
  const isConfident = isDisplayableEmotion(face);
  const color = isConfident && emotion.score >= 0.65 ? '#0f9f70' : '#d97706';
  $('primaryEmotion').textContent = emotion.label;
  $('primaryScore').textContent = `${(emotion.score * 100).toFixed(1)}%`;
  $('primaryScore').style.color = color;
  $('confidenceFill').style.width = `${emotion.score * 100}%`;
  $('confidenceFill').style.background = color;
  $('predictionStatus').textContent = isConfident
    ? '模型分数表示判断倾向，不代表识别准确率。'
    : isDisplayableEmotion({ emotion })
      ? '正在确认表情变化，当前显示候选结果。'
      : '置信度较低，当前仅显示模型倾向。';
  $('predictionStatus').classList.toggle('uncertain', !isConfident);
  const scoresById = Object.fromEntries(face.scores.map((score) => [score.id, score.score]));
  const sevenScores = EMOTION_ORDER.map((id) => ({
    id,
    label: EMOTION_LABELS[id],
    score: scoresById[id] ?? 0,
  }));
  $('scoreList').innerHTML = sevenScores.map((score) => `
    <div class="score-row ${score.id === emotion.id ? 'top' : ''}">
      <span class="score-name">${score.label}</span>
      <div class="score-track"><span style="width:${score.score * 100}%"></span></div>
      <span class="score-value">${(score.score * 100).toFixed(1)}%</span>
    </div>`).join('');
}

function boxIoU(first, second) {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first.width * first.height + second.width * second.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function resetCameraSmoothing() {
  state.cameraSmoothing = null;
  state.missingFaceFrames = 0;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function topEmotionId(scores) {
  return Object.entries(scores).reduce(
    (best, current) => (current[1] > best[1] ? current : best),
  )[0];
}

function scoreDistance(first, second) {
  return Object.keys(first).reduce(
    (total, id) => total + Math.abs(first[id] - (second[id] ?? first[id])),
    0,
  ) / 2;
}

function smoothBox(previous, current) {
  return Object.fromEntries(['x', 'y', 'width', 'height'].map((key) => [
    key,
    Math.round(BOX_EMA_ALPHA * current[key] + (1 - BOX_EMA_ALPHA) * previous[key]),
  ]));
}

function adaptiveEmotionState(previous, rawScores) {
  const displayedEmotion = previous.displayedEmotion ?? topEmotionId(previous.scores);
  const rawTop = topEmotionId(rawScores);
  const vectorChange = scoreDistance(rawScores, previous.rawScores ?? previous.scores);
  let alpha = clamp(
    MIN_EMOTION_ALPHA + vectorChange * 0.8,
    MIN_EMOTION_ALPHA,
    MAX_TRACKING_ALPHA,
  );
  let pendingEmotion = null;
  let pendingFrames = 0;
  const isLeadingPrediction = rawTop !== displayedEmotion
    && rawScores[rawTop] > (rawScores[displayedEmotion] ?? 0);
  const leadingFrames = isLeadingPrediction
    ? (rawTop === previous.rawLeader ? (previous.leadingFrames ?? 0) + 1 : 1)
    : 0;
  let nextDisplayedEmotion = displayedEmotion;

  if (rawTop === displayedEmotion) {
    alpha = Math.max(alpha, STABLE_EMOTION_ALPHA);
  } else {
    const switchMargin = rawScores[rawTop] - (rawScores[displayedEmotion] ?? 0);
    const instantSwitch = rawScores[rawTop] >= INSTANT_SWITCH_SCORE
      && switchMargin >= INSTANT_SWITCH_MARGIN;
    const strongSwitch = rawScores[rawTop] >= STRONG_SWITCH_SCORE
      && switchMargin >= STRONG_SWITCH_MARGIN;
    const mediumSwitch = rawScores[rawTop] >= MEDIUM_SWITCH_SCORE
      && switchMargin >= MEDIUM_SWITCH_MARGIN;
    const validChallenge = instantSwitch || strongSwitch || mediumSwitch || (
      rawScores[rawTop] >= MIN_SWITCH_SCORE && switchMargin >= MIN_SWITCH_MARGIN
    );
    // Tracking must follow persistent low-confidence changes too. Confidence
    // controls the uncertainty notice and speech, not whether a class can change.
    const persistentChallenge = leadingFrames >= PERSISTENT_SWITCH_CONFIRM_FRAMES;

    if (validChallenge || persistentChallenge) {
      pendingEmotion = rawTop;
      pendingFrames = previous.pendingEmotion === rawTop
        ? previous.pendingFrames + 1
        : 1;
      const requiredFrames = instantSwitch || persistentChallenge
        ? 1
        : strongSwitch
          ? STRONG_SWITCH_CONFIRM_FRAMES
          : mediumSwitch
            ? MEDIUM_SWITCH_CONFIRM_FRAMES
            : SWITCH_CONFIRM_FRAMES;
      if (pendingFrames >= requiredFrames) {
        nextDisplayedEmotion = rawTop;
        pendingEmotion = null;
        pendingFrames = 0;
        alpha = Math.max(
          alpha,
          instantSwitch || strongSwitch
            ? STRONG_SWITCH_ALPHA
            : CONFIRMED_SWITCH_ALPHA,
        );
      } else {
        alpha = Math.min(alpha, STABLE_EMOTION_ALPHA);
      }
    } else {
      alpha = Math.min(alpha, STABLE_EMOTION_ALPHA);
    }
  }

  const scores = Object.fromEntries(Object.keys(rawScores).map((id) => [
    id,
    alpha * rawScores[id] + (1 - alpha) * (previous.scores[id] ?? rawScores[id]),
  ]));
  const total = Object.values(scores).reduce((sum, value) => sum + value, 0);
  Object.keys(scores).forEach((id) => { scores[id] /= total; });
  return {
    scores,
    rawScores,
    rawLeader: rawTop,
    leadingFrames,
    pendingEmotion,
    pendingFrames,
    displayedEmotion: nextDisplayedEmotion,
    alpha,
  };
}

function smoothCameraResult(result) {
  const observedAt = performance.now();
  if (!result.faces.length) {
    state.missingFaceFrames += 1;
    const withinHoldTime = state.cameraSmoothing
      && observedAt - state.cameraSmoothing.lastSeenMs <= MAX_FACE_HOLD_MS;
    if (withinHoldTime && state.missingFaceFrames < MISSING_FACE_RESET_FRAMES) {
      return { ...result, faces: [state.cameraSmoothing.face] };
    }
    if (state.cameraSmoothing || state.missingFaceFrames >= MISSING_FACE_RESET_FRAMES) {
      resetCameraSmoothing();
    }
    return result;
  }

  state.missingFaceFrames = 0;
  const primaryFace = nearestFace(result.faces);
  const otherFaces = result.faces.filter((face) => face !== primaryFace);
  const previous = state.cameraSmoothing;
  const sameFace = previous
    && boxIoU(primaryFace.bbox, previous.bbox) >= SAME_FACE_IOU_THRESHOLD;
  const rawScores = Object.fromEntries(primaryFace.scores.map((item) => [item.id, item.score]));

  if (!sameFace) {
    state.cameraSmoothing = {
      bbox: primaryFace.bbox,
      scores: rawScores,
      rawScores,
      pendingEmotion: null,
      pendingFrames: 0,
      displayedEmotion: topEmotionId(rawScores),
      rawLeader: topEmotionId(rawScores),
      leadingFrames: 0,
      alpha: 1,
      face: primaryFace,
      lastSeenMs: observedAt,
    };
    return { ...result, faces: [primaryFace, ...otherFaces] };
  }

  const temporalState = adaptiveEmotionState(previous, rawScores);
  const blendedScores = primaryFace.scores.map((item) => ({
    ...item,
    score: temporalState.scores[item.id],
  }));
  blendedScores.sort((first, second) => second.score - first.score);
  const displayedScore = blendedScores.find(
    (item) => item.id === temporalState.displayedEmotion,
  ) ?? blendedScores[0];
  const displayOrderedScores = [
    displayedScore,
    ...blendedScores.filter((item) => item.id !== displayedScore.id),
  ];
  const smoothedBox = smoothBox(previous.bbox, primaryFace.bbox);

  const smoothedFace = {
    ...primaryFace,
    detectedBbox: primaryFace.bbox,
    bbox: smoothedBox,
    emotion: displayedScore,
    scores: displayOrderedScores,
  };
  state.cameraSmoothing = {
    bbox: smoothedBox,
    ...temporalState,
    face: smoothedFace,
    lastSeenMs: observedAt,
  };
  return { ...result, faces: [smoothedFace, ...otherFaces] };
}

async function analyzeBlob(blob) {
  if (!blob || state.busy) return;
  state.busy = true;
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'image/jpeg' },
      body: blob,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '识别失败');
    return result;
  } catch (error) {
    cameraState.textContent = error.message;
  } finally {
    state.busy = false;
  }
}

async function submitFeedback(label) {
  const face = nearestFace(state.feedbackResult?.faces ?? []);
  if (!state.feedbackBlob || !face) return;
  const emotion = visibleEmotion(face);
  const buttons = [...document.querySelectorAll('[data-feedback-label]')];
  buttons.forEach((button) => { button.disabled = true; });
  $('feedbackStatus').textContent = '正在保存…';
  try {
    const query = new URLSearchParams({
      label,
      session: state.feedbackSession,
      predicted: emotion.id,
      score: String(emotion.score),
      model: 'transformer',
    });
    const response = await fetch(`/api/feedback?${query}`, {
      method: 'POST',
      headers: { 'Content-Type': state.feedbackBlob.type || 'image/jpeg' },
      body: state.feedbackBlob,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '保存失败');
    $('feedbackStatus').textContent = `已记录：${EMOTION_LABELS[label]}（本次 ${payload.count} 张）`;
    $('feedbackChoices').classList.add('hidden');
    state.feedbackBlob = null;
    state.feedbackResult = null;
  } catch (error) {
    $('feedbackStatus').textContent = error.message;
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

function captureVideo() {
  if (!state.running || state.inputMode !== 'camera' || video.readyState < 2 || state.busy) return;
  const session = state.cameraSession;
  const isCurrentSession = () => state.running && state.inputMode === 'camera'
    && state.cameraSession === session;
  const canvas = document.createElement('canvas');
  const scale = Math.min(
    1,
    CAMERA_MAX_CAPTURE_SIDE / Math.max(video.videoWidth, video.videoHeight),
  );
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (!isCurrentSession()) return;
    analyzeBlob(blob).then((result) => {
      if (!isCurrentSession()) return;
      if (!result) {
        speech.reset();
        return;
      }
      const smoothedResult = smoothCameraResult(result);
      state.feedbackBlob = blob;
      state.feedbackResult = result.faces.length ? smoothedResult : result;
      showResult(smoothedResult);
      drawResults(overlay, video, smoothedResult);
      // Never announce a held face from a missed detection.
      speech.update(result.faces.length ? smoothedResult : result);
    });
  }, 'image/jpeg', CAMERA_JPEG_QUALITY);
}

function scheduleCapture() {
  clearTimeout(state.timer);
  if (state.running) {
    state.timer = setTimeout(() => {
      captureVideo();
      scheduleCapture();
    }, CAMERA_CAPTURE_INTERVAL_MS);
  }
}

async function startCamera() {
  const session = ++state.cameraSession;
  $('startCamera').disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    if (session !== state.cameraSession || state.inputMode !== 'camera') {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    video.srcObject = state.stream;
    await video.play();
    if (session !== state.cameraSession) return;
    resetCameraSmoothing();
    speech.reset();
    state.running = true;
    cameraEmpty.classList.add('hidden');
    cameraState.textContent = '实时识别中';
    $('startCamera').disabled = true;
    $('stopCamera').disabled = false;
    scheduleCapture();
  } catch (error) {
    if (session !== state.cameraSession) return;
    stopCamera();
    cameraState.textContent = `无法访问摄像头：${error.message}`;
  }
}

function stopCamera() {
  state.cameraSession += 1;
  state.running = false;
  speech.reset();
  clearTimeout(state.timer);
  resetCameraSmoothing();
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.srcObject = null;
  state.feedbackBlob = null;
  state.feedbackResult = null;
  $('feedbackChoices').classList.add('hidden');
  $('feedbackStatus').textContent = '';
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
  if (state.lastCanvas === overlay) {
    state.lastResult = null;
    state.lastCanvas = null;
    state.lastSource = null;
  }
  cameraEmpty.classList.remove('hidden');
  cameraState.textContent = '摄像头未开启';
  $('startCamera').disabled = false;
  $('stopCamera').disabled = true;
}

function setMode(mode) {
  state.inputMode = mode;
  $('cameraTab').classList.toggle('active', mode === 'camera');
  $('imageTab').classList.toggle('active', mode === 'image');
  $('cameraSource').classList.toggle('active', mode === 'camera');
  $('imageSource').classList.toggle('active', mode === 'image');
  if (mode === 'image') stopCamera();
}

function loadImage(file) {
  if (!file?.type.startsWith('image/')) return;
  state.image = file;
  const url = URL.createObjectURL(file);
  const image = $('imagePreview');
  image.onload = async () => {
    const result = await analyzeBlob(file);
    if (result) {
      state.feedbackBlob = file;
      state.feedbackResult = result;
      $('feedbackStatus').textContent = '';
      showResult(result);
      drawResults(imageOverlay, image, result);
    }
    URL.revokeObjectURL(url);
  };
  image.src = url;
  $('dropZone').classList.add('hidden');
  $('imageResult').classList.remove('hidden');
}

const speech = new window.EmotionSpeechController({
  onStatus: (message) => { $('speechStatus').textContent = message; },
});
$('speechToggle').disabled = !speech.available;
$('speechToggle').addEventListener('change', (event) => {
  speech.setEnabled(event.target.checked);
  if (speech.enabled && !state.running) {
    $('speechStatus').textContent = '播报已开启，打开摄像头后开始';
  }
});

$('cameraTab').addEventListener('click', () => setMode('camera'));
$('imageTab').addEventListener('click', () => setMode('image'));
$('startCamera').addEventListener('click', startCamera);
$('stopCamera').addEventListener('click', stopCamera);
$('feedbackToggle').addEventListener('click', () => {
  if (!state.feedbackBlob || !state.feedbackResult?.faces?.length) return;
  $('feedbackChoices').classList.toggle('hidden');
});
document.querySelectorAll('[data-feedback-label]').forEach((button) => {
  button.addEventListener('click', () => submitFeedback(button.dataset.feedbackLabel));
});
$('chooseImage').addEventListener('click', () => $('imageInput').click());
$('imageInput').addEventListener('change', (event) => loadImage(event.target.files[0]));
$('clearImage').addEventListener('click', () => {
  state.image = null;
  state.feedbackBlob = null;
  state.feedbackResult = null;
  $('feedbackChoices').classList.add('hidden');
  $('feedbackStatus').textContent = '';
  $('imagePreview').removeAttribute('src');
  $('dropZone').classList.remove('hidden');
  $('imageResult').classList.add('hidden');
  imageOverlay.getContext('2d').clearRect(0, 0, imageOverlay.width, imageOverlay.height);
  state.lastResult = null;
  state.lastCanvas = null;
  state.lastSource = null;
  showResult({ faces: [], latencyMs: 0 });
});
$('dropZone').addEventListener('dragover', (event) => { event.preventDefault(); $('dropZone').classList.add('dragging'); });
$('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('dragging'));
$('dropZone').addEventListener('drop', (event) => {
  event.preventDefault();
  $('dropZone').classList.remove('dragging');
  loadImage(event.dataTransfer.files[0]);
});
window.addEventListener('beforeunload', stopCamera);
window.addEventListener('resize', () => {
  if (state.lastResult && state.lastCanvas && state.lastSource) {
    drawResults(state.lastCanvas, state.lastSource, state.lastResult);
  }
});
checkHealth();
