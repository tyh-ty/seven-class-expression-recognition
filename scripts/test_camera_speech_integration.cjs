const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const webSource = (file) => fs.readFileSync(path.join(__dirname, '../web', file), 'utf8');
const labels = {
  anger: '愤怒', disgust: '厌恶', fear: '恐惧', happiness: '开心',
  neutral: '中性', sadness: '悲伤', surprise: '惊讶',
};
function face(id = 'anger', score = 0.8, bbox = {}) {
  return {
    bbox: { x: 20, y: 20, width: 150, height: 150, ...bbox },
    emotion: { id, label: labels[id], score },
    scores: Object.entries(labels).map(([key, label]) => ({
      id: key, label, score: key === id ? score : (1 - score) / 6,
    })),
    faceConfidence: 0.95,
  };
}
const result = (faces) => ({ width: 512, height: 384, faces, latencyMs: 30 });
const settle = () => new Promise((resolve) => setImmediate(resolve));

async function createBrowser() {
  let now = 0;
  let timerId = 0;
  const nodes = new Map();
  const encodings = [];
  const requests = [];
  const timers = new Map();
  const spoken = [];
  const tracks = [];
  const drawing = {
    scale() {}, clearRect() {}, strokeRect() {}, fillRect() {}, fillText() {}, drawImage() {},
    measureText(text) { return { width: text.length * 12 }; },
  };

  function element(id = '') {
    const classes = new Set();
    const events = new Map();
    const children = new Map();
    return {
      id, textContent: '', innerHTML: '', style: {}, dataset: {}, checked: false, disabled: false,
      width: 512, height: 384, videoWidth: 640, videoHeight: 480, readyState: 4,
      classList: {
        add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
        contains(name) { return classes.has(name); },
        toggle(name, force) {
          const value = force === undefined ? !classes.has(name) : force;
          if (value) classes.add(name); else classes.delete(name);
          return value;
        },
      },
      querySelector(selector) {
        if (!children.has(selector)) children.set(selector, element(selector));
        return children.get(selector);
      },
      addEventListener(name, callback) {
        if (!events.has(name)) events.set(name, []);
        events.get(name).push(callback);
      },
      async dispatch(name, detail = {}) {
        for (const callback of events.get(name) || []) {
          await callback({ target: this, preventDefault() {}, ...detail });
        }
      },
      click() { return this.dispatch('click'); },
      removeAttribute(name) { delete this[name]; },
      getContext() { return drawing; },
      getBoundingClientRect() { return { width: 512, height: 384 }; },
      toBlob(callback) { encodings.push(() => callback({ type: 'image/jpeg' })); },
      play: async () => {},
    };
  }
  const get = (id) => {
    if (!nodes.has(id)) nodes.set(id, element(id));
    return nodes.get(id);
  };
  const synth = {
    cancellations: 0,
    getVoices: () => [],
    addEventListener() {}, removeEventListener() {},
    speak(utterance) { spoken.push(utterance); },
    cancel() {
      this.cancellations += 1;
      spoken.at(-1)?.onerror?.({ error: 'interrupted' });
    },
  };
  const context = vm.createContext({
    document: {
      getElementById: get, createElement: element, querySelectorAll: () => [],
    },
    window: {
      speechSynthesis: synth,
      SpeechSynthesisUtterance: class { constructor(text) { this.text = text; } },
      addEventListener() {}, devicePixelRatio: 1,
    },
    navigator: { mediaDevices: { getUserMedia: async () => {
      const track = { stopped: false, stop() { this.stopped = true; } };
      tracks.push(track);
      return { getTracks: () => [track] };
    } } },
    fetch(url) {
      if (url === '/api/health') {
        return Promise.resolve({ ok: true, json: async () => ({ models: [{ id: 'transformer' }] }) });
      }
      assert.equal(url, '/api/analyze', 'unexpected request');
      return new Promise((resolve) => requests.push(resolve));
    },
    performance: { now: () => now },
    setTimeout(callback) { timers.set(++timerId, callback); return timerId; },
    clearTimeout(id) { timers.delete(id); },
    URL, URLSearchParams, console,
  });
  vm.runInContext(webSource('emotion-speech.js'), context, { filename: 'emotion-speech.js' });
  vm.runInContext(webSource('app.js'), context, { filename: 'app.js' });
  await settle();
  const evaluate = (code) => vm.runInContext(code, context);
  return {
    get, spoken, synth, tracks, evaluate, requests, encodings,
    async toggle(enabled) {
      get('speechToggle').checked = enabled;
      await get('speechToggle').dispatch('change');
    },
    async start() { await get('startCamera').click(); },
    begin(time) { now = time; evaluate('captureVideo()'); },
    encode() {
      assert.ok(encodings.length, 'a capture should have scheduled JPEG encoding');
      encodings.shift()();
    },
    async answer(time, faces = [face()]) {
      now = time;
      assert.ok(requests.length, 'an analysis request should be pending');
      requests.shift()({ ok: true, json: async () => result(faces) });
      await settle();
    },
    async frame(time, faces = [face()]) {
      this.begin(time); this.encode(); await this.answer(time, faces);
    },
  };
}

(async () => {
  {
    const h = await createBrowser();
    assert.equal(h.get('speechToggle').disabled, false);
    await h.start();
    await h.frame(0); await h.frame(700);
    assert.equal(h.spoken.length, 0, 'camera does not speak while the UI switch is off');
    await h.toggle(true);
    await h.frame(1000); await h.frame(1600);
    assert.equal(h.spoken[0].text, '当前情绪为愤怒', 'the actual checkbox event enables speech');
    await h.toggle(false);
    assert.equal(h.synth.cancellations, 1, 'the actual checkbox event cancels speech');
    await h.frame(4200); await h.frame(4800);
    assert.equal(h.spoken.length, 1);
    await h.get('stopCamera').click();
  }

  for (const action of ['stop', 'image', 'restart']) {
    const h = await createBrowser();
    await h.toggle(true); await h.start();
    h.begin(0); h.encode();
    assert.equal(h.requests.length, 1);
    if (action === 'image') await h.get('imageTab').click();
    else await h.get('stopCamera').click();
    if (action === 'restart') await h.start();
    await h.answer(1000);
    assert.equal(h.spoken.length, 0, `${action}: an old response cannot speak`);
    assert.equal(h.get('primaryEmotion').textContent, '', `${action}: an old response cannot render`);
    assert.equal(h.tracks[0].stopped, true, `${action}: the old camera track is stopped`);
    if (action === 'restart') {
      await h.frame(1600);
      assert.equal(h.spoken.length, 0, 'the old session must not seed speech stabilization');
      await h.frame(2200);
      assert.equal(h.spoken[0].text, '当前情绪为愤怒', 'the fresh camera session still works');
      await h.get('stopCamera').click();
    }
  }

  {
    const h = await createBrowser();
    await h.toggle(true); await h.start();
    h.begin(0);
    await h.get('stopCamera').click();
    h.encode(); await settle();
    assert.equal(h.requests.length, 0, 'an encoding completed after stop cannot submit a stale request');
    assert.equal(h.spoken.length, 0);
  }

  {
    const h = await createBrowser();
    await h.toggle(true); await h.start();
    await h.frame(0); await h.frame(500);
    await h.frame(650, []);
    assert.equal(h.evaluate('state.lastResult.faces.length'), 1, 'short missed detections still hold the visual box');
    assert.equal(h.spoken.length, 0, 'a held visual box must not complete speech stabilization');
    await h.frame(700); await h.frame(1200);
    assert.equal(h.spoken.length, 0, 'reappearing faces must stabilize from a fresh real observation');
    await h.frame(1300);
    assert.equal(h.spoken[0].text, '当前情绪为愤怒');
    await h.frame(1400, []);
    assert.equal(h.synth.cancellations, 1, 'a miss cancels ongoing speech even if the overlay retains the face');
    await h.get('stopCamera').click();
  }

  {
    const h = await createBrowser();
    await h.toggle(true); await h.start();
    const people = [
      face('anger', 0.9, { x: 250, width: 50, height: 50 }),
      face('fear', 0.25),
    ];
    await h.frame(0, people); await h.frame(700, people);
    assert.equal(h.get('resultContent').classList.contains('hidden'), false);
    assert.equal(h.get('noResult').classList.contains('hidden'), true);
    assert.match(h.get('predictionStatus').textContent, /置信度较低/);
    assert.equal(h.get('primaryEmotion').textContent, '恐惧', 'show the closest face even below threshold, never the distant confident face');
    assert.equal(h.get('primaryScore').textContent, '25.0%');
    assert.equal((h.get('scoreList').innerHTML.match(/class="score-row/g) || []).length, 7);
    assert.equal(h.spoken.length, 0, 'a visible low-confidence candidate must not be announced as certain');
    await h.get('stopCamera').click();
  }

  for (const [id, label] of Object.entries(labels)) {
    const h = await createBrowser();
    await h.toggle(true); await h.start();
    await h.frame(0, [face(id, 0.25)]); await h.frame(800, [face(id, 0.25)]);
    assert.equal(h.get('resultContent').classList.contains('hidden'), false, `${id}: a valid weak prediction is visible`);
    assert.equal(h.get('primaryEmotion').textContent, label);
    assert.equal(h.get('primaryScore').textContent, '25.0%');
    assert.equal(h.spoken.length, 0);
    await h.frame(1300, []);
    assert.equal(h.get('resultContent').classList.contains('hidden'), true, 'no face still clears the result');
    await h.get('stopCamera').click();
  }

  console.log('摄像头语音集成检查通过：真实开关绑定、异步请求隔离、停止/切图/重开、漏检保留框、最近人低分。');
})().catch((error) => { console.error(error); process.exitCode = 1; });
