const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const strokes = [];
const labels = [];
const drawing = {
  scale() {},
  clearRect() { strokes.length = 0; labels.length = 0; },
  strokeRect(...box) { strokes.push({ box, color: this.strokeStyle }); },
  fillRect() {},
  fillText(text) { labels.push(text); },
  measureText(text) { return { width: text.length * 13 }; },
};
const canvas = {
  getContext: () => drawing,
  getBoundingClientRect: () => ({ width: 1024, height: 640 }),
};
const context = vm.createContext({
  document: { getElementById: () => ({}) },
  window: { devicePixelRatio: 2 },
  canvas,
});
vm.runInContext(source.slice(0, source.indexOf('async function analyzeBlob')), context);
const evaluate = (code) => vm.runInContext(code, context);

// Reproduce a strong face detection with an uncertain expression, as in the
// reported webcam frame. Both input modes must preserve the detector's box.
for (const media of [{ videoWidth: 1280 }, { naturalWidth: 512 }]) {
  context.media = media;
  context.result = {
    width: 512, height: 288,
    faces: [{
      bbox: { x: 221, y: 45, width: 142, height: 189 },
      faceConfidence: 0.918,
      emotion: { id: 'neutral', label: '中性', score: 0.238 },
    }],
  };
  evaluate('drawResults(canvas, media, result)');
  assert.equal(strokes.length, 1, 'detected faces must remain visible below the emotion threshold');
  assert.deepEqual(strokes[0].box, [442, 122, 284, 378], 'box follows the image scale and letterbox offset');
  assert.deepEqual(labels, ['倾向：中性 24%']);
  assert.equal(canvas.width, 2048);
  assert.equal(canvas.height, 1280);

  // Every detected face gets a box, independently of its expression score.
  context.result.faces.push({
    bbox: { x: 10, y: 10, width: 50, height: 60 },
    faceConfidence: 0.95,
    emotion: { id: 'happiness', label: '开心', score: 0.8 },
  });
  evaluate('drawResults(canvas, media, result)');
  assert.equal(strokes.length, 2);
  assert.deepEqual(labels, ['倾向：中性 24%', '开心 80%']);

  for (const [id, label, score, expected] of [
    ['neutral', '中性', 0.399, '倾向：中性 40%'],
    ['neutral', '中性', 0.4, '中性 40%'],
    ['fear', '恐惧', 0.319, '倾向：恐惧 32%'],
    ['fear', '恐惧', 0.32, '恐惧 32%'],
  ]) {
    context.result.faces[0].emotion = { id, label, score };
    evaluate('drawResults(canvas, media, result)');
    assert.equal(strokes.length, 2, 'crossing an emotion threshold must never drop a face box');
    assert.equal(labels[0], expected);
  }

  context.result.faces[0].emotion = { id: 'neutral', label: '中性', score: 0.1 };
  context.result.faces[0].scores = [
    { id: 'neutral', label: '中性', score: 0.1 },
    { id: 'anger', label: '愤怒', score: 0.28 },
  ];
  evaluate('drawResults(canvas, media, result)');
  assert.equal(labels[0], '倾向：愤怒 28%', 'an uncertain preview must not retain the old lower-scoring class');

  context.result.faces = [];
  evaluate('drawResults(canvas, media, result)');
  assert.equal(strokes.length, 0, 'no detection clears the old boxes');
  assert.equal(labels.length, 0);
}

console.log('人脸画框检查通过：低表情置信度、多脸、阈值切换、摄像头/图片坐标与空画面。');
