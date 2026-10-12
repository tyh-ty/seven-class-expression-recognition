const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
let now = 0;
const context = vm.createContext({
  document: { getElementById: () => ({}) },
  performance: { now: () => now },
});
vm.runInContext(source.slice(0, source.indexOf('async function analyzeBlob')), context);
const evaluate = (code) => vm.runInContext(code, context);

assert.equal(evaluate("isDisplayableEmotion({ emotion: { id: 'fear', score: 0.32 } })"), true);
assert.equal(evaluate("isDisplayableEmotion({ emotion: { id: 'fear', score: 0.319 } })"), false);
assert.equal(evaluate("isDisplayableEmotion({ emotion: { id: 'neutral', score: 0.39 } })"), false);
assert.equal(evaluate("isDisplayableEmotion({ emotion: { id: 'neutral', score: 0.4 } })"), true);

evaluate(`
  const face = {
    bbox: { x: 10, y: 10, width: 100, height: 100 },
    emotion: { id: 'neutral', label: '中性', score: 0.7 },
    scores: EMOTION_ORDER.map((id) => ({ id, label: EMOTION_LABELS[id], score: id === 'neutral' ? 0.7 : 0.05 })),
    faceConfidence: 0.8,
  };
  const detected = { width: 512, height: 288, faces: [face], latencyMs: 20 };
  const missing = { ...detected, faces: [] };
`);

assert.equal(evaluate('smoothCameraResult(missing).faces.length'), 0);

evaluate('resetCameraSmoothing(); smoothCameraResult(detected);');
now += 401;
assert.equal(evaluate('smoothCameraResult(missing).faces.length'), 0);
assert.equal(evaluate('state.cameraSmoothing'), null);
evaluate('resetCameraSmoothing(); smoothCameraResult(detected);');
for (let frame = 0; frame < 3; frame += 1) {
  assert.equal(evaluate('smoothCameraResult(missing).faces[0].emotion.id'), 'neutral');
}
assert.equal(evaluate('smoothCameraResult(missing).faces.length'), 0);
assert.equal(evaluate('state.cameraSmoothing'), null);

evaluate('smoothCameraResult(detected); smoothCameraResult(missing);');
assert.equal(evaluate('smoothCameraResult(detected).faces.length'), 1);
assert.equal(evaluate('state.missingFaceFrames'), 0);
assert.equal(evaluate('smoothCameraResult(missing).faces[0].emotion.id'), 'neutral');
evaluate('resetCameraSmoothing();');
assert.equal(evaluate('smoothCameraResult(missing).faces.length'), 0);

const neutralScores = {
  anger: 0.03, disgust: 0.03, fear: 0.03, happiness: 0.03,
  neutral: 0.8, sadness: 0.05, surprise: 0.03,
};
const weakFearScores = {
  anger: 0.06, disgust: 0.04, fear: 0.34, happiness: 0.03,
  neutral: 0.26, sadness: 0.14, surprise: 0.13,
};
const closeAngerScores = {
  anger: 0.46, disgust: 0.02, fear: 0.02, happiness: 0.02,
  neutral: 0.42, sadness: 0.04, surprise: 0.02,
};
const belowThresholdFearScores = { ...weakFearScores, fear: 0.31, sadness: 0.17 };

function runEmotionSequence(frames) {
  return evaluate(`(() => {
    let previous = {
      scores: ${JSON.stringify(neutralScores)},
      rawScores: ${JSON.stringify(neutralScores)},
      displayedEmotion: 'neutral',
      rawLeader: 'neutral',
      leadingFrames: 1,
      pendingFrames: 0,
    };
    return ${JSON.stringify(frames)}.map((scores) => {
      previous = adaptiveEmotionState(previous, scores);
      return previous;
    });
  })()`);
}

// Small but persistent leads must replace an old label without increasing scores.
for (const [emotion, scores] of [['fear', weakFearScores], ['anger', closeAngerScores]]) {
  const frames = runEmotionSequence(Array(12).fill(scores));
  frames.slice(0, 3).forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));
  assert.equal(frames[3].displayedEmotion, emotion);
  frames.slice(3).forEach((frame) => {
    assert.equal(frame.displayedEmotion, emotion);
    assert.ok(frame.scores[emotion] <= scores[emotion] + 1e-12);
    assert.ok(Math.abs(Object.values(frame.scores).reduce((sum, score) => sum + score, 0) - 1) < 1e-12);
  });
  const last = frames.at(-1);
  assert.equal(evaluate(`isDisplayableEmotion({ emotion: ${JSON.stringify({ id: emotion, score: last.scores[emotion] })} })`), true);
}

// A changed leader breaks a streak, but low confidence must not lock the tracker.
const interrupted = runEmotionSequence([
  weakFearScores, weakFearScores, weakFearScores, neutralScores,
  weakFearScores, weakFearScores, weakFearScores, weakFearScores,
]);
interrupted.slice(0, 7).forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));
assert.equal(interrupted[7].displayedEmotion, 'fear');
const belowThenValid = runEmotionSequence([
  belowThresholdFearScores, belowThresholdFearScores, belowThresholdFearScores,
  weakFearScores, weakFearScores, weakFearScores, weakFearScores,
]);
belowThenValid.slice(0, 3).forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));
belowThenValid.slice(3).forEach((frame) => assert.equal(frame.displayedEmotion, 'fear'));
runEmotionSequence(Array.from({ length: 20 }, (_, index) => index % 2 ? closeAngerScores : weakFearScores))
  .forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));
const tentativeFear = runEmotionSequence(Array(20).fill(belowThresholdFearScores));
tentativeFear.slice(0, 3).forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));
tentativeFear.slice(3).forEach((frame) => {
  assert.equal(frame.displayedEmotion, 'fear');
  assert.ok(frame.scores.fear <= 0.31 + 1e-12, 'tracking a weak prediction must never inflate its confidence');
});

const lowNeutral = evaluate(`(() => {
  let previous = {
    scores: { anger: .8, disgust: .03, fear: .03, happiness: .03, neutral: .03, sadness: .05, surprise: .03 },
    displayedEmotion: 'anger', pendingFrames: 0,
  };
  const scores = { anger: .13, disgust: .13, fear: .10, happiness: .10, neutral: .28, sadness: .13, surprise: .13 };
  for (let i = 0; i < 20; i++) previous = adaptiveEmotionState(previous, scores);
  return previous;
})()`);
assert.equal(lowNeutral.displayedEmotion, 'neutral', 'neutral can return even when no class reaches 40%');
assert.ok(lowNeutral.scores.neutral <= .28 + 1e-12);

// A tied top score provides no evidence for replacing the current label.
const tiedScores = {
  anger: 0.45, disgust: 0.02, fear: 0.02, happiness: 0.02,
  neutral: 0.45, sadness: 0.02, surprise: 0.02,
};
runEmotionSequence(Array(20).fill(tiedScores))
  .forEach((frame) => assert.equal(frame.displayedEmotion, 'neutral'));

// Clear changes retain the existing immediate response.
const strongFearScores = {
  anger: 0.04, disgust: 0.04, fear: 0.6, happiness: 0.04,
  neutral: 0.12, sadness: 0.08, surprise: 0.08,
};
assert.equal(runEmotionSequence([strongFearScores])[0].displayedEmotion, 'fear');

// Selection depends on face area, including when response order is reversed.
evaluate(`
  const farFace = {
    ...face,
    bbox: { x: 350, y: 10, width: 50, height: 50 },
    emotion: { id: 'fear', label: '恐惧', score: 0.94 },
    scores: EMOTION_ORDER.map((id) => ({ id, label: EMOTION_LABELS[id], score: id === 'fear' ? 0.94 : 0.01 })),
  };
  const unorderedFaces = [farFace, face];
  resetCameraSmoothing();
`);
assert.equal(evaluate('nearestFace([])'), null);
assert.equal(evaluate('nearestFace(unorderedFaces) === face'), true);
assert.equal(evaluate('smoothCameraResult({ ...detected, faces: unorderedFaces }).faces[0].emotion.id'), 'neutral');
assert.equal(evaluate('unorderedFaces[0] === farFace'), true);
assert.equal(evaluate('smoothCameraResult({ ...detected, faces: unorderedFaces }).faces[0].emotion.id'), 'neutral');
assert.equal(evaluate('smoothCameraResult({ ...detected, faces: [farFace] }).faces[0].emotion.id'), 'fear');
assert.equal(evaluate('state.cameraSmoothing.pendingFrames'), 0);
assert.equal(evaluate('state.cameraSmoothing.alpha'), 1);

console.log('摄像头漏检、持续表情切换、置信度保持及最近人脸检查通过。');
