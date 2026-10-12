const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const input = process.argv[2];
const output = process.argv[3];
if (!input || !output) throw new Error('Usage: node evaluate_video_smoothing.cjs input.json output.json');
const report = JSON.parse(fs.readFileSync(input, 'utf8'));
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
let now = 0;
const context = vm.createContext({ document: { getElementById: () => ({}) }, performance: { now: () => now } });
vm.runInContext(source.slice(0, source.indexOf('async function analyzeBlob')), context);
const order = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'];
const clips = [];
for (const video of report.videos) {
  vm.runInContext('resetCameraSmoothing()', context);
  let streak = 0;
  let maxStreak = 0;
  let streakStartedAt = null;
  let maxCorrectDurationSeconds = 0;
  let previous = null;
  let switches = 0;
  let correct = 0;
  let inWindow = 0;
  let firstStableSeconds = null;
  const frames = [];
  for (const frame of video.frames) {
    now = frame.seconds * 1000;
    let faces = [];
    if (frame.probabilities) {
      const scores = order.map((id, i) => ({ id, label: id, score: frame.probabilities[i] }));
      const best = scores.reduce((a, b) => (b.score > a.score ? b : a));
      faces = [{ bbox: frame.bbox, detectedBbox: frame.bbox, emotion: best, scores, faceConfidence: frame.face_score }];
    }
    context.frameInput = { width: frame.width, height: frame.height, faces, latencyMs: 0 };
    const result = vm.runInContext(`(() => {
      const result = smoothCameraResult(frameInput);
      const face = nearestFace(result.faces);
      return face && isDisplayableEmotion(face) ? face.emotion.id : null;
    })()`, context);
    const active = frame.fraction >= video.known_label_window[0] && frame.fraction <= video.known_label_window[1];
    if (active) {
      inWindow += 1;
      if (result === video.label) {
        correct += 1; streak += 1;
        if (streakStartedAt === null) streakStartedAt = frame.seconds;
        maxCorrectDurationSeconds = Math.max(maxCorrectDurationSeconds, frame.seconds - streakStartedAt);
      } else { streak = 0; streakStartedAt = null; }
      maxStreak = Math.max(maxStreak, streak);
      if (streak === 3 && firstStableSeconds === null) firstStableSeconds = frame.seconds;
      if (previous !== null && result !== null && previous !== result) switches += 1;
      previous = result;
    }
    frames.push({ index: frame.index, seconds: frame.seconds, displayed: result });
  }
  clips.push({ id: video.id, source: video.source, label: video.label, evaluatedFrames: inWindow,
    correctDisplayedFrames: correct, correctDisplayFraction: inWindow ? correct / inWindow : 0,
    stableCorrectAtLeastThreeFrames: maxStreak >= 3, maxCorrectStreak: maxStreak,
    stableCorrectAtLeastHalfSecond: maxCorrectDurationSeconds >= 0.5 - 0.0001,
    maxCorrectDurationSeconds,
    firstStableSeconds, labelSwitchesInWindow: switches, frames });
}
const groups = {};
for (const clip of clips) {
  const key = `${clip.source}/${clip.label}`;
  groups[key] ??= { clips: 0, stableCorrectClips: 0, stableHalfSecondClips: 0, correctDisplayFractionSum: 0, switches: 0 };
  const group = groups[key]; group.clips += 1;
  group.stableCorrectClips += Number(clip.stableCorrectAtLeastThreeFrames);
  group.stableHalfSecondClips += Number(clip.stableCorrectAtLeastHalfSecond);
  group.correctDisplayFractionSum += clip.correctDisplayFraction;
  group.switches += clip.labelSwitchesInWindow;
}
for (const group of Object.values(groups)) {
  group.stableCorrectClipRate = group.stableCorrectClips / group.clips;
  group.stableHalfSecondClipRate = group.stableHalfSecondClips / group.clips;
  group.meanCorrectDisplayFraction = group.correctDisplayFractionSum / group.clips;
  delete group.correctDisplayFractionSum;
}
fs.writeFileSync(output, JSON.stringify({
  scope: 'Actual app.js smoothing on sampled validation-video score sequences. Metrics count confidence-qualified labels only (40%, fear32%); the UI also shows tentative lower-confidence predictions, which are excluded here. Clip labels are not per-frame human annotations; CAER is scene-label diagnostic only. Reports both three consecutive sampled frames and at least0.5s continuously correct confident display; use the duration metric when comparing sampling rates.',
  model: report.model, groups, clips,
}, null, 2));
console.log(JSON.stringify(groups));
