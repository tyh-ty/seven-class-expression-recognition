const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'emotion-video-timing-'));
const input = path.join(directory, 'input.json');
const output = path.join(directory, 'output.json');
const timestamps = [[0, 0.25, 0.5], [0, 0.1, 0.2], [0, 0.1, 0.2, 0.3, 0.4, 0.5]];
try {
  const videos = timestamps.map((times, index) => ({
    id: `sample-${index}`, source: 'synthetic-regression', label: 'happiness', known_label_window: [0, 1],
    frames: times.map((seconds, index) => ({
      index, seconds, fraction: index / (times.length - 1), width: 512, height: 384,
      probabilities: [0.01, 0.01, 0.01, 0.94, 0.01, 0.01, 0.01],
      bbox: { x: 100, y: 80, width: 100, height: 100 }, face_score: 0.95,
    })),
  }));
  fs.writeFileSync(input, JSON.stringify({ model: 'regression-only', videos }));
  execFileSync(process.execPath, [path.join(__dirname, 'evaluate_video_smoothing.cjs'), input, output], { stdio: 'pipe' });
  const result = JSON.parse(fs.readFileSync(output, 'utf8'));
  assert.deepEqual(result.clips.map(c => c.stableCorrectAtLeastThreeFrames), [true, true, true]);
  assert.deepEqual(result.clips.map(c => c.stableCorrectAtLeastHalfSecond), [true, false, true]);
  assert.equal(result.groups['synthetic-regression/happiness'].stableHalfSecondClipRate, 2 / 3);
  console.log('视频稳定显示时长检查通过：区分三帧与半秒，不混淆4fps和10fps。');
} finally {
  if (fs.existsSync(input)) fs.unlinkSync(input);
  if (fs.existsSync(output)) fs.unlinkSync(output);
  fs.rmdirSync(directory);
}
