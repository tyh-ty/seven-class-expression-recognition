const assert = require('node:assert/strict');
const EmotionSpeechController = require('../web/emotion-speech.js');

function face(id = 'anger', score = 0.8, box = {}) {
  return {
    bbox: { x: 20, y: 20, width: 100, height: 100, ...box },
    emotion: { id, score, label: '不要使用输入中的任意文本' },
  };
}

function setup({ voices = [] } = {}) {
  let time = 0;
  const spoken = [];
  const statuses = [];
  const listeners = new Map();
  const synth = {
    voices,
    cancellations: 0,
    getVoices() { return this.voices; },
    addEventListener(name, handler) { listeners.set(name, handler); },
    removeEventListener(name) { listeners.delete(name); },
    speak(utterance) { spoken.push(utterance); },
    cancel() {
      this.cancellations += 1;
      spoken.at(-1)?.onerror?.({ error: 'interrupted' });
    },
  };
  class Utterance {
    constructor(text) { this.text = text; }
  }
  const controller = new EmotionSpeechController({
    synth, Utterance, now: () => time, onStatus: (status) => statuses.push(status),
  });
  return {
    controller, synth, spoken, statuses, listeners,
    at(next, faces = [face()]) { time = next; controller.update({ faces }); },
    end() { spoken.at(-1)?.onend?.(); },
  };
}

{
  const h = setup();
  assert.equal(h.controller.available, true);
  assert.equal(h.controller.enabled, false);
  h.at(0); h.at(800);
  assert.equal(h.spoken.length, 0, 'default is off');
  h.controller.setEnabled(true);
  h.at(1000); h.at(1599);
  assert.equal(h.spoken.length, 0, 'require 600 ms of stable observations');
  h.at(1600);
  assert.equal(h.spoken[0].text, '当前情绪为愤怒');
  assert.equal(h.spoken[0].lang, 'zh-CN');
  h.end();
  h.at(2500); h.at(4500); h.at(6500);
  assert.equal(h.spoken.length, 1, 'unchanged expression is not repeated');
}

{
  const h = setup(); h.controller.setEnabled(true);
  const people = [face('anger', 0.99, { width: 50, height: 50 }), face('fear', 0.32)];
  h.at(0, people); h.at(600, people);
  assert.equal(h.spoken[0].text, '当前情绪为恐惧', 'largest box wins, independent of array order');
  h.end();
  const uncertainClosest = [people[0], face('fear', 0.319)];
  h.at(1000, uncertainClosest); h.at(1700, uncertainClosest);
  assert.equal(h.spoken.length, 1, 'an uncertain closest person must not fall back to a smaller face');
  assert.match(h.statuses.at(-1), /置信度不足/);
  h.at(2000, [face('anger', 0.399)]); h.at(2600, [face('anger', 0.399)]);
  assert.equal(h.spoken.length, 1, 'other emotions require 40%');
}

{
  const h = setup(); h.controller.setEnabled(true);
  const smoothedLarge = face('anger', 0.9, { width: 200, height: 200 });
  smoothedLarge.detectedBbox = { x: 20, y: 20, width: 50, height: 50 };
  const actualClosest = face('fear');
  h.at(0, [smoothedLarge, actualClosest]); h.at(600, [smoothedLarge, actualClosest]);
  assert.equal(h.spoken[0].text, '当前情绪为恐惧', 'raw detected size wins over a smoothed box');
  const moved = { ...actualClosest, detectedBbox: { ...actualClosest.bbox, x: 250 } };
  h.at(700, [moved]);
  assert.equal(h.synth.cancellations, 1, 'tracking also uses the raw detected box');
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.at(0); h.at(500, [face('fear')]); h.at(900); h.at(1499);
  assert.equal(h.spoken.length, 0, 'expression flicker restarts stabilization');
  h.at(1500);
  h.at(1800, [face('fear')]); h.at(2400, [face('fear')]);
  assert.equal(h.spoken.length, 1, 'never enqueue while an earlier utterance is active');
  h.end();
  h.at(3999, [face('fear')]);
  assert.equal(h.spoken.length, 1, 'at least 2.5 seconds between starts');
  h.at(4000, [face('fear')]);
  assert.equal(h.spoken[1].text, '当前情绪为恐惧');
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.at(0); h.at(600);
  const differentPerson = [face('anger', 0.8, { x: 250 })];
  h.at(1000, differentPerson);
  assert.equal(h.synth.cancellations, 1, 'changing person cancels the old utterance');
  h.at(1600, differentPerson);
  assert.equal(h.spoken.length, 1, 'person changes still respect the global interval');
  h.at(3100, differentPerson);
  assert.equal(h.spoken.length, 2, 'same emotion can be spoken for the new closest person');
  assert.equal(h.spoken[1].text, '当前情绪为愤怒');
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.at(0); h.at(600);
  const first = h.spoken[0];
  h.at(700, []);
  assert.equal(h.synth.cancellations, 1, 'no detected face immediately cancels speech');
  first.onend(); first.onerror({ error: 'interrupted' });
  assert.match(h.statuses.at(-1), /等待画面中出现人脸/, 'late events cannot overwrite current state');
  const statusCount = h.statuses.length;
  h.at(800, []); h.at(900, []);
  assert.equal(h.statuses.length, statusCount, 'missing faces do not cause repeated status announcements');
  h.at(1000); h.at(1599);
  assert.equal(h.spoken.length, 1, 'returning face must stabilize again');
  h.at(3100);
  assert.equal(h.spoken.length, 2);
  h.at(3200, [face('anger', 0.2)]);
  assert.equal(h.synth.cancellations, 2, 'low confidence cancels old speech');
  h.at(3300); h.controller.reset(); h.at(3900);
  assert.equal(h.spoken.length, 2, 'reset clears the previous candidate');
  h.at(4500); h.at(5600);
  assert.equal(h.spoken.length, 3);
  h.controller.setEnabled(false);
  assert.equal(h.synth.cancellations, 3, 'switching off cancels current speech');
  h.at(9000); h.at(9800);
  assert.equal(h.spoken.length, 3);
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.at(0); h.at(600);
  h.spoken[0].onerror({ error: 'not-allowed' });
  assert.match(h.statuses.at(-1), /浏览器未允许语音播报/);
  h.at(1000); h.at(4000); h.at(6000);
  h.controller.reset(); h.controller.setEnabled(true); h.at(8000); h.at(8600);
  assert.equal(h.spoken.length, 1, 'errors do not automatically retry, including after reset');
  h.controller.setEnabled(false); h.controller.setEnabled(true);
  h.at(9000); h.at(9600);
  assert.equal(h.spoken.length, 2, 'explicit off/on allows retry');
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.synth.speak = () => { throw new Error('unavailable engine'); };
  h.at(0); h.at(600);
  assert.match(h.statuses.at(-1), /语音播报失败/);
  h.at(5000); h.at(5600);
  assert.equal(h.synth.cancellations, 1, 'a synchronous failure is handled without a retry loop');
}

{
  const h = setup();
  const remote = { name: 'Remote Chinese', lang: 'zh-CN', localService: false };
  const english = { name: 'English', lang: 'en-US', localService: true };
  const local = { name: 'Local Chinese', lang: 'zh-CN', localService: true };
  h.synth.voices = [remote, english, local];
  h.listeners.get('voiceschanged')();
  h.controller.setEnabled(true); h.at(0); h.at(600);
  assert.equal(h.spoken[0].voice, local, 'late-loaded local Chinese voice has priority');
  h.controller.destroy();
  assert.equal(h.listeners.has('voiceschanged'), false);
  assert.equal(h.controller.available, false);
  assert.equal(h.synth.cancellations, 1);
}

{
  const h = setup(); h.controller.setEnabled(true);
  h.at(0); h.at(4000);
  assert.equal(h.spoken.length, 0, 'a long gap cannot count as sustained observation');
  h.at(4600);
  assert.equal(h.spoken.length, 1);
}

{
  const statuses = [];
  const controller = new EmotionSpeechController({
    synth: null, Utterance: null, onStatus: (status) => statuses.push(status),
  });
  assert.equal(controller.available, false);
  assert.equal(controller.setEnabled(true), false);
  assert.match(statuses.at(-1), /不支持/);
  controller.update({ faces: [face()] });
  controller.reset(); controller.destroy();
}

console.log('语音播报检查通过：最近人脸、置信度、稳定防抖、间隔、换人、取消、错误恢复、延迟音色。');
