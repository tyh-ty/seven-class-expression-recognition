(function attachEmotionSpeechController(browser, factory) {
  const EmotionSpeechController = factory(browser);
  if (typeof module === 'object' && module.exports) module.exports = EmotionSpeechController;
  if (browser) browser.EmotionSpeechController = EmotionSpeechController;
}(typeof window !== 'undefined' ? window : null, function createEmotionSpeechController(browser) {
  'use strict';

  const LABELS = {
    anger: '愤怒',
    disgust: '厌恶',
    fear: '恐惧',
    happiness: '开心',
    neutral: '中性',
    sadness: '悲伤',
    surprise: '惊讶',
  };
  const STABLE_MS = 600;
  const MIN_INTERVAL_MS = 2500;
  const SAME_FACE_IOU = 0.3;
  const MAX_OBSERVATION_GAP_MS = 3000;

  function validBox(box) {
    return box && ['x', 'y', 'width', 'height'].every((key) => Number.isFinite(box[key]))
      && box.width > 0 && box.height > 0;
  }

  function detectedBox(face) {
    return face && (face.detectedBbox ?? face.bbox);
  }

  function boxIoU(first, second) {
    const width = Math.max(0, Math.min(first.x + first.width, second.x + second.width)
      - Math.max(first.x, second.x));
    const height = Math.max(0, Math.min(first.y + first.height, second.y + second.height)
      - Math.max(first.y, second.y));
    const intersection = width * height;
    return intersection / (first.width * first.height + second.width * second.height - intersection);
  }

  return class EmotionSpeechController {
    constructor({
      synth = browser && browser.speechSynthesis,
      Utterance = browser && browser.SpeechSynthesisUtterance,
      now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      onStatus = () => {},
    } = {}) {
      this._synth = synth;
      this._Utterance = Utterance;
      this._now = now;
      this._onStatus = onStatus;
      this.enabled = false;
      this._disposed = false;
      this._blocked = false;
      this._box = null;
      this._candidate = null;
      this._lastSpoken = null;
      this._lastStartedAt = -Infinity;
      this._lastObservedAt = null;
      this._active = null;
      this._voice = null;
      this._status = null;
      this._voicesChanged = () => this._loadVoice();
      if (this.available) {
        this._loadVoice();
        if (typeof synth.addEventListener === 'function') {
          synth.addEventListener('voiceschanged', this._voicesChanged);
        } else {
          this._previousVoicesChanged = synth.onvoiceschanged;
          this._voicesChangedFallback = (event) => {
            if (typeof this._previousVoicesChanged === 'function') {
              this._previousVoicesChanged.call(synth, event);
            }
            this._loadVoice();
          };
          synth.onvoiceschanged = this._voicesChangedFallback;
        }
      }
      this._setStatus(this.available ? '语音播报已关闭' : '当前浏览器不支持语音播报');
    }

    get available() {
      return !this._disposed && Boolean(this._synth)
        && typeof this._synth.speak === 'function'
        && typeof this._synth.cancel === 'function'
        && typeof this._Utterance === 'function';
    }

    setEnabled(enabled) {
      const next = Boolean(enabled);
      if (next && !this.available) {
        this._setStatus('当前浏览器不支持语音播报');
        return false;
      }
      // An error is retried only after the user explicitly turns the switch off and on.
      if (next === this.enabled) return this.enabled;
      this.enabled = next;
      this._blocked = false;
      this.reset();
      return this.enabled;
    }

    reset() {
      this._clearTracking();
      // Keep the interval across person changes and quick off/on actions.
      if (!this._blocked) {
        this._setStatus(!this.available ? '当前浏览器不支持语音播报'
          : this.enabled ? '等待最近的人脸表情稳定' : '语音播报已关闭');
      }
    }

    update(result) {
      if (!this.enabled || !this.available || this._blocked) return;
      const faces = Array.isArray(result && result.faces) ? result.faces : [];
      // Select before checking emotion confidence, so a distant face never takes over.
      const closest = faces.reduce((largest, face) => {
        const box = detectedBox(face);
        if (!validBox(box)) return largest;
        const largestBox = detectedBox(largest);
        return !largest || box.width * box.height
          > largestBox.width * largestBox.height ? face : largest;
      }, null);
      if (!closest) {
        this._clearTracking();
        this._setStatus('等待画面中出现人脸');
        return;
      }
      const emotion = closest.emotion;
      const threshold = emotion && emotion.id === 'fear' ? 0.32 : 0.4;
      if (!emotion || !Object.hasOwn(LABELS, emotion.id) || !Number.isFinite(emotion.score)
        || emotion.score < threshold || emotion.score > 1) {
        this._clearTracking();
        this._setStatus('最近的人脸置信度不足，暂不播报');
        return;
      }

      const time = this._now();
      const closestBox = detectedBox(closest);
      if ((this._box && boxIoU(this._box, closestBox) < SAME_FACE_IOU)
        || (this._lastObservedAt !== null && time - this._lastObservedAt > MAX_OBSERVATION_GAP_MS)) {
        this._clearTracking();
      }
      this._box = { ...closestBox };
      this._lastObservedAt = time;
      if (!this._candidate || this._candidate.id !== emotion.id) {
        this._candidate = { id: emotion.id, since: time };
        this._setStatus('等待最近的人脸表情稳定');
        return;
      }
      if (time - this._candidate.since < STABLE_MS) return;
      if (this._lastSpoken === emotion.id) {
        if (!this._active) this._setStatus(`已播报：当前情绪为${LABELS[emotion.id]}`);
        return;
      }
      if (time - this._lastStartedAt < MIN_INTERVAL_MS || this._active) return;
      this._speak(emotion.id, time);
    }

    destroy() {
      this.enabled = false;
      this._blocked = false;
      this.reset();
      if (this._synth && typeof this._synth.removeEventListener === 'function') {
        this._synth.removeEventListener('voiceschanged', this._voicesChanged);
      } else if (this._synth && this._synth.onvoiceschanged === this._voicesChangedFallback) {
        this._synth.onvoiceschanged = this._previousVoicesChanged || null;
      }
      this._disposed = true;
    }

    _setStatus(text) {
      if (text === this._status) return;
      this._status = text;
      this._onStatus(text);
    }

    _clearTracking() {
      this._cancelSpeech();
      this._box = null;
      this._candidate = null;
      this._lastSpoken = null;
      this._lastObservedAt = null;
    }

    _loadVoice() {
      let voices = [];
      try {
        voices = typeof this._synth.getVoices === 'function' ? this._synth.getVoices() : [];
      } catch (_error) {
        // Some browsers make voices available only after their voiceschanged event.
      }
      const chinese = (voices || []).filter((voice) => /^zh(?:[-_]|$)/i.test(voice.lang || ''));
      const mainland = (voice) => /^zh[-_]CN$/i.test(voice.lang || '');
      this._voice = chinese.find((voice) => voice.localService && mainland(voice))
        || chinese.find((voice) => voice.localService)
        || chinese.find(mainland) || chinese[0] || null;
    }

    _cancelSpeech() {
      const active = this._active;
      this._active = null;
      if (!active) return;
      // Invalidate first: browsers may synchronously emit an interrupted error in cancel().
      try { this._synth.cancel(); } catch (_error) { /* The utterance is already invalidated. */ }
    }

    _speak(id, time) {
      let utterance;
      try {
        utterance = new this._Utterance(`当前情绪为${LABELS[id]}`);
        utterance.lang = 'zh-CN';
        utterance.rate = 1;
        utterance.pitch = 1;
        this._loadVoice();
        if (this._voice) utterance.voice = this._voice;
        this._active = utterance;
        utterance.onend = () => {
          if (this._active !== utterance) return;
          this._active = null;
          this._setStatus(`已播报：当前情绪为${LABELS[id]}`);
        };
        utterance.onerror = (event) => {
          if (this._active !== utterance) return;
          this._fail(event && event.error);
        };
        this._lastSpoken = id;
        this._lastStartedAt = time;
        this._setStatus(`正在播报：当前情绪为${LABELS[id]}`);
        this._synth.speak(utterance);
      } catch (_error) {
        this._fail();
      }
    }

    _fail(error) {
      this._blocked = true;
      this._cancelSpeech();
      this._candidate = null;
      const reason = error === 'not-allowed' ? '浏览器未允许语音播报'
        : error === 'voice-unavailable' || error === 'language-unavailable' ? '中文语音不可用'
          : error === 'audio-busy' || error === 'audio-hardware' ? '音频设备暂不可用'
            : '语音播报失败';
      this._setStatus(`${reason}，请关闭播报开关后重新开启`);
    }
  };
}));
