/**
 * Regression tests for the voice-to-voice transcript engine.
 *
 * These reproduce the two bugs that kept resurfacing:
 *   1. Answers being OVERWRITTEN when SpeechRecognition restarts.
 *   2. Auto-submit firing too early (mid-answer pauses ending the answer).
 *
 * Run: node src/__tests__/transcript.test.mjs
 */

import assert from 'node:assert/strict';

const SILENCE_SUBMIT_MS = 5000;
const MIN_AUTO_SUBMIT_CHARS = 10;

/**
 * Mirror of the append-only transcript logic used in CandidatePortal/index.jsx
 * and InterviewRoom.jsx.
 */
function createTranscriptEngine({ onSubmit = () => {} } = {}) {
  let committed = '';
  let sessionFinal = '';
  let visible = '';
  let silenceTimer = null;
  let now = 0;

  const norm = (s) => s.replace(/\s+/g, ' ').trim();

  const commit = () => {
    committed = norm(`${committed} ${sessionFinal}`);
    sessionFinal = '';
    if (committed) visible = committed;
  };

  return {
    /** Feed results from the CURRENT recognition instance. */
    result(results, { agentSpeaking = false } = {}) {
      if (agentSpeaking) return;
      let fin = '';
      let interim = '';
      for (const r of results) {
        if (r.isFinal) fin += r.transcript + ' ';
        else interim += r.transcript;
      }
      sessionFinal = fin.trim();
      visible = norm(`${committed} ${fin} ${interim}`);
      silenceTimer = now + SILENCE_SUBMIT_MS;
    },
    /** Recognition ended (network blip, echo-pause, browser timeout). */
    end() {
      commit();
    },
    /** Advance the virtual clock; fires auto-submit when silence elapses. */
    tick(ms) {
      now += ms;
      if (silenceTimer !== null && now >= silenceTimer) {
        silenceTimer = null;
        if (visible.trim().length >= MIN_AUTO_SUBMIT_CHARS) onSubmit(visible);
      }
    },
    manualEdit(text) {
      committed = text;
      sessionFinal = '';
      visible = text;
    },
    reset() {
      committed = '';
      sessionFinal = '';
      visible = '';
      silenceTimer = null;
    },
    get text() {
      return visible;
    },
  };
}

let passed = 0;
const test = (name, fn) => {
  fn();
  console.log(`  PASS  ${name}`);
  passed++;
};

console.log('\nvoice-to-voice transcript engine\n');

test('preserves earlier speech across a recognition restart (the overwrite bug)', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: 'I have five years of backend experience', isFinal: true }]);
  assert.equal(e.text, 'I have five years of backend experience');

  // Recognition drops and restarts — results array resets to empty.
  e.end();
  e.result([{ transcript: 'mostly in Python and Go', isFinal: true }]);

  assert.equal(
    e.text,
    'I have five years of backend experience mostly in Python and Go',
    'earlier speech must NOT be overwritten'
  );
});

test('survives many consecutive restarts without losing text', () => {
  const e = createTranscriptEngine();
  const parts = ['one', 'two', 'three', 'four', 'five'];
  for (const p of parts) {
    e.result([{ transcript: p, isFinal: true }]);
    e.end();
  }
  assert.equal(e.text, 'one two three four five');
});

test('does NOT submit during a natural 4-second mid-answer pause', () => {
  let submitted = null;
  const e = createTranscriptEngine({ onSubmit: (t) => (submitted = t) });

  e.result([{ transcript: 'Let me think about that for a moment', isFinal: true }]);
  e.tick(4000); // pause shorter than the 5s threshold
  assert.equal(submitted, null, 'must not cut the candidate off mid-thought');

  // Candidate resumes; timer resets.
  e.result([{ transcript: 'I would use a message queue', isFinal: true }]);
  e.tick(4000);
  assert.equal(submitted, null, 'resumed speech must reset the silence timer');
});

test('submits after a full 5 seconds of silence', () => {
  let submitted = null;
  const e = createTranscriptEngine({ onSubmit: (t) => (submitted = t) });
  e.result([{ transcript: 'That is my complete answer', isFinal: true }]);
  e.tick(5000);
  assert.equal(submitted, 'That is my complete answer');
});

test('submits exactly once, not repeatedly', () => {
  let count = 0;
  const e = createTranscriptEngine({ onSubmit: () => count++ });
  e.result([{ transcript: 'A sufficiently long answer here', isFinal: true }]);
  e.tick(5000);
  e.tick(5000);
  e.tick(5000);
  assert.equal(count, 1);
});

test('ignores a stray short utterance (cough / filler)', () => {
  let submitted = null;
  const e = createTranscriptEngine({ onSubmit: (t) => (submitted = t) });
  e.result([{ transcript: 'um', isFinal: true }]);
  e.tick(6000);
  assert.equal(submitted, null, 'too short to be a real answer');
});

test('interim results are replaced, not duplicated', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: 'I work with', isFinal: false }]);
  e.result([{ transcript: 'I work with React', isFinal: false }]);
  e.result([{ transcript: 'I work with React daily', isFinal: true }]);
  assert.equal(e.text, 'I work with React daily');
});

test('ignores results while the AI agent is speaking (anti-echo)', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: 'Tell me about yourself', isFinal: true }], { agentSpeaking: true });
  assert.equal(e.text, '', 'agent TTS must not be transcribed as the answer');
});

test('manual typing becomes the baseline for later speech', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: 'I know Java', isFinal: true }]);
  e.end();
  e.manualEdit('I know Java and Kotlin');
  e.result([{ transcript: 'and Swift', isFinal: true }]);
  assert.equal(e.text, 'I know Java and Kotlin and Swift', 'edits must not be reverted');
});

test('new question clears the transcript (no bleed between answers)', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: 'Answer to question one', isFinal: true }]);
  e.end();
  e.reset();
  e.result([{ transcript: 'Answer to question two', isFinal: true }]);
  assert.equal(e.text, 'Answer to question two');
});

test('whitespace stays normalised across merges', () => {
  const e = createTranscriptEngine();
  e.result([{ transcript: '  spaced   out  ', isFinal: true }]);
  e.end();
  e.result([{ transcript: '   words   ', isFinal: true }]);
  assert.equal(e.text, 'spaced out words');
  assert.ok(!/\s{2,}/.test(e.text));
});

console.log(`\n${passed} passed\n`);
