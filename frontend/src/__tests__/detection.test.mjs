/**
 * Tests for the proctoring detection helpers.
 * Run: node src/__tests__/detection.test.mjs
 */

import assert from 'node:assert/strict';
import {
  ViolationTracker,
  detectGazeAway,
  detectHeadTurned,
  countPeople,
  isFaceOutOfFrame,
  isEyeClosed,
  LM,
} from '../proctoring/detection.js';

let passed = 0;
const test = (name, fn) => {
  fn();
  console.log(`  PASS  ${name}`);
  passed++;
};

/** Build a synthetic landmark array. */
function makeFace({
  leftPupilH = 0.5,
  rightPupilH = 0.5,
  pupilV = 0.5,
  noseX = 0.5,
  noseY = 0.5,
  eyeOpen = 0.3,
} = {}) {
  const lm = new Array(478).fill(null).map(() => ({ x: 0.5, y: 0.5, z: 0 }));

  const setEye = (eye, pupilH) => {
    const innerX = 0.4;
    const outerX = 0.6;
    const width = outerX - innerX;
    lm[eye.inner] = { x: innerX, y: 0.5 };
    lm[eye.outer] = { x: outerX, y: 0.5 };
    const top = 0.5 - (eyeOpen * width) / 2;
    const bottom = 0.5 + (eyeOpen * width) / 2;
    lm[eye.top] = { x: 0.5, y: top };
    lm[eye.bottom] = { x: 0.5, y: bottom };
    lm[eye.pupil] = { x: innerX + pupilH * width, y: top + pupilV * (bottom - top) };
  };

  setEye(LM.LEFT_EYE, leftPupilH);
  setEye(LM.RIGHT_EYE, rightPupilH);

  lm[LM.NOSE_TIP] = { x: noseX, y: noseY };
  lm[LM.LEFT_EAR] = { x: 0.3, y: 0.5 };
  lm[LM.RIGHT_EAR] = { x: 0.7, y: 0.5 };
  lm[LM.FOREHEAD] = { x: 0.5, y: 0.3 };
  lm[LM.CHIN] = { x: 0.5, y: 0.7 };
  return lm;
}

console.log('\nproctoring detection\n');

// --------------------------------------------------------------------------
// ViolationTracker — the frame-count vs. time bug
// --------------------------------------------------------------------------

test('fires after the configured duration, not a frame count', () => {
  const t = new ViolationTracker({ requiredMs: 1500, cooldownMs: 5000 });
  let now = 0;
  let fired = false;
  // 4 FPS loop, same rate as the real detection loop.
  for (let i = 0; i < 8; i++) {
    now += 250;
    if (t.update(true, now)) fired = true;
  }
  assert.equal(fired, true, '1.5s of violation at 4 FPS must fire');
});

test('does not fire for a brief glance', () => {
  const t = new ViolationTracker({ requiredMs: 1500, cooldownMs: 5000 });
  let now = 0;
  let fired = false;
  for (let i = 0; i < 3; i++) {
    now += 250;
    if (t.update(true, now)) fired = true; // only 750ms
  }
  assert.equal(fired, false);
});

test('respects the cooldown between warnings', () => {
  const t = new ViolationTracker({ requiredMs: 1000, cooldownMs: 5000 });
  let now = 0;
  let count = 0;
  for (let i = 0; i < 80; i++) {
    now += 250;
    if (t.update(true, now)) count++;
  }
  // 20s of continuous violation, 5s cooldown => about 4 warnings, never 80.
  assert.ok(count >= 3 && count <= 5, `expected ~4 warnings, got ${count}`);
});

test('clean frames drain the accumulator', () => {
  const t = new ViolationTracker({ requiredMs: 1500, cooldownMs: 1000, decayFactor: 2 });
  let now = 0;
  let fired = false;
  // Alternating violation/clean should never accumulate to a warning.
  for (let i = 0; i < 40; i++) {
    now += 250;
    if (t.update(i % 2 === 0, now)) fired = true;
  }
  assert.equal(fired, false, 'intermittent noise must not trigger');
});

test('is independent of loop frame rate', () => {
  const run = (stepMs) => {
    const t = new ViolationTracker({ requiredMs: 2000, cooldownMs: 10000 });
    let now = 0;
    let fired = 0;
    while (now < 3000) {
      now += stepMs;
      if (t.update(true, now)) fired++;
    }
    return fired;
  };
  // 4 FPS and 30 FPS must agree: 3s of violation, 2s threshold => 1 warning.
  assert.equal(run(250), 1);
  assert.equal(run(33), 1);
});

// --------------------------------------------------------------------------
// Gaze
// --------------------------------------------------------------------------

test('centred gaze is not flagged', () => {
  assert.equal(detectGazeAway(makeFace()).away, false);
});

test('reading naturally around the screen is not flagged', () => {
  for (const h of [0.35, 0.42, 0.5, 0.58, 0.65]) {
    const r = detectGazeAway(makeFace({ leftPupilH: h, rightPupilH: h }));
    assert.equal(r.away, false, `h=${h} should be on-screen`);
  }
});

test('looking hard left or right is flagged', () => {
  assert.equal(detectGazeAway(makeFace({ leftPupilH: 0.1, rightPupilH: 0.1 })).away, true);
  assert.equal(detectGazeAway(makeFace({ leftPupilH: 0.92, rightPupilH: 0.92 })).away, true);
});

test('single-eye landmark jitter does NOT trigger a false positive', () => {
  // One iris glitches off-screen while the other stays centred.
  const r = detectGazeAway(makeFace({ leftPupilH: 0.05, rightPupilH: 0.5 }));
  assert.equal(r.away, false, 'both eyes must agree before flagging');
});

test('blinking is not mistaken for looking away', () => {
  const face = makeFace({ eyeOpen: 0.05, leftPupilH: 0.1, rightPupilH: 0.1 });
  assert.equal(isEyeClosed(face, LM.LEFT_EYE), true);
  assert.equal(detectGazeAway(face).away, false, 'a blink must never be a violation');
});

test('looking far down (at a phone in the lap) is flagged', () => {
  assert.equal(detectGazeAway(makeFace({ pupilV: 0.95 })).away, true);
});

test('missing landmarks are handled safely', () => {
  assert.equal(detectGazeAway(null).away, false);
  assert.equal(detectGazeAway([]).away, false);
  assert.equal(detectGazeAway(new Array(478).fill(null)).away, false);
});

// --------------------------------------------------------------------------
// Head pose
// --------------------------------------------------------------------------

test('facing the camera is not flagged', () => {
  assert.equal(detectHeadTurned(makeFace()).turned, false);
});

test('small natural head movement is tolerated', () => {
  assert.equal(detectHeadTurned(makeFace({ noseX: 0.54 })).turned, false);
});

test('turning away from the monitor is flagged', () => {
  assert.equal(detectHeadTurned(makeFace({ noseX: 0.68 })).turned, true);
  assert.equal(detectHeadTurned(makeFace({ noseX: 0.32 })).turned, true);
});

test('looking down is flagged via pitch', () => {
  const r = detectHeadTurned(makeFace({ noseY: 0.68 }));
  assert.equal(r.turned, true);
  assert.equal(r.reason, 'looking-down');
});

// --------------------------------------------------------------------------
// People counting
// --------------------------------------------------------------------------

test('single candidate counts as one', () => {
  assert.equal(countPeople({ faceCount: 1, detections: [], frameWidth: 640, frameHeight: 480 }), 1);
});

test('a second person in frame is detected', () => {
  const detections = [
    { categories: [{ categoryName: 'person', score: 0.9 }], boundingBox: { width: 200, height: 300 } },
    { categories: [{ categoryName: 'person', score: 0.85 }], boundingBox: { width: 180, height: 280 } },
  ];
  assert.equal(countPeople({ faceCount: 1, detections, frameWidth: 640, frameHeight: 480 }), 2);
});

test('a small poster or photo on the wall is ignored', () => {
  const detections = [
    { categories: [{ categoryName: 'person', score: 0.9 }], boundingBox: { width: 200, height: 300 } },
    { categories: [{ categoryName: 'person', score: 0.72 }], boundingBox: { width: 30, height: 40 } },
  ];
  assert.equal(countPeople({ faceCount: 1, detections, frameWidth: 640, frameHeight: 480 }), 1);
});

test('low-confidence detections are ignored', () => {
  const detections = [
    { categories: [{ categoryName: 'person', score: 0.4 }], boundingBox: { width: 200, height: 300 } },
  ];
  assert.equal(countPeople({ faceCount: 1, detections, frameWidth: 640, frameHeight: 480 }), 1);
});

test('non-person objects are not counted as people', () => {
  const detections = [
    { categories: [{ categoryName: 'chair', score: 0.99 }], boundingBox: { width: 300, height: 300 } },
  ];
  assert.equal(countPeople({ faceCount: 1, detections, frameWidth: 640, frameHeight: 480 }), 1);
});

// --------------------------------------------------------------------------
// Frame bounds
// --------------------------------------------------------------------------

test('centred face is in frame', () => {
  assert.equal(isFaceOutOfFrame(makeFace()), false);
});

test('face drifting to the edge is flagged', () => {
  assert.equal(isFaceOutOfFrame(makeFace({ noseX: 0.01 })), true);
  assert.equal(isFaceOutOfFrame(makeFace({ noseY: 0.98 })), true);
});

console.log(`\n${passed} passed\n`);
