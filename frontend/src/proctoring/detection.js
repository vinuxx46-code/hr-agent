/**
 * Time-based proctoring detection helpers.
 *
 * The previous implementation counted raw frames ("fire after 45 frames"), but
 * the detection loop runs at ~4 FPS, so 45 frames meant ~11 seconds of
 * completely uninterrupted violation before a warning appeared. With the decay
 * applied on every clean frame, gaze and head-turn warnings almost never fired.
 *
 * These helpers accumulate violation time in milliseconds instead, so the
 * thresholds mean what they say regardless of loop rate or device speed.
 */

/** Sustained-violation accumulator with independent per-detector cooldown. */
export class ViolationTracker {
  /**
   * @param {object}  opts
   * @param {number}  opts.requiredMs  continuous violation time before firing
   * @param {number}  opts.cooldownMs  minimum gap between two warnings
   * @param {number}  opts.decayFactor how fast clean time drains the accumulator
   */
  constructor({ requiredMs = 1500, cooldownMs = 6000, decayFactor = 2 } = {}) {
    this.requiredMs = requiredMs;
    this.cooldownMs = cooldownMs;
    this.decayFactor = decayFactor;
    this.accumulatedMs = 0;
    this.lastFiredAt = -Infinity;
    this.lastUpdateAt = null;
  }

  /**
   * Feed one observation.
   * @param {boolean} violated is the rule currently broken?
   * @param {number}  now      timestamp in ms (performance.now())
   * @returns {boolean} true exactly once per confirmed violation
   */
  update(violated, now) {
    const delta = this.lastUpdateAt === null ? 0 : Math.max(0, now - this.lastUpdateAt);
    this.lastUpdateAt = now;

    if (violated) {
      this.accumulatedMs += delta;
    } else {
      // Brief natural glances drain quickly so they never accumulate into a
      // false positive, but a persistent violation still builds up.
      this.accumulatedMs = Math.max(0, this.accumulatedMs - delta * this.decayFactor);
    }

    if (this.accumulatedMs >= this.requiredMs && now - this.lastFiredAt >= this.cooldownMs) {
      this.lastFiredAt = now;
      this.accumulatedMs = 0;
      return true;
    }
    return false;
  }

  reset() {
    this.accumulatedMs = 0;
    this.lastUpdateAt = null;
  }
}

// MediaPipe FaceLandmarker indices.
export const LM = {
  NOSE_TIP: 1,
  LEFT_EAR: 234,
  RIGHT_EAR: 454,
  CHIN: 152,
  FOREHEAD: 10,
  LEFT_EYE: { pupil: 468, inner: 133, outer: 33, top: 159, bottom: 145 },
  RIGHT_EYE: { pupil: 473, inner: 362, outer: 263, top: 386, bottom: 374 },
};

/**
 * Horizontal + vertical iris offset for one eye, normalised 0..1.
 * 0.5 means the pupil is centred. Returns null when landmarks are unusable.
 */
export function eyeGazeRatios(landmarks, eye) {
  if (!landmarks) return null;
  const pupil = landmarks[eye.pupil];
  const inner = landmarks[eye.inner];
  const outer = landmarks[eye.outer];
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  if (!pupil || !inner || !outer) return null;

  const width = Math.abs(outer.x - inner.x);
  if (width < 0.012) return null; // eye too small / head turned away

  const minX = Math.min(inner.x, outer.x);
  const h = (pupil.x - minX) / width;

  let v = 0.5;
  if (top && bottom) {
    const height = Math.abs(bottom.y - top.y);
    if (height > 0.004) {
      v = (pupil.y - Math.min(top.y, bottom.y)) / height;
    }
  }
  return { h, v };
}

/** True when the eye is closed (blink) — gaze is meaningless then. */
export function isEyeClosed(landmarks, eye) {
  if (!landmarks) return false;
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  const inner = landmarks[eye.inner];
  const outer = landmarks[eye.outer];
  if (!top || !bottom || !inner || !outer) return false;

  const height = Math.abs(bottom.y - top.y);
  const width = Math.abs(outer.x - inner.x);
  if (width <= 0) return false;
  // Eye aspect ratio; a blink collapses this well below the open-eye value.
  return height / width < 0.12;
}

/**
 * Decide whether the candidate is looking away from the screen.
 *
 * Combines both irises and requires agreement, which rejects the single-eye
 * landmark jitter that produced most false positives previously. Blinks are
 * ignored outright.
 */
export function detectGazeAway(landmarks, opts = {}) {
  const {
    hMin = 0.30,
    hMax = 0.70,
    vMin = 0.22,
    vMax = 0.80,
  } = opts;

  if (!landmarks || landmarks.length <= LM.RIGHT_EYE.pupil) {
    return { away: false, reason: 'no-landmarks' };
  }

  const leftClosed = isEyeClosed(landmarks, LM.LEFT_EYE);
  const rightClosed = isEyeClosed(landmarks, LM.RIGHT_EYE);
  if (leftClosed && rightClosed) {
    return { away: false, reason: 'blink' };
  }

  const left = leftClosed ? null : eyeGazeRatios(landmarks, LM.LEFT_EYE);
  const right = rightClosed ? null : eyeGazeRatios(landmarks, LM.RIGHT_EYE);
  const usable = [left, right].filter(Boolean);
  if (usable.length === 0) return { away: false, reason: 'unusable' };

  const offScreen = (r) => r.h < hMin || r.h > hMax || r.v < vMin || r.v > vMax;

  // With both eyes visible require agreement; a single eye disagreeing is
  // almost always landmark noise rather than a real glance away.
  const away = usable.length === 2 ? usable.every(offScreen) : offScreen(usable[0]);

  return {
    away,
    reason: away ? 'gaze-off-screen' : 'on-screen',
    left,
    right,
  };
}

/**
 * Head yaw from the nose position between the ears, normalised to roughly
 * -1 (turned hard left) .. +1 (turned hard right).
 */
export function headYaw(landmarks) {
  if (!landmarks) return null;
  const nose = landmarks[LM.NOSE_TIP];
  const leftEar = landmarks[LM.LEFT_EAR];
  const rightEar = landmarks[LM.RIGHT_EAR];
  if (!nose || !leftEar || !rightEar) return null;

  const dLeft = Math.abs(nose.x - leftEar.x);
  const dRight = Math.abs(nose.x - rightEar.x);
  const total = dLeft + dRight;
  if (total <= 0.0001) return null;
  return (dRight - dLeft) / total;
}

/** Head pitch (nodding down / looking at lap or a phone). */
export function headPitch(landmarks) {
  if (!landmarks) return null;
  const nose = landmarks[LM.NOSE_TIP];
  const chin = landmarks[LM.CHIN];
  const forehead = landmarks[LM.FOREHEAD];
  if (!nose || !chin || !forehead) return null;

  const faceHeight = Math.abs(chin.y - forehead.y);
  if (faceHeight < 0.02) return null;
  // 0.5 = nose centred vertically between forehead and chin.
  return (nose.y - Math.min(forehead.y, chin.y)) / faceHeight;
}

export function detectHeadTurned(landmarks, { yawLimit = 0.42, pitchMin = 0.28, pitchMax = 0.78 } = {}) {
  const yaw = headYaw(landmarks);
  const pitch = headPitch(landmarks);

  if (yaw !== null && Math.abs(yaw) > yawLimit) {
    return { turned: true, reason: yaw > 0 ? 'turned-right' : 'turned-left', yaw, pitch };
  }
  if (pitch !== null && (pitch < pitchMin || pitch > pitchMax)) {
    return { turned: true, reason: pitch > pitchMax ? 'looking-down' : 'looking-up', yaw, pitch };
  }
  return { turned: false, reason: 'facing-camera', yaw, pitch };
}

/**
 * Count people in frame, combining face landmarks with person detections.
 * Small or low-confidence person boxes (posters, photos on a wall) are ignored.
 */
export function countPeople({ faceCount = 0, detections = [], frameWidth = 0, frameHeight = 0 } = {}) {
  let persons = 0;
  for (const d of detections || []) {
    const cat = d?.categories?.[0];
    if (!cat || cat.categoryName !== 'person' || cat.score < 0.68) continue;
    const box = d.boundingBox;
    if (box && frameWidth && frameHeight) {
      const area = (box.width / frameWidth) * (box.height / frameHeight);
      if (area < 0.035) continue;
    }
    persons += 1;
  }
  return Math.max(faceCount, persons);
}

/** Face present but drifting out of the camera frame. */
export function isFaceOutOfFrame(landmarks, margin = 0.06) {
  if (!landmarks) return false;
  const nose = landmarks[LM.NOSE_TIP];
  if (!nose) return false;
  return nose.x < margin || nose.x > 1 - margin || nose.y < margin || nose.y > 1 - margin;
}
