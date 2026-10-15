/**
 * Web Audio API Audible Fall Alert Synthesizer
 * Generates an alternating two-tone emergency beeper without external audio files.
 * Includes mute/unmute persistence and browser autoplay policy recovery.
 */

let audioCtx = null;
let oscillator = null;
let gainNode = null;
let intervalId = null;
let isPlaying = false;

const STORAGE_KEY = 'fall_detection_alert_muted';

export function isAudioMuted() {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

export function setAudioMuted(muted) {
  try {
    localStorage.setItem(STORAGE_KEY, muted ? 'true' : 'false');
  } catch {
    // ignore local storage restrictions
  }
  if (muted && isPlaying) {
    stopAlarm();
  }
}

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

/**
 * Starts the audible emergency alarm tone if unmuted.
 */
export function startAlarm() {
  if (isAudioMuted() || isPlaying) return;

  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    isPlaying = true;

    oscillator = ctx.createOscillator();
    gainNode = ctx.createGain();

    oscillator.type = 'sawtooth';
    oscillator.frequency.setValueAtTime(880, ctx.currentTime); // High tone (A5)

    // Gentle gain ramp to avoid harsh clicks
    gainNode.gain.setValueAtTime(0.001, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.05);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    oscillator.start();

    // Pulse between 880Hz and 660Hz every 240ms
    let high = true;
    intervalId = setInterval(() => {
      if (!oscillator || !ctx || ctx.state === 'closed') return;
      high = !high;
      const freq = high ? 880 : 660;
      try {
        oscillator.frequency.setValueAtTime(freq, ctx.currentTime);
      } catch {
        // context might have closed
      }
    }, 240);
  } catch (err) {
    console.warn('[AudioAlert] Web Audio API init error:', err);
    isPlaying = false;
  }
}

/**
 * Stops the audible emergency alarm immediately.
 */
export function stopAlarm() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }

  if (oscillator) {
    try {
      if (gainNode && audioCtx) {
        gainNode.gain.setValueAtTime(0.001, audioCtx.currentTime);
      }
      oscillator.stop();
      oscillator.disconnect();
    } catch {
      // ignore already stopped error
    }
    oscillator = null;
  }

  if (gainNode) {
    try {
      gainNode.disconnect();
    } catch {
      // ignore
    }
    gainNode = null;
  }

  isPlaying = false;
}

export function isAlarmPlaying() {
  return isPlaying;
}
