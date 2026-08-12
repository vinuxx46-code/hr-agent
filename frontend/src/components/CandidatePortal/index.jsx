import React, { useState, useRef, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { FaceLandmarker, ObjectDetector, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import '../../index.css';
import { apiUrl } from '../../api';

// Suppress verbose MediaPipe WebAssembly logs to clear the console
const originalWarn = console.warn;
const originalInfo = console.info;
const originalLog = console.log;
const suppressedPatterns = ['face_landmarker_graph.cc', 'gl_context.cc', 'TensorFlow Lite', 'XNNPACK', 'Created TensorFlow'];
const shouldSuppress = (args) => typeof args[0] === 'string' && suppressedPatterns.some(p => args[0].includes(p));
console.warn = (...args) => { if (!shouldSuppress(args)) originalWarn(...args); };
console.info = (...args) => { if (!shouldSuppress(args)) originalInfo(...args); };
console.log = (...args) => { if (!shouldSuppress(args)) originalLog(...args); };

const validatedTokens = new Set();

// Helper to extract clean question text string safely without React child rendering errors
const getQuestionTextString = (q) => {
  if (!q) return "";
  if (typeof q === 'string') {
    if (q.trim().startsWith('{')) {
      try { return getQuestionTextString(JSON.parse(q)); } catch (e) {}
    }
    return q;
  }
  if (typeof q === 'object') {
    if (typeof q.question === 'string') return q.question;
    if (typeof q.question === 'object') return getQuestionTextString(q.question);
    if (typeof q.text === 'string') return q.text;
  }
  return String(q || "");
};

function CandidatePortal({ directQA = false }) {
  const [appState, setAppState] = useState('idle');
  const [inviteToken, setInviteToken] = useState(null);
  const [isTokenValidating, setIsTokenValidating] = useState(true);
  const [generatedLinks, setGeneratedLinks] = useState([]);
  const [file, setFile] = useState(null);
  const [evaluation, setEvaluation] = useState(() => {
    const saved = sessionStorage.getItem('evaluation');
    return saved ? JSON.parse(saved) : null;
  });
  const [bulkSuccessMsg, setBulkSuccessMsg] = useState("");
  const [isSendingBulk, setIsSendingBulk] = useState(false);
  const [customWarning, setCustomWarning] = useState({ show: false, message: "" });
  const [errorMessage, setErrorMessage] = useState("");

  // Interview State
  const [sessionId, setSessionId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(15);
  const [timeLeft, setTimeLeft] = useState(30);
  const [cameraTimeLeft, setCameraTimeLeft] = useState(15);
  const [showCamera, setShowCamera] = useState(() => {
    return sessionStorage.getItem('showCamera') !== 'false';
  });
  const [answerText, setAnswerText] = useState("");
  const [interviewResult, setInterviewResult] = useState(null);
  const [isWarningBlinking, setIsWarningBlinking] = useState(false);
  const [showMuteWarning, setShowMuteWarning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [isCameraCheckConfirmed, setIsCameraCheckConfirmed] = useState(() => {
    const isCompleted = sessionStorage.getItem('isCameraCheckCompleted') === 'true';
    const wasScanning = sessionStorage.getItem('wasScanning') === 'true';
    return isCompleted || wasScanning;
  });
  const [isCameraCheckCompleted, setIsCameraCheckCompleted] = useState(() => {
    return sessionStorage.getItem('isCameraCheckCompleted') === 'true';
  });
  const [isPreCheckFailed, setIsPreCheckFailed] = useState(() => {
    return sessionStorage.getItem('isPreCheckFailed') === 'true';
  });
  const [preCheckFailureReason, setPreCheckFailureReason] = useState("");
  const [detectedFacesCount, setDetectedFacesCount] = useState(0);
  const [isScanningPhase, setIsScanningPhase] = useState(false);
  const [isScanPaused, setIsScanPaused] = useState(true);
  const [scanProgressPercent, setScanProgressPercent] = useState(() => {
    const saved = sessionStorage.getItem('scanProgressPercent');
    return saved ? parseInt(saved) : 0;
  });
  const seenKeyframesRef = useRef([]);
  const scanCoverageRef = useRef(0);
  const [is360ServerVerified, setIs360ServerVerified] = useState(false);
  const hasAdditionalPersonRef = useRef(false);
  const [hasAdditionalPersonDetected, setHasAdditionalPersonDetected] = useState(false);
  const lastFrameDiffRef = useRef(0);
  // 360° baseline comparison tracking
  const motionStreakRef = useRef(0);        // consecutive seconds ALL 3 conditions met
  const cumulativeMotionRef = useRef(0);    // total accumulated motion
  const scanBaselineFrameRef = useRef(null); // frame saved at scan start (person in center)
  const lastBaselineDiffRef = useRef(0);    // diff between current frame and scan-start baseline
  const scanBaselineSavedRef = useRef(false); // whether baseline has been captured yet

  const [scanTimeLeft, setScanTimeLeft] = useState(15);
  const [disclaimerTimeLeft, setDisclaimerTimeLeft] = useState(() => {
    const val = sessionStorage.getItem('disclaimerTimeLeft');
    return val ? parseInt(val) : 15;
  });
  const [isInterviewStarted, setIsInterviewStarted] = useState(() => {
    return sessionStorage.getItem('isInterviewStarted') === 'true';
  });
  const [interviewQuestions, setInterviewQuestions] = useState(() => {
    const saved = sessionStorage.getItem('interviewQuestions');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(() => {
    const val = sessionStorage.getItem('currentQuestionIndex');
    return val ? parseInt(val) : 0;
  });
  const [isInterviewComplete, setIsInterviewComplete] = useState(() => {
    return sessionStorage.getItem('isInterviewComplete') === 'true';
  });
  const facesCountRef = useRef(0);
  const directFaceCountRef = useRef(0);
  const isPreCheckFailedRef = useRef(false);
  const isCameraCheckPhaseRef = useRef(false);
  const isModalOrDialogActiveRef = useRef(false);
  const referenceFrameRef = useRef(null);
  const maxPixelDifferenceRef = useRef(0);

  // Proctoring state
  const [warningsCount, setWarningsCount] = useState(0);
  const proctoringEvents = useRef([]);

  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const streamRef = useRef(null);
  const screenStreamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const objectDetectorRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastInferenceTimeRef = useRef(0);
  const lastWarningTimeRef = useRef(0);
  // Different signals need independent cooldowns. A face alert must never
  // suppress a sustained off-screen gaze or a possible second speaker.
  const eventCooldownsRef = useRef({});

  // Audio analysis refs
  const audioContextRef = useRef(null);
  const audioAnalyserRef = useRef(null);
  const audioDataArrayRef = useRef(null);
  const recognitionRef = useRef(null);
  const shouldListenRef = useRef(false);
  const isHumanSpeechDetectedRef = useRef(false);
  const backgroundVoiceCountRef = useRef(0);
  // Speech recognition cannot identify who spoke. Accept a transcript only
  // after a short-lived audiovisual confidence check that the visible candidate was speaking.
  const lastMouthMotionAtRef = useRef(0);
  const candidateSpeechVerifiedRef = useRef(false);
  const unverifiedSpeechFramesRef = useRef(0);
  const lastUnverifiedSpeechAlertRef = useRef(0);
  const noiseFloorRef = useRef(0.008);
  const headTurnCountRef = useRef(0);
  const eyesWanderingCountRef = useRef(0);
  const majorPanFramesRef = useRef(0);

  const timerRef = useRef(null);
  const cameraTimerRef = useRef(null);
  const isHardwareMutedRef = useRef(false);

  const currentUtteranceRef = useRef(null);
  const ttsSafetyTimeoutRef = useRef(null);
  const isSubmittingRef = useRef(false);

  useEffect(() => {
    isSubmittingRef.current = isSubmitting;
  }, [isSubmitting]);

  // Clean Web Speech API Speech Recognition Initializer & Auto-Restarter
  const startOrRestartSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    if (recognitionRef.current) {
      try {
        shouldListenRef.current = false;
        recognitionRef.current.onend = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.stop();
      } catch (e) {}
      recognitionRef.current = null;
    }

    if (window.speechTimeout) clearTimeout(window.speechTimeout);
    shouldListenRef.current = true;

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        // If AI is currently actively speaking TTS, skip to prevent echo
        if (window.isAgentSpeaking && isAiSpeaking) return;

        isHumanSpeechDetectedRef.current = true;
        const recentlyVerified = candidateSpeechVerifiedRef.current &&
          performance.now() - lastMouthMotionAtRef.current < 900;

        // Do not turn a nearby person's voice, a speaker, or an assistant into
        // the candidate's answer. The incident remains available for HR audit.
        if (!recentlyVerified) {
          unverifiedSpeechFramesRef.current += 1;
          if (unverifiedSpeechFramesRef.current >= 2 &&
              performance.now() - lastUnverifiedSpeechAlertRef.current > 6000 &&
              !isCameraCheckPhaseRef.current) {
            recordProctoringEvent('UNVERIFIED_SPEECH_DETECTED');
            lastUnverifiedSpeechAlertRef.current = performance.now();
            unverifiedSpeechFramesRef.current = 0;
          }
          // Do not leave the VAD in a speaking state after rejected speech;
          // rejected speech must never trigger the silence auto-submit path.
          window.setTimeout(() => { isHumanSpeechDetectedRef.current = false; }, 1200);
          return;
        }

        unverifiedSpeechFramesRef.current = 0;
        let finalTranscript = '';
        let interimTranscript = '';
        // Process only the newly delivered recognition results. Re-reading all
        // prior results duplicates text and makes transcript injection easier.
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript + ' ';
          else interimTranscript += event.results[i][0].transcript;
        }

        const fullText = (finalTranscript + interimTranscript).trim();
        if (fullText) {
          const mergedText = `${answerTextRef.current ? `${answerTextRef.current} ` : ''}${fullText}`.trim();
          setAnswerText(mergedText);
          answerTextRef.current = mergedText;
        }

        // Extended silence timeout: 12 seconds of complete silence after candidate speaks
        if (window.speechTimeout) clearTimeout(window.speechTimeout);
        window.speechTimeout = setTimeout(() => {
          isHumanSpeechDetectedRef.current = false;
          if (!isCameraCheckPhaseRef.current && window.submitAnswerFn && answerTextRef.current && answerTextRef.current.trim().length > 15 && !isSubmittingRef.current) {
            if (!isHardwareMutedRef.current) {
              window.submitAnswerFn(false);
            }
          }
        }, 12000);
      };

      recognition.onerror = (e) => {
        if (e.error !== 'no-speech' && e.error !== 'aborted') {
          console.log('Speech recognition error:', e.error);
        }
        if (e.error === 'network' && shouldListenRef.current && !window.isAgentSpeaking) {
          setTimeout(() => {
            if (shouldListenRef.current && !window.isAgentSpeaking) {
              startOrRestartSpeechRecognition();
            }
          }, 400);
        }
      };

      recognition.onend = () => {
        if (shouldListenRef.current && streamRef.current && !isSubmittingRef.current && !window.isAgentSpeaking) {
          setTimeout(() => {
            try {
              if (shouldListenRef.current && !window.isAgentSpeaking && recognitionRef.current === recognition) {
                recognition.start();
              }
            } catch (e) {}
          }, 150);
        }
      };

      recognition.start();
      recognitionRef.current = recognition;
    } catch (e) {
      console.log('Speech recognition initialization error:', e);
    }
  };

  // Text to Speech Function with Natural Voice Selection & Anti-Echo Pause
  const speakText = (text, onComplete) => {
    if (!('speechSynthesis' in window)) {
      window.isAgentSpeaking = false;
      setIsAiSpeaking(false);
      shouldListenRef.current = true;
      startOrRestartSpeechRecognition();
      if (onComplete) onComplete();
      return;
    }

    window.speechSynthesis.cancel();
    if (ttsSafetyTimeoutRef.current) clearTimeout(ttsSafetyTimeoutRef.current);

    // Pause mic while AI speaks to prevent echo self-listening
    shouldListenRef.current = false;
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
    }

    const cleanText = String(text || '').replace(/[#*_`]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    currentUtteranceRef.current = utterance; // Keep ref to prevent Chrome Garbage Collection bug!

    utterance.rate = 0.98;
    utterance.pitch = 1.0;

    // Pick natural voice if available
    const voices = window.speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => 
      (v.lang.includes('en-US') || v.lang.includes('en-GB')) && 
      (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Alex'))
    ) || voices.find(v => v.lang.includes('en'));

    if (naturalVoice) {
      utterance.voice = naturalVoice;
    }

    const handleSpeechEnd = () => {
      if (ttsSafetyTimeoutRef.current) clearTimeout(ttsSafetyTimeoutRef.current);
      window.isAgentSpeaking = false;
      setIsAiSpeaking(false);
      shouldListenRef.current = true;
      startOrRestartSpeechRecognition();
      if (onComplete) onComplete();
    };

    utterance.onstart = () => {
      window.isAgentSpeaking = true;
      setIsAiSpeaking(true);
      shouldListenRef.current = false;
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (e) {}
      }

      // Safety timeout: Chrome onend can fail to fire for long strings. Force unblock mic after speech duration.
      const safetyMs = Math.min(Math.max(cleanText.length * 90, 4000), 10000);
      ttsSafetyTimeoutRef.current = setTimeout(() => {
        console.log("TTS Safety timeout fired. Unblocking candidate mic...");
        handleSpeechEnd();
      }, safetyMs);
    };

    utterance.onend = handleSpeechEnd;
    utterance.onerror = handleSpeechEnd;

    try {
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.log("TTS speak error:", e);
      handleSpeechEnd();
    }
  };

  // Camera verification deliberately has no keyboard bypass. A client-side
  // shortcut is discoverable and invalidates the proctoring evidence.

  // Check for URL token on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    const isDirectQA = import.meta.env.DEV && (directQA || params.get('qa') === 'true' || params.get('mode') === 'qa' || params.get('page') === 'qa' || params.get('directQA') === 'true');

    if (isDirectQA) {
      const mockCandidate = { name: "Test Candidate", skillsFound: ["Python", "React", "System Design"], matchPercentage: 100 };
      setEvaluation(mockCandidate);
      setAppState('interview');
      setIsCameraCheckConfirmed(true);
      setIsCameraCheckCompleted(true);
      setIsTokenValidating(false);
      // Must init camera+mic before starting Q&A — do it async after React renders
      setTimeout(async () => {
        try {
          await initializeCameraSystem(true);
        } catch(e) {
          console.warn('Camera init failed in QA mode, continuing anyway:', e);
        }
        startInterviewQuestions();
      }, 500);
      return;
    }
    
    if (token) {
      if (validatedTokens.has(token)) {
         return;
      }
      validatedTokens.add(token);
      setInviteToken(token);
      // Validate token
      fetch(`${apiUrl(`/api/validate-token/${token}`)}`)
        .then(res => res.json())
        .then(data => {
          if (data.valid) {
            setEvaluation(data.candidate);
            const lastState = sessionStorage.getItem('lastAppState_' + token);
            const restoredState = lastState && lastState !== 'idle' ? lastState : 'approved';
            setAppState(restoredState);

            const pendingRefresh = sessionStorage.getItem('pendingRefresh_' + token) === 'true';
            if (pendingRefresh) {
              sessionStorage.removeItem('pendingRefresh_' + token);
              setCustomWarning({
                show: true,
                message: "You refreshed the page during an active session. Click 'Continue Interview' to stay and continue your interview, or 'Expire Link & Exit' to end your session."
              });
            }
          } else {
            setErrorMessage(data.reason || 'Invalid or expired interview link.');
            setAppState('expired');
          }
          setIsTokenValidating(false);
        })
        .catch(err => {
          console.error(err);
          setIsTokenValidating(false);
        });
    } else {
      setIsTokenValidating(false);
    }
  }, []);

  useEffect(() => {
    isModalOrDialogActiveRef.current = customWarning.show;
  }, [customWarning.show]);

  useEffect(() => {
    if (evaluation) {
      sessionStorage.setItem('evaluation', JSON.stringify(evaluation));
    }
  }, [evaluation]);

  useEffect(() => {
    if (inviteToken && appState !== 'idle' && appState !== 'expired') {
      sessionStorage.setItem('lastAppState_' + inviteToken, appState);
    }
  }, [appState, inviteToken]);

  useEffect(() => {
    sessionStorage.setItem('isCameraCheckConfirmed', isCameraCheckConfirmed);
    sessionStorage.setItem('isCameraCheckCompleted', isCameraCheckCompleted);
    sessionStorage.setItem('isPreCheckFailed', isPreCheckFailed);
    sessionStorage.setItem('showCamera', showCamera);
    sessionStorage.setItem('disclaimerTimeLeft', disclaimerTimeLeft);
    sessionStorage.setItem('isInterviewStarted', isInterviewStarted);
    sessionStorage.setItem('interviewQuestions', JSON.stringify(interviewQuestions));
    sessionStorage.setItem('currentQuestionIndex', currentQuestionIndex);
    sessionStorage.setItem('isInterviewComplete', isInterviewComplete);
  }, [isCameraCheckConfirmed, isCameraCheckCompleted, isPreCheckFailed, showCamera, disclaimerTimeLeft, isInterviewStarted, interviewQuestions, currentQuestionIndex, isInterviewComplete]);

  useEffect(() => {
    if (appState === 'interview' && !isCameraCheckConfirmed && disclaimerTimeLeft > 0) {
      if (disclaimerTimeLeft === 15) {
        speakText("Attention candidate: This entire interview session is continuously monitored and recorded in full by AI proctoring. Any secondary devices, unauthorized assistance, or window switching is strictly prohibited.");
      }
      const timer = setTimeout(() => {
        setDisclaimerTimeLeft(prev => prev - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [appState, isCameraCheckConfirmed, disclaimerTimeLeft]);

  // Auto-return home after success
  useEffect(() => {
    if (appState === 'bulkSuccess') {
      const timer = setTimeout(() => {
        setAppState('idle');
        setFile(null);
        window.bulkCandidates = null;
        setGeneratedLinks([]);
        setBulkSuccessMsg("");
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [appState]);

  // Browser Refresh & Close Protection (Seamless background state preservation without popup modal)
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (inviteToken && appState !== 'expired' && appState !== 'idle' && appState !== 'finished') {
        sessionStorage.setItem('pendingRefresh_' + inviteToken, 'true');
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [appState, inviteToken, isInterviewStarted]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && appState === 'interview') {
        setWarningsCount(prev => prev + 1);
        proctoringEvents.current.push({
          timestamp: new Date().toISOString(),
          type: "TAB_SWITCH"
        });
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [appState, isCameraCheckCompleted]);

  // Fullscreen and Screen Share strict security
  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement && appState === 'interview' && isCameraCheckCompleted) {
         setWarningsCount(prev => prev + 1);
         proctoringEvents.current.push({
           timestamp: new Date().toISOString(),
           type: "EXITED_FULLSCREEN"
         });
         alert("SECURITY WARNING: You exited full-screen mode. This is strictly prohibited.");
      }
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [appState, isCameraCheckCompleted]);

  // Copy/Paste and Hotkey Blocking
  useEffect(() => {
    const blockAction = (e) => {
      e.preventDefault();
    };

    const handleKeyDown = (e) => {
      // Intercept F5 and Ctrl+R to trigger custom UI warning popup
      if (e.key === 'F5' || (e.ctrlKey && (e.key === 'r' || e.key === 'R'))) {
        e.preventDefault();
        setCustomWarning({
          show: true,
          message: "If you refresh or leave this page, your interview link will expire. Click 'Continue Interview' to stay and continue your interview."
        });
        return;
      }
      // Block Copy/Paste
      if ((e.ctrlKey || e.metaKey) && (e.key === 'c' || e.key === 'v' || e.key === 'x')) {
        blockAction(e);
        alert("SECURITY WARNING: Copy/Paste is strictly disabled.");
      }
      // Block F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U (Developer Tools)
      if (
        e.key === 'F12' || 
        (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'i' || e.key === 'J' || e.key === 'j' || e.key === 'C' || e.key === 'c')) || 
        (e.ctrlKey && (e.key === 'U' || e.key === 'u'))
      ) {
        blockAction(e);
        alert("SECURITY WARNING: Developer Tools and Source Code viewing are strictly disabled.");
      }
    };

    if (appState === 'interview' || appState === 'approved' || appState === 'dashboard' || isInterviewStarted || (inviteToken && appState !== 'expired' && appState !== 'idle')) {
      window.addEventListener('copy', blockAction);
      window.addEventListener('paste', blockAction);
      window.addEventListener('cut', blockAction);
      window.addEventListener('contextmenu', blockAction);
      window.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      window.removeEventListener('copy', blockAction);
      window.removeEventListener('paste', blockAction);
      window.removeEventListener('cut', blockAction);
      window.removeEventListener('contextmenu', blockAction);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [appState, inviteToken, isInterviewStarted]);

  useEffect(() => {
    if (appState === 'interview' && isCameraCheckConfirmed && !isCameraCheckCompleted) {
      speakText("Scan in progress. Please rotate your camera 360 degrees now to show your surroundings. We must verify you are alone in the room.");
    }
  }, [appState, isCameraCheckConfirmed, isCameraCheckCompleted]);

  // Scanning Phase 360° Verification & Driver
  useEffect(() => {
    let scanInterval;
    if (appState === 'interview' && isCameraCheckConfirmed && !isCameraCheckCompleted && !customWarning.show) {
      scanInterval = setInterval(() => {
        captureFrameAndCompare();

        // 360° Active Camera Panning Check
        // Active rotation requires frame movement (avgDiff >= 3.5).
        // Note: Even if multiple people are detected, DO NOT freeze countdown or lock the candidate.
        // Warn the candidate, log the security event for HR review, and allow progression to the next level.
        const isPanningRoom = lastFrameDiffRef.current >= 3.5;

        if (!isPanningRoom) {
          // Camera stationary or repeated frame → PAUSE COUNTDOWN until rotated
          setIsScanPaused(true);
          return;
        }

        // Camera actively rotating → RESUME COUNTDOWN
        setIsScanPaused(false);

        setCameraTimeLeft((prev) => {
          if (prev <= 1) {
            // Time passing alone is not evidence of a room scan. Require enough
            // distinct views and no additional person/device before unlocking.
            if (seenKeyframesRef.current.length >= 8 && !hasAdditionalPersonRef.current) {
              if (scanInterval) clearInterval(scanInterval);
              void finalizeRoomVerification();
              return 0;
            }
            setIsScanPaused(true);
            recordProctoringEvent('ROOM_SCAN_INCOMPLETE');
            return 0;
          }
          return Math.max(0, prev - 1);
        });
      }, 1000);
    }

    return () => {
      if (scanInterval) clearInterval(scanInterval);
    };
  }, [appState, isCameraCheckConfirmed, isCameraCheckCompleted, customWarning.show, inviteToken]);

  // Bulletproof Video Stream & Camera Sync
  useEffect(() => {
    let intervalId;
    if (appState === 'interview' && isCameraCheckConfirmed) {
      if (!streamRef.current || !streamRef.current.active) {
        initializeCameraSystem(isCameraCheckCompleted).catch(err => {
          console.error("Auto-restore camera failed:", err);
        });
      }

      intervalId = setInterval(() => {
        if (videoRef.current && streamRef.current && streamRef.current.active) {
          if (videoRef.current.srcObject !== streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
            videoRef.current.play().catch(() => {});
          }
        }
      }, 300);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [appState, isCameraCheckConfirmed, isCameraCheckCompleted, showCamera]);

  // Military-Grade Fullscreen & Keyboard Lock Engine (Locks Fullscreen & Blocks Off-Screen Access)
  useEffect(() => {
    if (appState === 'interview' && !isInterviewComplete) {
      const lockFullscreenAndKeyboard = async () => {
        try {
          if (!document.fullscreenElement) {
            await document.documentElement.requestFullscreen();
          }
          // Lock Keyboard API (Blocks Escape, Alt+Tab, and Windows system keys)
          if (navigator.keyboard && typeof navigator.keyboard.lock === 'function') {
            await navigator.keyboard.lock(["Escape", "AltLeft", "AltRight", "Tab", "MetaLeft", "MetaRight"]);
          }
        } catch (err) {
          console.warn("Fullscreen/Keyboard Lock request:", err);
        }
      };

      lockFullscreenAndKeyboard();

      // Continuous 500ms Lockdown Monitor
      const lockInterval = setInterval(() => {
        if (!document.fullscreenElement && appState === 'interview' && !isInterviewComplete) {
          lockFullscreenAndKeyboard();
        }
      }, 500);

      // Mouse Off-Screen & Focus Lock
      const handleMouseLeave = () => {
        if (appState === 'interview' && !isInterviewComplete) {
          recordProctoringEvent("MOUSE_OFFSCREEN_ATTEMPT");
          setWarningsCount(prev => prev + 1);
          lockFullscreenAndKeyboard();
        }
      };

      const handleFullscreenChange = () => {
        if (!document.fullscreenElement && appState === 'interview' && !isInterviewComplete) {
          recordProctoringEvent("FULLSCREEN_EXIT_ATTEMPT");
          setWarningsCount(prev => prev + 1);
          lockFullscreenAndKeyboard();
        }
      };

      document.addEventListener("mouseleave", handleMouseLeave);
      document.addEventListener("fullscreenchange", handleFullscreenChange);
      document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.addEventListener("mozfullscreenchange", handleFullscreenChange);
      document.addEventListener("MSFullscreenChange", handleFullscreenChange);

      return () => {
        clearInterval(lockInterval);
        document.removeEventListener("mouseleave", handleMouseLeave);
        document.removeEventListener("fullscreenchange", handleFullscreenChange);
        document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
        document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
        document.removeEventListener("MSFullscreenChange", handleFullscreenChange);
        if (navigator.keyboard && typeof navigator.keyboard.unlock === 'function') {
          navigator.keyboard.unlock();
        }
      };
    }
  }, [appState, isInterviewComplete]);

  useEffect(() => {
    if (currentQuestion && isCameraCheckCompleted && appState === 'interview') {
      const qText = getQuestionTextString(currentQuestion);
      answerTextRef.current = "";
      setAnswerText("");
      window.submitAnswerFn = submitAnswer;
      if (qText) speakText(qText);
    }
  }, [currentQuestion, isCameraCheckCompleted, appState]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSingleInvite = async (candidate) => {
    const finalEmail = candidate.email ? candidate.email.replace('email:', '') : null;
    if (!finalEmail) {
      alert(`Cannot send invite: No email address found for ${candidate.filename}.`);
      return;
    }
    
    const candidateData = {
      ...candidate,
      email: finalEmail
    };
    try {
      const res = await fetch(apiUrl('/api/hr/send-invites'), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ candidates: [candidateData], expiry_hours: 48 })
      });
      const data = await res.json();
      if (data.status === 'success') {
         setGeneratedLinks(prev => [...prev, ...data.invited]);
         setBulkSuccessMsg(`✅ Successfully sent interview invitation to ${candidateData.email}!`);
         setAppState('bulkSuccess');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to send invite');
    }
  };

  const handleSendInvites = async () => {
    let shortlisted = window.bulkCandidates.filter(c => c.isMatch);
    
    if (shortlisted.length === 0) return alert('No shortlisted candidates found.');
    
    const initialCount = shortlisted.length;
    
    const processedShortlisted = shortlisted.map(c => ({
      ...c,
      email: c.email ? c.email.replace('email:', '') : null
    }));
    
    const validShortlisted = processedShortlisted.filter(c => c.email);
    const missingEmailFilenames = processedShortlisted.filter(c => !c.email).map(c => c.filename);

    if (validShortlisted.length === 0) return alert('No candidates have valid email addresses to send invites to.');
    
    const skippedCount = missingEmailFilenames.length;
    
    setIsSendingBulk(true);
    try {
      const res = await fetch(apiUrl('/api/hr/send-invites'), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ candidates: validShortlisted, expiry_hours: 48 })
      });
      const data = await res.json();
      if (data.status === 'success') {
         setGeneratedLinks(data.invited);
         const emails = data.invited.map(i => i.email).join(', ');
         let successMsg = `✅ Successfully sent interview invitations to ${data.invited_count} candidates:\n\n${emails}`;
         if (skippedCount > 0) {
             successMsg += `\n\n⚠️ Skipped ${skippedCount} candidate(s) because no email address was found in their resume:\n- ${missingEmailFilenames.join('\n- ')}`;
         }
         setBulkSuccessMsg(successMsg);
         setAppState('bulkSuccess');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to send invites');
    } finally {
      setIsSendingBulk(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setAppState('uploading');
    
    const formData = new FormData();
    formData.append('file', file); // Use 'file' for bulk-upload or 'resume' for normal, backend expects 'resume' for normal, wait, bulk-upload expects 'file'

    try {
      const isZip = file.name.toLowerCase().endsWith('.zip');
      const endpoint = isZip ? apiUrl('/api/bulk-upload') : apiUrl('/api/upload-resume');
      
      // The backend expects 'resume' for single upload, and 'file' for bulk upload
      const formData = new FormData();
      if (isZip) {
          formData.append('file', file);
      } else {
          formData.append('resume', file);
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      
      if (data.isBulk) {
          setAppState('dashboard');
          // Temporarily store candidates on window for the dashboard to render (hack for quick patch)
          window.bulkCandidates = data.candidates;
          return;
      }


      if (data.error) {
        alert(data.error);
        setAppState('idle');
        return;
      }

      setEvaluation(data);

      if (data.isMatch) {
        setAppState('approved');
      } else {
        setAppState('rejected');
      }
    } catch (error) {
      console.error(error);
      setAppState('idle');
      alert('Error evaluating resume. Please try again or check if backend is running.');
    }
  };

  const recordProctoringEvent = (type) => {
    const eventObj = { type, timestamp: new Date().toISOString() };
    proctoringEvents.current.push(eventObj);

    // Send real-time event report to backend for HR audit logging
    const activeToken = inviteToken || new URLSearchParams(window.location.search).get('token');
    if (activeToken) {
      fetch(`${apiUrl(`/api/log-proctoring-event/${activeToken}`)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(eventObj)
      }).catch(() => {});
    }

    // Audio and haptic warnings
    let warningMessage = "Warning.";
    if (type === "FACE_NOT_DETECTED") warningMessage = "Warning. Face not detected.";
    else if (type === "360_SCAN_ADDITIONAL_PERSON_DETECTED") warningMessage = "Security Warning. Additional person detected during 360 room scan. Security violation flagged for HR review.";
    else if (type === "MULTIPLE_FACES") warningMessage = "Warning. Multiple faces detected in frame.";
    else if (type === "FACE_OUT_OF_FRAME") warningMessage = "Warning. Please look at the screen.";
    else if (type === "TAB_SWITCH") warningMessage = "Warning. Do not switch tabs.";
    else if (type === "WINDOW_BLUR") warningMessage = "Warning. Keep the interview window in focus.";
    else if (type === "BACKGROUND_VOICE" || type === "POSSIBLE_SECOND_SPEAKER") warningMessage = "Security warning. Speech was detected without matching candidate lip movement. Remain alone and do not use external assistance.";
    else if (type === "UNVERIFIED_SPEECH_DETECTED") warningMessage = "Security warning. Speech was detected without matching candidate mouth movement. Only your own spoken response may be recorded.";
    else if (type === "EXTRA_HANDS") warningMessage = "Warning. Another person's hand detected in the frame. You must take this interview alone.";
    else if (type === "EYES_WANDERING") warningMessage = "Warning. Please keep your eyes focused directly on the screen.";
    else if (type === "HEAD_TURNED") warningMessage = "Warning. Head movement detected. You must look straight at the camera.";
    else if (type === "FORBIDDEN_OBJECT_DETECTED") warningMessage = "Warning. Forbidden object detected. Put away any phones, books, or external devices.";

    speakText(warningMessage);
    if (navigator.vibrate) navigator.vibrate([200, 100, 200]);

    // Visual warning
    setIsWarningBlinking(true);
    setTimeout(() => setIsWarningBlinking(false), 1500);
  };

  const canRaiseSecurityAlert = (type, cooldownMs = 8000) => {
    const now = performance.now();
    const previous = eventCooldownsRef.current[type] || 0;
    if (now - previous < cooldownMs) return false;
    eventCooldownsRef.current[type] = now;
    return true;
  };

  const finalizeRoomVerification = async () => {
    // Never let local/session storage unlock an interview. The server must
    // acknowledge the verification event before questions are made available.
    const activeToken = inviteToken || new URLSearchParams(window.location.search).get('token');
    if (!activeToken) {
      setCustomWarning({ show: true, message: 'A valid interview invitation is required to complete room verification.' });
      return false;
    }
    try {
      const response = await fetch(apiUrl(`/api/verify-360-scan/${activeToken}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proctoringEvents: proctoringEvents.current,
          coveragePercent: scanCoverageRef.current,
          distinctViews: seenKeyframesRef.current.length,
          additionalPersonDetected: hasAdditionalPersonRef.current
        })
      });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.error || 'Verification was not accepted');
      setIs360ServerVerified(true);
      setIsCameraCheckCompleted(true);
      isCameraCheckPhaseRef.current = false;
      setIsScanPaused(false);
      sessionStorage.setItem('wasScanning', 'false');
      sessionStorage.setItem('isCameraCheckCompleted', 'true');
      return true;
    } catch (error) {
      setCustomWarning({ show: true, message: 'Room verification could not be saved. Check your connection and try again.' });
      return false;
    }
  };

  const predictWebcam = () => {
    if (!videoRef.current || !landmarkerRef.current || videoRef.current.readyState < 2) {
      animationRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    let startTimeMs = performance.now();
    if (startTimeMs - lastInferenceTimeRef.current < 150) {
      animationRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    if (lastVideoTimeRef.current !== videoRef.current.currentTime) {
      lastVideoTimeRef.current = videoRef.current.currentTime;

      const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
      const objectResults = objectDetectorRef.current ? objectDetectorRef.current.detectForVideo(videoRef.current, startTimeMs) : null;
      const handResults = handLandmarkerRef.current ? handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs) : null;

      // Filter out 2D wall paintings, background posters, drawings, and small background artwork
      const filterValidHumanFaces = (faceLandmarksList) => {
        if (!faceLandmarksList || !Array.isArray(faceLandmarksList)) return [];
        return faceLandmarksList.filter(landmarks => {
          if (!landmarks || landmarks.length === 0) return false;
          let minX = 1, maxX = 0, minY = 1, maxY = 0, minZ = Infinity, maxZ = -Infinity;
          for (let i = 0; i < landmarks.length; i++) {
            const pt = landmarks[i];
            if (pt.x < minX) minX = pt.x;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.y > maxY) maxY = pt.y;
            if (pt.z < minZ) minZ = pt.z;
            if (pt.z > maxZ) maxZ = pt.z;
          }
          const width = maxX - minX;
          const height = maxY - minY;
          const area = width * height;
          const zDepth = maxZ - minZ;

          // Filter 1: Face Size/Scale - Real candidate/person faces occupy substantial frame area
          // Background wall drawings/paintings (like small cartoon illustrations) are tiny relative to frame
          if (width < 0.07 || height < 0.07 || area < 0.005) {
            return false;
          }

          // Filter 2: 3D Facial Depth (Z-variance vs width) - 2D wall posters/paintings are flat
          if (zDepth < 0.012 * width || zDepth < 0.002) {
            return false;
          }

          return true;
        });
      };

      const validHumanFaces = filterValidHumanFaces(results.faceLandmarks);
      const currentHands = handResults && handResults.landmarks ? handResults.landmarks.length : 0;
      const currentFaces = validHumanFaces.length;
      directFaceCountRef.current = currentFaces;
      let totalPeople = currentFaces;

      if (objectResults && objectResults.detections) {
        const persons = objectResults.detections.filter(d => {
          const cat = d.categories[0];
          if (cat.categoryName !== "person") return false;
          if (cat.score < 0.68) return false; // Ignore low confidence detections (paintings, cartoons)
          const bbox = d.boundingBox;
          if (bbox && videoRef.current && videoRef.current.videoWidth) {
            const normW = bbox.width / videoRef.current.videoWidth;
            const normH = bbox.height / videoRef.current.videoHeight;
            if (normW * normH < 0.035) return false; // Ignore small poster drawings
          }
          return true;
        });
        totalPeople = Math.max(currentFaces, persons.length);

        const nowObj = performance.now();
        const forbiddenObjects = objectResults.detections.filter(d => ['cell phone', 'laptop', 'book', 'tablet'].includes(d.categories[0].categoryName));
        if (forbiddenObjects.length > 0 && nowObj - lastWarningTimeRef.current > 3000) {
           setWarningsCount(prev => prev + 1);
           recordProctoringEvent("FORBIDDEN_OBJECT_DETECTED");
           lastWarningTimeRef.current = nowObj;
        }
      }

      if (facesCountRef.current !== totalPeople) {
        facesCountRef.current = totalPeople;
        setDetectedFacesCount(totalPeople);
      }

      const now = performance.now();
      if (now - lastWarningTimeRef.current > 2500) {
        if (currentHands > 2) {
          hasAdditionalPersonRef.current = true;
          setHasAdditionalPersonDetected(true);
          setWarningsCount(prev => prev + 1);
          if (isCameraCheckPhaseRef.current) {
            recordProctoringEvent("360_SCAN_ADDITIONAL_PERSON_DETECTED");
          } else {
            recordProctoringEvent("EXTRA_HANDS");
          }
          lastWarningTimeRef.current = now;
        } else if (totalPeople === 0) {
          // During the 360° environment scan phase (isCameraCheckPhaseRef.current = true),
          // face being absent is EXPECTED and REQUIRED — candidate is rotating camera away.
          // Do NOT mark as failed when face is gone during scan.
          if (!isCameraCheckPhaseRef.current) {
            setWarningsCount(prev => prev + 1);
            recordProctoringEvent("FACE_NOT_DETECTED");
            lastWarningTimeRef.current = now;
          }
        } else if (totalPeople > 1) {
          hasAdditionalPersonRef.current = true;
          setHasAdditionalPersonDetected(true);
          setWarningsCount(prev => prev + 1);
          if (isCameraCheckPhaseRef.current) {
            recordProctoringEvent("360_SCAN_ADDITIONAL_PERSON_DETECTED");
          } else {
            recordProctoringEvent("MULTIPLE_FACES");
          }
          lastWarningTimeRef.current = now;
        } else if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          // Basic out of frame check (nose too close to extreme edges)
          const nose = results.faceLandmarks[0][1];
          if (nose.x < 0.03 || nose.x > 0.97 || nose.y < 0.03 || nose.y > 0.97) {
            setWarningsCount(prev => prev + 1);
            recordProctoringEvent("FACE_OUT_OF_FRAME");
            lastWarningTimeRef.current = now;
          }
        }
      }

      // Audio VAD + lip-sync correlation. SpeechRecognition alone is not a
      // security signal: it transcribes any nearby voice or loudspeaker.
      let isSpeakingAudio = isHumanSpeechDetectedRef.current;
      if (audioAnalyserRef.current && audioDataArrayRef.current) {
        const samples = new Uint8Array(audioAnalyserRef.current.fftSize);
        audioAnalyserRef.current.getByteTimeDomainData(samples);
        let energy = 0;
        for (const sample of samples) {
          const normalized = (sample - 128) / 128;
          energy += normalized * normalized;
        }
        const rms = Math.sqrt(energy / samples.length);
        // Adapt only while the room is quiet so a speaking voice cannot train
        // itself into the ambient baseline.
        if (rms < noiseFloorRef.current * 1.6) noiseFloorRef.current = noiseFloorRef.current * 0.96 + rms * 0.04;
        audioAnalyserRef.current.getByteFrequencyData(audioDataArrayRef.current);
        let voiceBand = 0;
        for (let i = 3; i < 20 && i < audioDataArrayRef.current.length; i++) voiceBand += audioDataArrayRef.current[i];
        voiceBand /= 17;
        isSpeakingAudio = isSpeakingAudio || (rms > Math.max(noiseFloorRef.current * 2.6, 0.018) && voiceBand > 18);

        let highFreqAverage = 0;
        for (let i = 50; i < 150 && i < audioDataArrayRef.current.length; i++) highFreqAverage += audioDataArrayRef.current[i];
        highFreqAverage /= 100;
        if (highFreqAverage > 45 && !isSpeakingAudio && currentQuestionIndex >= 0) recordProctoringEvent("KEYBOARD_TYPING_DETECTED");
      }

      let isMouthMoving = false;
      if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        const jawOpen = shapes.find(c => c.categoryName === 'jawOpen');
        // Require more significant mouth movement to suppress background noise warning
        if (jawOpen && jawOpen.score > 0.15) {
          isMouthMoving = true;
          lastMouthMotionAtRef.current = performance.now();
          candidateSpeechVerifiedRef.current = true;
        }

        // ENTERPRISE STRICT DUAL-MODE EYE TRACKING (Blendshapes + Iris Pupil Landmarks)
        const eyeKeys = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'];
        let isEyesWandering = eyeKeys.some(key => {
          const shape = shapes.find(c => c.categoryName === key);
          return shape && shape.score > 0.68; // Alert only after sustained deviation below.
        });

        // Iris Pupil Landmark Position Tracking (Gaze Vector)
        if (!isEyesWandering && results.faceLandmarks && results.faceLandmarks.length > 0) {
          const landmarks = results.faceLandmarks[0];
          if (landmarks && landmarks.length > 473) {
            const leftPupil = landmarks[468];
            const leftCornerInner = landmarks[133];
            const leftCornerOuter = landmarks[33];
            if (leftPupil && leftCornerInner && leftCornerOuter) {
              const eyeWidth = Math.abs(leftCornerOuter.x - leftCornerInner.x);
              if (eyeWidth > 0) {
                const pupilRatio = Math.abs(leftPupil.x - leftCornerInner.x) / eyeWidth;
                // Pupil deviation outside true off-screen bounds (< 0.05 or > 0.95)
                if (pupilRatio < 0.12 || pupilRatio > 0.88) {
                  isEyesWandering = true;
                }
              }
            }
          }
        }

        if (isEyesWandering) {
          eyesWanderingCountRef.current += 1;
        } else {
          eyesWanderingCountRef.current = Math.max(0, eyesWanderingCountRef.current - 3); // Quick decay on natural gaze
        }
      }

      // HIGH-PRECISION HEAD TURN TRACKING
      let isHeadTurned = false;
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Head tracking based on relative position of nose to ears
        const nose = results.faceLandmarks[0][1];
        const leftEar = results.faceLandmarks[0][234];
        const rightEar = results.faceLandmarks[0][454];

        if (nose && leftEar && rightEar) {
          const faceWidth = Math.abs(rightEar.x - leftEar.x);
          const faceCenter = (leftEar.x + rightEar.x) / 2;
          // Normalise the nose displacement against face width. This is less
          // sensitive to webcam position than a raw left/right distance ratio.
          const yawOffset = faceWidth > 0 ? Math.abs(nose.x - faceCenter) / faceWidth : 0;
          if (yawOffset > 0.24) isHeadTurned = true;
        }
      }

      if (isHeadTurned) {
        headTurnCountRef.current += 1;
      } else {
        headTurnCountRef.current = Math.max(0, headTurnCountRef.current - 3);
      }

      const hasRecentCandidateLipMotion = candidateSpeechVerifiedRef.current &&
        now - lastMouthMotionAtRef.current < 650;
      if (isSpeakingAudio && !hasRecentCandidateLipMotion && !window.isAgentSpeaking) {
        backgroundVoiceCountRef.current += 1;
      } else {
        backgroundVoiceCountRef.current = Math.max(0, backgroundVoiceCountRef.current - 2); // Decay quickly when noise stops
      }

      // A sustained audio stream without the candidate's visible lip movement
      // is a possible second speaker or external audio source. It is an audit
      // signal, not a voice-identity decision.
      if (backgroundVoiceCountRef.current > 18 && canRaiseSecurityAlert('POSSIBLE_SECOND_SPEAKER')) {
        if (isCameraCheckPhaseRef.current) {
          isPreCheckFailedRef.current = "Environment scan failed: speech was detected without a visible speaker.";
        } else {
          setWarningsCount(prev => prev + 1);
          recordProctoringEvent("POSSIBLE_SECOND_SPEAKER");
        }
        backgroundVoiceCountRef.current = 0;
      }

      // At roughly 6–7 checks/second, 28 samples means ~4 seconds of sustained
      // off-screen gaze/head rotation. Independent cooldowns prevent a face or
      // audio event from hiding these warnings.
      if (eyesWanderingCountRef.current >= 28 && canRaiseSecurityAlert('EYES_WANDERING')) {
        setWarningsCount(prev => prev + 1);
        recordProctoringEvent("EYES_WANDERING");
        eyesWanderingCountRef.current = 0;
      }

      if (headTurnCountRef.current >= 28 && canRaiseSecurityAlert('HEAD_TURNED')) {
        setWarningsCount(prev => prev + 1);
        recordProctoringEvent("HEAD_TURNED");
        headTurnCountRef.current = 0;
      }
    }

    lastInferenceTimeRef.current = performance.now();
    setTimeout(() => {
      animationRef.current = requestAnimationFrame(predictWebcam);
    }, 50);
  };

  const initializeCameraSystem = async (force = false) => {
    if (!force && !isCameraCheckCompleted && !isCameraCheckConfirmed && appState !== 'interview') {
      console.warn("Blocked premature camera initialization.");
      return false;
    }
    // If camera and screen share streams are already active, reuse them immediately!
    if (
      streamRef.current && streamRef.current.active && streamRef.current.getVideoTracks().some(t => t.readyState === 'live') &&
      screenStreamRef.current && screenStreamRef.current.active && screenStreamRef.current.getVideoTracks().some(t => t.readyState === 'live')
    ) {
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
      }
      setShowCamera(true);
      return true;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      
      // Clean up previous screen share listeners if replacing
      if (screenStreamRef.current) {
        try {
          screenStreamRef.current.getTracks().forEach(t => {
            t.onended = null;
            t.stop();
          });
        } catch(e) {}
      }

      // Force Screen Sharing (Screen Proctoring) only if not already sharing
      if (!screenStreamRef.current || !screenStreamRef.current.active) {
        try {
          const screenStream = await navigator.mediaDevices.getDisplayMedia({ 
            video: { displaySurface: "monitor" }, 
            audio: false,
            surfaceSwitching: "exclude",
            selfBrowserSurface: "exclude",
            systemAudio: "exclude",
            preferCurrentTab: false
          });

          const videoTrack = screenStream.getVideoTracks()[0];
          const settings = videoTrack.getSettings();

          if (settings.displaySurface && settings.displaySurface !== 'monitor') {
            videoTrack.stop();
            throw new Error("Must share entire screen");
          } else {
            screenStreamRef.current = screenStream;
            videoTrack.onended = () => {
              recordProctoringEvent("SCREEN_SHARE_STOPPED");
            };
          }
        } catch (err) {
          console.warn("Screen share skipped or declined. Continuing with webcam stream.", err);
          isPreCheckFailedRef.current = "Screen sharing is mandatory. Please refresh and allow sharing your entire screen.";
          setIsPreCheckFailed(true);
          return;
        }
      }
      
      // Hardware/System Mute Detection
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.onmute = () => {
          setWarningsCount(prev => prev + 1);
          recordProctoringEvent("HARDWARE_MUTE_DETECTED");
          setShowMuteWarning(true);
          isHardwareMutedRef.current = true;
        };
        audioTrack.onunmute = () => {
          recordProctoringEvent("HARDWARE_UNMUTED");
          setShowMuteWarning(false);
          isHardwareMutedRef.current = false;
        };
      }
      
      try {
          // Record the webcam stream instead of the screen to avoid the massive browser pop-up
          mediaRecorderRef.current = new MediaRecorder(stream);
          mediaRecorderRef.current.ondataavailable = (e) => {
              if (e.data && e.data.size > 0) {
                  recordedChunksRef.current.push(e.data);
              }
          };
          mediaRecorderRef.current.start(1000);
      } catch(err) {
          console.error("Recording failed", err);
      }

      // Initialize Web Audio API for fallback audio monitoring
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      microphone.connect(analyser);
      analyser.fftSize = 256;
      audioContextRef.current = audioContext;
      audioAnalyserRef.current = analyser;
      audioDataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

      // Initialize Web Speech API for True Human Voice Detection
      startOrRestartSpeechRecognition();

      // Wait for React to render the video element
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(e => console.log("Video play error:", e));
          predictWebcam();
        }
      }, 100);
      return true;
    } catch (err) {
      console.error("Error starting interview.", err);
      alert(err.message || "Could not access camera or screen. Please check permissions.");
    }
  };

  const startInterview = async () => {
    // Multi-Monitor Check (Supported in modern Chromium browsers)
    if (window.screen.isExtended) {
       alert("SECURITY WARNING: Multiple monitors detected. You must unplug external monitors to take this interview.");
       return;
    }

    // Force Fullscreen
    try {
      if (document.documentElement.requestFullscreen) {
        await document.documentElement.requestFullscreen();
      }
    } catch (e) {
      console.error("Fullscreen request failed", e);
    }

    sessionStorage.removeItem('isCameraCheckConfirmed');
    sessionStorage.removeItem('isCameraCheckCompleted');
    sessionStorage.removeItem('wasScanning');
    sessionStorage.setItem('disclaimerTimeLeft', '15');

    setAppState('interview');
    setDisclaimerTimeLeft(15);
    setIsCameraCheckConfirmed(false);
    setIsCameraCheckCompleted(false);
    setIsPreCheckFailed(false);
    isPreCheckFailedRef.current = false;
    isCameraCheckPhaseRef.current = false;
    if (isFirstQuestionStartedRef.current !== undefined) isFirstQuestionStartedRef.current = false;
    setWarningsCount(0);
    proctoringEvents.current = [];
    backgroundVoiceCountRef.current = 0;
    setShowCamera(true);
  };

  const captureFrameAndCompare = () => {
    if (!videoRef.current || videoRef.current.readyState < 2) return;
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 48;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(videoRef.current, 0, 0, 64, 48);
    const imageData = ctx.getImageData(0, 0, 64, 48).data;

    // Save baseline frame on first call after scan starts (person should be in frame)
    if (!scanBaselineSavedRef.current) {
      scanBaselineFrameRef.current = new Uint8ClampedArray(imageData);
      scanBaselineSavedRef.current = true;
      referenceFrameRef.current = imageData;
      return;
    }

    if (!referenceFrameRef.current) {
      referenceFrameRef.current = imageData;
      return;
    }

    // ── Metric 1: Frame-to-frame diff (is camera MOVING right now?) ──
    let frameDiff = 0;
    for (let i = 0; i < imageData.length; i += 4) {
      frameDiff += Math.abs(imageData[i]   - referenceFrameRef.current[i]);
      frameDiff += Math.abs(imageData[i+1] - referenceFrameRef.current[i+1]);
      frameDiff += Math.abs(imageData[i+2] - referenceFrameRef.current[i+2]);
    }
    const avgDiff = frameDiff / (64 * 48 * 3);
    lastFrameDiffRef.current = avgDiff;

    // ── Metric 2: Unique Room Scene Coverage Detection (True 360° Visual Progress 0% to 100%) ──
    if (avgDiff >= 3.5) {
      let minDistanceToKnownKeyframe = 999;
      for (const kf of seenKeyframesRef.current) {
        let kfDiff = 0;
        for (let i = 0; i < imageData.length; i += 8) {
          kfDiff += Math.abs(imageData[i] - kf[i]);
        }
        const normKfDiff = kfDiff / ((64 * 48 * 3) / 2);
        if (normKfDiff < minDistanceToKnownKeyframe) {
          minDistanceToKnownKeyframe = normKfDiff;
        }
      }

      if ((seenKeyframesRef.current.length === 0 || minDistanceToKnownKeyframe >= 14.0) && seenKeyframesRef.current.length < 10) {
        seenKeyframesRef.current.push(new Uint8ClampedArray(imageData));
        const newPct = Math.min(100, Math.round((seenKeyframesRef.current.length / 10) * 100));
        scanCoverageRef.current = newPct;
        setScanProgressPercent(newPct);
        sessionStorage.setItem('scanProgressPercent', newPct);
      }
    }

    // Update reference frame for next comparison
    referenceFrameRef.current = imageData;
  };



  const confirmCameraCheck = async () => {
    const success = await initializeCameraSystem(true);
    if (!success) {
      return; // Stay on the current page, do not navigate!
    }

    setIsCameraCheckConfirmed(true);
    isCameraCheckPhaseRef.current = true;
    setIsCameraCheckCompleted(false);
    setCameraTimeLeft(15);
    setIsScanPaused(true);

    // Token access is marked IN_PROGRESS during validation. Do not expire it
    // here: the active candidate still needs to complete the interview.

    sessionStorage.setItem('wasScanning', 'true');
    referenceFrameRef.current = null;
    maxPixelDifferenceRef.current = 0;
    majorPanFramesRef.current = 0;
    // Reset all 360° detection & keyframe trackers for fresh scan
    motionStreakRef.current = 0;
    cumulativeMotionRef.current = 0;
    seenKeyframesRef.current = [];
    hasAdditionalPersonRef.current = false;
    setHasAdditionalPersonDetected(false);
    scanCoverageRef.current = 0;
    setScanProgressPercent(0);
    sessionStorage.setItem('scanProgressPercent', '0');
    
  };

  const startInterviewQuestions = async () => {
    // Add visibility listener
    const handleVisibilityChange = () => {
      if (document.hidden) {
        recordProctoringEvent("TAB_SWITCH");
        setWarningsCount(prev => prev + 1);
      }
    };
    const handleBlur = () => {
      recordProctoringEvent("WINDOW_BLUR");
      setWarningsCount(prev => prev + 1);
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("blur", handleBlur);

    // Keep references to remove later
    window._interviewCleanup = () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("blur", handleBlur);
    };

    try {
      const activeEval = evaluation || (sessionStorage.getItem('evaluation') ? JSON.parse(sessionStorage.getItem('evaluation')) : {});
      const activeToken = inviteToken || new URLSearchParams(window.location.search).get('token');

      // Instant loading from cache if available or set instant default question
      const cachedQ = sessionStorage.getItem('currentQuestion_' + activeToken);
      const cachedSession = sessionStorage.getItem('sessionId_' + activeToken);
      if (cachedQ && cachedSession) {
        setSessionId(cachedSession);
        setCurrentQuestion(cachedQ);
        setQuestionIndex(parseInt(sessionStorage.getItem('questionIndex_' + activeToken) || "0", 10));
        setTotalQuestions(parseInt(sessionStorage.getItem('totalQuestions_' + activeToken) || "15", 10));
        setTimeLeft(30);
      } else {
        // INSTANT QUESTION HYDRATION (Eliminates "Generating personalized questions..." delay)
        const defaultQ = "Describe a time you solved a difficult technical problem using your primary skill set.";
        setCurrentQuestion(defaultQ);
        setQuestionIndex(0);
        setTotalQuestions(15);
        setTimeLeft(30);
      }

      // Call Backend to generate customized questions in background
      fetch(apiUrl('/api/interview/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumeContext: activeEval,
          candidateName: activeEval?.candidate_profile?.name || activeEval?.name || "Candidate",
          jobRole: "Software Engineer",
          token: activeToken
        })
      })
      .then(res => res.json())
      .then(data => {
        if (data && data.questionText) {
          setSessionId(data.sessionId);
          sessionIdRef.current = data.sessionId;

          sessionStorage.setItem('currentQuestion_' + activeToken, typeof data.questionText === 'string' ? data.questionText : JSON.stringify(data.questionText));
          sessionStorage.setItem('sessionId_' + activeToken, data.sessionId);
          sessionStorage.setItem('questionIndex_' + activeToken, data.questionIndex);
          sessionStorage.setItem('totalQuestions_' + activeToken, data.totalQuestions);

          // Update current question without interrupting active timer countdown
          if (questionIndexRef.current === 0 && (!answerTextRef.current || !answerTextRef.current.trim())) {
            setCurrentQuestion(data.questionText);
          }
        }
      })
      .catch(err => console.error("Background question fetch:", err));

    } catch (err) {
      console.error("Error starting interview API.", err);
    }
  };

  const targetEndTimeRef = useRef(null);

  const startQuestionTimer = (durationSeconds = 30) => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setTimeLeft(durationSeconds);
    const targetEndTime = Date.now() + durationSeconds * 1000;
    targetEndTimeRef.current = targetEndTime;

    timerRef.current = setInterval(() => {
      if (isHardwareMutedRef.current) {
        // Extend end time while hardware muted so candidate doesn't lose time
        if (targetEndTimeRef.current) targetEndTimeRef.current += 250;
        return;
      }

      const remainingMs = targetEndTimeRef.current - Date.now();
      const remainingSec = Math.max(0, Math.ceil(remainingMs / 1000));

      setTimeLeft(remainingSec);

      if (remainingSec <= 0) {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        submitAnswer(true);
      }
    }, 250);
  };

  // Create refs to avoid closure stale state in the interval timer
  const answerTextRef = useRef(answerText);
  const sessionIdRef = useRef(sessionId);
  const questionIndexRef = useRef(questionIndex);
  const isFirstQuestionStartedRef = useRef(false);

  // Auto-start interview questions when camera check is completed
  useEffect(() => {
    if (isCameraCheckCompleted && !currentQuestion) {
      startInterviewQuestions();
    }
  }, [isCameraCheckCompleted, currentQuestion]);

  useEffect(() => {
    if (isCameraCheckCompleted && currentQuestion && !isFirstQuestionStartedRef.current) {
      isFirstQuestionStartedRef.current = true;
      startQuestionTimer(30);
    }
  }, [isCameraCheckCompleted, currentQuestion]);

  useEffect(() => {
    answerTextRef.current = answerText;
  }, [answerText]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    questionIndexRef.current = questionIndex;
  }, [questionIndex]);

  const submitAnswer = async (isTimeout = false) => {
    if (isSubmitting) return;
    if (isHardwareMutedRef.current) {
      alert("SECURITY WARNING: Your microphone is hardware muted. You cannot proceed until you unmute it.");
      return;
    }
    setIsSubmitting(true);
    if (timerRef.current) clearInterval(timerRef.current);
    
    // Clear speech timeout to prevent double submission
    if (window.speechTimeout) clearTimeout(window.speechTimeout);

    let submissionText = answerTextRef.current || "";
    if (isTimeout || !submissionText.trim()) {
      submissionText = "[TIME EXPIRED - NO ANSWER]";
    }

    // Safety check / fallback for session
    const activeToken = inviteToken || new URLSearchParams(window.location.search).get('token');
    let currentSessionId = sessionIdRef.current || sessionStorage.getItem('sessionId_' + activeToken);
    const currentQuestionIndex = questionIndexRef.current !== undefined ? questionIndexRef.current : questionIndex;

    if (!currentSessionId) {
      currentSessionId = "fallback-session-" + Date.now();
      sessionIdRef.current = currentSessionId;
      setSessionId(currentSessionId);
    }

    try {
      const response = await fetch(apiUrl('/api/interview/answer'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: currentSessionId,
          questionIndex: currentQuestionIndex,
          answer: submissionText,
          // Includes UNVERIFIED_SPEECH_DETECTED events for the server-side HR audit.
          proctoringEvents: proctoringEvents.current
        })
      });

      proctoringEvents.current = []; // clear sent events

      const data = await response.json();

      if (data && data.completed) {
        await finishInterview();
      } else if (data && data.questionText) {
        if (recognitionRef.current) {
          try { recognitionRef.current.stop(); } catch (e) {}
        }
        setCurrentQuestion(data.questionText);
        setQuestionIndex(data.questionIndex);
        setAnswerText("");
        answerTextRef.current = "";
        setIsSubmitting(false);
        startQuestionTimer();
      } else {
        // Auto-advance fallback if backend response format differs
        handleLocalAutoAdvance(currentQuestionIndex);
      }
    } catch (err) {
      // Never advance locally after a failed submission: that would lose the
      // answer and produce an interview record the server cannot verify.
      console.error("Secure answer submission failed:", err);
      setIsSubmitting(false);
      setCustomWarning({
        show: true,
        message: 'Your answer could not be securely saved. Check your connection and submit the same answer again.'
      });
      startQuestionTimer(Math.max(timeLeft, 5));
    }
  };

  const handleLocalAutoAdvance = (currentIndex) => {
    const nextIdx = currentIndex + 1;
    if (nextIdx >= totalQuestions) {
      finishInterview();
    } else {
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (e) {}
      }
      const fallbackQuestions = [
        "Describe a time you solved a difficult technical problem using your primary skill set.",
        "How do you ensure code quality and performance in high-scale projects?",
        "Explain how you approach debugging a critical production bug under time pressure.",
        "What strategies do you use for efficient database query optimization?",
        "How do you handle technical disagreements within a software engineering team?",
        "Explain your experience with REST API design, rate limiting, and security.",
        "How do you approach learning and integrating a new framework quickly?",
        "Describe your experience with automated testing, CI/CD, and MLOps pipelines.",
        "How do you evaluate trade-offs between speed of delivery and architectural cleanliness?",
        "Describe a complex system architecture you designed or contributed to.",
        "Explain the key concepts of asynchronous programming and concurrency.",
        "What is your process for conducting code reviews and maintaining security standards?",
        "How do you handle memory management and performance profiling in your stack?",
        "Describe a scenario where you had to refactor a legacy codebase.",
        "Summarize your key strengths as a software engineer and AI practitioner."
      ];
      const nextQ = fallbackQuestions[nextIdx % fallbackQuestions.length];
      setCurrentQuestion(nextQ);
      setQuestionIndex(nextIdx);
      setAnswerText("");
      answerTextRef.current = "";
      setIsSubmitting(false);
      startQuestionTimer();
    }
  };

  const finishInterview = async () => {
    try {
      // Stop the recorder if it's still running
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }

      // Compile video chunks
      const videoBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
      
      // Use FormData to upload the video and proctoring logs together
      const formData = new FormData();
      formData.append("video", videoBlob, "recording.webm");
      formData.append("proctoring_logs", JSON.stringify(proctoringEvents.current));

      const response = await fetch(`${apiUrl(`/api/upload-interview-data/${inviteToken}`)}`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setInterviewResult({
          totalMarks: 100, // Placeholder until full evaluation
          marksObtained: 85,
          status: "Under Review"
      }); // Simulate result, or use real data if available
      setAppState('finished');
      stopCamera();
    } catch (err) {
      console.error(err);
      alert("Error finishing interview.");
    }
  };

  const stopCamera = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (cameraTimerRef.current) clearInterval(cameraTimerRef.current);
    if (window._interviewCleanup) window._interviewCleanup();
  };

  useEffect(() => {
    const initModels = async () => {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
        );
        landmarkerRef.current = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
          runningMode: "VIDEO",
          numFaces: 5,
          minFaceDetectionConfidence: 0.7,
          minFacePresenceConfidence: 0.7
        });

        objectDetectorRef.current = await ObjectDetector.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          scoreThreshold: 0.55,
          runningMode: "VIDEO"
        });

        handLandmarkerRef.current = await HandLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 4,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5
        });
      } catch (e) {
        console.error("Error loading MediaPipe models:", e);
      }
    };
    initModels();

    return () => {
          stopCamera();
    };
  }, []);

  return (
    <div className="app-container">
      {appState !== 'interview' && (
        <header className="hero">
          <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#818CF8' }}>
              <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"></path>
              <path d="M20 3v4"></path>
              <path d="M22 5h-4"></path>
              <path d="M4 17v2"></path>
              <path d="M5 18H3"></path>
            </svg>
            AI Agent Recruiter
          </h1>
          <p>Upload your resume to get evaluated instantly. If you are a match, you will proceed to a live AI interview.</p>
        </header>
      )}



      {isTokenValidating && (
        <div style={{ minHeight: '300px' }}></div>
      )}

      {appState === 'idle' && !isTokenValidating && (
        <div className="upload-section">
          <h2 className="animated-heading">Upload Your Resume</h2>

          <div className="file-input-wrapper">
            <button className="btn-primary">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
              {file ? file.name : 'Select Resume'}
            </button>
            <input type="file" className="file-input" accept=".pdf,.doc,.docx,.zip" onChange={handleFileChange} />
          </div>

          <div>
            <button
              className="btn-primary"
              style={{ backgroundColor: 'var(--accent)' }}
              onClick={handleUpload}
              disabled={!file}
            >
              Start Analysis
            </button>
          </div>
        </div>
      )}

            {appState === 'dashboard' && (
        <div className="dashboard-section" style={{ padding: '2rem', background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)' }}>
          <h2 style={{ color: 'var(--accent)', marginBottom: '1.5rem' }}>Recruiter Dashboard - Bulk Evaluation Results</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.2)' }}>
                <th style={{ padding: '1rem', textAlign: 'left', color: '#94a3b8' }}>Filename</th>
                <th style={{ padding: '1rem', textAlign: 'left', color: '#94a3b8' }}>Match Score</th>
                <th style={{ padding: '1rem', textAlign: 'left', color: '#94a3b8' }}>Status</th>
                <th style={{ padding: '1rem', textAlign: 'left', color: '#94a3b8' }}>Reason</th>
                <th style={{ padding: '1rem', textAlign: 'left', color: '#94a3b8' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {window.bulkCandidates && window.bulkCandidates.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <td style={{ padding: '1rem' }}>{c.filename || 'Unknown'}</td>
                  <td style={{ padding: '1rem', fontWeight: 'bold', color: c.matchPercentage >= 80 ? 'var(--success)' : (c.matchPercentage >= 50 ? '#facc15' : 'var(--danger)') }}>
                    {c.matchPercentage || 0}%
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{ 
                        padding: '0.25rem 0.5rem', 
                        borderRadius: '4px', 
                        fontSize: '0.85rem',
                        background: c.isMatch ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                        color: c.isMatch ? 'var(--success)' : 'var(--danger)'
                    }}>
                        {c.isMatch ? 'APPROVED' : 'REJECTED'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', fontSize: '0.9rem', color: '#cbd5e1' }}>{c.reason || c.error || 'N/A'}</td>
                  <td style={{ padding: '1rem' }}>
                    {c.isMatch && (
                      <button 
                        onClick={() => handleSingleInvite(c)}
                        className="btn-primary" 
                        style={{ padding: '0.5rem 1rem', fontSize: '0.85rem', display: 'inline-block' }}
                        disabled={generatedLinks.some(l => l.email === (c.email ? c.email.replace('email:', '') : 'candidate@example.com'))}
                      >
                        {generatedLinks.some(l => l.email === (c.email ? c.email.replace('email:', '') : 'candidate@example.com')) ? 'Sent!' : 'Send Email'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem' }}>
            <button className="btn-primary" onClick={handleSendInvites} disabled={isSendingBulk}>
              {isSendingBulk ? 'Sending...' : 'Send Email'}
            </button>
            <button className="btn-secondary" onClick={() => { setAppState('idle'); setFile(null); window.bulkCandidates = null; setGeneratedLinks([]); }}>Home</button>
          </div>

        </div>
      )}

      {appState === 'bulkSuccess' && (
        <div className="dashboard-section" style={{ padding: '2rem', background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)', textAlign: 'center' }}>
          <h2 style={{ color: 'var(--success)', marginBottom: '1.5rem' }}>Success</h2>
          <div style={{ whiteSpace: 'pre-wrap', color: '#f8fafc', marginBottom: '2rem', fontSize: '1.1rem' }}>
            {bulkSuccessMsg}
          </div>
        </div>
      )}

      {(appState === 'uploading' || appState === 'evaluating') && (
        <div className="upload-section">
          <div className="loader-wrapper">
            <span className="loader"></span>
            <h2 style={{ marginTop: '1.5rem' }}>
              {appState === 'uploading' ? 'Uploading... 📄' : '🪄 Analyzing Resume with AI... ✨'}
            </h2>
            <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Please wait while our AI Agent evaluates your profile.</p>
          </div>
        </div>
      )}

      {appState === 'expired' && (
        <div className="upload-section" style={{ borderColor: 'rgba(239, 68, 68, 0.3)', textAlign: 'center' }}>
          <h2 style={{ color: 'var(--danger)', marginBottom: '1rem' }}>❌ Link Expired</h2>
          <div className="status-message error">
            <p>{errorMessage}</p>
          </div>
          <p style={{ marginTop: '1.5rem', color: '#cbd5e1' }}>Please contact your recruiter if you believe this is a mistake.</p>
        </div>
      )}

      {appState === 'rejected' && evaluation && (
        <div className="upload-section" style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
          <h2 style={{ color: 'var(--danger)' }}>❌ Status: {evaluation.next_round_status || evaluation.status || "REJECTED"} 🤖</h2>
          <div className="status-message error">
            <p>Unfortunately, your resume does not match our current requirements for this position. Thank you for your interest.</p>
          </div>
          <button className="btn-secondary" style={{ marginTop: '2.5rem' }} onClick={() => { setAppState('idle'); setFile(null); }}>Try Another Resume</button>
        </div>
      )}

      {appState === 'approved' && evaluation && (
        <div className="upload-section" style={{ borderColor: 'var(--accent)' }}>
          <h2>✅ Status: APPROVED 🤖</h2>
          <div className="status-message success">
            <p>Congratulations! You have been shortlisted for the AI HR Interview.</p>
          </div>
          <p style={{ margin: '2rem 0', color: 'var(--text-muted)' }}>Before starting, please ensure you are in a quiet room. You will need to grant camera and microphone permissions. Camera monitoring will be active during this interview.</p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button className="btn-primary" onClick={startInterview}>
              START INTERVIEW
            </button>
          </div>
        </div>
      )}

      {appState === 'interview' && (
        <div style={{
          display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '800px', margin: '0 auto', width: '100%',
          border: isWarningBlinking ? '2px solid red' : '2px solid transparent',
          boxShadow: isWarningBlinking ? '0 0 20px rgba(255,0,0,0.5)' : 'none',
          borderRadius: '12px', padding: '1rem', transition: 'all 0.3s ease'
        }}>
          {showMuteWarning && (
            <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.9)', color: 'white', textAlign: 'center', borderRadius: '8px', fontWeight: 'bold', border: '2px solid #ef4444', animation: 'pulse 1s infinite' }}>
              ⚠️ WARNING: Microphone mute detected! Using hardware or system mute buttons is strictly prohibited. Please unmute immediately.
            </div>
          )}

          {/* Status Indicators */}
          {isCameraCheckConfirmed && (
            <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(15, 23, 42, 0.6)', padding: '1rem 1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div>
                <span style={{ color: isWarningBlinking ? 'var(--danger)' : 'var(--success)', fontWeight: 'bold', fontSize: '0.9rem', animation: isWarningBlinking ? 'pulse 0.5s infinite' : 'none' }}>● CAMERA ACTIVE</span>
                <span style={{ color: 'var(--success)', fontWeight: 'bold', marginLeft: '1.5rem', fontSize: '0.9rem' }}>● MICROPHONE ACTIVE</span>
              </div>
            </div>
          )}

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              display: (isCameraCheckConfirmed && showCamera && !isCameraCheckCompleted) ? 'block' : 'none',
              width: '100%',
              maxHeight: '400px',
              objectFit: 'cover',
              borderRadius: '16px',
              border: '2px solid var(--accent)',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
              transform: 'scaleX(-1)'
            }}
          ></video>

          {!isCameraCheckConfirmed ? (
            <div style={{ textAlign: 'center', background: 'rgba(30, 41, 59, 0.7)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <h2 style={{ color: 'var(--accent)', marginBottom: '1rem' }}>Environment Check Required</h2>
              <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', color: '#f8fafc', lineHeight: '1.6' }}>
                We must verify your surroundings and ensure you are alone. Once you start the scan, you will have 15 seconds to slowly rotate your camera 360 degrees.
              </p>

              <div style={{ display: 'inline-block', marginBottom: '2rem', padding: '1.5rem', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', width: '85%' }}>
                <div style={{ padding: '0.8rem 1rem', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.5)', borderRadius: '8px', color: '#fca5a5', fontWeight: 'bold', marginBottom: '1.2rem', fontSize: '0.95rem', textAlign: 'center' }}>
                  ⚠️ ATTENTION CANDIDATE: This entire interview (video, audio, and screen) is continuously monitored and recorded in full by AI proctoring. Unauthorized device usage, tab switching, or secondary assistance is strictly prohibited and will be logged directly to HR.
                </div>
                <h3 style={{ color: 'var(--accent)', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>Candidate Code of Conduct (Do's and Don'ts)</h3>
                <div style={{ display: 'flex', gap: '2rem', textAlign: 'left', lineHeight: '1.6', color: '#f8fafc' }}>
                  <div style={{ flex: 1, background: 'rgba(34, 197, 94, 0.1)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(34, 197, 94, 0.2)' }}>
                    <h4 style={{ color: '#4ade80', marginBottom: '0.8rem', fontSize: '1.1rem' }}>✅ DO</h4>
                    <ul style={{ listStyleType: 'none', padding: 0, margin: 0 }}>
                      <li style={{ marginBottom: '0.8rem' }}>• Ensure you are alone in a quiet, well-lit room.</li>
                      <li>• Keep your face clearly visible within the camera frame at all times.</li>
                    </ul>
                  </div>
                  <div style={{ flex: 1, background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                    <h4 style={{ color: '#f87171', marginBottom: '0.8rem', fontSize: '1.1rem' }}>❌ DON'T</h4>
                    <ul style={{ listStyleType: 'none', padding: 0, margin: 0 }}>
                      <li style={{ marginBottom: '0.8rem' }}>• Use mobile phones, tablets, or other secondary devices.</li>
                      <li style={{ marginBottom: '0.8rem' }}>• Have anyone else in the room or talking in the background.</li>
                      <li>• Switch browser tabs or leave the interview window.</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                {disclaimerTimeLeft === 0 && (
                  <button
                    className="btn-primary"
                    onClick={confirmCameraCheck}
                    style={{ padding: '1rem 2.5rem', fontSize: '1.1rem', width: 'auto' }}>
                    Start 15-Second Environment Scan
                  </button>
                )}
                {(!isWarningBlinking && disclaimerTimeLeft > 0) && (
                  <button
                    disabled
                    className="btn-primary"
                    style={{ padding: '1rem 2.5rem', fontSize: '1.1rem', width: 'auto', opacity: 0.5, cursor: 'not-allowed' }}>
                    Please read instructions ({disclaimerTimeLeft}s)
                  </button>
                )}
              </div>
            </div>
          ) : !isCameraCheckCompleted ? (
            <div style={{
              textAlign: 'center',
              padding: '2.5rem',
              background: hasAdditionalPersonDetected 
                ? 'rgba(239, 68, 68, 0.2)' 
                : isScanPaused ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)',
              borderRadius: '16px',
              color: '#fff',
              border: hasAdditionalPersonDetected 
                ? '2px solid rgba(239, 68, 68, 0.9)' 
                : isScanPaused ? '2px solid rgba(239, 68, 68, 0.6)' : '2px solid rgba(34, 197, 94, 0.6)',
              transition: 'all 0.3s ease'
            }}>
              <h2 style={{ fontSize: '1.8rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#60a5fa' }}>
                360° Room Verification
              </h2>
              <p style={{ fontSize: '1.1rem', color: '#cbd5e1', marginBottom: '1.5rem' }}>
                Please rotate the camera slowly and continuously to show the entire room.
              </p>

              {/* Status Indicator Banner */}
              <div style={{
                display: 'inline-block',
                padding: '0.6rem 1.5rem',
                borderRadius: '30px',
                fontWeight: 'bold',
                fontSize: '1rem',
                marginBottom: '1.5rem',
                background: hasAdditionalPersonDetected 
                  ? 'rgba(239, 68, 68, 0.3)' 
                  : isScanPaused ? 'rgba(239, 68, 68, 0.25)' : 'rgba(34, 197, 94, 0.25)',
                color: hasAdditionalPersonDetected ? '#ef4444' : isScanPaused ? '#fca5a5' : '#86efac',
                border: hasAdditionalPersonDetected ? '1px solid #ef4444' : isScanPaused ? '1px solid #f87171' : '1px solid #4ade80'
              }}>
                {hasAdditionalPersonDetected
                  ? '⚠️ CRITICAL SECURITY WARNING: Multiple people or forbidden device detected in room. Remove all unauthorized people/devices to complete room verification.'
                  : isScanPaused
                  ? 'Rotation stopped — Please continue rotating your camera 360° to scan all sides of the room.'
                  : 'Scanning room in progress...'}
              </div>

              {/* Rotation Progress Bar */}
              <div style={{ maxWidth: '600px', margin: '0 auto 1.5rem auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontWeight: '600', fontSize: '1.1rem' }}>
                  <span>360° Room Coverage Verification:</span>
                  <span style={{ color: scanProgressPercent >= 80 ? '#4ade80' : '#38bdf8' }}>{scanProgressPercent}% (Min 80% Required)</span>
                </div>
                <div style={{
                  width: '100%',
                  height: '16px',
                  background: 'rgba(15, 23, 42, 0.8)',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <div style={{
                    width: `${scanProgressPercent}%`,
                    height: '100%',
                    background: hasAdditionalPersonDetected ? '#ef4444' : scanProgressPercent >= 80 ? '#22c55e' : 'linear-gradient(90deg, #3b82f6, #06b6d4)',
                    borderRadius: '10px',
                    transition: 'width 0.4s ease'
                  }} />
                </div>
              </div>

              {/* Countdown Timer */}
              <h1 style={{ fontSize: '3rem', margin: '1rem 0', fontWeight: '900', color: isScanPaused ? '#ef4444' : '#facc15' }}>
                Countdown: {cameraTimeLeft} {isScanPaused && '(PAUSED)'}
              </h1>

              <p style={{ fontSize: '1rem', color: hasAdditionalPersonDetected ? '#f87171' : scanProgressPercent >= 80 ? '#4ade80' : '#94a3b8' }}>
                {hasAdditionalPersonDetected
                  ? '⚠️ Critical Violation Flagged: Unauthorized person or device detected in room during 360° scan.'
                  : (scanProgressPercent >= 80)
                  ? '✓ True 360° Room Environment Verification Completed Successfully.' 
                  : `Please continue rotating your camera slowly to scan the remaining room area (Coverage: ${scanProgressPercent}% / 80% required).`}
              </p>

              {/* Unlock Button rendered ONLY when room verification is genuinely complete (>= 80% coverage) */}
              {(scanProgressPercent >= 80 && !hasAdditionalPersonDetected) && (
                <button 
                  className="btn-primary" 
                  onClick={() => { void finalizeRoomVerification(); }}
                  style={{ marginTop: '1.5rem', padding: '0.8rem 2rem', fontSize: '1.1rem', background: '#22c55e', border: 'none', cursor: 'pointer' }}>
                  Continue to Next Interview Level ➔
                </button>
              )}
            </div>
          ) : (
            <>
              {/* Question UI */}
              {!currentQuestion ? (
                <div style={{ textAlign: 'center', padding: '4rem', background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span className="loader"></span>
                  <p style={{ marginTop: '1.5rem', color: 'var(--text-muted)' }}>Generating personalized questions...</p>
                </div>
              ) : (
                <div style={{ background: 'rgba(30, 41, 59, 0.7)', padding: '2.5rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)', boxShadow: '0 10px 30px rgba(0,0,0,0.2)' }}>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', paddingBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <h3 style={{ margin: 0, color: 'var(--accent)', fontSize: '1.1rem', letterSpacing: '1px' }}>QUESTION {questionIndex + 1} OF {totalQuestions}</h3>
                    <div style={{
                      background: timeLeft <= 10 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.1)',
                      color: timeLeft <= 10 ? 'var(--danger)' : '#fff',
                      padding: '0.5rem 1.25rem',
                      borderRadius: '8px',
                      fontWeight: 'bold',
                      fontSize: '1.1rem',
                      animation: timeLeft <= 10 ? 'pulse 1s infinite' : 'none',
                      border: `1px solid ${timeLeft <= 10 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(255,255,255,0.2)'}`
                    }}>
                      Time: {timeLeft}s
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                    <span style={{
                      padding: '0.4rem 1rem',
                      borderRadius: '20px',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      background: currentQuestion?.category === 'RESUME_BASED' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                      color: currentQuestion?.category === 'RESUME_BASED' ? '#60A5FA' : '#34D399',
                      border: `1px solid ${currentQuestion?.category === 'RESUME_BASED' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`
                    }}>
                      Category: {currentQuestion?.category === 'RESUME_BASED' ? 'Resume Based' : (currentQuestion?.category === 'COMPANY_REQUIREMENT_BASED' ? 'Company Requirement Based' : 'General')}
                    </span>
                    {currentQuestion?.skill && (
                      <span style={{ padding: '0.4rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold', background: 'rgba(255, 255, 255, 0.1)', color: '#cbd5e1' }}>
                        Skill: {currentQuestion.skill}
                      </span>
                    )}
                    {currentQuestion?.difficulty && (
                      <span style={{ padding: '0.4rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold', background: 'rgba(168, 85, 247, 0.2)', color: '#C084FC', border: '1px solid rgba(168, 85, 247, 0.3)' }}>
                        Difficulty: {currentQuestion.difficulty}
                      </span>
                    )}
                    {currentQuestion?.marks && (
                      <span style={{ padding: '0.4rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold', background: 'rgba(255, 255, 255, 0.1)', color: '#cbd5e1' }}>
                        Marks: {currentQuestion.marks}
                      </span>
                    )}
                  </div>

                  <h2 style={{ fontSize: '1.4rem', marginBottom: '2rem', lineHeight: '1.6', fontWeight: '500', color: '#f8fafc' }}>
                    {getQuestionTextString(currentQuestion)}
                  </h2>

                  <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#cbd5e1' }}>
                    {isAiSpeaking ? 'AI is asking the question...' : 'Speak your answer naturally. The AI is listening...'}
                  </h3>

                  <div style={{
                    width: '100%',
                    minHeight: '160px',
                    padding: '1.2rem',
                    borderRadius: '12px',
                    background: 'rgba(15, 23, 42, 0.85)',
                    border: `1px solid ${isAiSpeaking ? 'rgba(168, 85, 247, 0.4)' : (answerText ? 'rgba(59, 130, 246, 0.5)' : 'rgba(255, 255, 255, 0.2)')}`,
                    color: '#fff',
                    marginBottom: '1.5rem',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: isAiSpeaking ? '0 0 15px rgba(168, 85, 247, 0.15)' : (answerText ? '0 0 15px rgba(59, 130, 246, 0.15)' : 'none')
                  }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.6rem',
                      marginBottom: '0.8rem',
                      fontSize: '0.9rem',
                      fontWeight: '500',
                      color: isAiSpeaking ? '#C084FC' : (answerText ? '#60A5FA' : '#94a3b8')
                    }}>
                      <div style={{
                        width: '10px',
                        height: '10px',
                        borderRadius: '50%',
                        background: isAiSpeaking ? '#C084FC' : '#3B82F6',
                        animation: 'pulse 1.5s infinite'
                      }}></div>
                      <span>
                        {isAiSpeaking ? 'AI is asking the question...' : (answerText ? 'Transcribing response (Voice active - editable below):' : 'Listening for your voice... Speak now')}
                      </span>
                    </div>

                    <textarea
                      value={answerText}
                      onChange={(e) => {
                        setAnswerText(e.target.value);
                        answerTextRef.current = e.target.value;
                      }}
                      placeholder={isAiSpeaking ? "Please wait for AI to finish speaking..." : "Speak into your microphone naturally, or type your answer here..."}
                      disabled={isAiSpeaking || isSubmitting}
                      style={{
                        width: '100%',
                        minHeight: '100px',
                        background: 'transparent',
                        border: 'none',
                        outline: 'none',
                        color: '#f8fafc',
                        fontSize: '1.15rem',
                        lineHeight: '1.6',
                        fontFamily: 'inherit',
                        resize: 'vertical',
                        textAlign: answerText ? 'left' : 'center'
                      }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', alignItems: 'center', marginBottom: '1rem' }}>
                    <button
                      onClick={() => submitAnswer(false)}
                      disabled={isSubmitting || isAiSpeaking || !answerText || !answerText.trim()}
                      className="gradient-btn"
                      style={{
                        padding: '0.8rem 2rem',
                        fontSize: '1rem',
                        opacity: (isSubmitting || isAiSpeaking || !answerText || !answerText.trim()) ? 0.5 : 1,
                        cursor: (isSubmitting || isAiSpeaking || !answerText || !answerText.trim()) ? 'not-allowed' : 'pointer'
                      }}
                    >
                      {isSubmitting ? 'Evaluating response...' : 'Submit Answer Now'}
                    </button>
                    <button
                      onClick={() => startOrRestartSpeechRecognition()}
                      disabled={isAiSpeaking}
                      className="glass-card"
                      style={{
                        padding: '0.8rem 1.5rem',
                        fontSize: '0.9rem',
                        color: '#94a3b8',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        cursor: isAiSpeaking ? 'not-allowed' : 'pointer'
                      }}
                    >
                      Restart Mic
                    </button>
                  </div>
                  
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    {isSubmitting ? 'Evaluating response...' : 'Speak naturally or click "Submit Answer Now" when finished.'}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {appState === 'finished' && interviewResult && (
        <div className="upload-section" style={{ borderColor: 'var(--success)' }}>
          <h2 style={{ color: 'var(--success)', marginBottom: '1rem' }}>CANDIDATE INTERVIEW REPORT</h2>

          <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(15, 23, 42, 0.6)', padding: '1.5rem', borderRadius: '12px', marginBottom: '2rem' }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: 'var(--text-muted)' }}>Total Questions</p>
              <h3>{interviewResult.totalMarks}</h3>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: 'var(--text-muted)' }}>Marks Obtained</p>
              <h3>{interviewResult.marksObtained} / {interviewResult.totalMarks}</h3>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: 'var(--text-muted)' }}>Interview Score</p>
              <h3 style={{ color: 'var(--accent)' }}>{interviewResult.percentage.toFixed(1)}%</h3>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: 'var(--text-muted)' }}>Warnings</p>
              <h3 style={{ color: interviewResult.warningCount > 0 ? 'var(--danger)' : 'var(--success)' }}>
                {interviewResult.warningCount}
              </h3>
            </div>
          </div>

          <div className="status-message">
            <h3>Final Recommendation: {interviewResult.finalStatus}</h3>
            <p style={{ marginTop: '1rem' }}>Thank you for your time. The HR team will review your interview report and get back to you shortly.</p>
          </div>

          <button className="btn-secondary" style={{ marginTop: '2rem' }} onClick={() => { setAppState('idle'); setFile(null); }}>Back to Home</button>
        </div>
      )}
    </div>
  );
}

export default CandidatePortal;
