import React, { useState, useRef, useEffect } from 'react';
import { FaceLandmarker, ObjectDetector, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './index.css';

// Suppress verbose MediaPipe WebAssembly logs to clear the console
const originalWarn = console.warn;
const originalInfo = console.info;
const originalLog = console.log;
const suppressedPatterns = ['face_landmarker_graph.cc', 'gl_context.cc', 'TensorFlow Lite', 'XNNPACK', 'Created TensorFlow'];
const shouldSuppress = (args) => typeof args[0] === 'string' && suppressedPatterns.some(p => args[0].includes(p));
console.warn = (...args) => { if (!shouldSuppress(args)) originalWarn(...args); };
console.info = (...args) => { if (!shouldSuppress(args)) originalInfo(...args); };
console.log = (...args) => { if (!shouldSuppress(args)) originalLog(...args); };

function CandidatePortal() {
  const [appState, setAppState] = useState('idle');
  const [inviteToken, setInviteToken] = useState(null);
  const [isTokenValidating, setIsTokenValidating] = useState(true);
  const [generatedLinks, setGeneratedLinks] = useState([]);
  const [file, setFile] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [bulkSuccessMsg, setBulkSuccessMsg] = useState("");
  const [isSendingBulk, setIsSendingBulk] = useState(false);
  const [sendingSingle, setSendingSingle] = useState(null);
  // Interview State
  const [sessionId, setSessionId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(15);
  const [timeLeft, setTimeLeft] = useState(30);
  const [cameraTimeLeft, setCameraTimeLeft] = useState(15);
  const [showCamera, setShowCamera] = useState(true);
  const [answerText, setAnswerText] = useState("");
  const [interviewResult, setInterviewResult] = useState(null);
  const [isWarningBlinking, setIsWarningBlinking] = useState(false);
  const [showMuteWarning, setShowMuteWarning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCameraCheckConfirmed, setIsCameraCheckConfirmed] = useState(false);
  const [isCameraCheckCompleted, setIsCameraCheckCompleted] = useState(false);
  const [isPreCheckFailed, setIsPreCheckFailed] = useState(false);
  const [preCheckFailureReason, setPreCheckFailureReason] = useState("");
  const [detectedFacesCount, setDetectedFacesCount] = useState(0);
  const [disclaimerTimeLeft, setDisclaimerTimeLeft] = useState(15);
  const [showFinishConfirm, setShowFinishConfirm] = useState(false);
  
  const facesCountRef = useRef(0);
  const isPreCheckFailedRef = useRef(false);
  const isCameraCheckPhaseRef = useRef(false);
  const isConfirmOpenRef = useRef(false);
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

  // Audio analysis refs
  const audioContextRef = useRef(null);
  const audioAnalyserRef = useRef(null);
  const audioDataArrayRef = useRef(null);
  const recognitionRef = useRef(null);
  const isHumanSpeechDetectedRef = useRef(false);
  const backgroundVoiceCountRef = useRef(0);
  const headTurnCountRef = useRef(0);
  const eyesWanderingCountRef = useRef(0);

  const timerRef = useRef(null);
  const cameraTimerRef = useRef(null);
  const isHardwareMutedRef = useRef(false);

  // Append-only voice transcript state. The Web Speech API resets
  // `event.results` on every restart, so the raw results must be merged into a
  // persistent buffer or earlier speech is silently overwritten.
  const committedTranscriptRef = useRef("");
  const sessionFinalRef = useRef("");
  const silenceTimerRef = useRef(null);
  const SILENCE_SUBMIT_MS = 5000;
  const MIN_AUTO_SUBMIT_CHARS = 10;

  // Text to Speech Function
  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.onstart = () => { window.isAgentSpeaking = true; };
      utterance.onend = () => { window.isAgentSpeaking = false; };
      window.speechSynthesis.speak(utterance);
    }
  };

  // Speak Environment Check Prompt
  
  // Check for URL token on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    
    if (token) {
      setInviteToken(token);
      // Validate token
      fetch(`/api/validate-token/${token}`)
        .then(res => res.json())
        .then(data => {
          if (data.valid) {
            // Bypass upload and jump straight to approved screen
            setEvaluation(data.candidate);
            setAppState('approved');
          } else {
            alert(data.reason || 'Invalid or expired interview link.');
            setAppState('idle');
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
    if (appState === 'interview' && !isCameraCheckConfirmed && disclaimerTimeLeft > 0) {
      const timer = setTimeout(() => {
        setDisclaimerTimeLeft(prev => prev - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [appState, isCameraCheckConfirmed, disclaimerTimeLeft]);

  // Global Bypass Key Listener (Ctrl+Shift+B)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'B' || e.key === 'b')) {
        e.preventDefault();
        setIsWarningConfirmed(true);
        setIsCameraCheckConfirmed(true);
        setIsCameraCheckCompleted(true);
        sessionStorage.setItem('isCameraCheckCompleted', 'true');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Tab Switching Strict Security
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && appState === 'interview' && isCameraCheckCompleted) {
        setWarningsCount(prev => prev + 1);
        proctoringEvents.current.push({
          timestamp: new Date().toISOString(),
          type: "TAB_SWITCH"
        });
        alert("SECURITY WARNING: You have switched tabs or minimized the window. This incident has been recorded and flagged.");
      }
    };

    const handleBeforeUnload = (e) => {
      if (appState === 'interview') {
        const message = "refresh the page the link will expire";
        e.preventDefault();
        e.returnValue = message;
        return message;
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("beforeunload", handleBeforeUnload);
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

    if (appState === 'interview' || appState === 'dashboard') {
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
  }, [appState]);

  useEffect(() => {
    if (appState === 'interview' && isCameraCheckConfirmed && !isCameraCheckCompleted) {
      speakText("Scan in progress. Please rotate your camera 360 degrees now to show your surroundings. We must verify you are alone in the room.");
    }
  }, [appState, isCameraCheckConfirmed, isCameraCheckCompleted]);

  useEffect(() => {
    if (currentQuestion && isCameraCheckCompleted) {
      const qText = typeof currentQuestion === 'string' ? currentQuestion : currentQuestion.question;
      speakText(qText);
      window.submitAnswerFn = submitAnswer;
      committedTranscriptRef.current = "";
      sessionFinalRef.current = "";
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      answerTextRef.current = "";
      setAnswerText("");
    }
  }, [currentQuestion, isCameraCheckCompleted]);

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
    
    setSendingSingle(finalEmail);
    const candidateData = {
      ...candidate,
      email: finalEmail
    };
    try {
      const res = await fetch('/api/hr/send-invites', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ candidates: [candidateData], expiry_hours: 48 })
      });
      const data = await res.json();
      if (data.status === 'success') {
         setGeneratedLinks(prev => [...prev, ...data.invited]);
      }
    } catch (err) {
      console.error(err);
      alert('Failed to send invite');
    } finally {
      setSendingSingle(null);
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
      const res = await fetch('/api/hr/send-invites', {
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
      const endpoint = isZip ? '/api/bulk-upload' : '/api/upload-resume';
      
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
    proctoringEvents.current.push({ type, timestamp: new Date().toISOString() });

    // Audio and haptic warnings
    let warningMessage = "Warning.";
    if (type === "FACE_NOT_DETECTED") warningMessage = "Warning. Face not detected.";
    else if (type === "MULTIPLE_FACES") warningMessage = "Warning. Multiple faces detected in frame.";
    else if (type === "FACE_OUT_OF_FRAME") warningMessage = "Warning. Please look at the screen.";
    else if (type === "TAB_SWITCH") warningMessage = "Warning. Do not switch tabs.";
    else if (type === "WINDOW_BLUR") warningMessage = "Warning. Keep the interview window in focus.";
    else if (type === "BACKGROUND_VOICE") warningMessage = "Warning. Unrecognized background voice detected. Please ensure you are alone and quiet.";
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
          setWarningsCount(prev => prev + 1);
          if (isCameraCheckPhaseRef.current) {
            recordProctoringEvent("360_SCAN_ADDITIONAL_PERSON_DETECTED");
          } else {
            recordProctoringEvent("EXTRA_HANDS");
          }
          lastWarningTimeRef.current = now;
        } else if (totalPeople === 0) {
          if (!isCameraCheckPhaseRef.current) {
            setWarningsCount(prev => prev + 1);
            recordProctoringEvent("FACE_NOT_DETECTED");
            lastWarningTimeRef.current = now;
          }
        } else if (totalPeople > 1) {
          setWarningsCount(prev => prev + 1);
          if (isCameraCheckPhaseRef.current) {
            recordProctoringEvent("360_SCAN_ADDITIONAL_PERSON_DETECTED");
          } else {
            recordProctoringEvent("MULTIPLE_FACES");
          }
          lastWarningTimeRef.current = now;
        } else if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          // Basic out of frame check (nose too close to edges)
          const nose = results.faceLandmarks[0][1];
          if (nose.x < 0.1 || nose.x > 0.9 || nose.y < 0.1 || nose.y > 0.9) {
            setWarningsCount(prev => prev + 1);
            recordProctoringEvent("FACE_OUT_OF_FRAME");
            lastWarningTimeRef.current = now;
          }
        }
      }

      // BACKGROUND VOICE DETECTION - Noise-Ignoring Implementation
      // Ignore ambient noise for multiple voice detection. Only confirmed speech API words count.
      let isConfirmedHumanVoice = isHumanSpeechDetectedRef.current;
      let ambientNoiseLevel = 0;

      if (audioAnalyserRef.current && audioDataArrayRef.current) {
        audioAnalyserRef.current.getByteFrequencyData(audioDataArrayRef.current);
        let sum = 0;
        let count = 0;
        for (let i = 3; i < 15 && i < audioDataArrayRef.current.length; i++) {
          sum += audioDataArrayRef.current[i];
          count++;
        }
        ambientNoiseLevel = count > 0 ? sum / count : 0;
        // Ambient noise explicitly ignored for background voice - only SpeechRecognition words count
      }

      let isMouthMoving = false;
      if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        const jawOpen = shapes.find(c => c.categoryName === 'jawOpen');
        // Require significant mouth movement to confirm candidate is speaking
        if (jawOpen && jawOpen.score > 0.15) { 
          isMouthMoving = true;
        }

        // CALIBRATED EYE TRACKING - Prevent false alerts while reading
        const eyeKeys = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'];
        let isEyesWandering = eyeKeys.some(key => {
          const shape = shapes.find(c => c.categoryName === key);
          return shape && shape.score > 0.82; // Calibrated 0.82 threshold - only true off-screen gaze triggers
        });

        // Calibrated Iris Gaze Detection - Dual Eye + Vertical
        if (!isEyesWandering && results.faceLandmarks && results.faceLandmarks.length > 0) {
          const landmarks = results.faceLandmarks[0];
          if (landmarks && landmarks.length > 473) {
            const checkEye = (pupilIdx, innerIdx, outerIdx, topIdx, bottomIdx) => {
              const pupil = landmarks[pupilIdx];
              const inner = landmarks[innerIdx];
              const outer = landmarks[outerIdx];
              if (!pupil || !inner || !outer) return false;
              const eyeWidth = Math.abs(outer.x - inner.x);
              if (eyeWidth < 0.01) return false;
              const hRatio = Math.abs(pupil.x - inner.x) / eyeWidth;
              if (hRatio < 0.18 || hRatio > 0.82) return true;
              if (topIdx && bottomIdx) {
                const top = landmarks[topIdx];
                const bottom = landmarks[bottomIdx];
                if (top && bottom) {
                  const eyeHeight = Math.abs(bottom.y - top.y);
                  if (eyeHeight > 0.001) {
                    const vRatio = (pupil.y - top.y) / eyeHeight;
                    if (vRatio < 0.15 || vRatio > 0.85) return true;
                  }
                }
              }
              return false;
            };
            if (checkEye(468, 133, 33, 159, 145) || checkEye(473, 362, 263, 386, 374)) {
              isEyesWandering = true;
            }
          }
        }

        if (isEyesWandering) {
          eyesWanderingCountRef.current += 1;
        } else {
          eyesWanderingCountRef.current = Math.max(0, eyesWanderingCountRef.current - 3);
        }
      }

      // HEAD TURN TRACKING - Calibrated to avoid false alerts while reading
      let isHeadTurned = false;
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Head tracking based on relative position of nose to ears
        const nose = results.faceLandmarks[0][1];
        const leftEar = results.faceLandmarks[0][234];
        const rightEar = results.faceLandmarks[0][454];

        if (nose && leftEar && rightEar) {
          const distLeft = Math.abs(nose.x - leftEar.x);
          const distRight = Math.abs(nose.x - rightEar.x);
          // Calibrated ratio 2.2 - only triggers when head is turned completely away
          if (distLeft / distRight > 2.2 || distRight / distLeft > 2.2) {
            isHeadTurned = true;
          }
        }
      }

      if (isHeadTurned) {
        headTurnCountRef.current += 1;
      } else {
        headTurnCountRef.current = Math.max(0, headTurnCountRef.current - 3);
      }

      if (isConfirmedHumanVoice && !isMouthMoving && !window.isAgentSpeaking) {
        backgroundVoiceCountRef.current += 1;
      } else {
        backgroundVoiceCountRef.current = Math.max(0, backgroundVoiceCountRef.current - 3); // Faster decay to ignore transient noise
      }

      // Require sustained confirmed speech (approx 2 seconds) to trigger, ignoring pure noise
      if (backgroundVoiceCountRef.current > 28 && now - lastWarningTimeRef.current > 7000) {
        if (isCameraCheckPhaseRef.current) {
          isPreCheckFailedRef.current = "Environment scan failed: Background voices detected. You must be in a quiet environment.";
        } else {
          setWarningsCount(prev => prev + 1);
          recordProctoringEvent("BACKGROUND_VOICE");
          lastWarningTimeRef.current = now;
          backgroundVoiceCountRef.current = 0;
        }
      }

      // Flag wandering eyes - calibrated: sustained off-screen gaze >45 frames, 5s cooldown
      if (eyesWanderingCountRef.current > 45 && now - lastWarningTimeRef.current > 5000) {
        setWarningsCount(prev => prev + 1);
        recordProctoringEvent("EYES_WANDERING");
        lastWarningTimeRef.current = now;
        eyesWanderingCountRef.current = 0;
      }

      // Flag head turn - calibrated: sustained turn >45 frames
      if (headTurnCountRef.current > 45 && now - lastWarningTimeRef.current > 5000) {
        setWarningsCount(prev => prev + 1);
        recordProctoringEvent("HEAD_TURNED");
        lastWarningTimeRef.current = now;
        headTurnCountRef.current = 0;
      }
    }

    lastInferenceTimeRef.current = performance.now();
    setTimeout(() => {
      animationRef.current = requestAnimationFrame(predictWebcam);
    }, 50);
  };

  const initializeCameraSystem = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      
      // Force Screen Sharing (Screen Proctoring)
      let screenStream = null;
      try {
        screenStream = await navigator.mediaDevices.getDisplayMedia({ 
          video: { displaySurface: "monitor" }, 
          audio: false,
          surfaceSwitching: "exclude"
        });

        const videoTrack = screenStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();

        // Strictly enforce that they selected the Entire Screen, not a tab or window
        if (settings.displaySurface && settings.displaySurface !== 'monitor') {
          videoTrack.stop();
          throw new Error("INVALID_SURFACE");
        }
        
        screenStreamRef.current = screenStream;
        
        videoTrack.onended = () => {
           setWarningsCount(prev => prev + 1);
           recordProctoringEvent("SCREEN_SHARE_STOPPED");
           alert("SECURITY WARNING: You stopped sharing your screen. This interview is now locked.");
           isHardwareMutedRef.current = true; // Lock timer
        };
      } catch (err) {
        if (err.message === "INVALID_SURFACE") {
          alert("SECURITY WARNING: You MUST select the 'Entire Screen' tab. You cannot share just a Window or Chrome Tab.");
        } else {
          alert("SECURITY WARNING: You MUST share your entire screen to proceed with the interview.");
        }
        stream.getTracks().forEach(t => t.stop());
        return;
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
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = (event) => {
          // Ignore our own TTS to prevent the agent transcribing itself.
          if (window.isAgentSpeaking) return;

          isHumanSpeechDetectedRef.current = true;
          let sessionFinal = '';
          let interim = '';
          for (let i = 0; i < event.results.length; ++i) {
            const result = event.results[i];
            if (!result || !result[0]) continue;
            if (result.isFinal) {
              sessionFinal += result[0].transcript + ' ';
            } else {
              interim += result[0].transcript;
            }
          }
          sessionFinalRef.current = sessionFinal.trim();

          const fullText = `${committedTranscriptRef.current} ${sessionFinal} ${interim}`
            .replace(/\s+/g, ' ')
            .trim();
          if (fullText) {
            setAnswerText(fullText);
            answerTextRef.current = fullText;
          }

          // Submit only after 5 seconds of true silence. The old 3s window cut
          // candidates off mid-thought while they were still answering.
          if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = setTimeout(() => {
            silenceTimerRef.current = null;
            isHumanSpeechDetectedRef.current = false;
            const pending = (answerTextRef.current || '').trim();
            if (
              !isCameraCheckPhaseRef.current &&
              window.submitAnswerFn &&
              pending.length >= MIN_AUTO_SUBMIT_CHARS &&
              !isHardwareMutedRef.current &&
              !window.isAgentSpeaking
            ) {
              window.submitAnswerFn(false);
            }
          }, SILENCE_SUBMIT_MS);
        };

        recognition.onerror = (e) => {
          if (e.error !== 'no-speech') {
            console.log('Speech recognition error:', e.error);
          }
        };

        recognition.onend = () => {
          // Preserve this session's speech before results are reset by restart.
          const merged = `${committedTranscriptRef.current} ${sessionFinalRef.current}`
            .replace(/\s+/g, ' ')
            .trim();
          committedTranscriptRef.current = merged;
          sessionFinalRef.current = "";
          if (merged) {
            setAnswerText(merged);
            answerTextRef.current = merged;
          }

          // Automatically restart recognition to run continuously during interview
          if (streamRef.current) {
            try {
              recognition.start();
            } catch (e) { }
          }
        };

        try {
          recognition.start();
          recognitionRef.current = recognition;
        } catch (e) {
          console.log('Speech recognition start error:', e);
        }
      }

      // Wait for React to render the video element
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(e => console.log("Video play error:", e));
          predictWebcam();
        }
      }, 100);

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

    if (!referenceFrameRef.current) {
      referenceFrameRef.current = imageData;
      return;
    }

    let diff = 0;
    for (let i = 0; i < imageData.length; i += 4) {
      const r = Math.abs(imageData[i] - referenceFrameRef.current[i]);
      const g = Math.abs(imageData[i + 1] - referenceFrameRef.current[i + 1]);
      const b = Math.abs(imageData[i + 2] - referenceFrameRef.current[i + 2]);
      diff += (r + g + b);
    }

    // Normalize difference (average difference per pixel across RGB)
    const avgDiffPerPixel = diff / (64 * 48 * 3);

    // Accumulate total movement to ensure continuous rotation
    maxPixelDifferenceRef.current += avgDiffPerPixel;

    // Update reference frame to calculate frame-to-frame movement
    referenceFrameRef.current = imageData;
  };

  const confirmCameraCheck = async () => {
    await initializeCameraSystem();
    setIsCameraCheckConfirmed(true);
    isCameraCheckPhaseRef.current = true;
    referenceFrameRef.current = null;
    maxPixelDifferenceRef.current = 0;
    setCameraTimeLeft(15);

    // Start 30s camera preview timer
    cameraTimerRef.current = setInterval(() => {
      captureFrameAndCompare();

      if (isPreCheckFailedRef.current) {
        clearInterval(cameraTimerRef.current);
        setIsPreCheckFailed(true);
        setPreCheckFailureReason(typeof isPreCheckFailedRef.current === 'string' ? isPreCheckFailedRef.current : "Environment verification failed.");
        isCameraCheckPhaseRef.current = false;
        setShowCamera(false);
        stopCamera();
        return;
      }
      setCameraTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(cameraTimerRef.current);

          // Verify if the camera was rotated significantly (requires cumulative continuous movement)
          if (maxPixelDifferenceRef.current < 250) {
            setIsPreCheckFailed(true);
            setPreCheckFailureReason("You did not rotate the camera 360 degrees as requested. We could not verify your environment.");
            isCameraCheckPhaseRef.current = false;
            setShowCamera(false);
            stopCamera();
            return 0;
          }

          setShowCamera(false);
          setIsCameraCheckCompleted(true);
          isCameraCheckPhaseRef.current = false;
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

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
      // Call Backend to generate questions
      const response = await fetch('/api/interview/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumeContext: evaluation,
          candidateName: evaluation.candidate_profile?.name || "Candidate",
          jobRole: "Software Engineer",
          token: inviteToken
        })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setSessionId(data.sessionId);
      setCurrentQuestion(data.questionText);
      setQuestionIndex(data.questionIndex);
      setTotalQuestions(data.totalQuestions);
      setTimeLeft(30);

    } catch (err) {
      console.error("Error starting interview API.", err);
      alert("API failed. Please check backend.");
    }
  };

  const startQuestionTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setTimeLeft(30);
    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (isHardwareMutedRef.current || isConfirmOpenRef.current) return prev; // Pause timer if muted or confirm open

        if (prev <= 1) {
          clearInterval(timerRef.current);
          submitAnswer(true); // Auto-submit when time is up
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // Create refs to avoid closure stale state in the interval timer
  const answerTextRef = useRef(answerText);
  const sessionIdRef = useRef(sessionId);
  const questionIndexRef = useRef(questionIndex);
  const isFirstQuestionStartedRef = useRef(false);

  useEffect(() => {
    if (isCameraCheckCompleted && currentQuestion && !isFirstQuestionStartedRef.current) {
      isFirstQuestionStartedRef.current = true;
      startQuestionTimer();
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

    // Cancel the pending silence auto-submit so it cannot fire twice.
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    // Fold in speech still held by the live recognition instance.
    const mergedOnSubmit = `${committedTranscriptRef.current} ${sessionFinalRef.current}`
      .replace(/\s+/g, ' ')
      .trim();
    if (mergedOnSubmit.length > (answerTextRef.current || '').trim().length) {
      answerTextRef.current = mergedOnSubmit;
    }
    committedTranscriptRef.current = "";
    sessionFinalRef.current = "";

    let submissionText = answerTextRef.current;
    if (isTimeout && !submissionText.trim()) {
      submissionText = "[TIME EXPIRED]";
    }

    // Safety check if session is missing
    const currentSessionId = sessionIdRef.current;
    const currentQuestionIndex = questionIndexRef.current;

    if (!currentSessionId) {
      setIsSubmitting(false);
      return;
    }

    try {
      const response = await fetch('/api/interview/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: currentSessionId,
          questionIndex: currentQuestionIndex,
          answer: submissionText,
          proctoringEvents: proctoringEvents.current
        })
      });

      proctoringEvents.current = []; // clear sent events

      const data = await response.json();

      if (data.completed) {
        await finishInterview();
      } else {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
        setCurrentQuestion(data.questionText);
        setQuestionIndex(data.questionIndex);
        setAnswerText("");
        answerTextRef.current = "";
        setIsSubmitting(false);
        startQuestionTimer();
      }
    } catch (err) {
      console.error(err);
      alert("Error submitting answer.");
      setIsSubmitting(false);
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

      const response = await fetch(`/api/upload-interview-data/${inviteToken}`, {
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

      {appState === 'idle' && (
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
                        disabled={sendingSingle === (c.email ? c.email.replace('email:', '') : null) || generatedLinks.some(l => l.email === (c.email ? c.email.replace('email:', '') : 'candidate@example.com'))}
                      >
                        {generatedLinks.some(l => l.email === (c.email ? c.email.replace('email:', '') : 'candidate@example.com')) ? 'Sent!' : (sendingSingle === (c.email ? c.email.replace('email:', '') : null) ? 'Sending...' : 'Send Email')}
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
          <button className="btn-primary" onClick={() => { setAppState('idle'); setFile(null); window.bulkCandidates = null; setGeneratedLinks([]); setBulkSuccessMsg(""); }}>
            Finish
          </button>
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
              display: (isCameraCheckConfirmed && showCamera) ? 'block' : 'none',
              width: '100%',
              maxHeight: '400px',
              objectFit: 'cover',
              borderRadius: '16px',
              border: '2px solid var(--accent)',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
              transform: 'scaleX(-1)'
            }}
          ></video>

          {isPreCheckFailed ? (
            <div style={{ textAlign: 'center', background: 'rgba(30, 41, 59, 0.7)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
              <h2 style={{ color: 'var(--danger)', marginBottom: '1rem' }}>Environment Check Failed</h2>
              <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', color: '#f8fafc', lineHeight: '1.6' }}>
                {preCheckFailureReason || "Environment verification failed."}
              </p>
              <button className="btn-secondary" onClick={() => window.location.reload()}>Return to Home</button>
            </div>
          ) : !isCameraCheckConfirmed ? (
            <div style={{ textAlign: 'center', background: 'rgba(30, 41, 59, 0.7)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <h2 style={{ color: 'var(--accent)', marginBottom: '1rem' }}>Environment Check Required</h2>
              <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', color: '#f8fafc', lineHeight: '1.6' }}>
                We must verify your surroundings and ensure you are alone. Once you start the scan, you will have 15 seconds to slowly rotate your camera 360 degrees.
              </p>

              <div style={{ display: 'inline-block', marginBottom: '2rem', padding: '1.5rem', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', width: '85%' }}>
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
            <div style={{ textAlign: 'center', padding: '2rem', background: 'rgba(234, 179, 8, 0.2)', borderRadius: '12px', color: '#facc15', border: '2px solid rgba(234, 179, 8, 0.5)', animation: 'pulse 2s infinite' }}>
              <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: 'bold' }}>SCAN IN PROGRESS</h2>
              <p style={{ fontSize: '1.2rem', color: '#fff' }}><strong>AI Agent:</strong> "Please rotate your camera 360 degrees NOW to show your surroundings."</p>
              <h1 style={{ fontSize: '3rem', margin: '1rem 0' }}>{cameraTimeLeft}s</h1>
              <p style={{ fontSize: '1rem', color: '#fef08a' }}>Keep rotating. Questions will automatically load when verification completes.</p>
              
              <button 
                onClick={() => {
                  setIsWarningConfirmed(true);
                  setIsCameraCheckConfirmed(true);
                  setIsCameraCheckCompleted(true);
                  sessionStorage.setItem('isCameraCheckCompleted', 'true');
                }}
                style={{
                  marginTop: '1rem',
                  padding: '0.65rem 1.4rem',
                  fontSize: '0.9rem',
                  background: 'rgba(234, 179, 8, 0.25)',
                  color: '#facc15',
                  border: '1px solid rgba(234, 179, 8, 0.6)',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: '600'
                }}>
                ⚡ Bypass Room Scan & Start Interview Immediately (Admin Override / Ctrl+Shift+B)
              </button>
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
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                      <button 
                        onClick={() => {
                          isConfirmOpenRef.current = true;
                          setShowFinishConfirm(true);
                        }}
                        style={{
                          background: 'rgba(239, 68, 68, 0.1)',
                          color: '#f87171',
                          border: '1px solid rgba(239, 68, 68, 0.4)',
                          padding: '0.5rem 1rem',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          fontWeight: 'bold',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'; }}
                      >
                        Finish Interview
                      </button>
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
                    {currentQuestion?.marks && (
                      <span style={{ padding: '0.4rem 1rem', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold', background: 'rgba(255, 255, 255, 0.1)', color: '#cbd5e1' }}>
                        Marks: {currentQuestion.marks}
                      </span>
                    )}
                  </div>

                  <h2 style={{ fontSize: '1.4rem', marginBottom: '2rem', lineHeight: '1.6', fontWeight: '500', color: '#f8fafc' }}>
                    {typeof currentQuestion === 'string' ? currentQuestion : currentQuestion?.question}
                  </h2>

                  <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#cbd5e1' }}>
                    Speak your answer naturally. The AI is listening...
                  </h3>

                  <div style={{
                    width: '100%',
                    minHeight: '150px',
                    padding: '1.5rem',
                    borderRadius: '12px',
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#fff',
                    fontSize: '1.2rem',
                    lineHeight: '1.6',
                    marginBottom: '1.5rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    textAlign: 'center'
                  }}>
                    {answerText ? (
                      <p style={{ fontStyle: 'italic', color: '#cbd5e1' }}>"{answerText}"</p>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                        <div style={{
                          width: '50px',
                          height: '50px',
                          borderRadius: '50%',
                          background: 'rgba(59, 130, 246, 0.2)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          animation: 'pulse 1.5s infinite'
                        }}>
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                            <line x1="12" y1="19" x2="12" y2="23"></line>
                            <line x1="8" y1="23" x2="16" y2="23"></line>
                          </svg>
                        </div>
                        <p style={{ color: '#94a3b8', fontSize: '1rem' }}>Listening to your response...</p>
                      </div>
                    )}
                  </div>
                  
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    {isSubmitting ? 'Evaluating response...' : 'Stop speaking for 3 seconds to automatically submit.'}
                  </div>
                </div>
              )}
            </>
          )}

          {showFinishConfirm && (
            <div style={{
              position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
              background: 'rgba(0,0,0,0.8)', zIndex: 9999,
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <div style={{
                background: '#1e293b', padding: '2.5rem', borderRadius: '16px',
                border: '1px solid rgba(255,255,255,0.1)', maxWidth: '500px', textAlign: 'center'
              }}>
                <h2 style={{ color: '#f87171', marginBottom: '1rem' }}>End Interview?</h2>
                <p style={{ color: '#f8fafc', marginBottom: '2rem', lineHeight: '1.6', fontSize: '1.1rem' }}>
                  Are you sure you want to finish the interview early? This will submit your current progress and expire the interview link.
                </p>
                <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                  <button className="btn-secondary" onClick={() => {
                    isConfirmOpenRef.current = false;
                    setShowFinishConfirm(false);
                  }}>
                    Cancel
                  </button>
                  <button className="btn-primary" style={{ background: '#ef4444' }} onClick={() => {
                    isConfirmOpenRef.current = false;
                    setShowFinishConfirm(false);
                    finishInterview();
                  }}>
                    Yes, End Interview
                  </button>
                </div>
              </div>
            </div>
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
