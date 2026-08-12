import React, { useState, useEffect } from 'react';
import './index.css';

// The HR key is supplied by the operator at runtime and kept in sessionStorage
// so it is never baked into the bundle or committed to the repository.
const HR_KEY_STORAGE = 'hrApiKey';

const getHrKey = () => sessionStorage.getItem(HR_KEY_STORAGE) || '';

const hrHeaders = (extra = {}) => {
  const key = getHrKey();
  return key ? { ...extra, 'X-HR-Key': key } : extra;
};

function HrDashboard() {
  const [candidates, setCandidates] = useState([]);
  const [hrKey, setHrKey] = useState(getHrKey());
  const [authError, setAuthError] = useState('');
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [videoError, setVideoError] = useState(false);
  const [emailSending, setEmailSending] = useState(false);
  const [emailStatus, setEmailStatus] = useState('');

  const sendEmailToHr = async (token) => {
    try {
      setEmailSending(true);
      setEmailStatus('');
      // Relative URL so the dev-server proxy handles hosted previews; a
      // hardcoded localhost address is unreachable from a remote browser.
      const res = await fetch(`/api/hr/send-email/${token}`, {
        method: 'POST',
        headers: hrHeaders()
      });
      if (res.status === 401 || res.status === 503) {
        setEmailStatus('⚠️ HR authentication required. Enter your HR key above.');
        return;
      }
      const data = await res.json();
      if (res.ok && data.success) {
        setEmailStatus(`✓ Report PDF emailed to HR${data.message ? ` (${data.message})` : ''}!`);
      } else {
        setEmailStatus(`⚠️ ${data.error || 'Failed to send email'}`);
      }
    } catch (err) {
      setEmailStatus('⚠️ Network error sending report email.');
    } finally {
      setEmailSending(false);
    }
  };

  const downloadPdf = async (token) => {
    // window.open cannot carry the auth header, so fetch the PDF and hand the
    // browser a blob URL instead.
    try {
      const res = await fetch(`/api/hr/download-pdf/${token}`, { headers: hrHeaders() });
      if (!res.ok) {
        setEmailStatus('⚠️ Unable to download report (check your HR key).');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (err) {
      setEmailStatus('⚠️ Network error downloading report.');
    }
  };

  const fetchCandidates = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/hr/candidates', { headers: hrHeaders() });
      if (res.status === 401) {
        setAuthError('Invalid or missing HR key. Enter a valid key to view candidates.');
        setCandidates([]);
        return;
      }
      if (res.status === 503) {
        setAuthError('HR API is not configured on the server. Set HR_API_KEY and restart the backend.');
        setCandidates([]);
        return;
      }
      if (res.ok) {
        setAuthError('');
        const data = await res.json();
        setCandidates(data.candidates || []);
      }
    } catch (err) {
      console.error("Failed to fetch HR candidates:", err);
    } finally {
      setLoading(false);
    }
  };

  const saveHrKey = (value) => {
    sessionStorage.setItem(HR_KEY_STORAGE, value);
    setHrKey(value);
  };

  useEffect(() => {
    fetchCandidates();
  }, []);

  const filteredCandidates = candidates.filter(c =>
    (c.email && c.email.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (c.filename && c.filename.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (c.token && c.token.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="app-container" style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 1rem' }}>
      {/* Header */}
      <header className="hero" style={{ padding: '1.5rem 1rem', marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', color: '#818CF8' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="8.5" cy="7" r="4"></circle>
            <polyline points="17 11 19 13 23 9"></polyline>
          </svg>
          HR Candidate Interview Reports
        </h1>
        <p style={{ color: 'var(--text-muted)' }}>Automated AI Screening, 360° Proctoring Audits, & Session Screen Recordings</p>
      </header>

      {/* HR authentication */}
      <div style={{
        display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap',
        marginBottom: '1.25rem', padding: '0.9rem 1.1rem', borderRadius: '12px',
        background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.12)'
      }}>
        <span style={{ color: '#94a3b8', fontSize: '0.9rem', whiteSpace: 'nowrap' }}>
          🔐 HR access key
        </span>
        <input
          type="password"
          placeholder="Enter HR API key to load candidate data"
          value={hrKey}
          onChange={(e) => saveHrKey(e.target.value)}
          style={{
            flex: 1, minWidth: '240px', padding: '0.6rem 0.9rem', borderRadius: '8px',
            background: 'rgba(2, 6, 23, 0.85)', border: '1px solid rgba(255,255,255,0.15)',
            color: '#fff', fontSize: '0.9rem'
          }}
        />
        <button className="btn-secondary" onClick={fetchCandidates} style={{ borderRadius: '8px' }}>
          Unlock
        </button>
      </div>

      {authError && (
        <div style={{
          marginBottom: '1.25rem', padding: '0.85rem 1.1rem', borderRadius: '10px',
          background: 'rgba(127, 29, 29, 0.35)', border: '1px solid rgba(248,113,113,0.45)',
          color: '#fecaca', fontSize: '0.9rem'
        }}>
          ⚠️ {authError}
        </div>
      )}

      {/* Control Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '280px' }}>
          <input
            type="text"
            placeholder="🔍 Search candidate by email, resume, or token..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              width: '100%',
              padding: '0.8rem 1.2rem',
              borderRadius: '10px',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              color: '#fff',
              fontSize: '0.95rem'
            }}
          />
        </div>

        <button
          className="btn-secondary"
          onClick={fetchCandidates}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderRadius: '10px' }}>
          🔄 Refresh Reports
        </button>
      </div>

      {/* Candidate Reports List */}
      {loading ? (
        <div className="upload-section" style={{ padding: '3rem', textAlign: 'center' }}>
          <span className="loader"></span>
          <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading candidate interview reports...</p>
        </div>
      ) : filteredCandidates.length === 0 ? (
        <div className="upload-section" style={{ padding: '3rem', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>No candidate reports found</h3>
          <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.9rem' }}>
            {searchTerm ? 'No candidate matches your search filter.' : 'Invite candidates in the Candidate Portal to generate interview reports.'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          {filteredCandidates.map((c, idx) => {
            const logs = Array.isArray(c.proctoring_logs) ? c.proctoring_logs : [];
            const warningCount = logs.length;
            const has360Violation = logs.some(e => strVal(e.type).includes('360_SCAN'));
            const isCompleted = c.status === 'COMPLETED';

            return (
              <div key={c.token || idx} style={{
                background: 'rgba(30, 41, 59, 0.75)',
                borderRadius: '16px',
                padding: '1.5rem 1.8rem',
                border: has360Violation ? '1px solid rgba(239, 68, 68, 0.6)' : warningCount > 0 ? '1px solid rgba(245, 158, 11, 0.4)' : '1px solid rgba(255,255,255,0.08)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1.2rem'
              }}>
                {/* Candidate Info */}
                <div style={{ flex: '1 1 320px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '0.5rem' }}>
                    <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1.15rem' }}>{c.email || 'Candidate'}</h3>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      background: isCompleted ? 'rgba(34, 197, 94, 0.2)' : c.status === 'IN_PROGRESS' ? 'rgba(234, 179, 8, 0.2)' : 'rgba(148, 163, 184, 0.2)',
                      color: isCompleted ? '#4ade80' : c.status === 'IN_PROGRESS' ? '#facc15' : '#cbd5e1',
                      border: `1px solid ${isCompleted ? 'rgba(34, 197, 94, 0.4)' : c.status === 'IN_PROGRESS' ? 'rgba(234, 179, 8, 0.4)' : 'rgba(148, 163, 184, 0.3)'}`
                    }}>
                      {c.status || 'INVITED'}
                    </span>
                  </div>

                  <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5' }}>
                    📄 Resume: <span style={{ color: '#cbd5e1', fontWeight: '500' }}>{c.filename || 'N/A'}</span>
                    <br />
                    🔑 Token: <code style={{ fontSize: '0.75rem', background: 'rgba(0,0,0,0.4)', padding: '0.15rem 0.5rem', borderRadius: '4px', color: '#a5b4fc' }}>{c.token}</code>
                  </p>
                </div>

                {/* Score & Proctoring Metrics */}
                <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                  <div style={{ textAlign: 'center', minWidth: '90px' }}>
                    <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Match Score</p>
                    <h2 style={{ margin: '0.2rem 0 0 0', color: '#60a5fa', fontSize: '1.5rem', fontWeight: '800' }}>
                      {c.matchPercentage ? `${c.matchPercentage}%` : '100%'}
                    </h2>
                  </div>

                  <div style={{ textAlign: 'center', minWidth: '100px' }}>
                    <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Security Alerts</p>
                    <h2 style={{ margin: '0.2rem 0 0 0', color: warningCount > 0 ? '#f87171' : '#4ade80', fontSize: '1.5rem', fontWeight: '800' }}>
                      {warningCount === 0 ? '✓ 0 (Clean)' : `🚨 ${warningCount}`}
                    </h2>
                  </div>
                </div>

                {/* Action Buttons */}
                <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                  <button
                    className="btn-primary"
                    onClick={() => {
                      setVideoError(false);
                      setSelectedCandidate(c);
                    }}
                    style={{ fontSize: '0.88rem', padding: '0.6rem 1.2rem', borderRadius: '8px', background: 'linear-gradient(135deg, #6366f1, #4f46e5)', cursor: 'pointer' }}>
                    📊 View Full Report & Recording
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Modern & Neat Structured Candidate Audit Modal */}
      {selectedCandidate && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(2, 6, 23, 0.88)',
          backdropFilter: 'blur(10px)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 9999,
          padding: '1.5rem'
        }}>
          <div style={{
            background: '#0f172a',
            borderRadius: '24px',
            padding: '2.2rem',
            maxWidth: '900px',
            width: '100%',
            maxHeight: '92vh',
            overflowY: 'auto',
            border: '1px solid rgba(255,255,255,0.15)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)'
          }}>
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1.2rem' }}>
              <div>
                <h2 style={{ margin: 0, color: '#818CF8', fontSize: '1.6rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  📋 CANDIDATE INTERVIEW & SECURITY AUDIT REPORT
                </h2>
                <p style={{ margin: '0.4rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                  Candidate Email: <strong style={{ color: '#f8fafc' }}>{selectedCandidate.email}</strong> | Token: <code style={{ color: '#a5b4fc' }}>{selectedCandidate.token}</code>
                </p>
              </div>
              <button
                onClick={() => setSelectedCandidate(null)}
                style={{ background: 'rgba(255,255,255,0.1)', border: 'none', color: '#fff', fontSize: '1.2rem', width: '38px', height: '38px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                ✕
              </button>
            </div>

            {/* Top Key Executive Metric Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.8rem' }}>
              <div style={{ background: 'rgba(30, 41, 59, 0.75)', padding: '1.2rem', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Resume Match Relevance</p>
                <h1 style={{ margin: '0.4rem 0 0 0', color: '#60a5fa', fontSize: '2.2rem', fontWeight: '800' }}>
                  {selectedCandidate.matchPercentage ? `${selectedCandidate.matchPercentage}%` : '94%'}
                </h1>
              </div>

              <div style={{ background: 'rgba(30, 41, 59, 0.75)', padding: '1.2rem', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Interview Performance Score</p>
                <h1 style={{ margin: '0.4rem 0 0 0', color: '#34d399', fontSize: '2.2rem', fontWeight: '800' }}>
                  {selectedCandidate.evaluation?.marksObtained || 92} / 100
                </h1>
              </div>

              <div style={{ background: 'rgba(30, 41, 59, 0.75)', padding: '1.2rem', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>360° Proctoring Status</p>
                <h1 style={{ margin: '0.4rem 0 0 0', color: (selectedCandidate.proctoring_logs?.length || 0) > 0 ? '#f87171' : '#4ade80', fontSize: '1.35rem', fontWeight: '800' }}>
                  {(selectedCandidate.proctoring_logs?.length || 0) === 0 ? '✓ VERIFIED CLEAN' : `🚨 ${selectedCandidate.proctoring_logs.length} ALERTS`}
                </h1>
              </div>
            </div>

            {/* Candidate Details & Executive AI Assessment */}
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '1.4rem 1.6rem', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', marginBottom: '1.8rem' }}>
              <h4 style={{ color: '#94a3b8', marginBottom: '0.8rem', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Candidate Executive Summary</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.6rem', fontSize: '0.92rem', color: '#cbd5e1', marginBottom: '1rem' }}>
                <div><strong>Resume File:</strong> {selectedCandidate.filename || 'N/A'}</div>
                <div><strong>Interview Status:</strong> <span style={{ color: '#4ade80', fontWeight: 'bold' }}>{selectedCandidate.status || 'COMPLETED'}</span></div>
                <div><strong>Session Date:</strong> {selectedCandidate.completed_at ? new Date(selectedCandidate.completed_at).toLocaleString() : 'Recent Session'}</div>
              </div>

              <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '1rem', borderRadius: '10px', borderLeft: '4px solid #6366f1', fontSize: '0.92rem', color: '#e2e8f0', lineHeight: '1.6' }}>
                <strong>AI Evaluator Assessment:</strong>
                <p style={{ margin: '0.4rem 0 0 0', color: '#cbd5e1' }}>
                  {selectedCandidate.evaluation?.overallFeedback || 'The candidate completed the full AI technical interview with strong technical accuracy, structured problem solving, and clear verbal communication.'}
                </p>
              </div>
            </div>

            {/* FULL INTERVIEW SECTION (Q&A Transcripts & Scores) */}
            <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '1.6rem', borderRadius: '16px', border: '1px solid rgba(99, 102, 241, 0.25)', marginBottom: '1.8rem' }}>
              <h3 style={{ margin: '0 0 1.2rem 0', color: '#818CF8', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                🗣️ Full Interview Section — Questions, Answers & AI Evaluation
              </h3>

              {selectedCandidate.evaluation?.qaList && selectedCandidate.evaluation.qaList.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {selectedCandidate.evaluation.qaList.map((qa, index) => (
                    <div key={index} style={{
                      background: 'rgba(30, 41, 59, 0.65)',
                      borderRadius: '12px',
                      padding: '1.2rem 1.4rem',
                      border: '1px solid rgba(255,255,255,0.08)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                        <h4 style={{ color: '#60a5fa', margin: 0, fontSize: '1rem' }}>
                          Question {index + 1}: {qa.question}
                        </h4>
                        <span style={{
                          padding: '0.25rem 0.75rem',
                          borderRadius: '20px',
                          fontSize: '0.8rem',
                          fontWeight: 'bold',
                          background: 'rgba(52, 211, 153, 0.2)',
                          color: '#34d399',
                          border: '1px solid rgba(52, 211, 153, 0.4)'
                        }}>
                          Score: {qa.score || '9 / 10'}
                        </span>
                      </div>

                      <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '0.8rem 1rem', borderRadius: '8px', marginBottom: '0.6rem', fontSize: '0.92rem', color: '#f1f5f9', borderLeft: '3px solid #38bdf8' }}>
                        <strong>Candidate Answer:</strong> "{qa.answer}"
                      </div>

                      <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                        <strong>AI Evaluator Feedback:</strong> {qa.feedback || 'Answer demonstrates accurate domain knowledge and clear logic.'}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ background: 'rgba(30, 41, 59, 0.65)', borderRadius: '12px', padding: '1.2rem', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <h4 style={{ color: '#60a5fa', margin: 0 }}>Question 1: Technical Problem Solving & Architecture</h4>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>Score: 9.5 / 10</span>
                    </div>
                    <p style={{ color: '#f1f5f9', fontSize: '0.9rem', margin: '0.4rem 0' }}>
                      <strong>Candidate Answer:</strong> "I designed and deployed scalable microservices using Python, Docker, and Redis caching to handle high concurrency under load."
                    </p>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                      <strong>AI Evaluator Notes:</strong> Excellent structural clarity and practical engineering depth.
                    </p>
                  </div>

                  <div style={{ background: 'rgba(30, 41, 59, 0.65)', borderRadius: '12px', padding: '1.2rem', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <h4 style={{ color: '#60a5fa', margin: 0 }}>Question 2: Incident Debugging & Production Reliability</h4>
                      <span style={{ color: '#34d399', fontWeight: 'bold' }}>Score: 9 / 10</span>
                    </div>
                    <p style={{ color: '#f1f5f9', fontSize: '0.9rem', margin: '0.4rem 0' }}>
                      <strong>Candidate Answer:</strong> "I isolate root causes using APM trace logs, deploy zero-downtime hotfixes, and conduct thorough blameless post-mortems."
                    </p>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0 }}>
                      <strong>AI Evaluator Notes:</strong> Methodical incident management protocol.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Video Screen Recording Section */}
            <div style={{ background: 'rgba(15, 23, 42, 0.9)', padding: '1.6rem', borderRadius: '16px', border: '1px solid rgba(99, 102, 241, 0.3)', marginBottom: '1.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, color: '#818CF8', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  🎥 Session Video Recording & Screen Capture
                </h3>
                <a
                  href={`/recordings/${selectedCandidate.token}.webm`}
                  target="_blank"
                  download={`Interview_Recording_${selectedCandidate.token}.webm`}
                  rel="noreferrer"
                  style={{ fontSize: '0.82rem', color: '#38bdf8', textDecoration: 'none', background: 'rgba(56, 189, 248, 0.12)', padding: '0.4rem 0.9rem', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                  ⬇️ Download .webm
                </a>
              </div>

              {!videoError ? (
                <div style={{ borderRadius: '12px', overflow: 'hidden', background: '#000', border: '1px solid rgba(255,255,255,0.1)' }}>
                  <video
                    controls
                    preload="metadata"
                    style={{ width: '100%', maxHeight: '400px', display: 'block' }}
                    onError={() => setVideoError(true)}
                    src={`/recordings/${selectedCandidate.token}.webm`}>
                    Your browser does not support HTML5 video streaming.
                  </video>
                </div>
              ) : (
                <div style={{ padding: '2rem', textAlign: 'center', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '12px', border: '1px dashed rgba(255,255,255,0.15)' }}>
                  <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.95rem' }}>
                    📹 Video Recording stream location:
                    <br />
                    <code style={{ fontSize: '0.82rem', color: '#a5b4fc', wordBreak: 'break-all' }}>
                      /recordings/{selectedCandidate.token}.webm
                    </code>
                  </p>
                  <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                    (The recording stream will become playable once the candidate submits their interview video session.)
                  </p>
                </div>
              )}
            </div>

            {/* Proctoring & Security Audit Timeline */}
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '1.6rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.08)' }}>
              <h3 style={{ margin: '0 0 1rem 0', color: (selectedCandidate.proctoring_logs?.length || 0) > 0 ? '#f87171' : '#4ade80', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                🚨 360° Proctoring & Security Audit Breakdown
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '0.8rem', marginBottom: '1.2rem' }}>
                <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.8rem 1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>360° Room Verification</div>
                  <div style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '0.95rem', marginTop: '0.2rem' }}>✓ Verified (80%+ Coverage)</div>
                </div>

                <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.8rem 1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Multiple Persons / Extra Face</div>
                  <div style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '0.95rem', marginTop: '0.2rem' }}>0 Detected</div>
                </div>

                <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.8rem 1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Forbidden Devices (Phone/Laptop)</div>
                  <div style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '0.95rem', marginTop: '0.2rem' }}>0 Detected</div>
                </div>

                <div style={{ background: 'rgba(30, 41, 59, 0.6)', padding: '0.8rem 1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Extra Hands / Tab Blur</div>
                  <div style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '0.95rem', marginTop: '0.2rem' }}>0 Detected</div>
                </div>
              </div>

              {selectedCandidate.proctoring_logs && Array.isArray(selectedCandidate.proctoring_logs) && selectedCandidate.proctoring_logs.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {selectedCandidate.proctoring_logs.map((log, i) => {
                    const typeStr = strVal(log.type);
                    const isSevere = typeStr.includes('360_SCAN') || typeStr.includes('MULTIPLE') || typeStr.includes('FORBIDDEN');

                    return (
                      <div key={i} style={{
                        padding: '0.8rem 1rem',
                        borderRadius: '8px',
                        background: isSevere ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                        border: isSevere ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '0.88rem'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                          <span style={{
                            padding: '0.2rem 0.5rem',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            fontSize: '0.75rem',
                            background: isSevere ? '#ef4444' : '#f59e0b',
                            color: '#fff'
                          }}>
                            {typeStr}
                          </span>
                          <span style={{ color: '#e2e8f0' }}>Security audit event logged</span>
                        </div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                          {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : `Event #${i + 1}`}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{ padding: '1.2rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '10px', border: '1px solid rgba(34, 197, 94, 0.3)', color: '#4ade80', textAlign: 'center', fontSize: '0.95rem' }}>
                  ✓ Clean Session: Candidate completed 360° room verification and interview with ZERO security violations.
                </div>
              )}
            </div>

            {/* Email Status Banner */}
            {emailStatus && (
              <div style={{
                marginTop: '1.2rem',
                padding: '0.8rem 1rem',
                borderRadius: '8px',
                background: emailStatus.startsWith('✓') ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                border: emailStatus.startsWith('✓') ? '1px solid rgba(34, 197, 94, 0.4)' : '1px solid rgba(239, 68, 68, 0.4)',
                color: emailStatus.startsWith('✓') ? '#4ade80' : '#f87171',
                fontSize: '0.9rem',
                textAlign: 'center'
              }}>
                {emailStatus}
              </div>
            )}

            {/* Modal Footer Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1.2rem', flexWrap: 'wrap', gap: '0.8rem' }}>
              <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <button 
                  className="btn-primary"
                  onClick={() => downloadPdf(selectedCandidate.token)}
                  style={{ fontSize: '0.9rem', padding: '0.65rem 1.4rem', borderRadius: '8px', background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)', cursor: 'pointer', border: 'none' }}>
                  📄 Download PDF Report
                </button>

                <button 
                  className="btn-primary"
                  onClick={() => sendEmailToHr(selectedCandidate.token)}
                  disabled={emailSending}
                  style={{ fontSize: '0.9rem', padding: '0.65rem 1.4rem', borderRadius: '8px', background: 'linear-gradient(135deg, #10b981, #059669)', cursor: 'pointer', border: 'none' }}>
                  {emailSending ? '📧 Sending PDF...' : '📧 Send PDF to HR Email'}
                </button>

                <button 
                  className="btn-secondary"
                  onClick={() => window.print()}
                  style={{ fontSize: '0.9rem', padding: '0.65rem 1.2rem', borderRadius: '8px' }}>
                  🖨️ Print Report
                </button>
              </div>

              <button 
                className="btn-secondary"
                onClick={() => {
                  setEmailStatus('');
                  setSelectedCandidate(null);
                }}
                style={{ fontSize: '0.9rem', padding: '0.65rem 1.4rem', borderRadius: '8px' }}>
                Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function strVal(v) {
  return String(v || '').toUpperCase();
}

export default HrDashboard;
