import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

function HrDashboard() {
  const [appState, setAppState] = useState('idle'); // idle, analyzing, result
  const [jobDescription, setJobDescription] = useState('');
  const [file, setFile] = useState(null);
  const [evaluation, setEvaluation] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file || !jobDescription.trim()) {
      alert("Please provide both a Job Description and a Candidate Resume.");
      return;
    }

    setAppState('analyzing');
    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jobDescription', jobDescription);

    try {
      const response = await fetch('http://localhost:8000/api/analyze-candidate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      
      if (data.error) {
        alert(data.error);
        setAppState('idle');
        return;
      }

      setEvaluation(data);
      setAppState('result');
    } catch (error) {
      console.error(error);
      alert('Error analyzing candidate. Please check the backend server.');
      setAppState('idle');
    }
  };

  const resetDashboard = () => {
    setAppState('idle');
    setFile(null);
    setEvaluation(null);
    // keeping JD in case they want to upload another resume against same JD
  };

  return (
    <div className="app-container" style={{ maxWidth: '1200px' }}>
      <header className="hero" style={{ padding: '2rem 1rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', color: '#818CF8' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="3" y1="9" x2="21" y2="9"></line>
            <line x1="9" y1="21" x2="9" y2="9"></line>
          </svg>
          HR Recruiter Dashboard
        </h1>
        <p>Advanced Candidate Screening & Semantic Skill Analysis</p>
      </header>

      {appState === 'idle' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', padding: '1rem' }}>
          
          <div className="upload-section" style={{ padding: '2rem', margin: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#f8fafc', textAlign: 'left' }}>1. Job Requirements</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem', textAlign: 'left' }}>
              Paste the complete Job Description or core requirements here.
            </p>
            <textarea 
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="e.g., We are looking for an AI Engineer with strong experience in Python, PyTorch, LLMs, and RAG architectures..."
              style={{
                width: '100%',
                flex: 1,
                minHeight: '250px',
                padding: '1rem',
                borderRadius: '8px',
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#f8fafc',
                fontFamily: 'inherit',
                resize: 'none',
                marginBottom: '1rem'
              }}
            />
          </div>

          <div className="upload-section" style={{ padding: '2rem', margin: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#f8fafc', textAlign: 'left' }}>2. Candidate Resume</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '2rem', textAlign: 'left' }}>
              Upload the candidate's resume (PDF, DOCX) for AI semantic analysis.
            </p>
            
            <div className="file-input-wrapper" style={{ margin: 'auto' }}>
              <button className="btn-primary" style={{ width: '100%' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                {file ? file.name : 'Select Resume'}
              </button>
              <input type="file" className="file-input" accept=".pdf,.doc,.docx,.txt" onChange={handleFileChange} />
            </div>

            <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
              <button 
                className="btn-primary" 
                style={{ backgroundColor: 'var(--accent)', width: '100%', opacity: (!file || !jobDescription) ? 0.5 : 1 }} 
                onClick={handleAnalyze}
                disabled={!file || !jobDescription}
              >
                Run AI Analysis
              </button>
            </div>
          </div>

        </div>
      )}

      {appState === 'analyzing' && (
        <div className="upload-section" style={{ marginTop: '2rem' }}>
          <div className="loader-wrapper">
            <span className="loader"></span>
            <h2 style={{ marginTop: '1.5rem', color: '#818CF8' }}>Deep AI Analysis in Progress...</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
              Extracting semantic skills, evaluating experience, and cross-referencing requirements.
            </p>
          </div>
        </div>
      )}

      {appState === 'result' && evaluation && (
        <div className="dashboard-results" style={{ animation: 'fadeIn 0.5s ease-out' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h2 style={{ fontSize: '2rem' }}>Recruiter Report</h2>
            <button className="btn-secondary" onClick={resetDashboard}>Analyze Another</button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem', alignItems: 'start' }}>
            
            {/* Main Report Column */}
            <div style={{ background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', padding: '2rem', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div className="markdown-content">
                <ReactMarkdown>{evaluation.reason}</ReactMarkdown>
              </div>
            </div>

            {/* Sidebar Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              
              {/* Score Card */}
              <div style={{ 
                  background: 'rgba(30, 41, 59, 0.7)', 
                  borderRadius: '16px', 
                  padding: '2rem', 
                  border: `2px solid ${evaluation.score >= 65 ? 'var(--success)' : 'var(--danger)'}`,
                  textAlign: 'center',
                  boxShadow: `0 0 20px ${evaluation.score >= 65 ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'}`
                }}>
                <h3 style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '1rem' }}>Overall Match Score</h3>
                <div style={{ fontSize: '4rem', fontWeight: 'bold', color: evaluation.score >= 65 ? 'var(--success)' : 'var(--danger)', lineHeight: 1 }}>
                  {evaluation.score}
                </div>
                <div style={{ marginTop: '1rem', fontSize: '1.1rem', fontWeight: 'bold', color: '#f8fafc' }}>
                  {evaluation.score >= 80 ? '✅ ' : (evaluation.score >= 65 ? '⚠️ ' : '❌ ')} 
                  {evaluation.next_round_status}
                </div>
              </div>

              {/* Quick Tags */}
              <div style={{ background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: '#cbd5e1' }}>Extracted Technical Skills</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {evaluation.skills?.length > 0 ? (
                    evaluation.skills.map((skill, i) => (
                      <span key={i} style={{ background: 'rgba(129, 140, 248, 0.2)', color: '#818CF8', padding: '0.2rem 0.6rem', borderRadius: '4px', fontSize: '0.8rem' }}>
                        {skill}
                      </span>
                    ))
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>None explicitly parsed into list.</span>
                  )}
                </div>
              </div>

              <div style={{ background: 'rgba(30, 41, 59, 0.7)', borderRadius: '16px', padding: '1.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: '#cbd5e1' }}>Identified Skill Gaps</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {evaluation.skill_gaps?.length > 0 ? (
                    evaluation.skill_gaps.map((gap, i) => (
                      <span key={i} style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5', padding: '0.2rem 0.6rem', borderRadius: '4px', fontSize: '0.8rem' }}>
                        {gap}
                      </span>
                    ))
                  ) : (
                    <span style={{ color: 'var(--success)', fontSize: '0.8rem' }}>No major gaps found in explicit list.</span>
                  )}
                </div>
              </div>

            </div>

          </div>
        </div>
      )}

    </div>
  );
}

export default HrDashboard;
