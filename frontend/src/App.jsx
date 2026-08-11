import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import CandidatePortal from './components/CandidatePortal';
import HrDashboard from './HrDashboard';
import './index.css';

function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<CandidatePortal />} />
            <Route path="/interview" element={<CandidatePortal directQA={true} />} />
            <Route path="/hr-dashboard" element={<HrDashboard />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
