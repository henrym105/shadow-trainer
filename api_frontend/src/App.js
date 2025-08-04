import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './components/pages/LandingPage';
import AppPage from './components/pages/AppPage';
import ProcessingPage from './components/pages/ProcessingPage';
import ResultsPage from './components/pages/ResultsPage';
import VisualizerPage from './components/pages/VisualizerPage';
import './App.css';
import './styles/LandingPage.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<AppPage />} />
        <Route path="/processing/:taskId" element={<ProcessingPage />} />
        <Route path="/results/:taskId" element={<ResultsPage />} />
        <Route path="/visualizer/:taskId" element={<VisualizerPage />} />
      </Routes>
    </Router>
  );
}

export default App;
