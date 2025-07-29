import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/pages/HomePage';
import ProcessingPage from './components/pages/ProcessingPage';
import ResultsPage from './components/pages/ResultsPage';
import VisualizerPage from './components/pages/VisualizerPage';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/processing/:taskId" element={<ProcessingPage />} />
        <Route path="/results/:taskId" element={<ResultsPage />} />
        <Route path="/visualizer/:taskId" element={<VisualizerPage />} />
      </Routes>
    </Router>
  );
}

export default App;
