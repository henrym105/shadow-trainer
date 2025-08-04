import React, { useState, useEffect } from 'react';
import '../../styles/PlotViewer.css';

function PlotViewer({ taskId, plotType, title }) {
  const [plotUrl, setPlotUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Plot explanations
  const plotExplanations = {
    hip_rotation: "NOTE: Relative to the starting position. When the red line is above the black line, you've rotated your hips more than the pro.",
    shoulder_rotation: "NOTE: Relative to the starting position. When the red line is above the black line, you've rotated your shoulders more than the pro.",
    hip_shoulder_separation: "NOTE: Positive when the hips are rotated ahead of your shoulders (more separation is better). Negative values when the shoulders are ahead of your hips."
  };

  useEffect(() => {
    const fetchPlot = async () => {
      if (!taskId || !plotType) return;
      
      try {
        const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiBaseUrl}/videos/${taskId}/plots/${plotType}`);
        
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          setPlotUrl(url);
        } else {
          setError('Plot not available');
        }
      } catch (error) {
        console.error('Failed to fetch plot:', error);
        setError('Failed to load plot');
      } finally {
        setLoading(false);
      }
    };

    fetchPlot();

    // Cleanup URL when component unmounts
    return () => {
      if (plotUrl) {
        URL.revokeObjectURL(plotUrl);
      }
    };
  }, [taskId, plotType]);

  if (loading) {
    return (
      <div className="plot-viewer loading">
        <div className="plot-spinner"></div>
        <p>Loading {title}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="plot-viewer error">
        <p>⚠️ {error}</p>
      </div>
    );
  }

  return (
    <div className="plot-viewer">
      <h4 className="plot-title">{title}</h4>
      {plotUrl && (
        <img 
          src={plotUrl} 
          alt={title}
          className="plot-image"
        />
      )}
      <p className="plot-explanation">
        {plotExplanations[plotType] || `${plotType} plot comparing you and a pro.`}
      </p>
    </div>
  );
}

export default PlotViewer;