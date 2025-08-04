import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function LandingPage() {
  const navigate = useNavigate();
  const [currentFeature, setCurrentFeature] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  const features = [
    {
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze your motion with professional-grade precision, for a fraction of the cost.",
      icon: "🤖"
    },
    {
      title: "3D Motion Tracking",
      description: "Accurate 3D pose estimation and skeleton tracking with any camera. No special equipment needed, just your smartphone or webcam.",
      icon: "🎯"
    },
    {
      title: "Professional Comparison",
      description: "Compare your technique against your favorite MLB pitchers to identify improvement areas, and get AI generated coaching tips to help you Play Like a Pro.",
      icon: "⚡"
    }
  ];

  useEffect(() => {
    setIsVisible(true);
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleGetStarted = () => {
    navigate('/app');
  };

  return (
    <div className="landing-page">
      <div className="landing-container">
        {/* Header */}
        <header className="landing-header">
          <div className="logo-section-large">
            <img 
              src="/Shadow Trainer Logo Only.png" 
              alt="Shadow Trainer Logo" 
              className="logo-large"
            />
            <div className="logo-text-large">
              <h1>Shadow Trainer</h1>
              <p>AI-Powered 3D Motion Analysis</p>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="hero-section">
          <div className={`hero-content ${isVisible ? 'visible' : ''}`}>
            <h2 className="hero-title">
              Perfect Your Athletic Performance with 
              <span className="highlight"> AI Precision</span>
            </h2>
            <p className="hero-subtitle">
              Upload your training videos and get instant professional-grade motion analysis. 
              Compare your technique against elite athletes and take your performance to the next level.
            </p>
            
            <div className="cta-section">
              <button className="cta-primary" onClick={handleGetStarted}>
                <span className="cta-icon">🚀</span>
                Get Started Now
              </button>
              <button className="cta-secondary" onClick={() => document.getElementById('features').scrollIntoView()}>
                Learn More
              </button>
            </div>
          </div>
        </main>

        {/* Features Section */}
        <section id="features" className="features-section">
          <h3 className="features-title">Why Choose Shadow Trainer?</h3>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div 
                key={index}
                className={`feature-card ${index === currentFeature ? 'active' : ''}`}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h4>{feature.title}</h4>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works */}
        <section className="how-it-works">
          <h3 className="section-title">How It Works</h3>
          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>Upload Your Video</h4>
                <p>Record your training session and upload it to our platform</p>
              </div>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>AI Analysis</h4>
                <p>Our AI analyzes your motion in 3D space with professional precision</p>
              </div>
            </div>
            <div className="step-arrow">→</div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Get Insights</h4>
                <p>Receive detailed feedback and comparisons with professional athletes</p>
              </div>
            </div>
          </div>
        </section>

        {/* Demo Section */}
        <section className="demo-section">
          <div className="demo-content">
            <h3>See Shadow Trainer in Action</h3>
            <p>Try our sample video to experience the power of AI motion analysis</p>
            <button className="demo-btn" onClick={handleGetStarted}>
              <span className="demo-icon">🎬</span>
              Try Sample Video
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer className="landing-footer">
          <div className="footer-content">
            <p>&copy; 2025 Shadow Trainer. Powered by cutting-edge AI technology.</p>
            <div className="footer-links">
              <a href="https://github.com/henrym105/shadow-trainer/tree/develop" target="_blank" rel="noopener noreferrer">
                View Source on GitHub
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default LandingPage;