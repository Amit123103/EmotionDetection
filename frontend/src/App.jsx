import React, { useState } from 'react';
import VideoFeed from './components/VideoFeed';
import EmotionDisplay from './components/EmotionDisplay';
import WelcomePage from './components/WelcomePage';
import './index.css';

function App() {
    const [emotionData, setEmotionData] = useState(null);
    const [showWelcome, setShowWelcome] = useState(true);

    if (showWelcome) {
        return <WelcomePage onComplete={() => setShowWelcome(false)} />;
    }

    return (
        <div className="container">
            <div className="header">
                <h1>Emotion AI</h1>
                <p>Real-time Deep Learning Emotion Recognition</p>
            </div>

            <div className="main-content">
                <VideoFeed onEmotionDetected={setEmotionData} />
                <EmotionDisplay emotionData={emotionData} />
            </div>
        </div>
    );
}

export default App;
