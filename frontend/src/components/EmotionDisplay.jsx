import React from 'react';

const EmotionDisplay = ({ emotionData }) => {
    if (!emotionData) {
        return (
            <div className="results-panel">
                <div className="emotion-card">
                    <div className="current-emotion">
                        <h2>Status</h2>
                        <div className="emotion-value" style={{ color: '#94a3b8', fontSize: '1.5rem' }}>
                            Waiting for face...
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    const { emotion, age, summary } = emotionData;

    const getEmotionColor = (e) => {
        const colors = {
            angry: '#ef4444', disgust: '#a3e635', fear: '#a855f7',
            happy: '#eab308', sad: '#3b82f6', surprise: '#f472b6',
            neutral: '#94a3b8', contempt: '#fb7185'
        };
        return colors[e] || '#fff';
    };

    return (
        <div className="results-panel">
            <div className="emotion-card" style={{ borderColor: getEmotionColor(emotion) }}>
                <div className="current-emotion">
                    <h2>Analyzed State</h2>

                    {/* Emotion */}
                    <div className="emotion-value" style={{ color: getEmotionColor(emotion), marginBottom: '0.5rem' }}>
                        {emotion}
                    </div>

                    {/* Age Badge */}
                    <div style={{ display: 'inline-block', background: 'rgba(255,255,255,0.1)', padding: '4px 12px', borderRadius: '20px', fontSize: '0.9rem', color: '#e2e8f0', marginBottom: '1rem', marginRight: '0.5rem' }}>
                        Age: {age || "N/A"}
                    </div>

                    {/* Gender Badge */}
                    <div style={{ display: 'inline-block', background: 'rgba(255,255,255,0.1)', padding: '4px 12px', borderRadius: '20px', fontSize: '0.9rem', color: '#e2e8f0', marginBottom: '1rem' }}>
                        Gender: {emotionData.gender || "N/A"}
                    </div>

                    {/* Summary */}
                    {summary && (
                        <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                            <p style={{ color: '#cbd5e1', fontSize: '1rem', fontStyle: 'italic', lineHeight: '1.5' }}>
                                "{summary}"
                            </p>
                        </div>
                    )}
                </div>
            </div>

            <div className="emotion-card">
                <p style={{ textAlign: 'center', opacity: 0.7, fontSize: '0.8rem' }}>
                    Multi-Modal Analysis Active
                    <br />
                    (Face + Eyes + Age + Voice)
                </p>
            </div>
        </div>
    );
};

export default EmotionDisplay;
