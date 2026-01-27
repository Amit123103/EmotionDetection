import React, { useEffect } from 'react';

const WelcomePage = ({ onComplete }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onComplete();
        }, 6000); // 6 seconds

        return () => clearTimeout(timer);
    }, [onComplete]);

    return (
        <div className="welcome-page">
            <h1 className="welcome-text">INITIALIZING BIOMETRIC SCANS...</h1>

            <div className="cyber-loader-container">
                <div className="cyber-loader"></div>
                <div className="cyber-loader-inner"></div>
                <div className="cyber-text">SYSTEM LOADING</div>
            </div>
        </div>
    );
};

export default WelcomePage;
