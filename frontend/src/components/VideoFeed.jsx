import React, { useRef, useEffect, useState } from 'react';

const VideoFeed = ({ onEmotionDetected }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const ws = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [error, setError] = useState(null);
    const recognitionRef = useRef(null);

    // Auto-start camera
    useEffect(() => {
        startVideo();
        startSpeechRecognition();

        return () => {
            stopVideo();
            if (ws.current) ws.current.close();
            if (recognitionRef.current) recognitionRef.current.stop();
        };
    }, []);

    const startSpeechRecognition = () => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;

            recognitionRef.current.onresult = (event) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        const finalTranscript = event.results[i][0].transcript;
                        // Send text to backend
                        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                            ws.current.send(JSON.stringify({
                                type: 'audio_text',
                                text: finalTranscript
                            }));
                        }
                    }
                }
            };

            recognitionRef.current.onerror = (event) => {
                console.log("Speech recognition error", event.error);
            };

            try {
                recognitionRef.current.start();
            } catch (e) {
                console.log("Speech recognition already started");
            }
        } else {
            console.log("Speech Recognition API not supported in this browser.");
        }
    };

    const startVideo = async () => {
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            // audio: true is good for permission, but we use Speech API for text
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsStreaming(true);
            }
        } catch (err) {
            console.error("Error accessing webcam:", err);
            setError("Camera/Mic access denied. Please allow permissions.");
        }
    };

    const stopVideo = () => {
        const stream = videoRef.current?.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setIsStreaming(false);
        }
    };

    // WebSocket Init
    useEffect(() => {
        // Connect to Live Render Backend
        ws.current = new WebSocket('wss://emotiondetection-w0t2.onrender.com/ws/detect');

        ws.current.onopen = () => console.log('Connected to Live Backend');

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.results && data.results.length > 0) {
                onEmotionDetected(data.results[0]);
                drawBoundingBox(data.results[0].box, data.results[0].eyes);
            } else {
                // If no face found for a frame, keep the old result for a purely visual "smoothness" 
                // OR clear it immediately. Let's clear it but maybe we can add a small debounce later.
                onEmotionDetected(null);
                clearCanvas();
            }
        };

        return () => { if (ws.current) ws.current.close(); };
    }, [onEmotionDetected]);

    const drawBoundingBox = (box, eyes) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Face
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 4;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // Eyes
        if (eyes) {
            ctx.strokeStyle = '#4ade80';
            eyes.forEach(eye => ctx.strokeRect(eye.x, eye.y, eye.w, eye.h));
        }
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    // Frame Loop
    useEffect(() => {
        if (!isStreaming) return;
        const interval = setInterval(() => {
            if (videoRef.current && ws.current && ws.current.readyState === WebSocket.OPEN) {
                const canvas = document.createElement('canvas');
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL('image/jpeg', 0.5);

                // Send Image Protocol
                ws.current.send(JSON.stringify({
                    type: 'image',
                    data: dataUrl.split(',')[1]
                }));
            }
        }, 200);
        return () => clearInterval(interval);
    }, [isStreaming]);

    return (
        <div className="video-container">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} />

            {error && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(15, 23, 42, 0.95)',
                    display: 'flex', flexDirection: 'column',
                    justifyContent: 'center', alignItems: 'center',
                    zIndex: 20, padding: '2rem', textAlign: 'center'
                }}>
                    <h3 style={{ color: '#ef4444' }}>Access Required</h3>
                    <p style={{ color: '#cbd5e1' }}>{error}</p>
                    <button onClick={startVideo} className="active">Try Again</button>
                </div>
            )}

            <div className="controls">
                <div style={{ color: 'white', background: 'rgba(0,0,0,0.5)', padding: '5px 10px', borderRadius: '15px', marginRight: '10px' }}>
                    🎤 Listening...
                </div>
                {!isStreaming ?
                    <button onClick={startVideo} className="active">Start Camera</button> :
                    <button onClick={stopVideo}>Stop Camera</button>
                }
            </div>
        </div>
    );
};

export default VideoFeed;
