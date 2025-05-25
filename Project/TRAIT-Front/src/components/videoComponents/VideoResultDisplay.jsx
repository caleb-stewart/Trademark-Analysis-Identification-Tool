import React from 'react';
import { useState, useEffect } from 'react';
import VideoResultDetails from './VideoResultDetails';
import ProgressBar from './ProgressBar';

const VideoResultDisplay = ({
    resultMessage,
    videoData,
    progress,
    setProgress,
}) => {
    // if (resultMessage === "Processing..." && !videoURL) {
    //     console.log("Haiiii")
    //     return <span class="loading loading-ring loading-md"></span>;
    // }

    if (!resultMessage) {
        return null;
    }

    useEffect(() => {
        // Update the loading bar progress every second
        if (resultMessage === 'Processing...') {
            const intervalId = setInterval(() => {
                fetch('http://localhost:5174/video/fetch-progress')
                    .then((response) => response.json())
                    .then((data) => {
                        setProgress(data);
                    })
                    .catch((error) => {
                        console.error('Error fetching progress:', error);
                    });
            }, 1000);

            // Clear interval when resultMessage updates or on unmount
            return () => clearInterval(intervalId);
        }
    }, [resultMessage]); // <- run effect when resultMessage changes

    return (
        <div>
            <fieldset className="fieldset w-l bg-base-200 border border-base-200 p-6 rounded-box justify-center">
                <legend className="fieldset-legend">Detected Logos</legend>

                {/* Red text if the message was not successful */}
                {resultMessage && (
                    <div className="flex justify-center items-center mb-4">
                        <p
                            className={`${
                                resultMessage !==
                                'Processing completed successfully!'
                                    ? 'text-red-500' // Red text for error
                                    : 'text-green-500' // Green text for success
                            } text-center text-xl`}
                        >
                            {resultMessage}
                        </p>
                    </div>
                )}

                {/* Loading Bar */}
                {resultMessage === 'Processing...' && progress && (
                    <ProgressBar
                        progress_percent={progress['progress_percentage']}
                    />
                )}

                {resultMessage === 'Processing completed successfully!' && (
                    <div className="w-full h-full relative">
                        <video
                            src="http://127.0.0.1:5174/video/fetch-processed-video"
                            controls
                            className="w-full h-full object-contain rounded-md"
                            style={{ maxWidth: '100%', maxHeight: '300px' }}
                        >
                            Your browser does not support the video tag.
                        </video>
                    </div>
                )}
            </fieldset>

            {resultMessage === 'Processing completed successfully!' && (
                <VideoResultDetails
                    croppedFramesSaved={videoData.saved_frames}
                    appearanceCounts={videoData.logo_appearance_count}
                    totalFrames={progress.total_frames}
                />
            )}
        </div>
    );
};

export default VideoResultDisplay;
