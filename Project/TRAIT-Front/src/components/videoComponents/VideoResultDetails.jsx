import React, { useState, useEffect } from 'react';

const VideoResultDetails = ({
    croppedFramesSaved = [],
    appearanceCounts = [],
    totalFrames = 0,
}) => {
    const [visibleFrames, setVisibleFrames] = useState(croppedFramesSaved);

    useEffect(() => {
        setVisibleFrames(croppedFramesSaved);
    }, [croppedFramesSaved]);

    if (!visibleFrames || visibleFrames.length === 0) {
        return null;
    }

    const confidenceScores = { 2: 'Low', 3: 'Medium', 4: 'High' };
    const getConfidenceScore = (confidence) => {
        if (confidence >= 4) {
            return 'High';
        }
        return confidenceScores[confidence];
    };

    const handleRemove = (indexToRemove) => {
        setVisibleFrames((prev) => prev.filter((_, i) => i !== indexToRemove));
    };

    return (
        <fieldset className="fieldset w-l bg-base-200 border border-base-200 p-6 rounded-box justify-center mb-4">
            <legend className="fieldset-legend">Detected Logo Metrics</legend>

            {visibleFrames.map((frame, index) => {
                const count = appearanceCounts[index];

                if (count < 5) return null;

                return (
                    <div key={index} className="flex items-center gap-4 mb-4">
                        <img
                            src={`data:image/jpeg;base64,${frame.logo_base64}`}
                            alt="Cropped Logo"
                            style={{ maxWidth: '100%', maxHeight: '200px' }}
                        />
                        <div className="text-base flex-1">
                            This logo first appeared in frame {frame.frame_idx}{' '}
                            and was in approximately {count} frames of the
                            video. ({((count / totalFrames) * 100).toFixed(3)}%)
                        </div>
                        <button
                            onClick={() => handleRemove(index)}
                            className="text-red-500 hover:text-red-700 text-xl font-bold"
                            title="Remove this logo"
                        >
                            ❌
                        </button>
                    </div>
                );
            })}
        </fieldset>
    );
};

export default VideoResultDetails;
