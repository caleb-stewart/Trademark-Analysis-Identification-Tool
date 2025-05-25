import React from 'react';
import ConfidenceThresholdInput from './ConfidenceThresholdInput';

// Displays the Embedding Algorithm and Confidence Threshold options for specific detection mode
const DetectionSpecificOptions = ({
    detectionMode,
    confidenceThreshold,
    setConfidenceThreshold,
}) => {
    return (
        <div>
            {/* The is the slider for the user to choose confidence threshold */}
            {detectionMode === 'specific' && (
                <ConfidenceThresholdInput
                    thresholdTitle="Confidence Threshold"
                    confidenceThreshold={confidenceThreshold}
                    setConfidenceThreshold={setConfidenceThreshold}
                    tooltipDescription="Confidence Threshold controls how visually similar a detected logo must be to the reference logo. Higher values require nearly identical matches, while lower values allow more loosely similar logos to be detected."
                />
            )}
        </div>
    );
};

export default DetectionSpecificOptions;
