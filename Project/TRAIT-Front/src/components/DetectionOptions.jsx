import React from 'react';
import SearchSpecificLogoOptions from './SearchSpecificLogoOptions';
import ThresholdInput from './ThresholdInput';
import ConfidenceThresholdInput from './ConfidenceThresholdInput';

const DetectionOptions = ({
    detectionMode,
    setDetectionMode,
    confidenceThreshold,
    setConfidenceThreshold,
    setBoundingBoxThreshold,
    boundingBoxThreshold,
    selectedBBColor,
    setSelectedBBColor,
}) => {
    const handleDetectionModeChange = (mode) => {
        setDetectionMode(mode);
    };

    return (
        <div>
            <fieldset className="fieldset w-l bg-base-200 border border-base-200 p-6 rounded-box">
                <legend className="fieldset-legend">Detection Mode</legend>
                <div>
                    <div className="flex justify-center detection-mode-toggle">
                        <button
                            className={`btn ${
                                detectionMode === 'all'
                                    ? 'btn-primary btn-active'
                                    : 'btn-outline btn-secondary'
                            } mr-4`}
                            onClick={() => handleDetectionModeChange('all')}
                        >
                            Search All Logos
                        </button>

                        <button
                            className={`btn ${
                                detectionMode === 'specific'
                                    ? 'btn-primary btn-active'
                                    : ' btn-outline btn-secondary'
                            }`}
                            onClick={() =>
                                handleDetectionModeChange('specific')
                            }
                        >
                            Search Specific Logo
                        </button>
                    </div>
                    <ThresholdInput
                        thresholdTitle={'Bounding Box Threshold'}
                        boundingThreshold={boundingBoxThreshold}
                        setBoundingThreshold={setBoundingBoxThreshold}
                        tooltipDescription={
                            'Bounding Box Threshold sets the minimum confidence score required for the model to consider a detected object as a logo. Higher values mean only highly confident detections are shown, while lower values allow more detections, including less certain ones.'
                        }
                        selectedBBColor={selectedBBColor}
                        setSelectedBBColor={setSelectedBBColor}
                    />
                </div>

                <SearchSpecificLogoOptions
                    detectionMode={detectionMode}
                    confidenceThreshold={confidenceThreshold}
                    setConfidenceThreshold={setConfidenceThreshold}
                />
            </fieldset>
        </div>
    );
};

export default DetectionOptions;
