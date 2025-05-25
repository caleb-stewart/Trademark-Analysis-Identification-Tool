import React, { useState, useEffect } from 'react';

const ResultDetails = ({ boundingBoxInfo = [] }) => {
    const [visibleBoxes, setVisibleBoxes] = useState(boundingBoxInfo);

    useEffect(() => {
        setVisibleBoxes(boundingBoxInfo);
    }, [boundingBoxInfo]);

    if (!visibleBoxes || visibleBoxes.length === 0) {
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
        setVisibleBoxes((prev) => prev.filter((_, i) => i !== indexToRemove));
    };

    return (
        <fieldset className="fieldset w-full bg-base-200 border border-base-300 p-6 rounded-box">
            <legend className="fieldset-legend">Detected Logo Metrics</legend>
            <div className="flex flex-col gap-6 mt-4">
                {visibleBoxes.map((box, index) => (
                    <div key={index} className="flex items-center gap-4">
                        <img
                            src={`data:image/jpeg;base64,${box.cropped_logo}`}
                            alt="Cropped Logo"
                            style={{ maxWidth: '100%', maxHeight: '200px' }}
                            className="rounded-md"
                        />
                        <div className="text-base flex-1">
                            This logo takes {box.box_coverage_percentage}% of
                            the main image.
                            {box.confidence && (
                                <div>
                                    Confidence:{' '}
                                    {getConfidenceScore(box.confidence)}
                                </div>
                            )}
                        </div>
                        <button
                            onClick={() => handleRemove(index)}
                            className="text-red-500 hover:text-red-700 text-xl font-bold"
                            title="Remove this logo"
                        >
                            ❌
                        </button>
                    </div>
                ))}
            </div>
        </fieldset>
    );
};

export default ResultDetails;
