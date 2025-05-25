import React from 'react';
import UploadSection from './UploadSection';

const ImageUploadSection = ({
    mainFile,
    setMainFile,
    referenceFile,
    setReferenceFile,
    detectionMode,
}) => {
    return (
        <div>
            <div className="mb-2 flex gap-4">
                <UploadSection
                    label="Upload Main Image"
                    file={mainFile}
                    setFile={setMainFile}
                    type="image" // new validation
                />

                {detectionMode === 'specific' && (
                    <UploadSection
                        label="Upload Reference Logo"
                        file={referenceFile}
                        setFile={setReferenceFile}
                        type="image" // new validation
                    />
                )}
            </div>
        </div>
    );
};

export default ImageUploadSection;
