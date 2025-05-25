import React from 'react';
import UploadSection from '../UploadSection';

// This is where the user will upload the main video and the reference image

const VideoUploadSection = ({
    mainVideo,
    setMainVideo,
    referenceFile,
    setReferenceFile,
    detectionMode,
}) => {
    return (
        <div>
            <div className="mb-2 flex gap-4">
                <UploadSection
                    label="Upload Main Video"
                    file={mainVideo}
                    setFile={setMainVideo}
                    type = "video" // new validation
                />

                {detectionMode === 'specific' && (
                    <UploadSection
                        label="Upload Reference Image"
                        file={referenceFile}
                        setFile={setReferenceFile}
                        type="image" // new validation
                    />
                )}
            </div>
        </div>
    );
};

export default VideoUploadSection;
