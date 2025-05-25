import React, { useState } from 'react';
import DetectionOptions from './components/DetectionOptions';
import ResultDisplay from './components/ResultDisplay';
import VideoResultDisplay from './components/videoComponents/VideoResultDisplay';
import ImageUploadSection from './components/ImageUploadSection';
import Header from './components/Header';
import VideoUploadSection from './components/videoComponents/VideoUploadSection';
import './App.css';
import ProcessingButton from './components/ProcessingButton';

function App() {
    const baseURL = 'http://127.0.0.1:5174';

    const [mainFile, setMainFile] = useState(null);
    const [mainVideo, setMainVideo] = useState(null);

    const [referenceFile, setReferenceFile] = useState(null);
    const [detectionMode, setDetectionMode] = useState('all');
    const [confidenceThreshold, setConfidenceThreshold] = useState(3);
    const [boundingBoxThreshold, setBoundingBoxThreshold] = useState(25);
    const [imageUrl, setImageUrl] = useState('None');
    const [imageResultMessage, setImageResultMessage] = useState('');
    const [videoResultMessage, setVideoResultMessage] = useState('');
    const [boundingBoxInfoImage, setBoundingBoxInfoImage] = useState([]);
    const [boundingBoxInfoVideo, setBoundingBoxInfoVideo] = useState([]);

    const [processingMode, setProcessingMode] = useState('Image');
    const [videoData, setVideoData] = useState('None');

    // Default white
    const [selectedBBColor, setSelectedBBColor] = useState('#FFFFFF');

    // This is used to show the progress of the video processing
    const [progress, setProgress] = useState({
        progress_percentage: 0,
        total_frames: 0,
    });

    const handleSubmit = async () => {
        if (
            (!mainFile || (detectionMode === 'specific' && !referenceFile)) &&
            processingMode === 'Image'
        ) {
            setImageResultMessage('Please upload the required images.');
            return;
        }
        if (
            (!mainVideo || (detectionMode === 'specific' && !referenceFile)) &&
            processingMode === 'Video'
        ) {
            setVideoResultMessage(
                'Please upload the required video and reference image.',
            );
            return;
        }

        const formData = new FormData();
        if (processingMode === 'Image') {
            formData.append('main_image', mainFile);
            if (detectionMode === 'specific') {
                formData.append('reference_image', referenceFile);
                formData.append('confidence', confidenceThreshold);
            }
        } else if (processingMode === 'Video') {
            formData.append('main_video', mainVideo);
            if (detectionMode === 'specific') {
                formData.append('reference_image', referenceFile); //reference image
                formData.append('confidence', confidenceThreshold);
            }
        }
        formData.append('bounding_box_threshold', boundingBoxThreshold);
        formData.append('bb_color', selectedBBColor);

        // Reset previous result state
        // setImageUrl('None');
        // setBoundingBoxInfo([]);
        if (processingMode === 'Image') {
            setImageResultMessage('Processing...');
            setImageUrl('None');
        } else {
            // Video processing
            setProgress({
                progress_percentage: 0,
                total_frames: 0,
            });
            setVideoResultMessage('Processing...');
        }

        const endpoint =
            processingMode === 'Image'
                ? detectionMode === 'all'
                    ? '/image/detect-all'
                    : '/image/detect-specific'
                : detectionMode === 'all'
                ? '/video/detect-all'
                : '/video/detect-specific';

        try {
            const response = await fetch(`${baseURL}${endpoint}`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                // Now fetch a json object
                // Object contains the output image and bouding box information

                const data = await response.json();
                console.log('Response data:', data);

                // check if the response contains a message
                // A message can be returned if the process is canceled
                if (data.message) {
                    if (processingMode === 'Image') {
                        setImageResultMessage(data.message);
                    } else {
                        setVideoResultMessage(data.message);
                    }
                    return;
                }

                // Save the image url from the data object.
                if (processingMode === 'Image') {
                    setImageUrl(`data:image/jpeg;base64,${data.image}`);
                    setBoundingBoxInfoImage(data.bounding_boxes);
                    setImageResultMessage('Processing completed successfully!');
                } else {
                    // Video processing
                    setVideoData(data);
                    setBoundingBoxInfoVideo(data.bounding_boxes);
                    setVideoResultMessage('Processing completed successfully!');
                }
            } else {
                // Response code not ok
                if (processingMode === 'Image') {
                    setImageResultMessage('Processing failed. Try again.');
                } else {
                    setVideoResultMessage('Processing failed. Try again.');
                }
            }
        } catch (error) {
            console.error(error);
            if (processingMode === 'Image') {
                setImageResultMessage('Error processing the file.');
            } else {
                setVideoResultMessage('Error processing the file.');
            }
        }
    };

    const handleCancelProcessing = async () => {
        if (processingMode === 'Image') {
            try {
                const response = await fetch(`${baseURL}/image/cancel`, {
                    method: 'POST',
                });

                if (response.ok) {
                    setImageResultMessage('Processing cancelled');
                } else {
                    setImageResultMessage('Failed to cancel processing.');
                }
            } catch (error) {
                console.error(error);
                setImageResultMessage('Error canceling processing.');
            }
        } else if (processingMode === 'Video') {
            // Cancel video processing
            try {
                const response = await fetch(`${baseURL}/video/cancel`, {
                    method: 'POST',
                });

                if (response.ok) {
                    setVideoResultMessage('Processing cancelled');
                } else {
                    setVideoResultMessage('Failed to cancel processing.');
                }
            } catch (error) {
                console.error(error);
                setVideoResultMessage('Error canceling processing.');
            }
        }
    };

    if (processingMode === 'Video') {
        return (
            <div>
                <Header
                    setProcessingMode={setProcessingMode}
                    processingMode={processingMode}
                />
                <div className="flex flex-col items-center justify-center text-white text-2xl">
                    <VideoUploadSection
                        mainVideo={mainVideo}
                        setMainVideo={setMainVideo}
                        referenceFile={referenceFile}
                        setReferenceFile={setReferenceFile}
                        detectionMode={detectionMode}
                    />

                    <DetectionOptions
                        detectionMode={detectionMode}
                        setDetectionMode={setDetectionMode}
                        setConfidenceThreshold={setConfidenceThreshold}
                        confidenceThreshold={confidenceThreshold}
                        setBoundingBoxThreshold={setBoundingBoxThreshold}
                        boundingBoxThreshold={boundingBoxThreshold}
                        setSelectedBBColor={setSelectedBBColor}
                        selectedBBColor={selectedBBColor}
                    />

                    <ProcessingButton
                        handleCancelProcessing={handleCancelProcessing}
                        handleSubmit={handleSubmit}
                        resultMessage={videoResultMessage}
                    />

                    <VideoResultDisplay
                        resultMessage={videoResultMessage}
                        videoData={videoData}
                        // boundingBoxInfo={boundingBoxInfoVideo}
                        progress={progress}
                        setProgress={setProgress}
                    />
                </div>
            </div>
        );
    }

    if (processingMode === 'Image') {
        return (
            <div>
                <Header
                    setProcessingMode={setProcessingMode}
                    processingMode={processingMode}
                />
                <div className="flex flex-col items-center justify-center text-white text-2xl">
                    <ImageUploadSection
                        mainFile={mainFile}
                        setMainFile={setMainFile}
                        referenceFile={referenceFile}
                        setReferenceFile={setReferenceFile}
                        detectionMode={detectionMode}
                    />

                    <DetectionOptions
                        detectionMode={detectionMode}
                        setDetectionMode={setDetectionMode}
                        setConfidenceThreshold={setConfidenceThreshold}
                        confidenceThreshold={confidenceThreshold}
                        setBoundingBoxThreshold={setBoundingBoxThreshold}
                        boundingBoxThreshold={boundingBoxThreshold}
                        setSelectedBBColor={setSelectedBBColor}
                        selectedBBColor={selectedBBColor}
                    />

                    <ProcessingButton
                        handleCancelProcessing={handleCancelProcessing}
                        handleSubmit={handleSubmit}
                        resultMessage={imageResultMessage}
                    />

                    <ResultDisplay
                        resultMessage={imageResultMessage}
                        imageUrl={imageUrl}
                        boundingBoxInfo={boundingBoxInfoImage}
                    />
                </div>
            </div>
        );
    }
}

export default App;
