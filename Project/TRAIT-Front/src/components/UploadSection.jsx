import React, { useState, useRef, useEffect } from 'react';

const UploadSection = ({ label, file, setFile, type = 'image' }) => {
    const [imagePreview, setImagePreview] = useState(null);
    const [videoPreview, setVideoPreview] = useState(null);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const allowedTypesMap = {
        image: ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'],
        video: ['video/mp4'],
    };
    const allowedTypes = allowedTypesMap[type] || [];

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (!selectedFile) return;

        if (!allowedTypes.includes(selectedFile.type)) {
            setError(`Only ${type.toUpperCase()} files are allowed.`);
            setFile(null);
            setImagePreview(null);
            setVideoPreview(null);
            return;
        }

        setError('');
        setFile(selectedFile);
    };

    useEffect(() => {
        if (!file) {
            setImagePreview(null);
            setVideoPreview(null);
            return;
        }

        if (type === 'image' && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onloadend = () => setImagePreview(reader.result);
            reader.readAsDataURL(file);
        } else if (type === 'video' && file.type === 'video/mp4') {
            const previewUrl = URL.createObjectURL(file);
            setVideoPreview(previewUrl);

            return () => {
                URL.revokeObjectURL(previewUrl);
            };
        }
    }, [file, type]);

    const handleClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    return (
        <div className="w-96 bg-base-100 border border-base-200 p-6 rounded-box">
            <div
                className={`border-4 border-dashed p-4 h-full rounded-md cursor-pointer ${
                    imagePreview || videoPreview ? 'border-transparent' : 'border-primary'
                }`}
                onClick={handleClick}
            >
                <div className="w-full aspect-video overflow-hidden rounded-md">
                    {type === 'image' && imagePreview ? (
                        <img
                            src={imagePreview}
                            alt="Preview"
                            className="w-full h-full object-cover rounded-md"
                        />
                    ) : type === 'video' && videoPreview ? (
                        <video
                            src={videoPreview}
                            autoPlay
                            muted
                            loop
                            className="w-full h-full object-cover rounded-md"
                        />
                    ) : (
                        <div className="flex justify-center items-center w-full h-full text-primary text-center">
                            {file ? file.name : label}
                        </div>
                    )}
                </div>
            </div>

            {error && (
                <div className="mt-2 text-red-500 text-sm text-center">{error}</div>
            )}

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept={allowedTypes.join(',')}
                hidden
            />
        </div>
    );
};

export default UploadSection;
