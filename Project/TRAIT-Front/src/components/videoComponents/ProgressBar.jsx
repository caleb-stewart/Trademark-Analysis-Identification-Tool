import React from 'react';

const ProgressBar = ({ progress_percent }) => {
    return (
        <div className="flex justify-center items-center h-full w-full">
            <div className="flex flex-col items-center space-y-2">
                <span className="text-lg font-medium">
                    {/* Display the % above the loading bar */}
                    {progress_percent !== 100 && progress_percent !== 0
                        ? progress_percent.toFixed(2) + '%'
                        : ' '}
                </span>

                {(progress_percent === 100 || progress_percent === 0) && (
                    <progress className="progress w-56"></progress>
                )}

                {progress_percent !== 100 && progress_percent !== 0 && (
                    <progress
                        className={'progress progress-primary w-56'}
                        value={progress_percent}
                        max="100"
                    ></progress>
                )}
            </div>
        </div>
    );
};

export default ProgressBar;
