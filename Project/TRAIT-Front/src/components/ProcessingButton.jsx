import React from 'react';

const ProcessingButton = ({
    resultMessage,
    handleSubmit,
    handleCancelProcessing,
}) => {
    {
        return resultMessage === 'Processing...' ? (
            <button
                className="mt-4 mb-4 btn btn-error btn-xl"
                onClick={handleCancelProcessing}
            >
                Cancel Processing
            </button>
        ) : (
            <button
                className="mt-4 mb-4 btn btn-success btn-xl"
                onClick={handleSubmit}
            >
                Submit for Detection
            </button>
        );
    }
};

export default ProcessingButton;
