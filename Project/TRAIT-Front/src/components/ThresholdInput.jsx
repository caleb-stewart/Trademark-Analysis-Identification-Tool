import React from 'react';
import ToolTipDropDown from './ToolTipDropDown';
import ColorPalette from './ColorPalette';

const ThresholdInput = ({
    boundingThreshold,
    setBoundingThreshold,
    thresholdTitle,
    tooltipDescription,
    selectedBBColor,
    setSelectedBBColor,
}) => {
    const handleInputChange = (e) => {
        const value = e.target.value;

        // Allow the value to be a valid float (with one decimal) or empty
        if (
            value === '' || // Allow empty input for deletion
            value === '.' || // Allow a single decimal point
            (!isNaN(value) && value.indexOf('.') === value.lastIndexOf('.')) // Ensure only one decimal point
        ) {
            // Parse the value as a float
            const parsedValue = parseFloat(value);

            // Ensure the value is within the range [0, 100]
            if (!isNaN(parsedValue)) {
                if (parsedValue <= 0) {
                    setBoundingThreshold(1);
                } else if (parsedValue >= 100) {
                    setBoundingThreshold(100);
                } else {
                    setBoundingThreshold(value);
                }
            } else if (value === '') {
                setBoundingThreshold(''); // Allow the input to be empty
            }
        }
    };

    // Input field and range slider for the threshold
    return (
        <div className="confidence-threshold mt-6">
            <div className="flex items-center gap-2 text-base font-medium mb-1">
                {thresholdTitle}
                {': '}
                <ToolTipDropDown description={tooltipDescription} />
                <input
                    className="input w-16 input-primary"
                    placeholder={boundingThreshold}
                    onChange={handleInputChange}
                    value={boundingThreshold}
                />
                %
                <ColorPalette
                    selectedBBColor={selectedBBColor}
                    setSelectedBBColor={setSelectedBBColor}
                />
            </div>

            <input
                type="range"
                min="1"
                max="100"
                value={boundingThreshold}
                onChange={(e) => setBoundingThreshold(e.target.value)}
                className="range range-primary"
            />
        </div>
    );
};

export default ThresholdInput;
