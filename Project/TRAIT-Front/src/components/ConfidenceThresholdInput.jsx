import React from 'react'
import ToolTipDropDown from './ToolTipDropDown'

const ConfidenceThresholdInput = ({
    confidenceThreshold,
    setConfidenceThreshold,
    thresholdTitle,
    tooltipDescription,
}) => {
    const mapping = { 2: 'Low', 3: 'Medium', 4: 'High' }

    const handleSliderChange = (e) => {
        const value = parseInt(e.target.value, 10)
        setConfidenceThreshold(value)
    }

    return (
        <div className="confidence-threshold mt-6">
            <div className="text-base font-medium mb-2">
                {thresholdTitle}
                {': '}
                <ToolTipDropDown description={tooltipDescription} />
            </div>
            <div className="flex justify-between text-sm font-medium mb-2">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
            </div>
            <input
                type="range"
                min="2"
                max="4"
                step="1"
                value={confidenceThreshold}
                onChange={handleSliderChange}
                className="range range-primary"
            />
            <div className="text-center mt-2 font-semibold">
                {mapping[confidenceThreshold] || ''}
            </div>
        </div>
    )
}

export default ConfidenceThresholdInput
