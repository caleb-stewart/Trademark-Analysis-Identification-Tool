import { useState } from 'react'

export default function ColorPalette({ selectedBBColor, setSelectedBBColor }) {
    // Color selector
    return (
        <div
            className="tooltip tooltip-bottom"
            data-tip="Select a color for the bounding box"
        >
            {/* Tooltip for the color selector */}
            <input
                type="color"
                value={selectedBBColor}
                onChange={(e) => setSelectedBBColor(e.target.value)}
                className="input w-10 h-10 p-0 cursor-pointer"
            />
        </div>
    )
}
