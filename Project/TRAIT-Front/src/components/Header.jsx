// Appears at the top of the page

// Set Image and Video to be buttons to switch between the two modes
const Header = ({ setProcessingMode, processingMode }) => {
    return (
        <div className="navbar bg-base-100 shadow-sm relative py-4 border-b-4 border-transparent mb-4 ">
            <div className="ml-2 navbar-start">
                <a className="font-semibold text-2xl">TRAIT</a>
            </div>

            {/* Image and Video buttons */}
            <div className="absolute left-1/2 transform -translate-x-1/2">
                <ul className="menu menu-horizontal px-1">
                    <li>
                        <a
                            className={`btn mr-4 ${
                                processingMode === 'Image'
                                    ? 'btn-active btn-primary'
                                    : ''
                            } `}
                            onClick={() => setProcessingMode('Image')}
                        >
                            Image
                        </a>
                    </li>
                    <li>
                        <a
                            className={`btn ${
                                processingMode === 'Video'
                                    ? 'btn-active btn-primary'
                                    : ''
                            } `}
                            onClick={() => setProcessingMode('Video')}
                        >
                            Video
                        </a>
                    </li>
                </ul>
            </div>

            
            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-blue-300 via-blue-400 to-blue-500 animate-pulse"></div>
        </div>
    );
};

export default Header;
