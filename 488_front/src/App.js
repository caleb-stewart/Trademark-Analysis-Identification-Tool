import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [resultMessage, setResultMessage] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setResultMessage("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        setImageUrl(URL.createObjectURL(blob));
        setResultMessage("File processed successfully!");
      } else {
        setResultMessage("Processing failed. Try again.");
      }
    } catch (error) {
      setResultMessage("Error processing the image.");
      console.error(error);
    }
  };

  return (
    <div className="App">
      <header className="main-header">
        <h1>Trademark Identification & Analysis Engine</h1>
        <p>Upload your trademark documents for analysis</p>
      </header>

      <div className="App-header">
        <div className="file-upload-container">
          <div className="file-box">
            <label htmlFor="fileUpload" className="file-label">
              {file ? file.name : "Click to Upload"}
            </label>
            <input id="fileUpload" type="file" onChange={handleFileChange} />
          </div>
        </div>

        <button className="submit-button" onClick={handleSubmit}>
          Submit
        </button>

        <div className="results-box">
          {resultMessage && <p>{resultMessage}</p>}
          {imageUrl && <img src={imageUrl} alt="Processed Result" className="result-image" />}
        </div>
      </div>
    </div>
  );
}

export default App;