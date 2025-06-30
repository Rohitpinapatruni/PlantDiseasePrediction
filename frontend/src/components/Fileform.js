import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { isMobile } from 'react-device-detect';
import './Fileform.css';

function Fileform() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [useCamera, setUseCamera] = useState(false);
  const webcamRef = useRef(null);

  const videoConstraints = {
    facingMode: isMobile ? { exact: 'environment' } : 'user',
    width: 350
  };

  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new File([u8arr], filename, { type: mime });
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setPreview(imageSrc);
      setFile(dataURLtoFile(imageSrc, 'captured.jpg'));
    }
  }, [webcamRef]);

  const handleFileInputChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('https://leafpredbackend.onrender.com/predict', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
      } else {
        console.error('Server error');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className="fileform">
      <h2>Upload or Capture Image</h2>

      <form onSubmit={handleSubmit} className="form-container">
        {!useCamera && (
          <div
            className="drag-drop-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <p>Drag and drop an image here</p>
          </div>
        )}

        <div className="upload-options">
          <button
            type="button"
            className="toggle-button"
            onClick={() => {
              setUseCamera(!useCamera);
              setFile(null);
              setPreview(null);
              setPrediction(null);
            }}
          >
            {useCamera ? '📁 Switch to File Upload' : '📷 Switch to Camera'}
          </button>

          {!useCamera && (
            <input
              type="file"
              accept="image/*"
              className="file-input"
              onChange={handleFileInputChange}
            />
          )}

          {useCamera && (
            <div className="camera-container">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
              />
              <br />
              <button
                type="button"
                className="capture-button"
                onClick={capture}
              >
                📸 Capture
              </button>
            </div>
          )}
        </div>

        <button type="submit" className="predict-button" disabled={!file}>
          🧠 Predict
        </button>
      </form>

      {preview && (
        <div className="preview-container">
          <h3>Image Preview:</h3>
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h2>Prediction Result:</h2>
          <p><strong>Class:</strong> {prediction.class}</p>
          <p><strong>Confidence:</strong> {prediction.confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default Fileform;
