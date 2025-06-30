import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import './Fileform.css';

function Fileform() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [useCamera, setUseCamera] = useState(false);
  const webcamRef = useRef(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setPreview(imageSrc);
    setFile(dataURLtoFile(imageSrc, 'captured.jpg'));
  }, [webcamRef]);

  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new File([u8arr], filename, { type: mime });
  };

  const HandleFileInputChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const HandleSubmit = async (event) => {
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
      <button onClick={() => setUseCamera(!useCamera)}>
        {useCamera ? 'Use File Upload' : 'Use Camera'}
      </button>

      <form onSubmit={HandleSubmit}>
        {useCamera ? (
          <>
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={350}
              videoConstraints={{
                    facingMode: { exact: "environment" } // Use "user" for front camera
                }}
            />
            <button type="button" onClick={capture}>Capture</button>
          </>
        ) : (
          <input type="file" accept="image/*" onChange={HandleFileInputChange} />
        )}
        <button type="submit">Predict</button>
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
