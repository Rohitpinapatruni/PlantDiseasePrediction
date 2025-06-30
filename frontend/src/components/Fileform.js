import React, { useState } from 'react';
import './Fileform.css';

function Fileform() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [prediction, setPrediction] = useState(null);

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
            <form onSubmit={HandleSubmit}>
                <input type="file" accept="image/*" onChange={HandleFileInputChange} />
                <button type="submit">Predict</button>
            </form>

            {file && <p>Selected file: {file.name}</p>}

            {preview && (
                <div className="preview-container">
                    <h3>Image Preview:</h3>
                    <img src={preview} alt="Preview" className="preview-image"/>
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
