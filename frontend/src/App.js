import React from 'react';
import './App.css'; // Import the CSS
import Fileform from "./components/Fileform";
import backgroundImage from './background.jpeg'; // 👈 Update this to your actual image file name

function App() {
  return (
    <div className="App">
      {/* Background Layer */}
      <div
        className="background-layer"
        style={{ backgroundImage: `url(${backgroundImage})` }}
      ></div>

      {/* Main Content */}
      <div className="App-content">
        <header className="App-header">
          <h1>Leaf Disease Prediction</h1>
          <h3>Note: This project may only work for the Potato plant.</h3>
        </header>
        <Fileform />
      </div>
    </div>
  );
}

export default App;
