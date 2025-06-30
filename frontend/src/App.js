import React from 'react';
import './App.css'; // Import the CSS
import Fileform from "./components/Fileform";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Leaf Disease Prediction</h1>
        <h3>Note: This project may only work for the Potato plant.</h3>
      </header>
      <Fileform />
    </div>
  );
}

export default App;
