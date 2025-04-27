import React from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import PoseDetection from "./PoseDetection";
import SoftwarePose from "./SoftwarePose";

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <NavLink to="/live" className="tab">Live</NavLink>
          <NavLink to="/software" className="tab">Software</NavLink>
          <NavLink to="/aiengine" className="tab">AI Engine</NavLink>
          <NavLink to="/hardware" className="tab">Hardware</NavLink>
        </nav>
        <Routes>
          <Route path="/live" element={<PoseDetection />} />
          <Route path="/software" element={<SoftwarePose />} />
          <Route path="/aiengine" element={<div><h2>AI Engine - Coming Soon</h2></div>} />
          <Route path="/hardware" element={<div><h2>Hardware - Coming Soon</h2></div>} />
          <Route path="/" element={<PoseDetection />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
