import React, { useState } from "react";
import Login from "./Login";
import TherapistApp from "./TherapistApp"; // This is where your AI Therapist functionality is

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleLoginSuccess = () => {
    setIsAuthenticated(true); // Change state to true once user logs in
  };

  return (
    <div className="App">
      {isAuthenticated ? (
        // If authenticated, render TherapistApp
        <TherapistApp />
      ) : (
        // If not authenticated, render Login
        <Login onLoginSuccess={handleLoginSuccess} />
      )}
    </div>
  );
}

export default App;
