import React, { useState } from "react";
import "./Login.css";

export default function Login({ onLoginSuccess }) {
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [registeredUsers, setRegisteredUsers] = useState({});

  const handleSubmit = (e) => {
    e.preventDefault();

    if (isRegistering) {
      // Register user
      if (registeredUsers[username]) {
        alert("Username already exists");
        return;
      }

      setRegisteredUsers((prev) => ({
        ...prev,
        [username]: password,
      }));
      alert("Registration successful! You can now log in.");
      setIsRegistering(false);
      setUsername("");
      setPassword("");
    } else {
      // Login
      if (registeredUsers[username] === password || (username === "user" && password === "pass")) {
        onLoginSuccess();
      } else {
        alert("Invalid credentials");
      }
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit} className="login-form">
        <h2>{isRegistering ? "Register" : "Welcome Back"}</h2>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">{isRegistering ? "Register" : "Login"}</button>

        <p className="toggle-auth" onClick={() => setIsRegistering(!isRegistering)}>
          {isRegistering
            ? "Already have an account? Log in"
            : "Don't have an account? Register"}
        </p>
      </form>
    </div>
  );
}
