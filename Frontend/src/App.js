import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';

// --- Login Component ---
// This component will be rendered when the user is not authenticated.
function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState(''); // For success messages like "Login successful!"
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    setMessage(''); // Clear previous messages
    setIsLoading(true);

    // Create FormData for sending username and password as form data
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
      // Send login request to your FastAPI backend
      const response = await axios.post('http://localhost:8000/login', formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // Important for FastAPI's Form(...)
        },
      });

      setMessage(response.data.message); // Display success message from backend
      localStorage.setItem('accessToken', response.data.access_token); // Store the token
      setTimeout(() => {
        onLoginSuccess(); // Call the parent's login success handler
      }, 500); // Small delay for UI feedback
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={loginStyles.container}>
      <h2 style={loginStyles.title}>Login to Cognito AI</h2>
      <form onSubmit={handleSubmit} style={loginStyles.form}>
        <div style={loginStyles.inputGroup}>
          <label htmlFor="username" style={loginStyles.label}>Username:</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={loginStyles.input}
            disabled={isLoading}
          />
        </div>
        <div style={loginStyles.inputGroup}>
          <label htmlFor="password" style={loginStyles.label}>Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={loginStyles.input}
            disabled={isLoading}
          />
        </div>
        {error && <p style={loginStyles.error}>{error}</p>}
        {message && <p style={loginStyles.successMessage}>{message}</p>}
        <button type="submit" style={loginStyles.button} disabled={isLoading}>
          {isLoading ? 'Logging In...' : 'Login'}
        </button>
      </form>
      <p style={{ marginTop: '20px', fontSize: '14px', color: '#777' }}>
        (Contact admin if password is forgotten.)
      </p>
    </div>
  );
}

// --- ChatAppContent Component (Your existing chat logic) ---
// This component now accepts an `onLogout` prop from the parent.
function ChatAppContent({ onLogout }) {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Use useCallback to memoize this function, preventing unnecessary re-renders
  const handleApiError = useCallback((err) => {
    setError(err.response?.data?.detail || 'Something went wrong with the request.');
    setMessages(prev => [...prev, {
      text: 'Sorry, there was an error processing your request. Please try again.',
      sender: 'bot',
      isError: true
    }]);
  }, []); // Empty dependency array means it only gets created once

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setMessages(prev => [...prev, {
        text: `Document uploaded: ${selectedFile.name}`,
        sender: 'user',
        isFile: true
      }]);
    }
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please upload a document first');
      return;
    }
    if (!query.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsLoading(true);
    setError('');

    const userMessage = { text: query, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('query', query);

      // IMPORTANT: If your /analyze endpoint in FastAPI eventually requires
      // authentication (e.g., a JWT token), you would add it here:
      // const token = localStorage.getItem('accessToken');
      // const config = {
      //   headers: {
      //     'Content-Type': 'multipart/form-data',
      //     'Authorization': `Bearer ${token}`
      //   },
      // };
      // const result = await axios.post('http://localhost:8000/analyze', formData, config);

      // For this current dummy setup, /analyze does not require a token
      const result = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMessages(prev => [...prev, {
        text: result.data.response,
        sender: 'bot'
      }]);
    } catch (err) {
      handleApiError(err); // Use the memoized error handler
    } finally {
      setIsLoading(false);
    }
  };

  // Call the onLogout prop when the logout button is clicked
  const handleLogout = () => {
    setFile(null); // Clear app state on logout
    setQuery('');
    setMessages([]);
    setError('');
    onLogout(); // This will clear the token in localStorage and navigate to login
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}> {/* Added a header div for title and logout button */}
        <h1 style={styles.title}>Cognito AI</h1>
        <button onClick={handleLogout} style={styles.logoutButton}>Logout</button>
      </div>

      <div style={styles.chatContainer}>
        <div style={styles.messagesContainer}>
          {messages.length === 0 ? (
            <div style={styles.welcomeMessage}>
              <p>Welcome! Upload a document and ask questions about its content.</p>
              <p>Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, TIF, DOCX, XLSX, PPTX, HTML</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                style={{
                  ...styles.message,
                  ...(message.sender === 'user' ? styles.userMessage : styles.botMessage),
                  ...(message.isError && styles.errorMessage)
                }}
              >
                {message.isFile ? (
                  <div style={styles.fileMessage}>
                    <span style={styles.fileIcon}>📄</span>
                    <span>{message.text}</span>
                  </div>
                ) : (
                  message.sender === 'bot' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
                  ) : (
                    message.text
                  )
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div style={styles.botMessage}>
              <div style={styles.typingIndicator}>
                <div style={styles.dot}></div>
                <div style={styles.dot}></div>
                <div style={styles.dot}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} style={styles.inputForm}>
          {!file && (
            <div style={styles.fileUploadContainer}>
              <label htmlFor="file" style={styles.fileUploadLabel}>
                <span style={styles.uploadIcon}>📁</span> Upload Document
              </label>
              <input
                type="file"
                id="file"
                onChange={handleFileChange}
                style={styles.hiddenFileInput}
                accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.docx,.xlsx,.pptx,.html"
              />
            </div>
          )}
          {file && (
            <div style={styles.uploadedFileName}>
              Document selected: <strong>{file.name}</strong>
              <button
                type="button"
                onClick={() => { setFile(null); setMessages(prev => prev.filter(msg => !msg.isFile)); }}
                style={styles.clearFileButton}
              >
                ✖
              </button>
            </div>
          )}
          <div style={styles.inputGroup}>
            <input
              type="text"
              value={query}
              onChange={handleQueryChange}
              style={styles.textInput}
              placeholder="Ask a question about the document..."
              disabled={!file || isLoading}
            />
            <button
              type="submit"
              style={styles.sendButton}
              disabled={isLoading || !file || !query.trim()}
            >
              <span style={styles.sendIcon}>➤</span>
            </button>
          </div>
          {error && <div style={styles.error}>{error}</div>}
        </form>
      </div>
    </div>
  );
}

// --- Main App Component (Controls authentication flow) ---
function App() {
  // State to track authentication status, checking localStorage for persistence
  // Now checks for 'accessToken' instead of 'dummyLoggedIn'
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem('accessToken') // Convert truthy/falsy to boolean
  );

  // Hook for programmatic navigation
  const navigate = useNavigate();

  // Callback for when login is successful
  const handleLoginSuccess = useCallback(() => {
    setIsAuthenticated(true);
    navigate('/app'); // Navigate to the app route on successful login
  }, [navigate]); // navigate is a dependency of useCallback

  // Callback for when logout occurs
  const handleLogout = useCallback(() => {
    setIsAuthenticated(false);
    localStorage.removeItem('accessToken'); // Remove the token
    navigate('/login'); // Navigate to the login route on logout
  }, [navigate]); // navigate is a dependency of useCallback

  // Conditional rendering based on authentication status
  return (
    <Routes>
      {/* Route for login page */}
      <Route
        path="/login"
        element={isAuthenticated ? <Navigate to="/app" replace /> : <Login onLoginSuccess={handleLoginSuccess} />}
      />

      {/* Route for the main application, protected */}
      <Route
        path="/app"
        element={
          isAuthenticated ? (
            <ChatAppContent onLogout={handleLogout} /> // Pass logout handler
          ) : (
            <Navigate to="/login" replace /> // Redirect to login if not authenticated
          )
        }
      />

      {/* Default route: Redirect based on authentication status */}
      <Route
        path="/"
        element={isAuthenticated ? <Navigate to="/app" /> : <Navigate to="/login" />}
      />

      {/* Fallback for any other undefined routes */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

// --- Wrapper component to include Router, as App itself uses Router hooks ---
// This is the component you would render in your index.js or main.js
function AppWithRouter() {
  return (
    <Router>
      <App />
    </Router>
  );
}

// --- Styles (kept separate for readability, same as before) ---
const loginStyles = {
    container: {
        maxWidth: '400px',
        margin: '80px auto',
        padding: '30px',
        border: '1px solid #ddd',
        borderRadius: '10px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
        backgroundColor: 'white',
        textAlign: 'center',
    },
    title: {
        marginBottom: '25px',
        color: '#34495e',
        fontSize: '28px',
    },
    form: {
        display: 'flex',
        flexDirection: 'column',
        gap: '15px',
    },
    inputGroup: {
        textAlign: 'left',
    },
    label: {
        display: 'block',
        marginBottom: '8px',
        fontSize: '15px',
        color: '#555',
        fontWeight: 'bold',
    },
    input: {
        width: 'calc(100% - 20px)',
        padding: '10px',
        borderRadius: '5px',
        border: '1px solid #ccc',
        fontSize: '16px',
        boxSizing: 'border-box',
    },
    button: {
        padding: '12px 20px',
        backgroundColor: '#3498db',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        fontSize: '18px',
        fontWeight: 'bold',
        marginTop: '15px',
        transition: 'background-color 0.3s ease',
        '&:hover': {
            backgroundColor: '#2980b9',
        },
        '&:disabled': {
            backgroundColor: '#a0cbe0',
            cursor: 'not-allowed',
        }
    },
    error: {
        color: '#e74c3c',
        fontSize: '14px',
        marginTop: '10px',
    },
    successMessage: {
        color: '#28a745',
        fontSize: '14px',
        marginTop: '10px',
    },
};

const styles = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: '#e9eff6',
    boxSizing: 'border-box',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  title: {
    textAlign: 'center',
    color: '#2c3e50',
    marginBottom: '0px',
    fontSize: '28px',
    fontWeight: '600',
    textShadow: '1px 1px 2px rgba(0,0,0,0.05)',
    flexGrow: 1,
  },
  logoutButton: {
    padding: '8px 15px',
    backgroundColor: '#e74c3c',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'background-color 0.3s ease',
    '&:hover': {
      backgroundColor: '#c0392b',
    }
  },
  chatContainer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid #c0c0c0',
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: '0 5px 20px rgba(0,0,0,0.15)',
    backgroundColor: '#ffffff',
  },
  messagesContainer: {
    flex: 1,
    padding: '20px',
    overflowY: 'auto',
    backgroundColor: '#f9fbff',
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
  },
  welcomeMessage: {
    textAlign: 'center',
    color: '#7f8c8d',
    marginTop: '20px',
    lineHeight: '1.6',
    fontSize: '16px',
    padding: '0 20px',
  },
  message: {
    maxWidth: '80%',
    padding: '12px 16px',
    borderRadius: '20px',
    lineHeight: '1.5',
    wordWrap: 'break-word',
    fontSize: '15px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#3498db',
    color: 'white',
    borderBottomRightRadius: '8px',
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#ecf0f1',
    color: '#2c3e50',
    borderBottomLeftRadius: '8px',
    // These styles apply to direct child elements of botMessage when rendered by ReactMarkdown
    // Note: These are for demonstration. For full Markdown styling, you might use a dedicated CSS library
    // like "github-markdown-css" and apply it to a div wrapping ReactMarkdown.
    '& p': {
      marginBottom: '10px',
    },
    '& ul, & ol': {
      paddingLeft: '20px',
      marginBottom: '10px',
    },
    '& pre': {
      backgroundColor: '#f8f8f8',
      padding: '10px',
      borderRadius: '5px',
      overflowX: 'auto',
      marginBottom: '10px',
    },
    '& code': {
      fontFamily: 'monospace',
      backgroundColor: '#e0e0e0',
      padding: '2px 4px',
      borderRadius: '3px',
    },
    '& table': {
      width: '100%',
      borderCollapse: 'collapse',
      marginBottom: '10px',
    },
    '& th, & td': {
      border: '1px solid #ccc',
      padding: '8px',
      textAlign: 'left',
    },
    '& th': {
      backgroundColor: '#f0f0f0',
    }
  },
  errorMessage: {
    backgroundColor: '#e74c3c',
    color: 'white',
  },
  fileMessage: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontWeight: 'bold',
  },
  fileIcon: {
    fontSize: '22px',
    color: '#2980b9',
  },
  typingIndicator: {
    display: 'flex',
    gap: '6px',
    padding: '12px',
    alignItems: 'center',
  },
  dot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    backgroundColor: '#ccc',
    animation: 'bounce 1.4s infinite ease-in-out both',
  },
  inputForm: {
    padding: '15px 20px',
    backgroundColor: '#ffffff',
    borderTop: '1px solid #e0e0e0',
  },
  fileUploadContainer: {
    marginBottom: '15px',
    textAlign: 'center',
  },
  fileUploadLabel: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '10px 20px',
    backgroundColor: '#f2f2f2',
    borderRadius: '25px',
    cursor: 'pointer',
    fontSize: '15px',
    color: '#555',
    transition: 'background-color 0.3s, box-shadow 0.3s',
    border: '1px solid #ddd',
    '&:hover': {
      backgroundColor: '#e0e0e0',
      boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
    }
  },
  hiddenFileInput: {
    display: 'none',
  },
  uploadIcon: {
    marginRight: '10px',
    fontSize: '18px',
    color: '#007bff',
  },
  uploadedFileName: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '8px 15px',
    backgroundColor: '#eaf4fd',
    borderRadius: '20px',
    marginBottom: '10px',
    fontSize: '14px',
    color: '#34495e',
    border: '1px solid #bbdffb',
    gap: '10px',
  },
  clearFileButton: {
    background: 'none',
    border: 'none',
    color: '#e74c3c',
    fontSize: '18px',
    cursor: 'pointer',
    marginLeft: '10px',
  },
  inputGroup: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    padding: '12px 18px',
    borderRadius: '25px',
    border: '1px solid #ccc',
    fontSize: '16px',
    outline: 'none',
    transition: 'border-color 0.3s, box-shadow 0.3s',
    '&:focus': {
      borderColor: '#007bff',
      boxShadow: '0 0 0 3px rgba(0, 123, 255, 0.25)',
    },
    '&:disabled': {
      backgroundColor: '#e9ecef',
      cursor: 'not-allowed',
    }
  },
  sendButton: {
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    border: 'none',
    backgroundColor: '#007bff',
    color: 'white',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    transition: 'background-color 0.3s, transform 0.2s',
    '&:hover': {
      backgroundColor: '#0056b3',
      transform: 'scale(1.05)',
    },
    '&:disabled': {
      backgroundColor: '#a0cbe0',
      cursor: 'not-allowed',
      transform: 'none',
    }
  },
  sendIcon: {
    // No specific style needed, using font size from button
  },
  error: {
    color: '#e74c3c',
    fontSize: '13px',
    marginTop: '10px',
    textAlign: 'center',
    fontWeight: 'bold',
  },
};

// Export the wrapper component that includes the Router
export default AppWithRouter;
