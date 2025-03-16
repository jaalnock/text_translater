import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import {
  Container,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Paper,
  CircularProgress,
} from "@mui/material";

function App() {
  const [inputType, setInputType] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("hi");
  const [result, setResult] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    setSummary(null);

    const formData = new FormData();
    formData.append("inputType", inputType);
    formData.append("language", language);

    if (inputType === "text") {
      formData.append("text", text);
    } else if (inputType === "link") {
      formData.append("url", url);
    } else if (inputType === "file" || inputType === "image") {
      formData.append("file", file);
    }

    try {
      const response = await axios.post(
        "http://localhost:5000/process",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = async () => {
    setLoading(true);
    setError("");
    setSummary(null);

    const formData = new FormData();
    formData.append("inputType", inputType);
    formData.append("language", language);

    if (inputType === "text") {
      formData.append("text", text);
    } else if (inputType === "link") {
      formData.append("url", url);
    } else if (inputType === "file" || inputType === "image") {
      formData.append("file", file);
    }

    try {
      const response = await axios.post(
        "http://localhost:5000/summarize",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setSummary(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" className="container">
      <Typography variant="h4" gutterBottom>
        Text Processing App
      </Typography>

      <FormControl fullWidth className="form-control">
        <InputLabel>Input Type</InputLabel>
        <Select
          value={inputType}
          onChange={(e) => setInputType(e.target.value)}
        >
          <MenuItem value="text">Text</MenuItem>
          <MenuItem value="file">File (PDF/TXT)</MenuItem>
          <MenuItem value="image">Image (PNG/JPG)</MenuItem>
          <MenuItem value="link">URL</MenuItem>
        </Select>
      </FormControl>

      <FormControl fullWidth className="form-control">
        <InputLabel>Target Language</InputLabel>
        <Select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <MenuItem value="hi">Hindi</MenuItem>
          <MenuItem value="ta">Tamil</MenuItem>
        </Select>
      </FormControl>

      {inputType === "text" && (
        <TextField
          label="Enter Text"
          multiline
          rows={4}
          fullWidth
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="text-field"
        />
      )}

      {inputType === "link" && (
        <TextField
          label="Enter URL"
          fullWidth
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="text-field"
        />
      )}

      {(inputType === "file" || inputType === "image") && (
        <input
          type="file"
          accept={inputType === "file" ? ".txt,.pdf" : ".png,.jpg"}
          onChange={(e) => setFile(e.target.files[0])}
          className="file-input"
        />
      )}

      <Box className="button-box">
        <Button variant="contained" onClick={handleSubmit} disabled={loading}>
          {loading ? <CircularProgress size={24} /> : "Process"}
        </Button>
        <Button
          variant="outlined"
          onClick={handleSummarize}
          disabled={loading}
          sx={{ ml: 2 }}
        >
          Summarize
        </Button>
      </Box>

      {error && <Typography className="error">{error}</Typography>}

      {result && (
        <Paper elevation={3} className="result-box">
          {result.extractedText && (
            <Typography variant="body1">
              <strong>Extracted Text:</strong> {result.extractedText}
            </Typography>
          )}
          <Typography variant="body1">
            <strong>Translated Text:</strong> {result.translated}
          </Typography>
          <Typography variant="body1">
            <strong>Key Details:</strong> {result.keyDetails.join(", ")}
          </Typography>
          <Typography variant="body1">
            <strong>Suggestions:</strong> {result.suggestions.join(", ")}
          </Typography>
        </Paper>
      )}

      {summary && (
        <Paper elevation={3} className="result-box">
          <Typography variant="body1">
            <strong>Summary (English):</strong> {summary.summaryEnglish}
          </Typography>
          <Typography variant="body1">
            <strong>Summary (Translated):</strong> {summary.summaryTranslated}
          </Typography>
        </Paper>
      )}
    </Container>
  );
}

export default App;
