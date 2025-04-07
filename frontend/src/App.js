import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mask, setMask] = useState(null);
  const [overlay, setOverlay] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setMask(null);
      setOverlay(null);
    }
  };

  const handleInference = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/segment/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setMask(`data:image/png;base64,${data.mask_base64}`);
      setOverlay(`data:image/jpeg;base64,${data.overlay_base64}`);
    } catch (err) {
      alert("Failed to segment image.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2>Segmentation App</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      {preview && (
        <div>
          <h3>Uploaded Image:</h3>
          <img src={preview} alt="Preview" width="400" />
        </div>
      )}
      <button onClick={handleInference} disabled={loading}>
        {loading ? "Processing..." : "Run Inference"}
      </button>
      {mask && (
        <div>
          <h3>Mask Output:</h3>
          <img src={mask} alt="Mask" width="400" />
        </div>
      )}
      {overlay && (
        <div>
          <h3>Overlay Output:</h3>
          <img src={overlay} alt="Overlay" width="400" />
        </div>
      )}
    </div>
  );
}

export default App;
