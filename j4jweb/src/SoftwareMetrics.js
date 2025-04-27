import React, { useRef, useState, useEffect } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const SoftwareMetrics = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [recording, setRecording] = useState(false);
  const [fps, setFps] = useState(0);
  const [count, setCount] = useState(0);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      const model = poseDetection.SupportedModels.BlazePose;
      const detectorConfig = { runtime: "tfjs", modelType: "full" };
      setDetector(await poseDetection.createDetector(model, detectorConfig));
    };

    loadModel();
  }, []);

  useEffect(() => {
    const startVideo = async () => {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error("Camera access error:", error);
      }
    };

    startVideo();
  }, []);

  const recordSession = async () => {
    if (!detector || !videoRef.current) return;
    setRecording(true);
    setCount(0);
    let frames = 0;
    const startTime = performance.now();

    const detect = async () => {
      if (!recording) return;
      const poses = await detector.estimatePoses(videoRef.current);
      frames++;

      // Calculate FPS every second
      const elapsedTime = performance.now() - startTime;
      if (elapsedTime > 1000) {
        setFps(frames);
        frames = 0;
      }

      drawSkeleton(poses, canvasRef.current.getContext("2d"));

      if (count < 10) {
        setTimeout(detect, 100); // Run detection every 100ms
        setCount(count + 1);
      } else {
        setRecording(false);
      }
    };

    detect();
  };

  const drawSkeleton = (poses, ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    poses.forEach((pose) => {
      pose.keypoints.forEach((keypoint) => {
        if (keypoint.score > 0.3) {
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        }
      });
    });
  };

  return (
    <div>
      <h2>Software Metrics</h2>
      <button onClick={recordSession} disabled={recording}>
        {recording ? "Recording..." : "Record 10s Session"}
      </button>
      <p>FPS: {fps}</p>
      <video ref={videoRef} width="640" height="480" autoPlay playsInline />
      <canvas ref={canvasRef} width="640" height="480" />
    </div>
  );
};

export default SoftwareMetrics;