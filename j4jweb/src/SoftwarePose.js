import React, { useRef, useEffect, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const SoftwarePose = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [poseCount, setPoseCount] = useState(0);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      console.log("TensorFlow.js WebGL backend set.");

      const model = poseDetection.SupportedModels.BlazePose;
      const detectorConfig = { runtime: "tfjs", modelType: "full" };
      detectorRef.current = await poseDetection.createDetector(model, detectorConfig);
      console.log("BlazePose model loaded.");
    };

    loadModel();
  }, []);

  useEffect(() => {
    const startVideo = async () => {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
        });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
          videoRef.current.play();
          console.log("Camera stream started.");
        };
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    startVideo();
  }, []);

  const recordPose = async () => {
    if (!detectorRef.current || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    setRecording(true);
    setPoseCount(0);
    let startTime = Date.now();

    const runDetection = async () => {
      if (Date.now() - startTime >= 10000) {
        setRecording(false);
        console.log(`Pose detection completed. Frames detected: ${poseCount}`);
        return;
      }

      try {
        const poses = await detectorRef.current.estimatePoses(video);
        console.log("Detected Poses:", poses);
        setPoseCount((prev) => prev + 1);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (poses.length > 0) {
          drawSkeleton(poses, ctx);
        }
      } catch (error) {
        console.error("Pose detection error:", error);
      }

      requestAnimationFrame(runDetection);
    };

    console.log("Starting 10-second pose recording...");
    runDetection();
  };

  const drawSkeleton = (poses, ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(videoRef.current, 0, 0, 1280, 720);

    poses.forEach((pose) => {
      if (!pose.keypoints || pose.keypoints.length === 0) return;

      pose.keypoints.forEach((keypoint) => {
        if (keypoint.score < 0.3) return;
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });
    });
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h2>Software Pose Recording</h2>
      <button onClick={recordPose} disabled={recording}>
        {recording ? "Recording..." : "Start 10-sec Recording"}
      </button>
      <p>Frames Captured: {poseCount}</p>
      <div style={{ position: "relative", width: "1280px", height: "720px", margin: "auto" }}>
        <video
          ref={videoRef}
          width="1280"
          height="720"
          autoPlay
          playsInline
          style={{ position: "absolute", top: 0, left: 0, zIndex: 1 }}
        />
        <canvas
          ref={canvasRef}
          width="1280"
          height="720"
          style={{ position: "absolute", top: 0, left: 0, zIndex: 2 }}
        />
      </div>
    </div>
  );
};

export default SoftwarePose;