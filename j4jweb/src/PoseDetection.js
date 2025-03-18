import React, { useRef, useEffect, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const PoseDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);

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
          detectPose(); // Start pose detection when the video is ready
        };
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    startVideo();
  }, []);

  const detectPose = async () => {
    if (!detectorRef.current || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const runDetection = async () => {
      if (video.paused || video.ended) return requestAnimationFrame(runDetection);

      try {
        const poses = await detectorRef.current.estimatePoses(video);
        console.log("Detected Poses:", poses);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (poses.length > 0) {
          drawSkeleton(poses, ctx);
        } else {
          console.log("No keypoints detected.");
        }
      } catch (error) {
        console.error("Pose detection error:", error);
      }

      requestAnimationFrame(runDetection);
    };

    console.log("Starting Pose Detection...");
    runDetection();
  };

  const drawSkeleton = (poses, ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(videoRef.current, 0, 0, 1280, 720);

    poses.forEach((pose) => {
      if (!pose.keypoints || pose.keypoints.length === 0) {
        console.log("No keypoints found for drawing.");
        return;
      }

      pose.keypoints.forEach((keypoint) => {
        if (!keypoint || !keypoint.x || !keypoint.y || keypoint.score < 0.3) {
          console.log("Skipping invalid keypoint:", keypoint);
          return;
        }
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });

      drawLines(ctx, pose.keypoints);
    });
  };

  const drawLines = (ctx, keypoints) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 7], // Left eye to ear
      [0, 4], [4, 5], [5, 6], [6, 8], // Right eye to ear
      [9, 10], // Mouth
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms
      [11, 23], [12, 24], [23, 24], // Hips
      [23, 25], [25, 27], [24, 26], [26, 28], // Legs
      [27, 31], [28, 32], [29, 31], [30, 32] // Feet
    ];

    ctx.strokeStyle = "blue";
    ctx.lineWidth = 3;

    connections.forEach(([i, j]) => {
      if (keypoints[i] && keypoints[j] && keypoints[i].score > 0.3 && keypoints[j].score > 0.3) {
        ctx.beginPath();
        ctx.moveTo(keypoints[i].x, keypoints[i].y);
        ctx.lineTo(keypoints[j].x, keypoints[j].y);
        ctx.stroke();
      }
    });
  };

  return (
    <div style={{ position: "relative", width: "1280px", height: "720px" }}>
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
  );
};

export default PoseDetection;