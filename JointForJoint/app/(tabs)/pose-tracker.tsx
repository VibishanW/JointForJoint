import React, { useEffect, useRef, useState } from "react";
import { View, StyleSheet, Text } from "react-native";
import { Camera } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { Canvas, Circle, Line } from "@shopify/react-native-skia";

const PoseTracker = () => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const [poses, setPoses] = useState([]);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        { runtime: "tfjs", modelType: "lite" }
      );

      setInterval(async () => {
        if (!cameraRef.current) return;
        const photo = await cameraRef.current.takePictureAsync();
        const img = await fetch(photo.uri);
        const blob = await img.blob();

        const detectedPoses = await detector.estimatePoses(blob);
        if (detectedPoses.length > 0) {
          setPoses(detectedPoses[0].keypoints);
        }
      }, 1000 / 10);
    };

    loadModel();
  }, []);

  if (hasPermission === null) return <Text>Requesting Camera Permission...</Text>;
  if (hasPermission === false) return <Text>No access to camera</Text>;

  return (
    <View style={{ flex: 1 }}>
      <Camera ref={cameraRef} style={StyleSheet.absoluteFill} type={"front"} />
      <Canvas style={StyleSheet.absoluteFill}>
        {poses.map((point, index) => (
          <Circle key={index} cx={point.x} cy={point.y} r={5} color="red" />
        ))}
        {drawSkeleton(poses)}
      </Canvas>
    </View>
  );
};

const drawSkeleton = (poses: any) => {
  const connections = [
    [11, 12], [11, 13], [13, 15], 
    [12, 14], [14, 16], 
    [11, 23], [12, 24], 
    [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], 
    [24, 26], [26, 28], [28, 30], [30, 32]
  ];

  return connections.map(([p1, p2], index) => {
    if (poses[p1] && poses[p2]) {
      return (
        <Line key={index} p1={{ x: poses[p1].x, y: poses[p1].y }} p2={{ x: poses[p2].x, y: poses[p2].y }} strokeWidth={2} color="blue" />
      );
    }
    return null;
  });
};

export default PoseTracker;
