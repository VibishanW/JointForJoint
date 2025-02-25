import React, { useEffect, useRef, useState } from 'react';
import { Text, View, StyleSheet, Dimensions, Platform } from 'react-native';
import Constants from 'expo-constants';

// Load native modules only if not on Web
let Camera, cameraWithTensors, tf, poseDetection, TensorCamera;
if (Platform.OS !== 'web') {
  const CameraModule = require('expo-camera');
  if (CameraModule?.Camera) {
    Camera = CameraModule.Camera;
    cameraWithTensors = require('@tensorflow/tfjs-react-native').cameraWithTensors;
    tf = require('@tensorflow/tfjs-react-native');
    poseDetection = require('@tensorflow-models/pose-detection');
    TensorCamera = cameraWithTensors(Camera);
  }
}

const { width, height } = Dimensions.get('window');

export default function ExploreScreen() {
  const [hasPermission, setHasPermission] = useState(null);
  const [isTfReady, setIsTfReady] = useState(false);
  const [pose, setPose] = useState(null);
  const cameraRef = useRef(null);
  const detectorRef = useRef(null);

  // Check if running in Expo Go
  const isExpoGo = Constants.appOwnership === 'expo';

  useEffect(() => {
    if (Platform.OS === 'web' || !Camera) return; // Skip TensorFlow setup on Web

    (async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');

        if (isExpoGo) return; // Skip TensorFlow.js in Expo Go

        // Ensure TensorFlow backend is correctly set
        await tf.ready();
        await tf.setBackend('rn-webgl');

        setIsTfReady(true);

        // Load BlazePose model
        detectorRef.current = await poseDetection.createDetector(
          poseDetection.SupportedModels.BlazePose,
          { runtime: 'tfjs', enableSmoothing: true }
        );
      } catch (error) {
        console.error('TensorFlow.js Init Error:', error);
      }
    })();
  }, []);

  // Process camera stream (Only if NOT Web)
  const handleCameraStream = (images) => {
    if (Platform.OS === 'web' || !detectorRef.current) return;

    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (nextImageTensor) {
        try {
          const poses = await detectorRef.current.estimatePoses(nextImageTensor);
          if (poses.length > 0) setPose(poses[0]);
        } catch (error) {
          console.error('Pose Detection Error:', error);
        } finally {
          tf.dispose(nextImageTensor); // Free memory
        }
      }
      requestAnimationFrame(loop);
    };
    loop();
  };

  // Handle camera permissions
  if (hasPermission === null) return <View />;
  if (hasPermission === false) return <Text>No access to camera</Text>;

  // Web Fallback: Prevent crash
  if (Platform.OS === 'web') {
    return (
      <View style={styles.container}>
        <Text style={styles.header}>Joint-4-Joint (Web Mode)</Text>
        <Text style={styles.status}>Camera & Pose Detection are not supported on Web.</Text>
      </View>
    );
  }

  // Expo Go Fallback: Show Camera without TensorFlow
  if (isExpoGo) {
    return (
      <View style={styles.container}>
        <Text style={styles.header}>Joint-4-Joint (Expo Go Mode)</Text>
        {Camera ? (
          <Camera
            ref={cameraRef}
            style={styles.camera}
            type={Camera.Constants.Type.front}
          />
        ) : (
          <Text style={styles.status}>Camera module not available</Text>
        )}
        <Text style={styles.status}>Pose detection disabled in Expo Go</Text>
      </View>
    );
  }

  // Full Functionality for Expo Dev Build
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Joint-4-Joint</Text>
      {isTfReady ? (
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          type={Camera?.Constants?.Type?.front ?? 1}
          cameraTextureHeight={height * 0.5}
          cameraTextureWidth={width * 0.5}
          resizeHeight={200}
          resizeWidth={152}
          resizeDepth={3}
          onReady={handleCameraStream}
          autorender={true}
        />
      ) : (
        <Text style={styles.status}>Loading TensorFlow.js...</Text>
      )}
      {pose && (
        <View style={styles.poseInfo}>
          <Text style={styles.poseText}>Pose Detected!</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000', alignItems: 'center', justifyContent: 'center' },
  header: { fontSize: 28, color: '#4CAF50', fontWeight: 'bold', marginTop: 20 },
  camera: { width: width, height: height },
  status: { color: '#fff', fontSize: 18, marginTop: 20 },
  poseInfo: { position: 'absolute', top: 40, left: 20, backgroundColor: 'rgba(0, 0, 0, 0.5)', padding: 10, borderRadius: 10 },
  poseText: { color: '#fff', fontSize: 18 },
});
