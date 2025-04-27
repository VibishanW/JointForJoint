import mediapipe as mp
import logging

# Enable logging to track file access
logging.basicConfig(level=logging.DEBUG)

# Initialize BlazePose
pose = mp.solutions.pose.Pose()

print("BlazePose is initialized. Check the logs for model file locations.")
