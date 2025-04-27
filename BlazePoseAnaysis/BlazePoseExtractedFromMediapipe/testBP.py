import mediapipe as mp

# Initialize BlazePose
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose()

print("BlazePose is successfully loaded!")