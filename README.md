# ultralyticsYOLO_ros

ros2 implementation of vanilla ultralytics YOLO models
This node:
1. reads a lidar pointcloud and an rgb image
2. detects humans using ultralytics YOLO (you can change the detection class by modifying the line self.model.predict() according to the ultralytics documentation
3. projects pointcloud into the image plane (pixel coordinates)
4. annotates the lidar points that corrospond to a human
5. projects the pointcloud back into world frame and publishes a pointcloud 2 message which includes a bool to specify if a lidar point is a human or not

To run:
connect a realsense camera
put the extrinsic calibration of lidar in camera frame in an npz file (with ["R"] and ['T'] as the fields) and place in the resource folder
put the ultralytics YOLO model file (.pt, .onnx, .engine) in a folder called models and appropriately set the path in scripts/detection.py
appropriately set the names of the topics, models, extrinsic calibration file in the detection.py script
build the package (colcon build)
launch using ros2 launch ultralyticsYOLO_ros detections_launch.py
