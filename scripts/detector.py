#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs_py.point_cloud2 import read_points
from sensor_msgs_py import point_cloud2
import tf2_ros
from scipy.spatial.transform import Rotation as R
import geometry_msgs.msg
from std_msgs.msg import Header


class Detector(Node):
    def __init__(self, vis=False):
        super().__init__('human_detector_node')
        
        self.use_bboxes = True
        self.use_segmentation = not self.use_bboxes
        self.publish_annotated_imgs = True

        self.rgb_sub = self.create_subscription(Image, '/camera/realsense_camera/color/image_raw', self.rgb_callback, 10)
        self.rgb_cam_info = self.create_subscription(CameraInfo, '/camera/realsense_camera/color/camera_info', self.cam_info_callback, 10)
        self.lidar_sub = self.create_subscription(PointCloud2, '/cloud_all_fields_fullframe', self.lidar_callback, 10)
        
        self.annotated_pc_publisher = self.create_publisher(PointCloud2, 'annotated_pointcloud', 10)
        
        if self.publish_annotated_imgs == True:
            self.annotated_image_publisher = self.create_publisher(Image, 'annotated_rgb', 10)
        
        if self.use_bboxes:
            self.model = YOLO("/ultralytics/workspace/ros2_ws/src/ultralyticsYOLO_ros/models/yolo11n.engine")
            print("LOADED YOLO11N - USING BBOXES")
        
        elif self.use_segmentation:
            self.model = YOLO("/ultralytics/workspace/ros2_ws/src/ultralyticsYOLO_ros/models/yolo11n-seg.engine")
            print("LOADED YOLO11NSEG - USING SEGMENTATION")
        
        self.bridge = CvBridge()
        self.vis = vis
        
        self.cam_matrix = None
        self.image = None
        
        T_lidar_in_cam_R = np.load("/ultralytics/workspace/ros2_ws/src/ultralyticsYOLO_ros/resource/extrinsics.npz")["R"]
        # T_lidar_in_cam_T = np.load("/ultralytics/workspace/ros2_ws/src/ultralyticsYOLO_ros/resource/extrinsics.npz")["T"].T
        T_lidar_in_cam_T = np.array([0.045, -0.11, -0.09]).reshape(3,1)
        
        self.lidar_in_cam = np.hstack((T_lidar_in_cam_R, T_lidar_in_cam_T))
        self.lidar_in_cam = np.vstack((self.lidar_in_cam, np.array([0, 0, 0, 1])))    #4x4
        
        self.detected = False		## Bool flag to check if there is a detection in current frame
        
        # self.br = tf2_ros.TransformBroadcaster(self)
        # self.broadcast_tf(self.lidar_in_cam)
        
    def cam_info_callback(self, msg):
        # print(self.cam_matrix)
        if self.cam_matrix is None:
            self.cam_matrix = np.array(msg.k).reshape(3,3) ## fetch camera calibration matrix from message
            
            self.image_shape = [msg.width, msg.height]
            # print(self.image_shape)
            
            
    def broadcast_tf(self, T_lidar_in_cam):
        # Create a TF broadcaster
        

        # Prepare the transformation message
        transform = geometry_msgs.msg.TransformStamped()
    
        # Header (ROS time)
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"  # Parent frame (LiDAR)
        transform.child_frame_id = "camera_color_optical_frame"  # Child frame (Camera)

        # Fill in the translation (position)
        transform.transform.translation.x = T_lidar_in_cam[0, -1]  # x position
        transform.transform.translation.y = T_lidar_in_cam[1, -1]  # y position
        transform.transform.translation.z = T_lidar_in_cam[2, -1]  # z position

        # Fill in the rotation (orientation) - Convert from rotation matrix to quaternion
        q = R.from_matrix(T_lidar_in_cam[:3, :3]).as_quat()  # Assuming T_lidar_in_cam is a 4x4 matrix
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
    
        # Broadcast the transform
        self.br.sendTransform(transform)
    
    
    def rgb_callback(self, msg):
        ## get image from message
        # print(msg.height, msg.width)
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # cv2.imshow("Camera", self.image)
        # cv2.waitKey(1)
        # print("capturing frames")
        
        self.results = self.model.predict(self.image, classes=[0], stream=True, show=self.vis, iou=0.4, verbose=False)
        
        for result in self.results:
            if self.publish_annotated_imgs == True:
                self.annotated_img = result.plot()
                
            if self.use_bboxes:
                self.detection_boxes = result.boxes.xyxy.cpu().numpy()		## (n,4) array - n=num of detections, 4- x1,y1,x2,y2 (img pixel coords)
                if self.detection_boxes.shape[0] != 0:
                    self.detected = True
                
                else:
                    self.detected = False
                
            if self.use_segmentation:
                self.segmentation = result.masks
                if self.segmentation is not None:
                    self.detected = True
                    self.segmentation = self.segmentation.xy
                    # print(len(self.segmentation))
                    # print(type(self.segmentation[0]))
                    # print(self.segmentation[0].shape)
                    print
                else:
                    self.detected = False
                
        if self.publish_annotated_imgs == True:        
            annotated_img_msg = self.bridge.cv2_to_imgmsg(self.annotated_img, encoding='bgr8')
            self.annotated_image_publisher.publish(annotated_img_msg)

    
    
    def lidar_callback(self, msg):
        if self.image is not None and self.cam_matrix is not None:
            # Extract metadata
            points = read_points(msg, field_names=("x", "y", "z", "range"), skip_nans=True)
            points_arr = np.stack((points['x'], points['y'], points['z'], points["range"]), axis=-1).T
            
            xyz_array = points_arr[0:3, :]
            ranges = points_arr[3, :]
            
            # Optionally filter out NaNs
            xyz_array = xyz_array[~np.isnan(xyz_array).any(axis=1)]
        
            ## append 1 to homogenize
            xyz_homo = np.vstack((xyz_array, np.ones_like(xyz_array[0]))) ## 4, N
                                   
            xyz_cam_frame = self.lidar_in_cam @ xyz_homo	## Convert to camera frame (extrinsic b/w cam and lidar)
            
            # Filter out points where z is negative
            valid_mask_positive_z = xyz_cam_frame[2, :] >= 0  # Points with non-negative z in camera frame
        
            # Apply the mask to remove points with negative z
            xyz_behind_cam = xyz_cam_frame[:, ~valid_mask_positive_z]		## store to complete pointcloud later
            xyz_cam_frame = xyz_cam_frame[:, valid_mask_positive_z] 
            ranges_valid = ranges[valid_mask_positive_z]
            ranges_behind = ranges[~valid_mask_positive_z]             
            
            
            
            lidar_as_pixels = self.cam_matrix @ xyz_cam_frame[0:3, :] ## Pixel projection coordinates of lidar points
            lidar_as_pixels = (lidar_as_pixels/xyz_cam_frame[2]).astype(int)	##normalize to get pixel coords
        
            u = lidar_as_pixels[1]
            v = lidar_as_pixels[0]
        
            valid_mask = (u >= 0) & (u < self.image_shape[1]) & (v >= 0) & (v < self.image_shape[0])		## mask out lidar points that are not within camera fov
        
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            valid_pixels = np.stack((v_valid, u_valid), axis=-1)
            valid_struct = valid_pixels.view([('x', valid_pixels.dtype), ('y', valid_pixels.dtype)])
            '''	
            ## For visulaization of img with lidar overlay only
            
            
            ranges_valid_in_frame = ranges_valid[valid_mask]
            
            assert xyz_cam_frame.shape[1] == ranges_valid.shape[0]
            
            # Normalize ranges to 0–255 for color mapping
            ranges_normalized = cv2.normalize(ranges_valid_in_frame, None, 0, 255, cv2.NORM_MINMAX)
            ranges_normalized = ranges_normalized.astype(np.uint8)

            # Apply a colormap (returns BGR colors)
            colors = cv2.applyColorMap(ranges_normalized, cv2.COLORMAP_JET)
            colors = np.squeeze(colors, axis=1)
            
            for i in range(len(v_valid)):
                x = int(v_valid[i])
                y = int(u_valid[i])
                b, g, r = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
                cv2.circle(self.image, (x, y), radius=2, color=(b, g, r), thickness=-1)
        
            cv2.imshow("1", self.image)
            cv2.waitKey(1)
            '''
           
            is_human = np.zeros_like(ranges)       ### if no detections - is_human = all zeros
            reconstructed_lidar_pts = xyz_array    ### if no detections - just publish the points from og pointcloud
            ranges_reconstructed = ranges           ### if no detections - just publish the points from og pointcloud
            
            if self.detected:
                # print("HUMAN DETECTED", self.detected, self.detection_boxes.shape)
                
                ## using bounding box
                if self.use_bboxes == True:
                    human_mask = np.zeros_like(u_valid, dtype=bool)
                    for i in range(self.detection_boxes.shape[0]):
                        x1, y1, x2, y2 = self.detection_boxes[i]
                        human_mask = human_mask | ( (v_valid >= x1) & (v_valid <= x2) & (u_valid >= y1) & (u_valid <= y2) )
                        
                    u_human = u_valid[human_mask]
                    v_human = v_valid[human_mask]
                
                elif self.use_segmentation:
                    u_human = []
                    v_human = []
                    human_mask = np.zeros_like(u_valid, dtype=bool)
                    for i in range(len(self.segmentation)):
                        black = np.zeros((480, 640), dtype=np.uint8)
                        polygon = self.segmentation[i].reshape((-1, 1, 2)).astype(int)
                        cv2.drawContours(black, [polygon], contourIdx=-1, color=255, thickness=cv2.FILLED)
                        non_zero_pixels = cv2.findNonZero(black).reshape(-1, 2)
                        non_zero_struct = non_zero_pixels.view([('x', non_zero_pixels.dtype), ('y', non_zero_pixels.dtype)])
                        human_mask = human_mask | np.isin(valid_struct, non_zero_struct).flatten()
                    
                    u_human = u_valid[human_mask]
                    v_human = v_valid[human_mask]

                all_pixels = np.stack((u, v), axis=-1)           # Shape (N, 2)
                human_pixels = np.stack((u_human, v_human), axis=-1)  # Shape (M, 2)
                print(human_pixels.shape)
                
                # View as structured array of two fields so np.isin works on rows
                all_struct = all_pixels.view([('u', u.dtype), ('v', v.dtype)]).reshape(-1)
                human_struct = human_pixels.view([('u', u.dtype), ('v', v.dtype)]).reshape(-1)
                '''
                print("u dtype:", u.dtype, "v dtype:", v.dtype)
                print("u_human dtype:", u_human.dtype, "v_human dtype:", v_human.dtype)
                print("u range:", u.min(), u.max())
                print("u_human range:", u_human.min(), u_human.max())
                print("v range:", v.min(), v.max())
                print("v_human range:", v_human.min(), v_human.max())
                
                print("all_struct.shape:", all_struct.shape)
                print("human_struct.shape:", human_struct.shape)
                '''
                
                # Vectorized boolean mask
                is_human = np.isin(all_struct, human_struct).flatten()
                # print(np.count_nonzero(is_human), self.segmentation[i].shape)
                
                reconstructed_lidar_pts_cam_frame = np.linalg.inv(self.cam_matrix) @ (np.vstack((v, u, np.ones_like(u))) * xyz_cam_frame[2]).astype(float)  #convert to camera frame from pixel coords
                
                reconstructed_lidar_pts_cam_frame = np.vstack((reconstructed_lidar_pts_cam_frame, np.ones_like(reconstructed_lidar_pts_cam_frame[0])))  ## homogenize
                
                reconstructed_lidar_pts_cam_frame = np.hstack((reconstructed_lidar_pts_cam_frame, xyz_behind_cam))  ## complete pc by adding back the points behind the camera (that we removed earlier) 
                is_human = np.hstack((is_human, np.zeros_like(xyz_behind_cam[0])))		## add "no humans" for points behind camera
                                
                reconstructed_lidar_pts = (np.linalg.inv(self.lidar_in_cam) @ reconstructed_lidar_pts_cam_frame)[0:3, :]
                ranges_reconstructed = np.hstack((ranges_valid, ranges_behind))

            self.publish_point_cloud(reconstructed_lidar_pts, ranges_reconstructed, is_human)
                
                
                
                
                # self.publish_point_cloud(xyz_cam_frame, is_human)
        
    def publish_point_cloud(self, xyz_cam_frame, ranges, detections_bool, frame="world"):
        # Create a Header for the PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()  # Get current time
        header.frame_id = frame				# "camera_color_optical_frame"  # The frame of reference for the point cloud
        
        # Prepare point cloud data in the required format (list of tuples)
        points = np.vstack((xyz_cam_frame[0, :], xyz_cam_frame[1, :], xyz_cam_frame[2, :], ranges.reshape(1, -1), detections_bool.reshape(1, -1))).T
        point_cloud_data = xyz_cam_frame[:3, :].T
        
        # Define the fields for the PointCloud2 message (x, y, z)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1), 
            PointField(name='is_human', offset=16, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Create the PointCloud2 message
        pc2_msg = point_cloud2.create_cloud(header, fields, points)

        # Publish the PointCloud2 message
        self.annotated_pc_publisher.publish(pc2_msg)
        self.get_logger().info('Publishing point cloud with %d points' % len(point_cloud_data))
        
         
        
        



def main():
    print("starting node")
    try:
        rclpy.init(args=None)
        human_detector_node = Detector(vis=False)
        rclpy.spin(human_detector_node)
        
    finally:
        human_detector_node.destroy_node()
        rclpy.shutdown()
    
        

if __name__ == '__main__':
    main()
    
    
    
'''
                if self.use_bboxes == True:
                    human_mask = np.zeros_like(u_valid, dtype=bool)
                    for i in range(self.detection_boxes.shape[0]):
                        x1, y1, x2, y2 = self.detection_boxes[i]

                        # Find points inside this bounding box
                        in_box_mask = (v_valid >= x1) & (v_valid <= x2) & (u_valid >= y1) & (u_valid <= y2)
                        in_box_indices = np.where(in_box_mask)[0]

                        # Compute center of bbox (in pixel coords)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + 2*y2) / 3)

                        # Find closest projected point to center pixel
                        dists_to_center = np.sqrt((v_valid[in_box_indices] - cx) ** 2 + (u_valid[in_box_indices] - cy) ** 2)
                        if dists_to_center.size == 0:
                            continue  # skip empty boxes

                        center_idx = np.argmin(dists_to_center)
                        center_range = ranges_valid[in_box_indices[center_idx]]

                        # Range mask: keep points within 10cm = 0.1m of center range
                        range_mask = np.abs(ranges_valid[in_box_indices] - center_range) <= 0.05
                        
                        # Combine in_box_mask and range_mask (must map back to full array)
                        # full_mask = np.zeros_like(human_mask)
                        # full_mask_indices = np.where(in_box_mask)[0][range_mask]

                        # human_mask[full_mask_indices] = True  # update final mask
                        human_mask[in_box_indices[range_mask]] = True
                        
                    u_human = u_valid[human_mask]
                    v_human = v_valid[human_mask]
                '''
