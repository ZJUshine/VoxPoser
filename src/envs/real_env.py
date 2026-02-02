import json
import os
import numpy as np
import open3d as o3d
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from perception.camera_tracker_d435i import CameraTracker
import cv2
from franky import Affine, Robot, Gripper, CartesianMotion, JointMotion

class CoordinateTransformer:
    def __init__(self, transform_matrix):
        """
        Initialize with the calibration transformation matrix
        Args:
            transform_matrix: 4x4 homogeneous transformation matrix from camera to robot base frame
        """
        self.T_camera_to_base = transform_matrix
        
    def camera_to_robot(self, point_camera):
        """
        Transform points from camera coordinates to robot base coordinates
        Args:
            point_camera: Array of shape (N, 3) containing N points in camera frame,
                         or single point of shape (3,)
        Returns:
            Array of shape (N, 3) containing points in robot base frame,
            or single point of shape (3,)
        """
        # Handle single point case
        single_point = len(point_camera.shape) == 1
        if single_point:
            point_camera = point_camera.reshape(1, 3)
        
        # Convert to homogeneous coordinates (N, 4)
        points_homog = np.hstack([point_camera, np.ones((point_camera.shape[0], 1))])
        
        # Apply transformation (N, 4)
        points_robot_homog = np.dot(points_homog, self.T_camera_to_base.T)
        
        # Convert back to 3D coordinates (N, 3)
        points_robot = points_robot_homog[:, :3] / points_robot_homog[:, 3:]
        
        return points_robot[0] if single_point else points_robot


    def robot_to_camera(self, point_robot):
        """
        Transform points from robot base coordinates to camera coordinates
        Args:
            point_robot: Array of shape (N, 3) containing N points in robot base frame,
                        or single point of shape (3,)
        Returns:
            Array of shape (N, 3) containing points in camera frame,
            or single point of shape (3,)
        """
        # Handle single point case
        single_point = len(point_robot.shape) == 1
        if single_point:
            point_robot = point_robot.reshape(1, 3)
        
        # Convert to homogeneous coordinates (N, 4)
        points_homog = np.hstack([point_robot, np.ones((point_robot.shape[0], 1))])
        
        # Compute inverse transformation
        T_base_to_camera = np.linalg.inv(self.T_camera_to_base)
        
        # Apply transformation (N, 4)
        points_camera_homog = np.dot(points_homog, T_base_to_camera.T)
        
        # Convert back to 3D coordinates (N, 3)
        points_camera = points_camera_homog[:, :3] / points_camera_homog[:, 3:]
        
        return points_camera[0] if single_point else points_camera

class RealRobotEnv():
    def __init__(self, visualizer=None, debug=False, force_vertical_gripper=True):
        """
        Initializes the RealRobotEnv environment.

        Args:
            visualizer: Visualization interface, optional.
            debug: Whether to run in debug mode.
            force_vertical_gripper: If True, gripper Z-axis always points downward (vertical).
        """
        self.force_vertical_gripper = force_vertical_gripper

        calibration_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'configs', 'calibration_result.json')
            )

        self.T_camera_to_base = self.load_T_camera_to_base(calibration_path)

        self.robot = Robot("172.16.0.2")
        self.gripper = Gripper("172.16.0.2")


        self.visualizer = visualizer

        # Define workspace bounds (in meters)
        self.workspace_bounds_max = np.array([1.0, 1.0, 1.0])
        self.workspace_bounds_min = np.array([-1.0, -1.0, -0.1])

        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)


        self.tracker = CameraTracker()
        self.cam2robot = CoordinateTransformer(self.T_camera_to_base)

        self.gripper_speed = 0.02
        self.gripper_force = 20.0
        self._last_gripper_state = 1  # 1=open, 0=close

        self.reset_robot()

        self.reset_task_variables()
        
        self._update_visualizer()

    def reset_robot(self):
        """
        Resets the robot in the environment.
        """
        # Recover from any errors
        self.robot.recover_from_errors()

        # Reduce the acceleration and velocity dynamic
        self.robot.relative_dynamics_factor = 0.05

        # Go to initial position
        self.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))

        # Open the gripper
        self.gripper.open(speed=self.gripper_speed)
        self._last_gripper_state = 1

    def reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """

        self.init_obs = self.get_ee_pose()
        self.latest_obs = self.get_ee_pose()
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name
        self.first_flag_dict = {}

    def load_T_camera_to_base(self, calibration_path):
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        if 'T_cam_base' in data:
            matrix = data['T_cam_base']
            T_camera_to_base = np.array(matrix, dtype=float)
        else:
            raise KeyError('calibration_result.json missing T_cam_base')
        return T_camera_to_base

    def get_3d_obs_by_name(self, name):
        """Get 3D position of specified object"""
        target = [name]
        dict_objects = self.tracker.get_latest_objects(target)
        self.tracker.process_3d()
        count = 0
        while name not in dict_objects or dict_objects[name] is None or dict_objects[name]['center3d'] is None:
            dict_objects = self.tracker.get_latest_objects(target)
            self.tracker.process_3d()
            count += 1
            if count > 5:
                print(f"[INFO]: Restarting tracker for {name} count: {count}")
                self.tracker.get_latest_objects(target, restart=True)
                count = 0
            time.sleep(1)

        # save segmentation image
        if name not in self.first_flag_dict.keys():
            image = self.tracker.latest_color_image.copy()
            mask_2d = dict_objects[name]['mask2d']
            image[mask_2d > 0] = image[mask_2d > 0] * 0.7 + np.array((255, 255, 0), dtype=np.uint8) * 0.3
            cv2.imwrite(f'{name}_detection.png', image)
            self.first_flag_dict[name] = False

        # Bug 1 fix: Return full mask3d point cloud (not just center3d)
        mask3d_camera = dict_objects[name]['mask3d']  # (N, 3) in camera frame
        mask3d_robot = self.cam2robot.camera_to_robot(mask3d_camera)  # (N, 3) in robot frame

        # Note: Removed forced z-minimum threshold (0.45) as it was causing ~0.5m offset
        # Safety limits should be handled by the controller/planner instead

        # Transform normal from camera frame to robot frame using rotation matrix,
        # then ensure it points upward relative to robot base (+Z direction)
        normal_camera = dict_objects[name]['normal'].reshape(1, 3)
        R = self.T_camera_to_base[:3, :3]
        normal_robot = (R @ normal_camera.T).T  # (1, 3)
        normal_robot = normal_robot / np.linalg.norm(normal_robot)
        # Flip normal if it points downward (negative Z in robot frame)
        if normal_robot[0, 2] < 0:
            normal_robot = -normal_robot

        return mask3d_robot, normal_robot
    
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Get 3D positions and colors of all objects in scene.
        """
        while self.tracker.latest_pointcloud is None:
            self.tracker.process_3d()
            time.sleep(1)
        points = self.tracker.latest_pointcloud.reshape(-1, 3)
        colors = self.tracker.latest_color_image
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
        colors = colors.reshape(-1, 3)

        points = self.cam2robot.camera_to_robot(points)

        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]


        if len(points) == 0:
            return np.zeros((1,3)), np.zeros((1,3), dtype=np.uint8)

        # Voxel downsample using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)

        points = np.asarray(pcd_downsampled.points)
        colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)
        return points, colors
    
    def apply_action(self, action):
        """
        Applies an action to the robot.

        Args:
            action: Array containing [x, y, z, qw, qx, qy, qz, gripper_state] (wxyz format from VoxPoser)
                    gripper_state: 1 = open, 0 = close
        """
        # 转换四元数格式: wxyz (VoxPoser) → xyzw (franky)
        action = self._process_action(action)

        # 提取位置和姿态
        pos = action[:3]  # [x, y, z] 单位: 米
        quat = action[3:7]  # [qx, qy, qz, qw] - franky使用xyzw格式
        gripper_state = action[7] if len(action) > 7 else None

        # 创建目标位姿并执行运动
        target_pose = Affine(pos.tolist(), quat.tolist())
        motion = CartesianMotion(target_pose)

        try:
            self.robot.move(motion)
        except Exception as e:
            print(f"[ERROR]: Motion failed: {e}")
            self.robot.recover_from_errors()
            return None

        # 控制夹爪（仅在状态变化时才执行，避免重复调用阻塞的 grasp）
        if gripper_state is not None:
            if gripper_state >= 1 and self._last_gripper_state < 1:
                self.open_gripper()
                self._last_gripper_state = 1
            elif gripper_state < 1 and self._last_gripper_state >= 1:
                self.close_gripper()
                self._last_gripper_state = 0

        # 更新状态
        self.latest_obs = self.get_ee_pose()
        self.latest_reward = 0
        self.latest_terminate = False
        self.latest_action = action
        self._update_visualizer()                                                                                       


    def move_to_pose(self, pose):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        self.apply_action(pose)
    
    def reset_to_default_pose(self):
        """
        Resets the robot to the default pose.
        """
        self.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))
        
    def open_gripper(self):
        """Open gripper"""
        self.gripper.open(speed=self.gripper_speed)
        
    def close_gripper(self):
        """Close gripper"""
        self.gripper.grasp(width = 0.0, epsilon_outer = 1.0, speed=self.gripper_speed, force=self.gripper_force)
        
    def set_gripper_state(self, gripper_state):
        """Set gripper to specified state"""
        action = np.concatenate([self.latest_obs, [gripper_state]])

        return self.apply_action(action)

    def get_ee_pose(self):
        """Get current end effector pose as numpy array [x, y, z, qw, qx, qy, qz] (wxyz format for VoxPoser)"""
        cartesian_state = self.robot.current_cartesian_state
        ee_affine = cartesian_state.pose.end_effector_pose

        # 从 Affine 对象提取 translation 和 quaternion
        # 逐个索引并用 float() 转换为本地 Python 对象，避免 RPyC pickling 错误
        translation = np.array([float(ee_affine.translation[i]) for i in range(3)])  # [x, y, z]
        quaternion = np.array([float(ee_affine.quaternion[i]) for i in range(4)])    # [qx, qy, qz, qw] (xyzw格式)

        obs = np.concatenate([translation, quaternion])
        # 转换四元数格式: xyzw (franky) → wxyz (VoxPoser)
        return self._process_obs(obs)


    def get_ee_pos(self):
        """Get current end effector position [x, y, z]"""
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        """Get current end effector quaternion [qw, qx, qy, qz] (wxyz format for VoxPoser)"""
        return self.get_ee_pose()[3:]

    def get_vertical_down_quat(self):
        """
        Get quaternion for gripper pointing vertically downward (Z-axis pointing down).
        Returns [qw, qx, qy, qz] in wxyz format for VoxPoser.
        """
        # Rotation matrix: gripper Z points down (-Z world), X points forward (+X world)
        # This corresponds to a 180 degree rotation around X-axis
        # quat (wxyz) = [0, 1, 0, 0] or equivalent: 180° around X
        return np.array([0.0, 1.0, 0.0, 0.0])

    def get_default_rotation_quat(self):
        """
        Get default rotation quaternion based on force_vertical_gripper setting.
        Returns [qw, qx, qy, qz] in wxyz format.
        """
        if self.force_vertical_gripper:
            return self.get_vertical_down_quat()
        else:
            return self.get_ee_quat()

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return False

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_action(self, action):
        """
        Processes the action, converts quaternion format from wxyz (VoxPoser) to xyzw (franky).

        Args:
            action: The action to process, [x, y, z, qw, qx, qy, qz, gripper_state]

        Returns:
            The processed action with quaternion in xyzw format.
        """
        action = action.copy()
        quat_wxyz = action[3:7]  # [qw, qx, qy, qz]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])  # [qx, qy, qz, qw]
        action[3:7] = quat_xyzw
        return action
    
    
    def _process_obs(self, obs):
        """
        Processes the observations, converts quaternion format from xyzw (franky) to wxyz (VoxPoser).

        Args:
            obs: The observation to process, [x, y, z, qx, qy, qz, qw]

        Returns:
            The processed observation with quaternion in wxyz format.
        """
        obs = obs.copy()
        quat_xyzw = obs[3:]  # [qx, qy, qz, qw]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])  # [qw, qx, qy, qz]
        obs[3:] = quat_wxyz
        return obs