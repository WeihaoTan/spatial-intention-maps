from cgi import test
import math
import pkgutil
import sys
from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from pathlib import Path
from pprint import pprint
from turtle import done, position

import anki_vector
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage.morphology import distance_transform_edt
from skimage.draw import line
from skimage.morphology import binary_dilation, dilation
from skimage.morphology.selem import disk
from gym import spaces
import vector_utils


class VectorEnv:
    WALL_HEIGHT = 0.1
    CUBE_WIDTH = 0.044
    #RECEPTACLE_WIDTH = 0.15
    RECEPTACLE_WIDTH = 0.3
    IDENTITY_QUATERNION = (0, 0, 0, 1)
    REMOVED_BODY_Z = -1000  # Hide removed bodies 1000 m below
    CUBE_COLOR = (237.0 / 255, 201.0 / 255, 72.0 / 255, 1)  # Yellow
    DEBUG_LINE_COLORS = [
        (78.0 / 255, 121.0 / 255, 167.0 / 255),  # Blue
        (89.0 / 255, 169.0 / 255, 79.0 / 255),  # Green
        (176.0 / 255, 122.0 / 255, 161.0 / 255),  # Purple
        (242.0 / 255, 142.0 / 255, 43.0 / 255),  # Orange
    ]

    def __init__(
        # This comment is here to make code folding work
            self, robot_config=None, room_length=1.0, room_width=0.5, num_cubes=0, env_name='small_empty',
            use_robot_map=True, use_distance_to_receptacle_map=False, distance_to_receptacle_map_scale=0.25,
            use_shortest_path_to_receptacle_map=True, use_shortest_path_map=True, shortest_path_map_scale=0.25,
            use_intention_map=False, intention_map_encoding='ramp',
            intention_map_scale=1.0, intention_map_line_thickness=2,
            use_history_map=False,
            use_intention_channels=False, intention_channel_encoding='spatial', intention_channel_nonspatial_scale=0.025,
            use_shortest_path_partial_rewards=True, success_reward=1.0, partial_rewards_scale=2.0,
            lifting_pointless_drop_penalty=0.25, obstacle_collision_penalty=0.25, robot_collision_penalty=1.0,
            use_shortest_path_movement=True, use_partial_observations=True,
            inactivity_cutoff_per_robot=1000,
            random_seed=None, use_egl_renderer=False,
            show_gui=False, show_debug_annotations=False, show_occupancy_maps=False,
            real=False, real_robot_indices=None, real_cube_indices=None, real_debug=False,
            obs_radius = 0.1, termination_step = 2000
        ):

        ################################################################################
        # Arguments

        # Room configuration
        self.robot_config = robot_config
        self.room_length = room_length
        self.room_width = room_width
        self.env_name = env_name

        # Misc
        self.use_egl_renderer = use_egl_renderer
        self.random_seed = random_seed

        # Debugging
        self.show_gui = show_gui
        self.show_debug_annotations = show_debug_annotations

        self.obs_radius = obs_radius
        self.radiusIds = []
        self.termination_step = termination_step
        # self.n_agent = n_agent

        # self.action_space = spaces.Discrete(5)

        pprint(self.__dict__)

        ################################################################################
        # Set up pybullet

        if self.show_gui:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            if self.use_egl_renderer:
                assert sys.platform == 'linux'  # Linux only
                self.plugin_id = self.p.loadPlugin(pkgutil.get_loader('eglRenderer').get_filename(), "_eglRendererPlugin")

        self.p.resetDebugVisualizerCamera(
            0.47 + (5.25 - 0.47) / (10 - 0.7) * (self.room_length - 0.7), 0, -70,
            (0, -(0.07 + (1.5 - 0.07) / (10 - 0.7) * (self.room_width - 0.7)), 0))

        # Used to determine whether robot poses are out of date
        self.step_simulation_count = 0

        ################################################################################
        # Robots and room configuration

        # Random placement of robots, cubes, and obstacles
        self.room_random_state = np.random.RandomState(self.random_seed)
        self.robot_spawn_bounds = None
        self.cube_spawn_bounds = None

        # Robots
        if self.robot_config is None:
            self.robot_config = [{'pushing_robot': 2}]
        self.num_robots = sum(sum(g.values()) for g in self.robot_config)
        self.robot_group_types = [next(iter(g.keys())) for g in self.robot_config]
        self.robot_ids = None
        self.robots = None
        self.robot_groups = None
        self.robot_random_state = np.random.RandomState(self.random_seed + 1 if self.random_seed is not None else None)  # Add randomness to throwing
        self.n_agent = self.num_robots

        # Room
        self.obstacle_ids = None
        self.receptacle_id = None
        if not any('rescue_robot' in g for g in self.robot_config):
            self.receptacle_position = (self.room_length / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, self.room_width / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, 0)
            print("self.receptacle_position", self.receptacle_position)

        # Collections for keeping track of environment state
        self.obstacle_collision_body_b_ids_set = None  # For collision detection
        self.robot_collision_body_b_ids_set = None  # For collision detection

        ################################################################################
        # Misc
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_agent * 2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3 * self.n_agent + 2,), dtype=np.float32)

        # Stats
        self.simulation_steps = None

    def reset(self):
        # Reset pybullet
        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0, 0, -9.8)

        # Create env
        self._create_env()

        self.drawRadius()

        self._reset_poses()
        
        

        # Stats
        self.simulation_steps = 0

        
        return self.get_macro_obs()

    def store_new_action(self, action):
        for i in range(self.num_robots):
            if self.robots[i].is_idle():
                self.robots[i].store_new_action([action[i * 2] * self.room_length / 2, action[i * 2 + 1] * self.room_width / 2])

    def step(self, action):
        ################################################################################
        # Setup before action execution

        self.store_new_action(action)

        ################################################################################
        # Execute actions
        self._execute_actions()

        # Increment counters
        self.simulation_steps += 1

        done = self.if_done()

        if done:
            reward = [1] * self.num_robots
        else:
            reward = [0] * self.num_robots
        
        info = {}

        return self.get_macro_obs(), reward, done, info

    def get_state(self):
        state = []
        for robot1 in self.robots:
            obs = []
            position1, heading1 = robot1.get_position(), robot1.get_heading()
            for robot2 in self.robots:
                position2, heading2 = robot2.get_position(), robot2.get_heading()
                if distance(position1, position2) <= self.obs_radius:
                    obs += [position2[0], position2[1], heading2]
                else:
                    obs += [-1, -1, 0]
            if distance(position1, self.receptacle_position) <= self.obs_radius:
                obs += [self.receptacle_position[0], self.receptacle_position[1]] 
            else:
                obs += [-1, -1]
            state.append(np.array(obs))
        return state


    def get_macro_obs(self):
        state = []
        for robot1 in self.robots:
            obs = []
            if robot1.is_idle():
                position1, heading1 = robot1.get_position(), robot1.get_heading()
                for robot2 in self.robots:
                    position2, heading2 = robot2.get_position(), robot2.get_heading()
                    if distance(position1, position2) <= self.obs_radius:
                        obs += [position2[0], position2[1], heading2]
                    else:
                        obs += [-1, -1, 0]
                if distance(position1, self.receptacle_position) <= self.obs_radius:
                    obs += [self.receptacle_position[0], self.receptacle_position[1]] 
                else:
                    obs += [-1, -1]
                robot1.obs = obs
            else:
                obs = robot1.obs
            state.append(np.array(obs))
        return state

    def drawRadius(self):
        self.p.removeAllUserDebugItems()
        colors = [[1, 0, 0], [0, 1, 0]]
        for robot, color, radiusIds in zip(self.robots, colors, self.radiusIds):
            x, y, _ = robot.get_position()
            t = 0
            pre_pos1 = [np.cos(t) * self.obs_radius + x, np.sin(t) * self.obs_radius + y, 0.01]
            for i in range(21):
                target_pos1 = [np.cos(t) * self.obs_radius + x , np.sin(t)  * self.obs_radius + y, 0.01]
                #radiusIds.append(self.p.addUserDebugLine(pre_pos1, target_pos1, color, lineWidth = 3, parentObjectUniqueId=robot.id, ))
                radiusIds.append(self.p.addUserDebugLine(pre_pos1, target_pos1, color, lineWidth = 3))
                pre_pos1 = target_pos1
                t += math.pi / 10

    def updateRadius(self):
        colors = [[1, 0, 0], [0, 1, 0]]
        for robot, color, radiusIds in zip(self.robots, colors, self.radiusIds):
            x, y, _ = robot.get_position()
            t = 0
            pre_pos1 = [np.cos(t) * self.obs_radius + x, np.sin(t) * self.obs_radius + y, 0.01]
            for i in range(21):
                target_pos1 = [np.cos(t) * self.obs_radius + x , np.sin(t)  * self.obs_radius + y, 0.01]
                #self.p.addUserDebugLine(pre_pos1, target_pos1, color, lineWidth = 3, replaceItemUniqueId=radiusIds[i], parentObjectUniqueId=robot.id)
                self.p.addUserDebugLine(pre_pos1, target_pos1, color, lineWidth = 3, replaceItemUniqueId=radiusIds[i])
                pre_pos1 = target_pos1
                t += math.pi / 10

    def if_done(self):
        done = True
        for robot in self.robots:
            if not self.robot_in_receptacle(robot):
                done = False
        if self.simulation_steps >= self.termination_step:
            done = True
        return done 

    def robot_in_receptacle(self, robot):
        rx, ry, _ = robot.get_position()
        tx, ty, _ = self.receptacle_position
        x_min = tx - VectorEnv.RECEPTACLE_WIDTH / 2
        x_max = tx + VectorEnv.RECEPTACLE_WIDTH / 2
        y_min = ty - VectorEnv.RECEPTACLE_WIDTH / 2
        y_max = ty + VectorEnv.RECEPTACLE_WIDTH / 2
        return (rx >= x_min and rx <= x_max and ry >= y_min and ry <= y_max)

    def close(self):
        if not self.show_gui and self.use_egl_renderer:
            self.p.unloadPlugin(self.plugin_id)
        self.p.disconnect()

    def step_simulation(self):
        self.p.stepSimulation()
        import time; time.sleep(1.0 / 60)
        self.step_simulation_count += 1
        self.updateRadius()
        #self.drawRadius()

    def get_robot_group_types(self):
        return self.robot_group_types

    def get_camera_image(self, image_width=1024, image_height=768):
        renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.show_gui else pybullet.ER_TINY_RENDERER
        return self.p.getCameraImage(image_width, image_height, flags=pybullet.ER_NO_SEGMENTATION_MASK, renderer=renderer)[2]

    def start_video_logging(self, video_path):
        assert self.show_gui
        return self.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, video_path)

    def stop_video_logging(self, log_id):
        self.p.stopStateLogging(log_id)

    def _create_env(self):

        # Create floor
        floor_thickness = 10
        wall_thickness = 1.4
        room_length_with_walls = self.room_length + 2 * wall_thickness
        room_width_with_walls = self.room_width + 2 * wall_thickness
        floor_half_extents = (room_length_with_walls / 2, room_width_with_walls / 2, floor_thickness / 2)
        floor_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        floor_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        self.p.createMultiBody(0, floor_collision_shape_id, floor_visual_shape_id, (0, 0, -floor_thickness / 2))

        # Create obstacles (including walls)
        obstacle_color = (0.9, 0.9, 0.9, 1)
        rounded_corner_path = str(Path(__file__).parent / 'assets' / 'rounded_corner.obj')
        self.obstacle_ids = []
        for obstacle in self._get_obstacles(wall_thickness):
            if obstacle['type'] == 'corner':
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_MESH, fileName=rounded_corner_path)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_MESH, fileName=rounded_corner_path, rgbaColor=obstacle_color)
            else:
                half_height = VectorEnv.CUBE_WIDTH / 2 if 'low' in obstacle else VectorEnv.WALL_HEIGHT / 2
                obstacle_half_extents = (obstacle['x_len'] / 2, obstacle['y_len'] / 2, half_height)
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)

            obstacle_id = self.p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                (obstacle['position'][0], obstacle['position'][1], VectorEnv.WALL_HEIGHT / 2), heading_to_orientation(obstacle['heading']))
            self.obstacle_ids.append(obstacle_id)

        # Create target receptacle
        if not any('rescue_robot' in g for g in self.robot_config):
            receptacle_color = (1, 87.0 / 255, 89.0 / 255, 1)  # Red
            receptacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=(0, 0, 0))
            receptacle_visual_shape_id = self.p.createVisualShape(
                #pybullet.GEOM_BOX, halfExtents=(VectorEnv.RECEPTACLE_WIDTH / 2, VectorEnv.RECEPTACLE_WIDTH / 2, 0),  # Gets rendered incorrectly in EGL renderer if height is 0
                pybullet.GEOM_BOX, halfExtents=(VectorEnv.RECEPTACLE_WIDTH / 2, VectorEnv.RECEPTACLE_WIDTH / 2, 0.0001),
                rgbaColor=receptacle_color, visualFramePosition=(0, 0, 0.0001))
            self.receptacle_id = self.p.createMultiBody(0, receptacle_collision_shape_id, receptacle_visual_shape_id, self.receptacle_position)

        # Create robots
        self.robot_collision_body_b_ids_set = set()
        self.robot_ids = []
        self.robots = []  # Flat list
        self.robot_groups = [[] for _ in range(len(self.robot_config))]  # Grouped list
        for robot_group_index, g in enumerate(self.robot_config):
            robot_type, count = next(iter(g.items()))
            for _ in range(count):
                robot = Robot.get_robot(robot_type, self, robot_group_index)
                self.robots.append(robot)
                self.robot_groups[robot_group_index].append(robot)
                self.robot_ids.append(robot.id)

        # Initialize collections
        self.obstacle_collision_body_b_ids_set = set(self.obstacle_ids)
        self.robot_collision_body_b_ids_set.update(self.robot_ids)

        self.radiusIds = [[] for _ in range(len(self.robots))]

    def _get_obstacles(self, wall_thickness):
        # if self.env_name.startswith('small'):
        #     assert math.isclose(self.room_length, 1)
        #     assert math.isclose(self.room_width, 0.5)
        # elif self.env_name.startswith('large'):
        #     assert math.isclose(self.room_length, 1)
        #     assert math.isclose(self.room_width, 1)

        def add_divider(x_offset=0):
            divider_width = 0.05
            opening_width = 0.16
            obstacles.append({'type': 'divider', 'position': (x_offset, 0), 'heading': 0, 'x_len': divider_width, 'y_len': self.room_width - 2 * opening_width})
            self.robot_spawn_bounds = (x_offset + divider_width / 2, None, None, None)
            self.cube_spawn_bounds = (None, x_offset - divider_width / 2, None, None)

        def add_tunnels(tunnel_length, x_offset=0, y_offset=0):
            tunnel_width = 0.18
            tunnel_x = (self.room_length + tunnel_width) / 6 + x_offset
            outer_divider_len = self.room_length / 2 - tunnel_x - tunnel_width / 2
            divider_x = self.room_length / 2 - outer_divider_len / 2
            middle_divider_len = 2 * (tunnel_x - tunnel_width / 2)
            obstacles.append({'type': 'divider', 'position': (-divider_x, y_offset), 'heading': 0, 'x_len': outer_divider_len, 'y_len': tunnel_length})
            obstacles.append({'type': 'divider', 'position': (0, y_offset), 'heading': 0, 'x_len': middle_divider_len, 'y_len': tunnel_length})
            obstacles.append({'type': 'divider', 'position': (divider_x, y_offset), 'heading': 0, 'x_len': outer_divider_len, 'y_len': tunnel_length})
            self.robot_spawn_bounds = (None, None, y_offset + tunnel_length / 2, None)
            self.cube_spawn_bounds = (None, None, None, y_offset - tunnel_length / 2)

        def add_rooms(x_offset=0, y_offset=0):
            divider_width = 0.05
            opening_width = 0.18
            divider_len = self.room_width / 2 - opening_width - divider_width / 2
            top_divider_len = divider_len - y_offset
            bot_divider_len = divider_len + y_offset
            top_divider_y = self.room_width / 2 - opening_width - top_divider_len / 2
            bot_divider_y = -self.room_width / 2 + opening_width + bot_divider_len / 2
            obstacles.append({'type': 'divider', 'position': (0, y_offset), 'heading': 0, 'x_len': self.room_length - 2 * opening_width, 'y_len': divider_width})
            obstacles.append({'type': 'divider', 'position': (x_offset, top_divider_y), 'heading': 0, 'x_len': divider_width, 'y_len': top_divider_len, 'snap_y': y_offset + divider_width / 2})
            obstacles.append({'type': 'divider', 'position': (x_offset, bot_divider_y), 'heading': 0, 'x_len': divider_width, 'y_len': bot_divider_len, 'snap_y': y_offset - divider_width / 2})

        # Walls
        obstacles = []
        for x, y, length, width in [
                (-self.room_length / 2 - wall_thickness / 2, 0, wall_thickness, self.room_width),
                (self.room_length / 2 + wall_thickness / 2, 0, wall_thickness, self.room_width),
                (0, -self.room_width / 2 - wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
                (0, self.room_width / 2 + wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'x_len': length, 'y_len': width})

        # Other obstacles
        if self.env_name == 'small_empty':
            pass

        elif self.env_name == 'small_divider_norand':
            add_divider()

        elif self.env_name == 'small_divider':
            add_divider(x_offset=self.room_random_state.uniform(-0.1, 0.1))

        elif self.env_name == 'large_empty':
            pass

        elif self.env_name == 'large_doors_norand':
            add_tunnels(0.05)

        elif self.env_name == 'large_doors':
            add_tunnels(0.05, x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.1, 0.1))

        elif self.env_name == 'large_tunnels_norand':
            add_tunnels(0.25)

        elif self.env_name == 'large_tunnels':
            add_tunnels(0.25, x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.05, 0.05))

        elif self.env_name == 'large_rooms_norand':
            add_rooms()

        elif self.env_name == 'large_rooms':
            add_rooms(x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.05, 0.05))

        else:
            raise Exception(self.env_name)

        ################################################################################
        # Rounded corners

        rounded_corner_width = 0.1006834873
        # Room corners
        for i, (x, y) in enumerate([
                (-self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, -self.room_width / 2),
                (-self.room_length / 2, -self.room_width / 2),
            ]):
            if any('rescue_robot' in g for g in self.robot_config) or distance((x, y), self.receptacle_position) > (1 + 1e-6) * (VectorEnv.RECEPTACLE_WIDTH / 2) * math.sqrt(2):
                heading = -math.radians(i * 90)
                offset = rounded_corner_width / math.sqrt(2)
                adjusted_position = (x + offset * math.cos(heading - math.radians(45)), y + offset * math.sin(heading - math.radians(45)))
                obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})

        # Corners between walls and dividers
        new_obstacles = []
        for obstacle in obstacles:
            if obstacle['type'] == 'divider':
                position, length, width = obstacle['position'], obstacle['x_len'], obstacle['y_len']
                x, y = position
                corner_positions = None
                if math.isclose(x - length / 2, -self.room_length / 2):
                    corner_positions = [(-self.room_length / 2, y - width / 2), (-self.room_length / 2, y + width / 2)]
                    corner_headings = [0, 90]
                elif math.isclose(x + length / 2, self.room_length / 2):
                    corner_positions = [(self.room_length / 2, y - width / 2), (self.room_length / 2, y + width / 2)]
                    corner_headings = [-90, 180]
                elif math.isclose(y - width / 2, -self.room_width / 2):
                    corner_positions = [(x - length / 2, -self.room_width / 2), (x + length / 2, -self.room_width / 2)]
                    corner_headings = [180, 90]
                elif math.isclose(y + width / 2, self.room_width / 2):
                    corner_positions = [(x - length / 2, self.room_width / 2), (x + length / 2, self.room_width / 2)]
                    corner_headings = [-90, 0]
                elif 'snap_y' in obstacle:
                    snap_y = obstacle['snap_y']
                    corner_positions = [(x - length / 2, snap_y), (x + length / 2, snap_y)]
                    corner_headings = [-90, 0] if snap_y > y else [180, 90]
                if corner_positions is not None:
                    for position, heading in zip(corner_positions, corner_headings):
                        heading = math.radians(heading)
                        offset = rounded_corner_width / math.sqrt(2)
                        adjusted_position = (
                            position[0] + offset * math.cos(heading - math.radians(45)),
                            position[1] + offset * math.sin(heading - math.radians(45))
                        )
                        obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})
        obstacles.extend(new_obstacles)

        return obstacles

    def _reset_poses(self):
        # Reset robot poses
        for robot in self.robots:
            pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
            robot.reset_pose(pos_x, pos_y, heading)

        # Check if any robots need another pose reset
        done = False
        while not done:
            done = True
            self.step_simulation()
            for robot in self.robots:
                reset_robot_pose = False

                # Check if robot is stacked on top of a cube
                if robot.get_position(set_z_to_zero=False)[2] > 0.001:  # 1 mm
                    reset_robot_pose = True

                # Check if robot is inside an obstacle or another robot
                for contact_point in self.p.getContactPoints(robot.id):
                    if contact_point[2] in self.obstacle_collision_body_b_ids_set or contact_point[2] in self.robot_collision_body_b_ids_set:
                        reset_robot_pose = True
                        break

                if reset_robot_pose:
                    done = False
                    pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
                    robot.reset_pose(pos_x, pos_y, heading)

    def _get_random_robot_pose(self, padding=0, bounds=None):
        position_x, position_y = self._get_random_position(padding=padding, bounds=bounds)
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return position_x, position_y, heading

    def _get_random_position(self, padding=0, bounds=None):
        low_x = -self.room_length / 2 + padding
        high_x = self.room_length / 2 - padding
        low_y = -self.room_width / 2 + padding
        high_y = self.room_width / 2 - padding
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if x_min is not None:
                low_x = x_min + padding
            if x_max is not None:
                high_x = x_max - padding
            if y_min is not None:
                low_y = y_min + padding
            if y_max is not None:
                high_y = y_max - padding
        position_x, position_y = self.room_random_state.uniform((low_x, low_y), (high_x, high_y))
        return position_x, position_y

    # def _execute_actions(self):
    #     sim_steps = 0
    #     while True:
    #         if any(robot.is_idle() for robot in self.robots):
    #             break
    #         sim_steps += 1
    #         for robot in self.robots:
    #             robot.step()
    #         self.step_simulation()

    #     return sim_steps

    def _execute_actions(self):
        # while True:
        #     if any(robot.is_idle() for robot in self.robots):
        #         break
        for robot in self.robots:
            robot.step()
        self.step_simulation()

    @property
    def state_size(self):
        return self.get_state()[0].shape[0] * self.n_agent

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return self.action_space.shape[0]

    @property
    def action_spaces(self):
        return [[self.action_space.low[0], self.action_space.high[0]]] * self.action_space.shape[0]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [self.action_spaces[nth * 2], self.action_spaces[nth * 2 + 1]]

    def action_space_sample(self, i):
        return [np.random.uniform(self.action_spaces[i][0], self.action_spaces[i][1]), np.random.uniform(self.action_spaces[i][0], self.action_spaces[i][1])]
    
    def _collectCurMacroActions(self):
        # loop each agent
        cur_mac = []
        for robot in self.robots:
            cur_mac.append(robot.action)
        return cur_mac

    def _computeMacroActionDone(self):
        # loop each agent
        mac_done = []
        for robot in self.robots:
            mac_done.append(True if robot.is_idle() else False)
        return mac_done

    def macro_action_sample(self):
        mac_actions = []
        for i in range(self.n_agent):
            mac_actions += self.action_space_sample(i)
        return mac_actions

    def build_agents(self):
        raise

    def build_macro_actions(self):
        raise








class Robot(ABC):
    HALF_WIDTH = 0.03
    BACKPACK_OFFSET = -0.0135
    BASE_LENGTH = 0.065  # Does not include the hooks
    TOP_LENGTH = 0.057  # Leaves 1 mm gap for lifted cube
    END_EFFECTOR_LOCATION = BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    HEIGHT = 0.07
    NUM_OUTPUT_CHANNELS = 1
    COLOR = (0.3529, 0.3529, 0.3529, 1)  # Gray
    CONSTRAINT_MAX_FORCE = 10

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, env, group_index, obs_radius=0.2, real=False, real_robot_index=None):
        self.env = env
        self.group_index = group_index
        self.real = real
        self.obs_radius = obs_radius
        self.id = self._create_multi_body()
        self.cid = self.env.p.createConstraint(self.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0))
        self._last_step_simulation_count = -1  # Used to determine whether pose is out of date
        self._position_raw = None  # Most current position, not to be directly accessed (use self.get_position())
        self._position = None  # Most current position (with z set to 0), not to be directly accessed (use self.get_position())
        self._heading = None  # Most current heading, not to be directly accessed (use self.get_heading())
        self.state = None
        

        # Movement
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller = RobotController(self)

        # Collision detection
        self.collision_body_a_ids_set = set([self.id])

        # State representation
        #self.mapper = Mapper(self.env, self)

        # Step variables and stats
        self.awaiting_new_action = False  # Only one robot at a time can be awaiting new action
        self.distance = 0
        self.prev_waypoint_position = None  # For tracking distance traveled over the step
        self.collided_with_obstacle = False
        self.collided_with_robot = False


    def store_new_action(self, action):
        # Action is specified as an index specifying an end effector action, along with (row, col) of the selected pixel location
        #self.action = tuple(np.unravel_index(action, (self.NUM_OUTPUT_CHANNELS, Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH)))  # Immutable tuple
        self.action = action

        # Get current robot pose
        current_position, current_heading = self.get_position(), self.get_heading()

        ################################################################################
        # Step variables and stats

        self.target_position = (action[0], action[1], 0)
        self.waypoint_positions = [current_position, self.target_position]

        self.waypoint_headings = [current_heading]
        for i in range(1, len(self.waypoint_positions)):
            dx = self.waypoint_positions[i][0] - self.waypoint_positions[i - 1][0]
            dy = self.waypoint_positions[i][1] - self.waypoint_positions[i - 1][1]
            self.waypoint_headings.append(restrict_heading_range(math.atan2(dy, dx)))

        # Reset controller
        self.controller.reset()
        self.controller.new_action()

        # Reset step variables and stats
        self.awaiting_new_action = False
        self.cubes = 0
        self.reward = None
        self.cubes_with_reward = 0
        self.distance = 0
        self.prev_waypoint_position = current_position
        self.collided_with_obstacle = False
        self.collided_with_robot = False

    def step(self):
        self.controller.step()

    def reset(self):
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller.reset()

    def is_idle(self):
        return self.controller.state == 'idle'

    def get_position(self, set_z_to_zero=True):
        # Returned position is immutable tuple
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        if not set_z_to_zero:
            return self._position_raw
        return self._position

    def get_heading(self):
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        return self._heading

    def reset_pose(self, position_x, position_y, heading):
        # Reset robot pose
        position = (position_x, position_y, 0)
        orientation = heading_to_orientation(heading)
        self.env.p.resetBasePositionAndOrientation(self.id, position, orientation)
        self.env.p.changeConstraint(self.cid, jointChildPivot=position, jointChildFrameOrientation=orientation, maxForce=Robot.CONSTRAINT_MAX_FORCE)
        self._last_step_simulation_count = -1

    def check_for_collisions(self):
        for body_a_id in self.collision_body_a_ids_set:
            for contact_point in self.env.p.getContactPoints(body_a_id):
                body_b_id = contact_point[2]
                if body_b_id in self.collision_body_a_ids_set:
                    continue
                if body_b_id in self.env.obstacle_collision_body_b_ids_set:
                    self.collided_with_obstacle = True
                if body_b_id in self.env.robot_collision_body_b_ids_set:
                    self.collided_with_robot = True
                if self.collided_with_obstacle or self.collided_with_robot:
                    break

    def update_distance(self):
        current_position = self.get_position()
        self.distance += distance(self.prev_waypoint_position, current_position)
        if self.env.show_debug_annotations:
            self.env.p.addUserDebugLine(
                (self.prev_waypoint_position[0], self.prev_waypoint_position[1], 0.001),
                (current_position[0], current_position[1], 0.001),
                VectorEnv.DEBUG_LINE_COLORS[self.group_index]
            )
        self.prev_waypoint_position = current_position

    def _update_pose(self):
        position, orientation = self.env.p.getBasePositionAndOrientation(self.id)
        self._position_raw = position
        self._position = (position[0], position[1], 0)  # Use immutable tuples to represent positions
        self._heading = orientation_to_heading(orientation)
        self._last_step_simulation_count = self.env.step_simulation_count

    def _create_multi_body(self):
        base_height = 0.035
        mass = 0.180
        shape_types = [pybullet.GEOM_CYLINDER, pybullet.GEOM_BOX, pybullet.GEOM_BOX]
        radii = [Robot.HALF_WIDTH, None, None]
        half_extents = [
            None,
            (self.BASE_LENGTH / 2, Robot.HALF_WIDTH, base_height / 2),
            (Robot.TOP_LENGTH / 2, Robot.HALF_WIDTH, Robot.HEIGHT / 2),
        ]
        lengths = [Robot.HEIGHT, None, None]
        rgba_colors = [self.COLOR, None, None]  # pybullet seems to ignore all colors after the first
        frame_positions = [
            (Robot.BACKPACK_OFFSET, 0, Robot.HEIGHT / 2),
            (Robot.BACKPACK_OFFSET + self.BASE_LENGTH / 2, 0, base_height / 2),
            (Robot.BACKPACK_OFFSET + Robot.TOP_LENGTH / 2, 0, Robot.HEIGHT / 2),
        ]
        collision_shape_id = self.env.p.createCollisionShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, collisionFramePositions=frame_positions)
        visual_shape_id = self.env.p.createVisualShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, rgbaColors=rgba_colors, visualFramePositions=frame_positions)
        return self.env.p.createMultiBody(mass, collision_shape_id, visual_shape_id)

    @staticmethod
    def get_robot_cls(robot_type):
        if robot_type == 'pushing_robot':
            return PushingRobot
        raise Exception(robot_type)

    @staticmethod
    def get_robot(robot_type, *args, real=False, real_robot_index=None):
        return Robot.get_robot_cls(robot_type)(*args, real=real, real_robot_index=real_robot_index)

class PushingRobot(Robot):
    BASE_LENGTH = Robot.BASE_LENGTH + 0.005  # 5 mm blade
    END_EFFECTOR_LOCATION = Robot.BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(Robot.HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    COLOR = (0.1765, 0.1765, 0.1765, 1)  # Dark gray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_dist_closer = 0

    def store_new_action(self, action):
        super().store_new_action(action)
        self.cube_dist_closer = 0

class RobotController:
    DRIVE_STEP_SIZE = 0.005  # 5 mm results in exactly 1 mm per simulation step
    TURN_STEP_SIZE = math.radians(15)  # 15 deg results in exactly 3 deg per simulation step

    def __init__(self, robot):
        self.robot = robot
        self.state = 'idle'
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.prev_position = None  # Position before call to p.stepSimulation()
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = None  # Used to detect if robot is stuck and oscillating
        self.manipulation_sim_step_target = 0
        self.manipulation_sim_steps = 0

    def reset(self):
        self.state = 'idle'
        self.waypoint_index = 1
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = 0

    def new_action(self):
        self.state = 'moving'

    def step(self):
        # States: idle, moving, manipulating

        assert not self.state == 'idle'
        self.sim_steps += 1

        if self.state == 'moving':
            current_position, current_heading = self.robot.get_position(), self.robot.get_heading()

            # First check change after sim step
            if self.prev_position is not None:

                # Detect if robot is still moving
                driving = distance(self.prev_position, current_position) > 0.0005  # 0.5 mm
                turning = abs(heading_difference(self.prev_heading, current_heading)) > math.radians(1)  # 1 deg
                self.consecutive_turning_sim_steps = (self.consecutive_turning_sim_steps + 1) if turning else 0
                stuck_oscillating = self.consecutive_turning_sim_steps > 100  # About 60 sim steps is sufficient for turning 180 deg
                not_moving = (not driving and not turning) or stuck_oscillating

                # Check for collisions
                if distance(self.robot.waypoint_positions[0], current_position) > RobotController.DRIVE_STEP_SIZE or not_moving:
                    self.robot.check_for_collisions()

                # Check if step limit exceeded (expect this won't ever happen, but just in case)
                step_limit_exceeded = self.sim_steps > 3200

                if self.robot.collided_with_obstacle or self.robot.collided_with_robot or step_limit_exceeded:
                    self.robot.update_distance()
                    self.state = 'idle'

                if self.state == 'moving' and not_moving:
                    # Reached current waypoint, move on to next waypoint
                    self.robot.update_distance()
                    if self.waypoint_index == len(self.robot.waypoint_positions) - 1:
                        self._done_moving()
                    else:
                        self.waypoint_index += 1

            # If still moving, set constraint for new pose
            if self.state == 'moving':
                new_position, new_heading = current_position, current_heading

                # Determine whether to turn or drive
                heading_diff = heading_difference(current_heading, self.robot.waypoint_headings[self.waypoint_index])
                if abs(heading_diff) > RobotController.TURN_STEP_SIZE:
                    new_heading += math.copysign(1, heading_diff) * RobotController.TURN_STEP_SIZE
                else:
                    curr_waypoint_position = self.robot.waypoint_positions[self.waypoint_index]
                    dx = curr_waypoint_position[0] - current_position[0]
                    dy = curr_waypoint_position[1] - current_position[1]
                    if distance(current_position, curr_waypoint_position) < RobotController.DRIVE_STEP_SIZE:
                        new_position = curr_waypoint_position
                    else:
                        move_sign = 1
                        new_heading = math.atan2(move_sign * dy, move_sign * dx)
                        new_position = (
                            new_position[0] + move_sign * RobotController.DRIVE_STEP_SIZE * math.cos(new_heading),
                            new_position[1] + move_sign * RobotController.DRIVE_STEP_SIZE * math.sin(new_heading),
                            new_position[2]
                        )

                # Set constraint
                self.robot.env.p.changeConstraint(
                    self.robot.cid, jointChildPivot=new_position, jointChildFrameOrientation=heading_to_orientation(new_heading), maxForce=Robot.CONSTRAINT_MAX_FORCE)

            self.prev_position, self.prev_heading = current_position, current_heading

        elif self.state == 'manipulating':
            self.manipulation_sim_steps += 1
            if self.manipulation_sim_steps >= self.manipulation_sim_step_target:
                self.manipulation_sim_step_target = 0
                self.manipulation_sim_steps = 0
                self.state = 'idle'

    def get_intention_path(self):
        return [self.robot.get_position()] + self.robot.waypoint_positions[self.waypoint_index:-1] + [self.robot.target_end_effector_position]

    def get_history_path(self):
        return self.robot.waypoint_positions[:self.waypoint_index] + [self.robot.get_position()]

    def _done_moving(self):
        self.state = 'idle'

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def orientation_to_heading(o):
    # Note: Only works for z-axis rotations
    return 2 * math.acos(math.copysign(1, o[2]) * o[3])

def heading_to_orientation(h):
    return pybullet.getQuaternionFromEuler((0, 0, h))

def restrict_heading_range(h):
    return (h + math.pi) % (2 * math.pi) - math.pi

def heading_difference(h1, h2):
    return restrict_heading_range(h2 - h1)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]
