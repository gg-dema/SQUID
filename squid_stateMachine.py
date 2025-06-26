
import sys
import os
import rospy
import cor_tud_msgs.msg as cor_msg

sys.path.append('../../')
from inverse_kinematics import inverse_kinematics_init

sys.path.append('/catkin_ros1_ws/src/squid/src')
os.chdir('/catkin_ros1_ws/src/squid/src')
from initializer import initialize_framework
from agent.utils.dynamical_system_operations import normalize_state, denormalize_state
import torch
import numpy as np
from spatialmath import UnitQuaternion
import cor_tud_msgs.msg as cor_msg
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
import importlib
import math
from pynput.keyboard import Listener, KeyCode


ROBOT = sys.argv[1]
# aka iiwa14



class PUMAControl():
	def __init__(self):
		"""PUMA to control the KUKA robot
		Parameters to check:
		- self.q_initial: configuration robot will move to when code starts
		- self.q_home: home configuration, can be useful for state machine
		- self.speed_factor: increases speed output from PUMA
		- self.delta_t: should be equal to data sampling frequency usFed for training
		"""

		# Important parameters
		self.q_home = np.array([0, 1, 0, 1.0, 0.0, 0.5, 0])

		self.q_2 = np.array([1.513038071462094, 0.9885089509512737, 0.5220451413876885, 1.36919827717424, 0.35048540073838463, 0.14629154536127498, 2.461169508133411])
		self.q_1 = np.array([1.513038071462094, -0.9885089509512737, -0.5220451413876885, -1.5976919827717424, -0.35048540073838463, 0.14629154536127498, 2.461169508133411])
		#self.q_home = self.q_initial
		self.speed_factor = 20
		self.delta_t = 0.03

		# Parameter LinearDS joint control
		self.gain_joint = 0.2

		# Threshold for goal reach method
		self.goal_reach_thr = 0.5

		# Offsets
		self.quat_offset = UnitQuaternion([1.0, 0.0, 0.0, 0.0])
		self.x_offset = -0.1
		self.z_offset = +0.3
		self.y_offset = 0.0

		self.x_base = 0.015712
		self.y_base = -0.76491
		self.z_base = 0.38089
		# no base quat because it starts at 1, 0, 0, 0

		# Init parent
		super(PUMAControl, self).__init__()

		self.IK = None

		# Load model parameters
		params_name = 'Discrete_Condor_s2_kuka'
		Params = getattr(importlib.import_module('params.' + params_name), 'Params')
		params = Params('./')

		# Modify some parameters
		self.start = False
		self.stuck = False
		self.go_to_next_goal = False
		params.load_model = True
		params.selected_primitives_ids = self._select_primitive_ids(params)
		params.results_path = params.results_path + params.selected_primitives_ids + '/'

		self.init_primitive_id = torch.FloatTensor([[1]]).cuda()
		self.dynamical_system = None
		self.encode_primitive_type = True
		self.home_reached = False

		# Initialize framework
		self.agent, _, self.info_demos = initialize_framework(params, params_name, verbose=False)

		self.q_d_traj = []
		self.rot_error_traj = []

		# ROS and robot control variables
		rospy.Subscriber('/%s/end_effector_state' % ROBOT, cor_msg.CartesianState, self._callback_end_effector_state, queue_size=10)
		rospy.Subscriber('/%s/joint_states' % ROBOT, JointState, self._callback_joint_states, queue_size=10)
		self.request_pub = rospy.Publisher('/%s/control_request' % ROBOT, cor_msg.ControlRequest, queue_size=10)
		self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

		# Start keyboard listener
		self.listener = Listener(on_press=self._on_press)
		self.listener.start()

		self.q = None
		self.q_d_stuck = None
		self.q_d_prev = None
		self.q_dot = None
		self.ee_pose = None
		self.ee_velocity = None
		self.position_initialized = False
		self.go_to_initial_position = False
		self.first_quaternion_checked = False

	def _callback_publish_marker(self, data, id_marker=1):
		marker = Marker()
		marker.header.frame_id = "world"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "goal_point"
		marker.id = id_marker
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD

		# Set pose from [x, y, z]
		marker.pose = Pose()
		marker.pose.position = Point(data[0], data[1], data[2])
		marker.pose.orientation = Quaternion(0, 0, 0, 1)

		# Set scale
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.1

		# Set color (green and opaque)
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0

		marker.lifetime = rospy.Duration(0)  # 0 means forever

		self.marker_pub.publish(marker)

	def _callback_end_effector_state(self, data):
		self.ee_pose = np.array(data.pose.data)
		self.ee_velocity = np.array(data.velocity.data)

	def _callback_joint_states(self, data):
		self.q = np.array(data.position)
		self.q_dot = np.array(data.velocity)

	def _on_press(self, key):
		# This function runs on the background and checks if a keyboard key was pressed
		if key == KeyCode.from_char('s'):
			self.start = True
			print('PUMA initialized.')
		if key == KeyCode.from_char('c'):
			self.go_to_next_goal = True

	def switch_to_velocity_control(self):
		# Switch joint impedance control to velocity mode
		damping_increase_factor = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6]) * 1  # TODO: do not have this hardcoded
		self.joint_impedance_controller.damping = self.joint_impedance_controller.damping * damping_increase_factor
		#self.joint_impedance_controller.stiffness = np.array([0, 0, 0, 0, 0, 0, 0])

	def send_request(self, q_d, q_dot_d=[0, 0, 0, 0, 0, 0, 0]):
		msg = cor_msg.ControlRequest()
		msg.header.stamp = rospy.Time.now()
		msg.q_d.data = q_d
		msg.q_dot_d.data = q_dot_d
		msg.control_type = 'joint impedance'
		self.request_pub.publish(msg)
		return True

	def _select_primitive_ids(self, params):
		return params.selected_primitives_ids

	def _callback_box_tracker(self, msg_stamped_pose):
		self.x_offset = msg_stamped_pose.pose.position.x - self.x_base
		self.y_offset = msg_stamped_pose.pose.position.y - self.y_base
		self.z_offset = msg_stamped_pose.pose.position.z - self.z_base

		w = msg_stamped_pose.pose.orientation.w
		x_quat = msg_stamped_pose.pose.orientation.x
		y_quat = msg_stamped_pose.pose.orientation.y
		z_quat = msg_stamped_pose.pose.orientation.z

		self.quat_offset = UnitQuaternion((w, x_quat, y_quat, z_quat))


	def get_offset(self, final_q):
		ee_pose_SE3 = self.robot.fkine(final_q, end='iiwa_link_7', start='iiwa_link_0')
		position = ee_pose_SE3.t
		orientation = UnitQuaternion(ee_pose_SE3.R)
		return position, orientation

	def _preprocess_x(self, x_raw):
		x = np.zeros(7)
		x_norm = np.zeros(7)

		# Translate DS reference frame
		x[0] = x_raw[0] - self.x_offset
		x[1] = x_raw[1] - self.y_offset
		x[2] = x_raw[2] - self.z_offset

		# Normalize translated position
		x_norm =  normalize_state(x, self.info_demos['x min'], self.info_demos['x max'])  # normalize state

		# Rotate DS reference frame
		orientation = UnitQuaternion([x_raw[3], x_raw[4], x_raw[5], x_raw[6]])

		# Check that we start in hemisphere nearest to goal
		if not self.first_quaternion_checked:
			orientation_1 = orientation.A
			orientation_2 = -orientation.A

			# Calculate the dot product
			dot_product_1 = orientation_1.dot(self.quat_offset.A)
			dot_product_2 = orientation_2.dot(self.quat_offset.A)

			# Calculate the angular distance
			angular_distance_1 = math.acos(dot_product_1)
			angular_distance_2 = math.acos(dot_product_2)

			if angular_distance_1 < angular_distance_2:
				orientation = UnitQuaternion(orientation_1)
				print('Not flipped', angular_distance_1)
			else:
				orientation = UnitQuaternion(orientation_2)
				print('Flipped', angular_distance_2)

			# Change flag
			self.first_quaternion_checked = True

		orientation_new = (orientation / self.quat_offset).A  # Rotation applied here
		x_norm[3] = orientation_new[0]
		x_norm[4] = orientation_new[1]
		x_norm[5] = orientation_new[2]
		x_norm[6] = orientation_new[3]

		# Create tensor
		x_tensor = torch.from_numpy(x_norm).reshape(1, -1).float().cuda()  # transform to tensor
		return x_tensor

	def _get_DS_initial_variables(self):
		# Get initial states through the workspace
		position = self.ee_pose[:3]
		quaternion = UnitQuaternion.RPY(self.ee_pose[3:]).A
		initial_pose_raw = np.concatenate([position, quaternion])
		initial_pose = self._preprocess_x(initial_pose_raw)  # get initial position from ROS topic and preprocess

		# Get Initial derivatives and append to initial states
		initial_derivatives = torch.zeros([1, self.agent.dynamical_system_order * self.agent.dim_state - self.agent.dim_state]).cuda()

		# Get initial states
		initial_state = torch.cat([initial_pose, initial_derivatives], dim=1)

		return initial_state

	def _initialize_DS(self):
		with torch.no_grad():
			initial_states = self._get_DS_initial_variables()

			# Initialize dynamical system
			self.dynamical_system = self.agent.init_dynamical_system(initial_states=initial_states)
			self.dynamical_system.primitive_type = self.init_primitive_id
		return True

	def check_goal_reached(self, error):
		# Get norm error
		norm_error = np.linalg.norm(error)
		print(norm_error)

		# If close to goal, reached
		if norm_error < self.goal_reach_thr:
			return True
		else:
			return  False

	def linear_DS_joint(self, q_goal):
		# Linear DS
		error = q_goal - self.q

		# Get delta q
		delta_q_d = self.gain_joint * error

		# Integrate
		q_d = self.q + delta_q_d

		return q_d, error

	def _request_PUMA(self, position, orientation):
		# Preprocess inputs
		quaternion = UnitQuaternion.RPY(orientation).A
		pose = np.concatenate([position, quaternion])

		x_processed = self._preprocess_x(pose)

		# Check that quaternion did not flip sign
		distance_quat = torch.linalg.norm(x_processed[0, 3:] - self.dynamical_system.x_t_d[0, 3:])
		if distance_quat > 1.0:
			x_processed[0, 3:] = -x_processed[0, 3:]

		# Get desired pose from PUMA
		transition = self.dynamical_system.transition(x_processed)
		pose_normalized_dot_d = transition['desired velocity']
		delta_t_model = 1

		# Integrate system and include speed factor
		pose_d_normalized = x_processed + pose_normalized_dot_d * delta_t_model * self.speed_factor

		# denormalize_pose
		pose_d = denormalize_state(pose_d_normalized[0].cpu().detach().numpy(), self.info_demos['x min'], self.info_demos['x max'])  # denormalize

		# Convert to quaternion object (which also projects quaternion back to the manifold)
		orientation_d = UnitQuaternion(pose_d[3:])

		# Add offset back
		pose_d[0] = pose_d[0] + self.x_offset
		pose_d[1] = pose_d[1] + self.y_offset
		pose_d[2] = pose_d[2] + self.z_offset
		pose_d[3:] = (orientation_d * self.quat_offset).A

		max_speed = 0.004
		velocity = (pose_d[:3] - pose[:3]) / delta_t_model
		speed = np.linalg.norm(velocity)
		if speed > max_speed:
			velocity_clipped = (velocity / speed) * max_speed
			pose_d[:3] = pose[:3] + velocity_clipped * delta_t_model
			print('Speed cliped!')

		# Compute inverse kinematics
		q_d = self.IK.compute(pose_d[:3], UnitQuaternion(pose_d[3:]).rpy(), self.q)

		if self.q_d_prev is None:
			self.q_d_prev = q_d

		ik_dist = np.linalg.norm(self.q_d_prev - q_d)

		thr = 0.15

		if ik_dist > thr:
			if not self.stuck:
				self.q_d_stuck = self.q
				q_d = self.q
			else:
				q_d = self.q_d_stuck

			if self.q_d_stuck is None:
				print('q_d_stuck None!')
				q_d = self.q

			self.stuck = True
			print('stuck', ik_dist)
		else:
			self.stuck = False

		self.q_d_prev = q_d

		# Compute velocity in joint space
		q_dot_d = (q_d - self.q) / self.delta_t

		# Run extra loop if near joint limit
		#self._fix_joint_limits(q_d, q_dot_d)

		return q_d, q_dot_d, pose_d


	def run_dema(self, control_frequency):

		ik_name = "ranged ik"
		rate = rospy.Rate(control_frequency)
		rospy.sleep(0.1)

		print("Let's start: READY")
		state = "init"

		while not rospy.is_shutdown():
			try:
				# waint until q is received
				if self.q is None: continue

				if state == "init":
					self.home_reached = False

					q_d, error = self.linear_DS_joint(self.q_home)
					self.send_request(q_d)

					if self.check_goal_reached(error):
						self.home_reached = True
						state = "start_q_conf_1"

				if state == "start_q_conf_1":

					q_d, error = self.linear_DS_joint(self.q_1)
					self.send_request(q_d)
					if self.check_goal_reached(error):
						state = "reach_q_conf_1"
						print("digit 's' for start squid")
						self.start = False
						self._initialize_DS()
						self.IK = inverse_kinematics_init(ik_name, self.q, ROBOT)  # always init IK when jumping to PUMA

				if state == "reach_q_conf_1":  # we can start squid :
					if self.start:
						ee_position = self.ee_pose[:3]
						ee_orientation = self.ee_pose[3:]

						q_des, q_dot_des, pose_des = self._request_PUMA(ee_position, ee_orientation)
						self.send_request(q_des, q_dot_des)

						# check if we reached the end point
						position_des = pose_des[:3]
						orientation_des = UnitQuaternion(pose_des[3:]).eul()
						pose_des = np.concatenate([position_des, orientation_des])
						error = self.ee_pose - pose_des

						if self.check_goal_reached(error) or self.go_to_next_goal:
							state = "reached_goal_1"
							self.start = False # wait user input before start the new puma
							self.go_to_next_goal = False
							self._callback_publish_marker(self.ee_pose[:3], id_marker=1)

				if state == "reached_goal_1": # go to conf 2

					q_d, error = self.linear_DS_joint(self.q_2)
					self.send_request(q_d)

					if self.check_goal_reached(error):
						state = "reach_q_conf_2"
						print("digit 's' for start squid")
						self.start = False
						self._initialize_DS()
						self.IK = inverse_kinematics_init(ik_name, self.q, ROBOT)  # always init IK when jumping to PUMA

				if state == "reach_q_conf_2":	# go to goal 2

					if self.start:
						ee_position = self.ee_pose[:3]
						ee_orientation = self.ee_pose[3:]
						q_des, q_dot_des, pose_des = self._request_PUMA(ee_position, ee_orientation)
						self.send_request(q_des, q_dot_des)

						# check if we reached the goal
						position_des = pose_des[:3]
						orientation_des = UnitQuaternion(pose_des[3:]).eul()
						pose_des = np.concatenate([position_des, orientation_des])
						error = self.ee_pose - pose_des

						if self.check_goal_reached(error) or self.go_to_next_goal:
							self._callback_publish_marker(self.ee_pose[:3], id_marker=2)
							state = "reached_goal_2"
							self.start = False
							self.go_to_next_goal = False

				if state == "reached_goal_2":
					q_d, error = self.linear_DS_joint(self.q_home)
					self.send_request(q_d)

					if self.check_goal_reached(error):
						state = "finish"

				if state == "finish":
					print('End reached')
					exit()

				# Sleep
				rate.sleep()
			except rospy.ROSInterruptException:
				pass

	def run(self, control_frequency):

		ik_name = 'ranged ik'  # options: ranged ik, track ik
		rate = rospy.Rate(control_frequency)

		rospy.sleep(0.1)

		print('Ready')
		state = 'init_conf_1_home'
		while not rospy.is_shutdown():
			try:
				# Do nothing until q is received
				if self.q is None:
					continue

				if state == 'init_conf_1_home':
					self.home_reached = False

					# Send initial position to low level controller
					q_d, error = self.linear_DS_joint(self.q_home)

					self.send_request(q_d)

					if self.check_goal_reached(error):
						self.home_reached = True
						print('Home configuration 1 reached')
						state = 'init_q_conf_1'

				if state == 'init_q_conf_1':
					# Send initial position to low level controller
					q_d, error = self.linear_DS_joint(self.q_home)
					self.send_request(q_d)

					if self.start:
						print('Home configuration 1 reached again')
						self.start = False

						# Initialize DS
						self._initialize_DS()
						self.IK = inverse_kinematics_init(ik_name, self.q, ROBOT)  # always init IK when jumping to PUMA
						state = 'puma_1'

				if state == 'puma_1':
					ee_position = self.ee_pose[:3]
					ee_orientation = self.ee_pose[3:]

					# Get desired position from CONDOR
					q_d, q_dot_d, _ = self._request_PUMA(ee_position, ee_orientation)

					self.send_request(q_d, q_dot_d)

					if self.start:
						print('End reached')
						exit()

				# Sleep
				rate.sleep()
			except rospy.ROSInterruptException:
				pass


if __name__ == '__main__':
	control_frequency = 500

	# Init ROS
	rospy.init_node('PUMA')
	PUMA_control = PUMAControl()
	PUMA_control.run_dema(control_frequency)
