from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
from .robot_locomotors import  HalfCheetah
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from gym import spaces

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):
  def __init__(self, robot, render=False):
    # print("WalkerBase::__init__ start")
    MJCFBaseBulletEnv.__init__(self, robot, render)

    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId=-1
    self._projectM = None
    self._param_init_camera_width = 320
    self._param_init_camera_height = 200
    self._param_camera_distance = 2.0


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
    return self.stadium_scene

  def reset(self):
    if (self.stateId>=0):
      # print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r = MJCFBaseBulletEnv.reset(self)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
      self.stadium_scene.ground_plane_mjcf)
    # self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
    #              self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
    if (self.stateId<0):
      self.stateId=self._p.saveState()
      #print("saving state self.stateId:",self.stateId)
    self._p.setGravity(0,0,-.5)
    for _ in range(200):
      self.robot.reset_position()
      self.scene.global_step()
    self.robot.reset_position_final()
    self._p.setGravity(0,0,-9.81)
    r = self.robot.calc_state()
    self.robot._initial_z = r[-1]
    # for _ in range(20):
    #   self.scene.global_step()
    #   self.robot.reset_position_final()
    #   time.sleep(0.1)
    # self.robot.reset_position()
    # self.scene.global_step()
    self.robot.initial_z = None
    r = self.robot.calc_state()

    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost   = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost  = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost  = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i,f in enumerate(self.robot.feet): # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
                          #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0


    electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode=0
    if(debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
      self._alive,
      progress,
      electricity_cost,
      joints_at_limit_cost,
      feet_collision_cost
      ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  # def camera_adjust(self):
  #   x, y, z = self.robot.body_xyz
  #   self.camera._p = self._p
  #   self.camera_x = 0.98*self.camera_x + (1-0.98)*x
  #   self.camera.move_and_look_at(self.camera_x, y-1.0, 1.4, x, y, 1.0)
  def camera_adjust(self):
        if self._p is None :
            return
        self.camera._p = self._p
        x, y, z = self.robot.body_xyz
        if self.camera_x is not None:
            self.camera_x = x # 0.98*self.camera_x + (1-0.98)*x
        else:
            self.camera_x = x
        # self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)
        lookat = [self.camera_x, y, z]
        distance = self._param_camera_distance
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)

  def render_camera_image(self, pixelWidthHeight = None):
    if pixelWidthHeight is not None or self._projectM is None:
        if self._projectM is None:
            self._pixelWidth = self._param_init_camera_width
            self._pixelHeight = self._param_init_camera_height
        else:
            self._pixelWidth = pixelWidthHeight[0]
            self._pixelHeight = pixelWidthHeight[1]
        nearPlane = 0.01
        farPlane = 10
        aspect = self._pixelWidth / self._pixelHeight
        fov = 60
        self._projectM = self._p.computeProjectionMatrixFOV(fov, aspect,
            nearPlane, farPlane)

    x, y, z = self.robot.robot_body.pose().xyz()
    lookat = [x, y, 0.5]
    distance = 2.0
    yaw = -20
    viewM = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=lookat,
        distance=distance,
        yaw=10.,
        pitch=yaw,
        roll=0.0,
        upAxisIndex=2)

    # img_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=1,lightDirection=[1,1,1],renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    img_arr = pybullet.getCameraImage(self._pixelWidth, self._pixelHeight, viewM, self._projectM, shadow=False, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL, flags=pybullet.ER_NO_SEGMENTATION_MASK)

    w=img_arr[0] #width of the image, in pixels
    h=img_arr[1] #height of the image, in pixels
    rgb=img_arr[2] #color data RGB

    image = np.reshape(rgb, (h, w, 4)) #Red, Green, Blue, Alpha
    image = image * (1./255.)
    image = image[:,:,0:3]
    return image

class HalfCheetahBulletEnv(WalkerBaseBulletEnv):
  def __init__(self, render=False, design = None):
    self.robot = HalfCheetah(design)
    WalkerBaseBulletEnv.__init__(self, self.robot, render)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=[17], dtype=np.float32)

  def _isDone(self):
    return False

  def disconnect(self):
      self._p.disconnect()

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = 0
    reward = max(state[-5]/10.0, 0.0)

    return state, reward, bool(done), {}

  def reset_design(self, design):
      self.stateId = -1
      self.scene = None
      self.robot.reset_design(self._p, design)


#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#            ______   __    __   ______   __    __  ________ 
#           /      \ |  \  |  \ /      \ |  \  /  \|        \
#           |  $$$$$$\| $$\ | $$|  $$$$$$\| $$ /  $$| $$$$$$$$
#           | $$___\$$| $$$\| $$| $$__| $$| $$/  $$ | $$__    
#           \$$    \ | $$$$\ $$| $$    $$| $$  $$  | $$  \   
#            _\$$$$$$\| $$\$$ $$| $$$$$$$$| $$$$$\  | $$$$$   
#           |  \__| $$| $$ \$$$$| $$  | $$| $$ \$$\ | $$_____ 
#            \$$    $$| $$  \$$$| $$  | $$| $$  \$$\| $$     \
#           \$$$$$$  \$$   \$$ \$$   \$$ \$$   \$$ \$$$$$$$$
#
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------   
import numpy as np
import mujoco as mujoco
import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from os import path
from os import listdir
import sys
import pandas as pd
import random


DEFAULT_CAMERA_CONFIG = {}

gymnasium.envs.register(
    id="Snake-v14",
    entry_point=f"{__name__}:SnakeEnv",
    max_episode_steps=750,  #30 sec
    reward_threshold=1000,
)

xml_filename = "snake_v14.xml"
directory, filename = path.split(__file__)
xml_file_path = path.join(directory, xml_filename)


class SnakeEnvMujoco(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_file=xml_file_path,
        forward_reward_weight=1.0,
        #ctrl_cost_weight=1e-1,
        reset_noise_scale=0.01,
        render=False,   #---------------------added
        design = None,  #--------------------added
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            #ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        self.max_vel = 0.3      #m/s
        self.max_ctrl_cost = 1.4
        #---------------------------------------------------------------------------- WEIGHT
        self._forward_reward_weight = forward_reward_weight #not used
        self._ctrl_cost_weight = 0.02
        self.goal_reward_weight = 0.98
        self.vel_tracking_weight = 0.05
        #self._ctrl_cost_weight = ctrl_cost_weight
        self.x_vel_tracking_weight = 0.5
        self.y_vel_tracking_weight = 0.3
        #---------------------------------------------------------------------------- 
        
        self.goal_used_for_training = [0,0]

        self._prev_action = 0
        self._prev_x_pos = 0

        self.goal_pos = [0,0]
        self.goal_bound = 0.01        #goal boundaries
        self.goal_distance = 0
        self.goal_reached = False
        self.initial_goal_distance = 0

        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64     #it stands for: [x,y,z,theta0,t1,t2,t3,t4,t5,t6,t7,d_x,d_y,d_z,d_t1, d_t2, ... , x_goal-current_x_pos, y_goal-current_y_pos] positions, velocities and goal distance
        )

        MujocoEnv.__init__(
            self, xml_file, 20, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.square(np.sum(np.square(action)))/self.max_ctrl_cost        #normalized
        return control_cost

    def step(self, action):
        action = action * 1.4         #---------------------------------action * gain
        xy_position_before = self.data.qpos[0:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        #get the new step observation
        observation = self._get_obs()

        #velocity
        tot_velocity = np.sum(np.square(xy_velocity))
        vel_tracking_reward = abs(self.max_vel - tot_velocity)/self.max_vel*self.vel_tracking_weight       #normalized

        x_vel_tracking_reward = abs(self.max_vel - x_velocity)/self.max_vel*self.x_vel_tracking_weight       #normalized
        y_vel_tracking_reward = abs(0 - y_velocity)/self.max_vel*self.y_vel_tracking_weight       #normalized

    

        #goal distance
        x_goal_dist = abs(self.goal_pos[0]-observation[0])   #should go before get obs
        y_goal_dist = abs(self.goal_pos[1]-observation[1])
        self.goal_distance = np.sqrt(np.square(x_goal_dist) + np.square(y_goal_dist))

        goal_reward = self.goal_reward(goal_distance = self.goal_distance)

        if self.goal_distance < self.goal_bound:
            self.goal_reached = True
        
        #-------REWARD-------
        if not self.goal_reached:
            reward = goal_reward - ctrl_cost #- vel_tracking_reward
        else:
            reward = 100
            print(f"---------------------------------------------------------------------GOAL REACHED in {self.goal_pos}------------------------------------------------------------------------")

        self._prev_x_pos = observation[0]
        self._prev_action = action

        info = {
            "goal_reward": goal_reward,
            "reward_ctrl": -ctrl_cost,
            "velocity_tracking_reward": vel_tracking_reward,
            "total_reward": reward,
            "distance_from_goal": self.goal_distance,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "tot_velocity": tot_velocity,
        }


        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        position = np.delete(position,2)    #delete z obs
        velocity = self.data.qvel.flat.copy()
        velocity = np.delete(velocity,2)    #delete z_dot obs
        goal_dist_state = [position[0]-self.goal_pos[0], position[1]-self.goal_pos[1]]   #add 2 state
        observation = np.concatenate([position, velocity, goal_dist_state]).ravel()

        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        #GOAL
        self.goal_pos = self.set_rndm_goal(min_dist = 1.50, max_dist = 2.10)  #sphere goal xy position,
        #test
        self.goal_reached = False
        self.goal_pos = [2,2]           #---------------------------fixed goal
        # self.goal_pos = [1.66,1.56]

        self.goal_used_for_training = np.vstack((self.goal_used_for_training,self.goal_pos))
        goal_df = pd.DataFrame(self.goal_used_for_training)
        goal_df.to_csv("goal_used.csv", sep=',', header=False, float_format='%.2f', index=False)
    
        #print("goal set here: ", self.goal_pos)

        initial_goal_distance_x = self.goal_pos[0] - qpos[0]
        initial_goal_distance_y = self.goal_pos[1] - qpos[1]
        self.initial_goal_distance = np.sqrt(np.square(initial_goal_distance_x) + np.square(initial_goal_distance_y))


        #FRICTION
        #place her friction randomness
        # f1_rndm = round(random.uniform(0.10, 1.00), 2)
        # f2_rndm = round(random.uniform(0.10, 1.00), 2) 
        # f3_rndm = round(random.uniform(0.10, 0.50), 2)
        f1_rndm = 0.3
        f2_rndm = 0.7
        f3_rndm = 0.05
        friction_values = [f1_rndm, f2_rndm, f3_rndm, 0.0001, 0.0001]
        #set friction
        for idx in range(8):
            self.set_pair_friction(geom_name1=f"caps{idx+1}", geom_name2="floor", new_pair_friction=friction_values)


        self.set_state(qpos, qvel)

        observation = self._get_obs()
        #print("Env resetted")
        return observation
    
    def reset_design(self, design):   #-----------------------------------------------------------reset design
      self.stateId = -1
      self.scene = None
      self.robot.reset_design(self._p, design)
    
    def set_rndm_goal(self, min_dist, max_dist):
        """Set a rndm position goal between min e max distance.

        :min_dist = minimum distance in x and y, ex min_value = 1 then x and y are more than 1
        :max_dist = maximum distance in x and y, ex max_value = 2 then x and y are less than 2

        """
        #set random goal sphere position
        x_goal_rndm = 0.00
        y_goal_rndm = 0.00
        #while goal is between x and y [-min_dist, min_dist]
        while x_goal_rndm < min_dist and x_goal_rndm > -min_dist:
            x_goal_rndm = round(random.uniform(-max_dist, max_dist), 2)
        while y_goal_rndm < min_dist and y_goal_rndm > -min_dist:
            y_goal_rndm = round(random.uniform(-max_dist, max_dist), 2)
        goal_pos_rndm = [abs(x_goal_rndm),y_goal_rndm]                  #only x positive

        return goal_pos_rndm
    
    def get_goal_pos(self):
        goal_pos = self.goal_pos
        return goal_pos

    def goal_reward(self, goal_distance, w=1, v=1, alpha=1e-4):
        goal_reward = -w*goal_distance**2 -v*np.log(goal_distance**2 + alpha)
        min_goal_reward = -w*self.initial_goal_distance**2 -v*np.log(self.initial_goal_distance**2 + alpha)
        max_goal_reward = -w*0**2 -v*np.log(0**2 + alpha)
        normalized_goal_reward = (goal_reward - min_goal_reward)/(abs(max_goal_reward)+abs(min_goal_reward))
        return normalized_goal_reward*self.goal_reward_weight

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
        

    #----------------------------VARYING FRICTION---------------------------------------


    def get_id(self, name, obj_type):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    

    def get_contact_pair(self, geom_name1, geom_name2):
        """Gets the pair ID of two objects that are in contact. 
        If the objects are not in contact it wil raise a key error.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The contact pair ID
        :rtype: int
        """
        # Find the proper geom ids
        geom_id1 = self.get_id(geom_name1, 'geom')
        geom_id2 = self.get_id(geom_name2, 'geom')

        # Find the right pair id
        pair_geom1 = self.model.pair_geom1
        pair_geom2 = self.model.pair_geom2
        pair_id = None
        for i, (g1, g2) in enumerate(zip(pair_geom1, pair_geom2)):
            if g1 == geom_id1 and g2 == geom_id2 \
                or g2 == geom_id1 and g1 == geom_id2:
                pair_id = i
                #print("pair_id n=", pair_id)  #i should get 9 values
                break 
        if pair_id is None: 
            raise KeyError("No contact between %s and %s defined."
                            % (geom_name1, geom_name2))
        return pair_id

    def get_pair_solref(self, geom_name1, geom_name2):
        """Gets the solver parameters between two objects in the enviroment.
        The solref represents the stiffness and damping between two object
        (mass-spring-damper system). See Mujoco documentation.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The solref between two objects
        :rtype: ???
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_solref = self.model.pair_solref[pair_id]
        return pair_solref

    def get_pair_friction(self, geom_name1, geom_name2):
        """Gets the friction between two objects in the 
        enviroment with geom_names specified.

        :param geom_name1: The geom name of the first object in enviroment
        :type geom_name1: str
        :param geom_name2: The geom name of the second object in enviroment
        :type geom_name2: str
        :return: The friction between two objects
        :rtype: ???
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_friction = self.model.pair_friction[pair_id]
        return pair_friction
    
    #set pair friction
    def set_pair_friction(self, geom_name1, geom_name2, new_pair_friction):
        """    
        :description: Sets the friction between 2 geoms
        :param new_pair_friction: New friction value. Has to be an array of 5 elements
        """
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        self.model.pair_friction[pair_id] = new_pair_friction
        return self.model.pair_friction[pair_id]
    


    @property
    def skin_friction(self):
        return self.get_pair_friction("seg1_body", "floor")     #i have from seg1 to seg9!!!!!

    @skin_friction.setter
    def skin_friction(self, value):
        """    
        :description: Sets the friction between the puck and the sliding surface
        :param value: New friction value. Can either be an array of 2 floats
            (to set the linear friction) or an array of 5 float (to set the
            torsional and rotational friction values as well)
        :raises ValueError: if the dim of ``value`` is other than 2 or 5
        """
        pair_fric = self.get_pair_friction("seg1_body", "floor")  #i have from seg1 to seg9!!!!!
        if value.shape[0] == 2:
            # Only set linear friction
            pair_fric[:2] = value
        elif value.shape[0] == 3:
            # linear friction + torsional
            pair_fric[:3] = value
        elif value.shape[0] == 5:
            # Set all 5 friction components
            pair_fric[:] = value
        else:  
            raise ValueError("Friction should be a vector or 2 or 5 elements.")
