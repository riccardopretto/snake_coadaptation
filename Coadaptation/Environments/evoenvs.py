from gym import spaces
import numpy as np
#from .pybullet_evo.gym_locomotion_envs import HalfCheetahBulletEnv, SnakeEnvMujoco
import copy
from utils import BestEpisodesVideoRecorder
import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/home/riccardo/Desktop/kevin_exp/Coadaptation/Environments')
import snake_v14
import gymnasium as gym

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
                                       
                                                  
class SnakeEnv(object):
    def __init__(self, config = {'env' : {'render' : True, 'record_video': False}}):
        self._config = config
        self._render = self._config['env']['render']
        self._record_video = self._config['env']['record_video']
        self._current_design = [.1] * 8    #---------------------------------number of segments
        self._config_numpy = np.array(self._current_design)
        self.design_params_bounds = [(0.1, 1.5)] * 8  #------------------------lenght size bound
        #self._env = HalfCheetahBulletEnv(render=self._render, design=self._current_design)  #---------------still to figure it out
        # self._env = SnakeEnvMujoco(render=self._render, design=self._current_design)
        self._env = gym.make(f"Snake-v14", render_mode = "rgb_array", width = 1080, height = 720)
        self._env.reset()

        self.init_sim_params = [
            [0.50] * 8,
            [1.43, 1.33, 1.16, 1.55, .72, 1.14, 1.42, 1.25],
	    [.20, .39, 1.28, .22, 1.45, .59, 1.02, .49],
	    [1.49, .84, 1.15, .64, .19, .55, .29, .23],
	    [.53, .79, 1.08, .85, .71, 1.14, .89, 1.38],
        ]
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0] + 21], dtype=np.float32)#env.observation_space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._env.observation_space.shape, dtype=np.float32)#env.observation_space
        self.action_space = self._env.action_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)#---------------------------------------it should be just self.action_space = self._env.action_space
        
        self._initial_state = self._env.reset()

        if self._record_video:
            self._video_recorder = BestEpisodesVideoRecorder(path=config['data_folder_experiment'], max_videos=5)

        # Which dimensions in the state vector are design parameters?
        self._design_dims = list(range(self.observation_space.shape[0] - len(self._current_design), self.observation_space.shape[0]))
        assert len(self._design_dims) == 8      #-----------------------------8 segments

    def render(self):
        pass

    def step(self, a):
        info = {}
        state, reward, done, _, info = self._env.step(a)
        state = np.append(state, self._config_numpy)
        info['orig_action_cost'] = 0.1 * np.mean(np.square(a))
        info['orig_reward'] = reward

        if self._record_video:
            self._video_recorder.step(env=self._env, state=state, reward=reward, done=done)

        return state, reward, False, info


    def reset(self):
        state = self._env.reset() #
        self._initial_state = state
        state = np.append(state[0], np.array(self._config_numpy))

        # if self._record_video:
        #     self._video_recorder.reset(env=self._env, state=state, reward=0, done=False)
        return state

    def set_new_design(self, vec):      #do i need to do something if I already get the new design through a xml file in random design?
        vec = np.array(vec) * 100.0
        vec= vec.astype(int)
        self._env.reset_design(vec) #
        #restart gym with the new design
        self._env = gym.make(f"Snake-v14", render_mode = "rgb_array", width = 1080, height = 720)
        print("new enviroment started")

        self._current_design = vec.astype(float) / 100.0
        self._config_numpy = np.array(vec.astype(float) / 100.0)          #so here i have to restrt the gym env with the new design

        if self._record_video:
            self._video_recorder.increase_folder_counter()

    def get_random_design(self):                 #maybe call the function that generate the xml file?
        optimized_params = np.random.uniform(low=0.1, high=150, size=8) / 100.0   #-----------------from 0 to 149 for 8 segments
        return optimized_params

    def get_current_design(self):
        return copy.copy(self._current_design)

    def get_design_dimensions(self):
        return copy.copy(self._design_dims)
    




# class HalfCheetahEnv(object):
#     def __init__(self, config = {'env' : {'render' : True, 'record_video': False}}):

#         print("evoenvs not selectiong snake--------------------------------------------------------------------")
#         self._config = config
#         self._render = self._config['env']['render']
#         self._record_video = self._config['env']['record_video']
#         self._current_design = [1.0] * 6
#         self._config_numpy = np.array(self._current_design)
#         self.design_params_bounds = [(0.8, 2.0)] * 6
#         self._env = HalfCheetahBulletEnv(render=self._render, design=self._current_design)
#         self.init_sim_params = [
#             [1.0] * 6,
#             [1.41, 0.96, 1.97, 1.73, 1.97, 1.17],
#             [1.52, 1.07, 1.11, 1.97, 1.51, 0.99],
#             [1.08, 1.18, 1.39, 1.76 , 1.85, 0.92],
#             [0.85, 1.54, 0.97, 1.38, 1.10, 1.49],
#         ]
#         self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._env.observation_space.shape[0] + 6], dtype=np.float32)#env.observation_space
#         self.action_space = self._env.action_space
#         self._initial_state = self._env.reset()

#         if self._record_video:
#             self._video_recorder = BestEpisodesVideoRecorder(path=config['data_folder_experiment'], max_videos=5)

#         # Which dimensions in the state vector are design parameters?
#         self._design_dims = list(range(self.observation_space.shape[0] - len(self._current_design), self.observation_space.shape[0]))
#         assert len(self._design_dims) == 6

#     def render(self):
#         pass

#     def step(self, a):
#         info = {}
#         state, reward, done, _ = self._env.step(a)
#         state = np.append(state, self._config_numpy)
#         info['orig_action_cost'] = 0.1 * np.mean(np.square(a))
#         info['orig_reward'] = reward

#         if self._record_video:
#             self._video_recorder.step(env=self._env, state=state, reward=reward, done=done)

#         return state, reward, False, info


#     def reset(self):
#         state = self._env.reset()
#         self._initial_state = state
#         state = np.append(state, self._config_numpy)

#         if self._record_video:
#             self._video_recorder.reset(env=self._env, state=state, reward=0, done=False)

#         return state

#     def set_new_design(self, vec):
#         self._env.reset_design(vec)
#         self._current_design = vec
#         self._config_numpy = np.array(vec)

#         if self._record_video:
#             self._video_recorder.increase_folder_counter()

#     def get_random_design(self):
#         optimized_params = np.random.uniform(low=0.8, high=2.0, size=6)
#         return optimized_params

#     def get_current_design(self):
#         return copy.copy(self._current_design)

#     def get_design_dimensions(self):
#         return copy.copy(self._design_dims)
