import random

import numpy as np
from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
from gym.spaces import Discrete
import utilities
import math
import random
from controller import Supervisor
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from config import get_config
class Epuck2Supervisor(CSVSupervisorEnv):
    def __init__(self,all_args=None):
        super().__init__(timestep=all_args.timestep)
        self.args = all_args
        self.num_agents = self.args.num_agents
        self.num_stags = self.args.num_stags
        self.num_obs_targets = self.args.num_obs_targets
        self.num_obs_agents = self.args.num_obs_agents
        self.radius_view = self.args.radius_view
        self.catch_distance = self.args.catch_distance
        self.episode_length = self.args.episode_length
        self.miscapture_punishment = self.args.miscapture_punishment
        self.reward_time = self.args.reward_time
        self.reward_stag = self.args.reward_stag
        self.collision_distance = self.args.collision_distance
        self.collision_reward = self.args.collision_reward
        self.capture_action_conditions = self.args.capture_action_conditions[0]
        # self.num_agents = 3
        # self.num_stags = 3
        self.num_observations = 11 + 4*(self.num_obs_targets+self.num_obs_agents)  # The agent has 4 inputs
        self.num_actions = 10  # The agent can perform 2 actions
        self.num_states = 3*self.num_agents+2*self.num_stags
        self.num_envs = 1
        self.timestep = self.args.timestep
        self.interval = self.args.interval
        self.add_agents()
        self.robot = []
        for i in range(1,self.num_agents+1):
            self.robot.append(self.getFromDef(f"epuck{i}"))
        self.target = []
        for i in range(1,self.num_stags+1):
            self.target.append(self.getFromDef(f"target{i}"))
        self.action_space = []
        self.observation_space = []
        self.state_space = []
        for i in range(self.num_agents):
            self.action_space.append(Discrete(self.num_actions))
            self.observation_space.append([self.num_observations])
            self.state_space.append([self.num_states])

        self.ps_sensor_mm = {'min': 0, 'max': 1023}
        self.angle_mm = {'min': -np.pi, 'max': np.pi}
        self.dis_mm = {'min': 0, 'max': 2.23}
        self.pos_x_mm = {'min': -1, 'max': 1}
        self.pos_y_mm = {'min': -0.5, 'max': 0.5}
        self.steps = 0
        self.m_cMuValues = [(0, 7.646890), (2, 7.596525), (5, 7.249550),
                            (10, 7.084636), (15, 6.984497), (30, 6.917447),
                            (45, 6.823188), (60, 6.828551), (80, 6.828551)]

        self.m_cSigmaValues = [(0, 0.3570609), (2, 0.3192310), (5, 0.1926492),
                               (10, 0.1529397), (15, 0.1092330), (30, 0.1216533),
                               (45, 0.1531546), (60, 0.1418425), (80, 0.1418425)]

        self.m_fExpA = 9.06422181283387
        self.m_fExpB = -0.00565074879677167
        self.radius_epuck = 0.035

        self.cleanup()
        self.emitter_arb = []
        for i in range(1,self.num_agents+1):
            self.emitter_arb.append(self.initialize_emitter(i))

    def random_pos(self):
        per_row = int(math.ceil(math.sqrt(self.num_agents + self.num_stags)))
        x_spacing = 0.3
        z_spacing = 0.3
        xmin = -0.5 * x_spacing * per_row
        zmin = -0.5 * z_spacing * (per_row - 2)
        count = 0
        # import pdb;pdb.set_trace()
        pos = []
        for j in range(per_row):
            agent_up = zmin + j * z_spacing
            for k in range(per_row):
                if count >= self.num_agents + self.num_stags:
                    break
                agentx = xmin + k * x_spacing
                pos.append([agentx, agent_up, 0.0])

                count += 1
        rand_idx = np.random.permutation(self.num_agents + self.num_stags)
        agent_pos = np.array(pos)[rand_idx[:self.num_agents]]
        target_pos = np.array(pos)[rand_idx[self.num_agents:]]
        return agent_pos, target_pos

    def add_agents(self):

        agent_pos, target_pos = self.random_pos()
        for i in range(1, self.num_agents + 1):
            self.importRobot(i, agent_pos[i - 1, 0], agent_pos[i - 1, 1], 0.05, random.uniform(-np.pi, np.pi), 3 + i)

        for i in range(1, self.num_stags + 1):
            self.importTarget(i, target_pos[i - 1, 0], target_pos[i - 1, 1], 0.05, random.uniform(-np.pi, np.pi))

    def importRobot(self, id, x, y, z, ro, channel_idx):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        DEF epuck%d E-puck{
                            translation %f %f %f
                            rotation 0 0 1 %f
                            name "e-puck%d"
                            controller "epuck_controller"
                            supervisor FALSE
                            version "2"
                            emitter_channel 1
                            receiver_channel 2
                            receiver_rab_channel %d           
                        }
                        """ % (id, x, y, z, ro, id, channel_idx)
        chFd.importMFNodeFromString(-1, line_String)

    def importTarget(self, id, x, y, z, ro):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        DEF target%d E-puck{
                            translation %f %f %f
                            rotation 0 0 1 %f
                            name "target%d"
                            controller "target_controller"
                            supervisor FALSE
                            version "1"
                            emitter_channel 3
                            receiver_channel 2
                        }
                        """ % (id, x, y, z, ro, id)
        chFd.importMFNodeFromString(-1, line_String)
        
    def initialize_emitter(self, id):
        emitter = self.getDevice('emitter0'+str(id))
        return emitter

    def Interpolate(self, Range, Values):
        Points, Values = zip(*Values)
        f = interpolate.interp1d(Points, Values, fill_value="extrapolate")
        return f(Range*100)
    def handle_emitter_arb(self):
        epuck_pos = np.zeros((self.num_agents, 1, 3), dtype=np.float32)
        rotation_vector = np.zeros((self.num_agents, 4), dtype=np.float32)
        for i in range(self.num_agents):
            epuck_pos[i] = self.robot[i].getField('translation').getSFVec3f()
            rotation_vector[i] = self.robot[i].getField('rotation').getSFRotation()
        CVectorRtoS = epuck_pos - epuck_pos.transpose(1,0,2)
        r = np.random.normal(loc=0,scale=0.01,size=(self.num_agents,self.num_agents))
        inclination = np.random.uniform(low=0,high=np.pi,size=(self.num_agents,self.num_agents))
        azimuth = np.random.uniform(low=0, high=np.pi*2, size=(self.num_agents, self.num_agents))
        CVectorRtoS[..., 0] += r * np.sin(inclination) * np.cos(azimuth)
        CVectorRtoS[..., 1] += r * np.sin(inclination) * np.sin(azimuth)
        CVectorRtoS[..., 2] += r * np.cos(inclination)
        CVectorStoR = -CVectorRtoS
        rotation = R.from_rotvec(rotation_vector[...,:3]*rotation_vector[...,-1][:,np.newaxis])
        rotation_matrix_inv = rotation.inv().as_matrix().astype(np.float32)
        CVectorStoR_rot = np.matmul(np.tile(rotation_matrix_inv,(self.num_agents,1,1,1)),CVectorStoR[:,:,:,np.newaxis])
        dis = np.clip(np.linalg.norm(CVectorStoR_rot,axis=2)[...,0]-self.radius_epuck*2, a_min=0.0,a_max=None)
        angle = np.arctan2(CVectorStoR_rot[:,:,1],CVectorStoR_rot[:,:,0])[...,0]
        # fMu = torch.from_numpy(self.Interpolate(dis, self.m_cMuValues))
        # fSigma = torch.from_numpy(self.Interpolate(dis, self.m_cSigmaValues))
        # fPower = torch.exp(torch.distributions.Normal(fMu, fSigma).sample())
        # real_dis = np.clip(np.exp(self.m_fExpA + self.m_fExpB * fPower.numpy()), a_min=0.0,a_max=None)
        fMu = self.Interpolate(dis, self.m_cMuValues)
        fSigma = self.Interpolate(dis, self.m_cSigmaValues)
        fPower = np.zeros((self.num_agents,self.num_agents),dtype=np.float32)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                fPower[i][j] = np.random.lognormal(fMu[i][j], fSigma[i][j])
        real_dis = np.clip(np.exp(self.m_fExpA + self.m_fExpB * fPower), a_min=0.0, a_max=None) / 100
        #np.set_printoptions(suppress=True)
        #import pdb;pdb.set_trace()

        message = np.stack((real_dis,angle),axis=-1)
        receive_sign = np.eye(self.num_agents,dtype=np.bool_) | (real_dis > 0.5)
        for i in range(0, self.num_agents):
            real_message = np.where(receive_sign[:,i,np.newaxis],np.zeros_like(message[:,i]),message[:,i]).reshape(-1)
            string_message = ",".join(map(str, real_message.tolist()))
            self.emitter_arb[i].send(string_message.encode("utf-8"))


    def step(self,action):
        for _ in range(self.interval):
            self.handle_emitter_arb()
            if super(Supervisor, self).step(self.timestep//self.interval) == -1:
                exit()

        self.handle_emitter(action)



        return (
            self.get_observations(),
            self.get_state(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
            self.get_availactions()
        )
    # def get_default_observation(self):
    #     return [0 for i in range(OBSERVATION_SPACE)]
    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.states_buf = np.zeros((self.num_envs, self.num_states), dtype=np.float32)
        self.alive_target_buf = np.ones((self.num_envs, self.num_stags), dtype=np.bool_)
        self.avail_actions_buf = np.ones((self.num_envs, self.num_agents, self.num_actions),dtype=np.int32)
        self.extras = {}

    def get_default_observation(self):
        agent_pos, target_pos = self.random_pos()
        #import pdb;pdb.set_trace()
        for i in range(self.num_agents):
            epuck_default_pos = self.robot[i].getField('translation').getSFVec3f()
            robot_default_rotation = self.robot[i].getField('rotation').getSFRotation()
            epuck_default_pos[:2] = agent_pos[i][:2]
            robot_default_rotation[3] = random.uniform(-np.pi, np.pi)
            self.robot[i].getField('translation').setSFVec3f(epuck_default_pos)
            self.robot[i].getField('rotation').setSFRotation(robot_default_rotation)

        for i in range(self.num_stags):
            target_default_pos = self.target[i].getField('translation').getSFVec3f()
            target_default_rotation = self.target[i].getField('rotation').getSFRotation()
            target_default_pos[:2] = target_pos[i][:2]
            target_default_rotation[3] = random.uniform(-np.pi, np.pi)
            self.target[i].getField('translation').setSFVec3f(target_default_pos)
            self.target[i].getField('rotation').setSFRotation(target_default_rotation)
        return None

    def get_state(self):
        return self.states_buf

    def get_availactions(self):
        return self.avail_actions_buf

    def handle_receiver(self):
        message = np.zeros((self.num_agents,8))
        #print("supervisor",self.receiver.getQueueLength())
        # while True:
        #     if (self.receiver.getQueueLength() == self.num_agents) | (self.receiver.getQueueLength() == 0):
        #         break
        for i in range(self.num_agents):
            if self.receiver.getQueueLength() > 0:
                try:
                    string_message = self.receiver.getString().split(',')
                except AttributeError:
                    string_message = self.receiver.getData().decode("utf-8")
                self.receiver.nextPacket()
                idx = int(string_message[0][1])-1
                message[idx] = np.array(string_message[1:]).astype(np.float32)
        return message

    def handle_emitter(self, action):
        message = (",".join(map(str, action))).encode("utf-8")
        #for i in range(self.num_agents+self.num_stags):
        self.emitter.send(message)

    def get_observations(self):
        message = self.handle_receiver()
        self.message = message
        observation = np.zeros((self.num_envs, self.num_agents, self.num_observations),dtype=np.float32)
        self.avail_actions_buf[:] = 1

        if message is not None:
            epuck_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
            epuck_angle = np.zeros((self.num_agents, 1), dtype=np.float32)
            for i in range(self.num_agents):
                epuck_pos[i] = self.robot[i].getField('translation').getSFVec3f()[:-1]
                robot_rotation = self.robot[i].getField('rotation').getSFRotation()
                epuck_angle[i] = robot_rotation[3] if robot_rotation[2] > 0 else -robot_rotation[3]
            target_pos = np.zeros((self.num_stags, 2), dtype=np.float32)
            for i in range(self.num_stags):
                target_pos[i] = self.target[i].getField('translation').getSFVec3f()[:-1]
            dis_to_epuck = utilities.get_distance_from_target(epuck_pos,epuck_pos)
            angle_to_epuck = utilities.get_angle_from_target(epuck_pos,epuck_pos,epuck_angle)
            dis_to_target = utilities.get_distance_from_target(epuck_pos, target_pos)
            self.agent_dis = dis_to_epuck
            self.distance = dis_to_target
            self.agent_pos = epuck_pos
            angle_to_target = utilities.get_angle_from_target(epuck_pos, target_pos, epuck_angle)

            epuck_pos[...,0] = utilities.normalize_to_range(epuck_pos[...,0],self.pos_x_mm['min'],self.pos_x_mm['max'],0,1)
            epuck_pos[..., 1] = utilities.normalize_to_range(epuck_pos[..., 1], self.pos_y_mm['min'],self.pos_y_mm['max'], 0, 1)
            epuck_angle = utilities.normalize_to_range(epuck_angle, self.angle_mm['min'], self.angle_mm['max'], 0, 1)
            message = utilities.normalize_to_range(message, self.ps_sensor_mm['min'], self.ps_sensor_mm['max'], 0, 1)
            target_pos[..., 0] = utilities.normalize_to_range(target_pos[..., 0], self.pos_x_mm['min'],self.pos_x_mm['max'], 0, 1)
            target_pos[..., 1] = utilities.normalize_to_range(target_pos[..., 1], self.pos_y_mm['min'],self.pos_y_mm['max'], 0, 1)
            angle_to_epuck = utilities.normalize_to_range(angle_to_epuck, self.angle_mm['min'],self.angle_mm['max'], 0, 1)
            angle_to_target = utilities.normalize_to_range(angle_to_target, self.angle_mm['min'], self.angle_mm['max'], 0,1)

            observation[...,:2] = epuck_pos
            observation[..., 2] = epuck_angle[...,0]
            observation[..., 3:11] = message
            for i in range(self.num_agents):
                other_agent_pos = np.concatenate((epuck_pos[:i],epuck_pos[i+1:]),axis=0)
                other_agent_dis = np.concatenate((dis_to_epuck[i][:i],dis_to_epuck[i][i+1:]),axis=-1)
                other_agent_angle = np.concatenate((angle_to_epuck[i][:i], angle_to_epuck[i][i + 1:]), axis=-1)
                other_message = np.concatenate((other_agent_pos,other_agent_angle[:,np.newaxis],utilities.normalize_to_range(other_agent_dis, self.dis_mm['min'],self.dis_mm['max'], 0, 1)[:,np.newaxis]),axis=-1)
                obs_agent = other_agent_dis < self.radius_view
                near_agent = np.zeros_like(obs_agent)
                near_agent_indices = np.argsort(other_agent_dis, axis=0)[:self.num_obs_agents]
                near_agent[near_agent_indices] = 1
                is_obs_agent = obs_agent & near_agent
                num_observable_agent = is_obs_agent.sum(-1)

                target_dis = dis_to_target[i]
                target_angle = angle_to_target[i]
                target_message = np.concatenate((target_pos, target_angle[:, np.newaxis],utilities.normalize_to_range(target_dis, self.dis_mm['min'],self.dis_mm['max'], 0, 1)[:, np.newaxis]),axis=-1)
                obs_target = (target_dis < self.radius_view) & self.alive_target_buf[0]
                near_target = np.zeros_like(obs_target)
                near_target_indices = np.argsort(target_dis, axis=0)[:self.num_obs_targets]
                near_target[near_target_indices] = 1
                is_obs_target = obs_target & near_target
                num_observable_target = is_obs_target.sum(-1)
                #self.avail_actions_buf[~(np.where(self.alive_target_buf, target_dis, 10) < self.catch_distance).any(-1),i,-1] = 0
                if num_observable_agent > 0:
                    observation[:, i, 11:11+num_observable_agent*4] = other_message[is_obs_agent].reshape(1,-1)
                if num_observable_target > 0:
                    observation[:, i, 11+self.num_obs_agents*4:11+4*(self.num_obs_agents+num_observable_target)] = target_message[is_obs_target].reshape(1,-1)

            agent_all_message = np.concatenate((epuck_pos, epuck_angle),axis=-1).reshape(-1)
            target_all_message = target_pos.reshape(-1)
            state = np.concatenate((agent_all_message,target_all_message),axis=-1)
            self.states_buf[:] = state[np.newaxis,:]

        return observation

    def get_reward(self, action):
        '''if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0'''
        action = np.array(action[:self.num_agents])
        rew_all = np.zeros((self.num_agents,1))
        reward = 0
        #self.agent_dis
        reward += ((self.agent_dis < self.collision_distance*2).sum()-self.num_agents)*self.collision_reward
        reward += (self.distance < self.collision_distance*2).sum() * self.collision_reward
        reward += ((self.agent_pos[:,0] < self.collision_distance) | (self.agent_pos[:,0] > 1-self.collision_distance) | (self.agent_pos[:,1] < self.collision_distance) | (self.agent_pos[:,1] > 1-self.collision_distance)).sum() * self.collision_reward
        for i in range(self.num_stags):
            #capture_agents = ((action == self.num_actions-1) & (self.distance[:,i] < self.catch_distance)).sum(-1)
            capture_agents = (self.distance[:, i] < self.catch_distance).sum(-1)
            reward += np.where(self.alive_target_buf[:,i] & (capture_agents > 0) & (capture_agents < self.capture_action_conditions),self.miscapture_punishment,0)
            reward += np.where(self.alive_target_buf[:, i] & (capture_agents >= self.capture_action_conditions),self.reward_stag, 0)
            self.alive_target_buf[:, i] &= capture_agents < self.capture_action_conditions
        reward += self.reward_time
        rew_all[:] = reward


        return rew_all

    def is_done(self):
        self.steps += 1
        dies = np.ones((1,self.num_agents,1), dtype=np.bool_)
        alives = np.zeros((1,self.num_agents,1), dtype=np.bool_)
        if (~self.alive_target_buf.any(-1)) | (self.steps >= self.episode_length + 2):
            return dies
        else:
            return alives

    def reset(self):
        self.steps = 0
        self.alive_target_buf = np.ones((self.num_envs, self.num_stags), dtype=np.bool_)

        #self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep//self.interval)

        self.get_default_observation()
        for _ in range(self.interval-1):
            super(Supervisor, self).step(self.timestep//self.interval)
        self.handle_receiver()
        return None

    def get_info(self):
        '''right_obstacle = ((self.message[:,0] > 80.0) | (self.message[:,1] > 80.0) | (self.message[:,2] > 80.0))
        left_obstacle = ((self.ps_sensor[5].getValue() > 80.0) | (self.ps_sensor[6].getValue() > 80.0) | (
                    self.ps_sensor[7].getValue() > 80.0))'''
        return np.zeros((self.num_envs, self.num_stags), dtype=np.int_)[0]




