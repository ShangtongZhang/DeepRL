from ale_python_interface import ALEInterface
from gym import spaces
from gym import utils
from gym.envs.atari import AtariEnv
from gym.utils import seeding
import numpy as np
import os

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class UpdatedAtariEnv(AtariEnv):
    def __init__(self, rom_path, obs_type, frameskip=(2,5), repeat_action_probability=0., mode=0, difficulty=0):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, rom_path, obs_type)
        assert obs_type in ('ram', 'image')

        self.rom_path = rom_path
        if not os.path.exists(self.rom_path):
            raise IOError('You asked for ROM %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        # Load new ALE interface, instead of atari-py
        self.ale = ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()
        
        # Set mode and difficulty
        self.ale.setMode(mode)
        self.ale.setDifficulty(difficulty)

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))
            
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        # Load game from ROM instead of game path
        self.ale.loadROM(self.rom_path)
        return [seed1, seed2]
            
    def _get_image(self):
        return self.ale.getScreenRGB()
    
ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
