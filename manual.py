import gym
import numpy as np
from gym.utils import seeding
from gym import spaces
from scipy.misc import imresize
import sys
import copy
from deep_rl import *


class Push(gym.Env):
    def __init__(self, mode="default"):
        if mode == "default":
            self.w = 8
            self.step_limit = 75
            self.n_boxes = 12
            self.n_obstacles = 6
            self.n_goals = 5
            self.box_block = True
            self.walls = False
            self.soft_obstacles = True
        else:
            raise ValueError("Mode must be default. Implement some more variants if you want to experiment!")

        self.use_obst = self.n_obstacles > 0 or self.walls

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (self.w, self.w, 4 + self.use_obst))
        # channels are 0: agent, 1: goals, 2: boxes, 3: obstacles, 4: time remaining

        self.state = None
        self.steps_taken = 0
        self.pos = None
        self.image = None

        self._seed()

    def _seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        state = np.zeros([self.w, self.w, 4 + self.use_obst])
        # initialise time-remaining to 1.
        state[:, :, 3 + self.use_obst].fill(1)

        # fill in walls at borders
        if self.walls:
            state[0, :, 3].fill(1)
            state[self.w - 1, :, 3].fill(1)
            state[:, 0, 3].fill(1)
            state[:, self.w - 1, 3].fill(1)

        # sample random locations for self, goals, and walls.
        locs = np.random.choice((self.w - 2 - (2 * self.walls)) ** 2, 1 + self.n_goals + self.n_boxes + self.n_obstacles, replace=False)

        xs, ys = np.unravel_index(locs, [self.w - 2 - (2 * self.walls), self.w - 2 - (2 * self.walls)])
        xs += 1 + self.walls
        ys += 1 + self.walls

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == 0:
                state[x, y, 0] = 1
                self.pos = np.asarray((x, y))
            elif i <= self.n_goals:
                state[x, y, 1] = 1
            elif i <= self.n_goals + self.n_boxes:
                state[x, y, 2] = 1
            else:
                assert self.n_obstacles > 0
                state[x, y, 3] = 1

        self.state = state
        self.steps_taken = 0
        self.boxes_left = self.n_boxes
        self.edge_boxes = 0

        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        vec = self.directions[action]
        done = False
        pos = self.pos
        old_pos = self.pos
        reward = -0.01

        # where we are about to step to
        step_pos = pos + vec

        if not self.is_in_grid(step_pos):
            reward += -1.0
            done = True
        elif self.use_obst and not self.soft_obstacles and self.state[step_pos[0], step_pos[1], 3] == 1:
            # walking into a hard obstacle, no move
            reward += -0.1
        elif self.state[step_pos[0], step_pos[1], 2] == 1:
            # pushing a box
            # push_pos is where it will be pushed
            push_pos = step_pos + vec
            if not self.is_in_grid(push_pos):
                # push out of grid, destroy
                self.state[step_pos[0], step_pos[1], 2] = 0
                self.boxes_left -= 1
                self.edge_boxes -= 1
                reward += -0.1
                pos = step_pos
            elif self.state[push_pos[0], push_pos[1], 2] == 1:
                # push into another box,
                if self.box_block:
                    # blocking, so no movement
                    reward += -0.1
                else:
                    # not blocking, so destroy
                    self.state[step_pos[0], step_pos[1], 2] = 0
                    self.boxes_left -= 1
                    reward += -0.1
                    pos = step_pos
            elif self.use_obst and self.state[push_pos[0], push_pos[1], 3] == 1:
                # push into obstacle
                if self.soft_obstacles:
                    reward += -0.2
                    self.state[step_pos[0], step_pos[1], 2] = 0
                    self.state[push_pos[0], push_pos[1], 2] = 1
                    pos = step_pos
                else:
                    reward += -0.1
            elif self.state[push_pos[0], push_pos[1], 1] == 1:
                # pushed into goal, get reward
                pos = step_pos
                self.boxes_left -= 1
                self.state[step_pos[0], step_pos[1], 2] = 0
                reward += 1.0
            else:
                # pushed into open space, move box
                self.state[step_pos[0], step_pos[1], 2] = 0
                self.state[push_pos[0], push_pos[1], 2] = 1
                if self.is_on_edge(push_pos):
                    self.edge_boxes += 1
                pos = step_pos
        else:
            #step into open space, just move
            pos = step_pos

        if np.all(pos == step_pos) and self.use_obst and self.soft_obstacles:
            if self.state[step_pos[0], step_pos[1], 3] == 1:
                reward += -0.2

        if self.boxes_left <= self.edge_boxes:
            done = True

        self.steps_taken += 1
        if self.steps_taken >= self.step_limit:
            done = True

        # update player position
        self.state[old_pos[0], old_pos[1], 0] = 0
        self.state[pos[0], pos[1], 0] = 1
        # update timelimit channel
        self.state[:, :, 3 + self.use_obst].fill((self.step_limit - self.steps_taken)/self.step_limit)

        self.pos = pos

        return self.state, reward, done, {}

    def is_in_grid(self, point):
        return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

    def is_on_edge(self, point):
        return (0 == point[0]) or (0 == point[1]) or (self.w - 1 == point[0]) or (self.w - 1 == point[1])

    def print(self):
        out = np.zeros([self.w, self.w])
        if self.use_obst:
            out[self.state[:, :, 3].astype(bool)] = 4
        out[self.state[:, :, 1].astype(bool)] = 2
        out[self.state[:, :, 2].astype(bool)] = 3
        out[self.state[:, :, 0].astype(bool)] = 1
        chars = {0: ".", 1: "x", 2: "O", 3: "#", 4:"@"}
        pretty = "\n".join(["".join([chars[x] for x in row]) for row in out])
        print(pretty)
        print("TIMELEFT ", self.state[0, 0, 3 + self.use_obst])

    def clone_full_state(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def restore_full_state(self, state_dict):
        self.__dict__.update(state_dict)

    def get_action_meanings(self):
        return ["down", "left", "up", "right"]


if __name__ == "__main__":
    env = IceCliffWalking()

    if True:
        all_r = []
        n_episodes = 1000
        for i in range(n_episodes):
            s = env.reset()
            done = False
            episode_r =0
            while not done:
                s, r, done, _ = env.step(np.random.randint(4))
                episode_r += r
            all_r.append(episode_r)
        print(np.mean(all_r), np.std(all_r), np.std(all_r)/np.sqrt(n_episodes), np.max(all_r))
    else:
        s = env.reset()
        env.print()
        episode_r = 0

        while True:
            key = input()
            if key == "q":
                break
            elif key == "a":
                a = 1
            elif key == "s":
                a = 0
            elif key == "d":
                a = 3
            elif key == "w":
                a = 2

            s, r, d, _ = env.step(a)
            episode_r += r
            if not d:
                env.print()
            if d or key == "r":
                print("Done with {} points. Resetting!".format(episode_r))
                s = env.reset()
                episode_r = 0
                env.print()