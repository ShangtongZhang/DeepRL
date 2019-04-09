# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Walker Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control.rl import control
from dm_control.utils import rewards
from dm_control.suite.walker import Physics
from dm_control.suite.walker import SUITE
from dm_control.suite.walker import get_model_and_assets
from dm_control.suite.walker import PlanarWalker
from dm_control.suite.walker import _DEFAULT_TIME_LIMIT
from dm_control.suite.walker import _WALK_SPEED
from dm_control.suite.walker import _CONTROL_TIMESTEP
from dm_control.suite.walker import _STAND_HEIGHT

_SQUAT_HEIGHT = 0.6


@SUITE.add('benchmarking')
def backward(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalkerBackward(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def squat(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalkerSquat(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class PlanarWalkerBackward(PlanarWalker):
    def __init__(self, move_speed, random=None):
        super(PlanarWalkerBackward, self).__init__(move_speed, random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(-physics.horizontal_velocity(),
                                            bounds=(self._move_speed, float('inf')),
                                            margin=self._move_speed / 2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6


class PlanarWalkerSquat(PlanarWalker):
    def __init__(self, move_speed, random=None):
        super(PlanarWalkerSquat, self).__init__(move_speed, random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_SQUAT_HEIGHT, float('inf')),
                                     margin=_SQUAT_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(self._move_speed, float('inf')),
                                            margin=self._move_speed / 2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6
