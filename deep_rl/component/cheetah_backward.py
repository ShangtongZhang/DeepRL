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

"""Cheetah Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.cheetah import Cheetah
from dm_control.suite.cheetah import Physics
from dm_control.suite.cheetah import SUITE
from dm_control.suite.cheetah import _DEFAULT_TIME_LIMIT
from dm_control.suite.cheetah import _RUN_SPEED
from dm_control.suite.cheetah import get_model_and_assets


@SUITE.add('benchmarking')
def backward(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CheetahBackward(random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class CheetahBackward(Cheetah):
    def __init__(self, random=None):
        Cheetah.__init__(self, random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return rewards.tolerance(-physics.speed(),
                                 bounds=(_RUN_SPEED, float('inf')),
                                 margin=_RUN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')
