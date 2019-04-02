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

"""Fish Domain."""

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
from dm_control.suite.fish import SUITE
from dm_control.suite.fish import _DEFAULT_TIME_LIMIT
from dm_control.suite.fish import Physics
from dm_control.suite.fish import get_model_and_assets
from dm_control.suite.fish import Upright
from dm_control.suite.fish import _CONTROL_TIMESTEP


@SUITE.add('benchmarking')
def downleft(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
    """Returns the Fish Upright task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Downleft(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
        **environment_kwargs)


class Downleft(Upright):
    def __init__(self, random=None):
        Upright.__init__(self, random)

    def get_reward(self, physics):
        """Returns a smooth reward."""
        return rewards.tolerance(-physics.upright(), bounds=(1, 1), margin=1)
