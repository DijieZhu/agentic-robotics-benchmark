# Copyright (c) 2017, the operator's government, as represented by the
# Administrator of the operator agency.
# 
# All rights reserved.
# 
# The Cabinflyer platform is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# Command from the Force Allocation Module to the PMs

# Timestamp, frame_id, sequential number
std_msgs/Header header

# Commands to the PMs
ff_hw_msgs/PmGoal[] goals
