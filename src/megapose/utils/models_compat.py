"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def change_keys_of_older_models(state_dict):
    new_state_dict = dict()
    print("SHOULD CHANGE")
    for k, v in state_dict.items():
        # print(k)
        # print(v.shape)

        if k.startswith("backbone.backbone"):
            new_k = "backbone." + k[len("backbone.backbone.") :]
        elif k.startswith("backbone.head.0."):
            new_k = "views_logits_head." + k[len("backbone.head.0.") :]
        

        else:
            new_k = k

        if new_k in  ["views_logits_head.weight"]:
            print("-------CHANGING STUFF")
            new_k = "pose_fc.weight"
        elif new_k in ["views_logits_head.bias"]:
            new_k = "pose_fc.bias"

        new_state_dict[new_k] = state_dict[k]
    return new_state_dict
