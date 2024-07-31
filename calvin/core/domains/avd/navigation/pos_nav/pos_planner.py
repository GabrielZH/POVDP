import argparse
import numpy as np

from core.domains.avd.navigation.planner import AVDPlannerBase, AVDMDPMetaBase
from core.domains.avd.navigation.pos_nav.position_map import PositionNode, create_position_map
# from swin3d.Swin3D.SemanticSeg.model.Swin3D_RGB import Swin3D
# from swin3d.Swin3D.SemanticSeg.util import config


class AVDPosMDPMeta(AVDMDPMetaBase):
    def __init__(self, *args, segment_config_path=None, **kwargs):
        super(AVDPosMDPMeta, self).__init__(*args, segment_config_path=None, **kwargs)
        self.maps = {}
        for scene_name, scene in self.scenes.items():
            pos_map = create_position_map(scene, interval=2)
            self.maps[scene_name] = pos_map
    #     self.segment_config = self.get_segment_arg_parser(segment_config_path)
    #     self.segmentation_model = Swin3D(
    #         depths=self.segment_config.depths, 
    #         channels=self.segment_config.channels, 
    #         num_heads=self.segment_config.num_heads, 
    #         window_sizes=self.segment_config.window_sizes, 
    #         up_k=self.segment_config.up_k, 
    #         quant_size=self.segment_config.quant_size, 
    #         drop_path_rate=self.segment_config.drop_path_rate, 
    #         num_layers=self.segment_config.num_layers, 
    #         num_classes=self.segment_config.num_classes, 
    #         stem_transformer=self.segment_config.stem_transformer, 
    #         upsample=self.segment_config.upsample, 
    #         down_stride=self.segment_config.down_stride, 
    #         knn_down=self.segment_config.knn_down, 
    #         signal=self.segment_config.signal, 
    #         in_channels=self.segment_config.fea_dim, 
    #         use_offset=self.segment_config.use_offset, 
    #         fp16_mode=self.segment_config.fp16_mode
    #     )

    # def get_segment_arg_parser(self, segment_config_path):
    #     return config.load_cfg_from_cfg_file(segment_config_path)

    def get_states_from_scene(self, scene_name):
        pos_map = self.maps[scene_name]
        return list(pos_map.nodes.values())

    def get_states_from_objects(self, objects):
        return [self.maps[obj.image_node.scene.name].image_to_pos_node[obj.image_node.image_name] for obj in objects]

    def state_to_index(self, state: PositionNode):
        h, _, w = state.position
        h1, w1, h2, w2 = self.map_bbox
        assert h1 <= h < h2, f"h dimension {h} out of range [{h1}, {h2})"
        assert w1 <= w < w2, f"w dimension {w} out of range [{w1}, {w2})"
        x_size, z_size = self.state_shape
        x_ind = int((h - h1) / (h2 - h1) * x_size)
        z_ind = int((w - w1) / (w2 - w1) * z_size)
        return x_ind, z_ind

    def state_index_to_grid_index(self, state_index):
        return state_index


class AVDPosPlanner(AVDPlannerBase):
    def transition(self, curr_state: PositionNode, action, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        if action == self.meta.actions.done: return None
        if reverse:
            raise NotImplementedError
        new_state = curr_state.transitions.get(action)
        if new_state:
            cost = np.linalg.norm(new_state.position - curr_state.position)
            return new_state, action, cost
        return None

    def get_motion(self, state_index, next_state_index, i):
        return next_state_index[i] - state_index[i]

    def get_values_and_best_actions(self, target):
        meta = self.meta
        cost, best_trans, _ = self.get_transition_tree(target, is_root_target=True)
        values = np.ones(meta.state_shape, dtype=float) * -np.inf
        counts = np.zeros(meta.state_shape, dtype=int)
        best_actions = np.zeros((len(meta.actions), *meta.state_shape), dtype=bool)
        motion = np.zeros((len(meta.actions), len(meta.state_shape) + 1, *meta.state_shape), dtype=int)
        for state in self.states:
            state_index = meta.state_to_index(state)
            values[state_index] = - cost[state]
            if best_trans[state]:
                _, action = best_trans[state]
                best_actions[(meta.action_to_index(action), *state_index)] = True
                while best_trans[state]:
                    next_state, action = best_trans[state]
                    state_index = meta.state_to_index(state)
                    next_state_index = meta.state_to_index(next_state)
                    counts[state_index] += 1
                    for i in range(len(meta.state_shape)):
                        motion[(meta.action_to_index(action), i, *state_index)] \
                            = self.get_motion(state_index, next_state_index, i)
                    motion[(meta.action_to_index(action), len(meta.state_shape), *state_index)] = 1
                    state = next_state
            state_index = meta.state_to_index(state)
            counts[state_index] += 1
        target_index = meta.state_to_index(target)
        done_index = meta.action_to_index(meta.actions.done)
        best_actions[(done_index, *target_index)] = True
        for i in range(len(meta.state_shape)):
            motion[(done_index, i, *target_index)] = self.get_motion(target_index, target_index, i)
        motion[(done_index, len(meta.state_shape), *target_index)] = 1
        return values, best_actions, counts, motion

    def get_valid_actions(self, target):
        meta = self.meta
        vaild_actions = np.zeros((len(meta.actions), *meta.state_shape), dtype=bool)
        for state in self.states:
            state_index = meta.state_to_index(state)
            for _, action, _ in self.transitions(state):
                vaild_actions[(meta.action_to_index(action), *state_index)] = True
        target_index = meta.state_to_index(target)
        done_index = meta.action_to_index(meta.actions.done)
        vaild_actions[(done_index, *target_index)] = True
        return vaild_actions
