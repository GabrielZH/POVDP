# POVDP
Project repository of the CVPR 2024 paper: Versatile Navigation under Partial Observability via Value-Guided Diffusion Policy. <br>
[NOTE] Due to an institutional administrative issue, we are currently unable to access the server to retrieve the code for this paper. We will release the code as soon as the server is back online. We apologize for the inconvenience.

## Abstract
Route planning for navigation under partial observability plays a crucial role in modern robotics and autonomous driving. Existing route planning approaches can be categorized into two main classes: traditional autoregressive and diffusion-based methods. The former often fails due to its myopic nature, while the latter either assumes full observability or struggles to adapt to unfamiliar scenarios, due to strong couplings with behavior cloning from experts. To address these deficiencies, we propose a versatile diffusion-based approach for both 2D and 3D route planning under partial observability. Specifically, our value-guided diffusion policy first generates plans to predict actions across various timesteps, providing ample foresight to the planning. It then employs a differentiable planner with state estimations to derive a value function, directing the agent’s exploration and goal-seeking behaviors without seeking experts while explicitly addressing partial observability. During inference, our policy is further enhanced by a best-plan-selection strategy, substantially boosting the planning success rate. Moreover, we propose projecting point clouds, derived from RGB-D inputs, onto 2D grid-based bird-eye-view maps via semantic segmentation, generalizing to 3D environments. This simple yet effective adaption enables zero-shot transfer from 2D policy to 3D, cutting across the laborious training for 3D policy, and thus certifying our versatility. Experimental results demonstrate our superior performance, particularly in navigating situations beyond expert demonstrations, surpassing state-of-the-art autoregressive and diffusion-based baselines for both 2D and 3D scenarios.

![Diffusion Policy](figs/diffusion_policy.pdf)
*Fig. 1 Closed-loop conditional diffusion plan generator.*

![Reward Function](figs/reward_function.pdf)
![Value Iteration](figs/value_iteration.pdf)
*Fig. 2 Value guidance via QMDP value iteration network.*

## Paper link: [arXiv](https://arxiv.org/abs/2404.02176).
