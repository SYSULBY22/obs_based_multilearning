在评论-指挥专家的仿真训练架构里，obs是模型训练的重要参数，在这个项目里，我们将实现一个基于obs编码的多任务学习任务，核心目的是，在保持一个模型不变的基础上，通过输入不同的指令来控制机器人完成不同的运动（仿真）。具体方法是将指令集编码成向量（one-hot），接入双专家架构中的输入端，通过均匀训练次数来保证模型能够同时学到多个指令任务。
可支持下游学习，即微调，模型训练完成之后随时可以添加新的指令继续学习，只需要微调其中的训练分配，该目的在于使得模型不产生遗忘。
项目源码已放在：
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/flat_multitask.py`
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/interactive_play.py`
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/__init__.py`
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/flat_env_cfg.py`
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/rough_env_cfg.py`
- `project_files/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/agents/rsl_rl_ppo_cfg.py`

## 功能摘要

- 多（三）技能多任务训练（forward / right / left）与 10 维 task instruction obs。
- 按任务掩码启用奖励簇，单模型混合训练。
- `interactive_play.py` 支持终端实时切换技能。
- `interactive_play.py` 支持中文自然语言输入并映射到技能命令。

## 典型启动示例

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/s_0/interactive_play.py \
  --task=Isaac-Velocity-Flat-S-0-MultiTask-v0 \
  --headless \
  --livestream 1 \
  --enable_cameras \
  --device cuda:2 \
  --num_envs 300 \
  --checkpoint /mnt/data/LBY/IsaacLab/logs/rsl_rl/s0_flat_velocity/<your_run>/model_950.pt
```

终端可输入：

- `forward`
- `right`
- `left`
- `random`
- `quit`

也支持自然语言，例如：`全部向前移动`、`往右转`、`向左走`。

## 视频展示

- 前进演示：[rl-video-step-950-forward.mp4](play/rl-video-step-950-forward.mp4)
- 右转演示：[rl-video-step-950-right.mp4](play/rl-video-step-950-right.mp4)
- 左转演示：[rl-video-step-950-left.mp4](play/rl-video-step-950-left.mp4)
