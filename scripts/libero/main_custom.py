import collections
import dataclasses
import datetime
import enum
import json
import logging
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R
import tqdm
import tyro
import xyg_scripts.rotate_recolor_dataset as rotate_recolor_dataset
import xyg_scripts.cross_embodiment_utils as cross_embodiment_utils
# IF this import fails, please add xyg_scripts folder in libero. Contact Jiyun if you have any questions.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class PolicyType(str, enum.Enum):
    """Supported policy serving modes for LIBERO eval."""

    LAP = "LAP"
    LAP_AR = "LAP_AR"


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    policy_type: PolicyType = PolicyType.LAP

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    control_mode: str = "OSC_POSE"  # Controller type. Options: OSC_POSE, IK_POSE, OSC_POSITION, JOINT_POSITION, etc.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "data/libero/results"  # Path to save evaluation results
    log_out_path: str = "data/libero/logs"  # Path to save per-step action + reasoning logs
    camera_name: str = "agentview"  # hw_review : 카메라 시점 회전 위한 세팅. 얘도 agentview가 XML에 정의된 카메라 이름 libero_tabletop_warm_style.xml 가면 볼 수 있음 Name of camera to use for policy input and video recording
    robot_base_name: str = "robot0_link0" # hw_review : Franka 로봇 body 구조가 MuJoCo 환경에 XML 데이터로 들었는데, link0(제일 prefix 몸통 최하단) 임. 이거 기준으로 카메라 회전.
    viewpoint_rotate: float = 0.0  # Rotate camera viewpoint by this many degrees (for testing generalization to new viewpoints)
    gripper_type: str = "Robotiq85Gripper"  # hw_review : LIBERO 기본값 Franka 기본 그리퍼(PandaGripper) -> Robotiq85 로 교체. gripper type to use in the environment. Options: ['default', 'RethinkGripper', 'PandaGripper', 'JacoThreeFingerGripper', 'JacoThreeFingerDexterousGripper', 'WipingGripper', 'Robotiq85Gripper', 'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 'RobotiqThreeFingerDexterousGripper', None] 
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.log_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize results tracking
    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_suite": args.task_suite_name,
            "policy_type": args.policy_type.value,
            "control_mode": args.control_mode,
            "seed": args.seed,
            "num_trials_per_task": args.num_trials_per_task,
            "replan_steps": args.replan_steps,
        },
        "episodes": [],
        "per_task_results": [],
        "summary": {},
    }

    if args.task_suite_name == "libero_spatial":
        max_steps = 200  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("viewpoint rotate:", args.viewpoint_rotate, "camera name:", args.camera_name)
    print("gripper type:", args.gripper_type)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env_kwargs = {}
        env_kwargs['gripper_types'] = [args.gripper_type] # hw_review : Robotiq85
        if args.gripper_type != "default":
            franka_env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_mode) # hw_review : 밑에서 initial state 변환할 때 기준으로 쓰려고
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_mode, **env_kwargs) # hw_review : get libero env에 **kwargs 추가로 gripper types 동적 조절

        task_description = "pick up the black bowl"

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            try:
                obs = env.set_init_state(initial_states[episode_idx])
            except: # hw_review : 그리퍼 교체 시 initial state 변환 로직. 변환 함수는 xyg 내부 모듈에 들었음. 
                cross_embodiment_initial_states = cross_embodiment_utils.convert_franka_flat_to_franka_robotiq85_flat(src_env=franka_env, tgt_env=env, src_flat=initial_states[episode_idx])
                obs = env.set_init_state(cross_embodiment_initial_states)
            camera_id = env.sim.model.camera_name2id(args.camera_name) #hw_review : 매 에피소드마다 카메라 회전 적용. MUJOCO에서 현재 카메라 pose 읽고, 로봇 베이스 pose 좌표계로 변환
            env = rotate_recolor_dataset.rotate_camera(env, camera_id=camera_id, camera_name=args.camera_name, robot_base_name=args.robot_base_name, 
                                                              theta=args.viewpoint_rotate, debug=False)
            # Setup
            t = 0
            replay_images = []
            wrist_replay_images = []
            step_logs = []
            current_reasoning = None
            episode_start_time = datetime.datetime.now()

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img, wrist_img = get_images_from_obs(obs, args.resize_size)

                    reasoning_this_step = None
                    action_chunk_sum = None
                    if not action_plan:
                        # Query model to get action
                        request = obs_to_request(obs, args.policy_type, img, wrist_img, task_description)
                        response = client.infer(request)
                        current_reasoning = response.get("reasoning")
                        reasoning_this_step = current_reasoning
                        single_action_or_chunk = np.asarray(response["actions"], dtype=np.float32)
                        if single_action_or_chunk.ndim == 1:
                            assert args.policy_type == PolicyType.LAP_AR
                            print(response)
                            action_chunk = get_action_from_response(
                                args.replan_steps, response, request["observation"]["state"]
                            )
                        else:
                            action_chunk = single_action_or_chunk
                        action_chunk = invert_and_scale_gripper(action_chunk)
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        # debug: gripper values across all 16 steps
                        print(f"[DEBUG] step={t - args.num_steps_wait} action_chunk.shape={action_chunk.shape} gripper_per_step: {action_chunk[:, -1].tolist()}")

                        # sum of full 16-step chunk, converted to physical units
                        s = action_chunk.sum(axis=0)
                        action_chunk_sum = {
                            "dx_cm": float(s[0] * _OSC_POS_OUTPUT_MAX * 100),
                            "dy_cm": float(s[1] * _OSC_POS_OUTPUT_MAX * 100),
                            "dz_cm": float(s[2] * _OSC_POS_OUTPUT_MAX * 100),
                            "droll_rad": float(s[3] * _OSC_ROT_OUTPUT_MAX),
                            "dpitch_rad": float(s[4] * _OSC_ROT_OUTPUT_MAX),
                            "dyaw_rad": float(s[5] * _OSC_ROT_OUTPUT_MAX),
                            "gripper": float(s[6]),
                        }
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    wrist_replay_images.append(wrist_img)

                    action = action_plan.popleft()

                    step_logs.append({
                        "step": t - args.num_steps_wait,
                        "action": action.tolist(),
                        "reasoning": reasoning_this_step,
                        "action_chunk_sum": action_chunk_sum,
                    })

                    # Execute action in environment
                    print("Step:", t, "Action:", action)
                    print(action[-1])
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Calculate episode duration
            episode_duration = (datetime.datetime.now() - episode_start_time).total_seconds()

            # Record episode results
            episode_result = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_id": episode_idx,
                "global_episode_id": total_episodes - 1,
                "success": bool(done),
                "num_steps": t - args.num_steps_wait,
                "total_steps_with_wait": t,
                "duration_seconds": episode_duration,
            }
            all_results["episodes"].append(episode_result)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_wrist_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in wrist_replay_images],
                fps=10,
            )

            # Save per-step action + reasoning log
            log_entry = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_id": episode_idx,
                "success": bool(done),
                "steps": step_logs,
            }
            log_path = pathlib.Path(args.log_out_path) / f"task{task_id}_ep{episode_idx}.json"
            with open(log_path, "w") as f:
                json.dump(log_entry, f, indent=2)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Record per-task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_result = {
            "task_id": task_id,
            "task_description": task_description,
            "num_episodes": task_episodes,
            "num_successes": task_successes,
            "success_rate": task_success_rate,
        }
        all_results["per_task_results"].append(task_result)

        # Log final results
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Calculate and save final summary
    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    all_results["summary"] = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "num_tasks": num_tasks_in_suite,
    }

    # Save results to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{args.task_suite_name}_{args.policy_type.value}_{timestamp}.json"
    results_path = pathlib.Path(args.results_out_path) / results_filename
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Total success rate: {overall_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Results saved to: {results_path}")

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    print(os.getenv("CUDA_VISIBLE_DEVICES"), os.getenv("MUJOCO_GL"), os.getenv("MUJOCO_EGL_DEVICE_ID"))

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_libero_env_multi_embodiment(task, model_family, resolution=256, **kwargs):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env_args.update(kwargs)
    print(os.getenv("CUDA_VISIBLE_DEVICES"), os.getenv("MUJOCO_GL"), os.getenv("MUJOCO_EGL_DEVICE_ID"))
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _get_libero_env(task, resolution, seed, controller="OSC_POSE", **kwargs):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "controller": controller,
    }
    env_args.update(kwargs)
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_images_from_obs(obs, resize_size):
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    return img, wrist_img


def obs_to_request(obs, policy_type: PolicyType, img, wrist_img, task_description: str):
    # Prepare observations dict
    assert policy_type in (PolicyType.LAP, PolicyType.LAP_AR), f"Unsupported policy type: {policy_type}"
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_rot6d = _quat2rot6d(obs["robot0_eef_quat"]).astype(np.float32, copy=False)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper_state = np.clip(gripper_qpos[-2:-1] / 0.04, 0, 1)  # normalize to [0, 1]
    state = np.concatenate((eef_pos, eef_rot6d, gripper_state)).astype(np.float32, copy=False)
    return {
        "observation": {
            "base_0_rgb": img,
            "left_wrist_0_rgb": wrist_img,
            "state": state,
        },
        "prompt": str(task_description),
    }


def invert_and_scale_gripper(action_chunk):
    action_chunk[:, -1:] = 1 - 2 * action_chunk[:, -1:]
    action_chunk[:, -1:] = np.sign(action_chunk[:, -1:])
    return action_chunk


def _quat2rot6d(quat):
    "Convert quaternion to 6D rotation representation."
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,), ordered as [x, y, z, w]")
    rot_matrix = R.from_quat(q).as_matrix()
    # rot6d = rot_matrix[:, :2].flatten()
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
    return rot6d


_OSC_POS_OUTPUT_MAX = 0.05  # meters: OSC_POSE scales [-1, 1] input → [-0.05, 0.05] m
_OSC_ROT_OUTPUT_MAX = 0.5  # radians: OSC_POSE scales [-1, 1] input → [-0.5, 0.5] rad


def get_action_from_response(replan_steps, response, state):
    action = np.asarray(response["actions"])
    grip_action = action[-1]

    # Policy outputs real-world deltas (meters, radians). LIBERO controller expects normalized
    # [-1, 1] inputs which it scales by output_max. Divide total delta evenly across
    # replan_steps (equivalent to uniform SLERP for rotation).

    # Position: normalize by OSC output_max (0.05 m), split across steps
    pos_per_step = (action[:3] / _OSC_POS_OUTPUT_MAX) / replan_steps
    pos_actions = np.tile(pos_per_step, (replan_steps, 1))

    # Rotation: convert delta extrinsic Euler XYZ → axis-angle, normalize by OSC
    # output_max (0.5 rad), split across steps
    delta_rotvec = R.from_euler("xyz", action[3:6]).as_rotvec()
    rot_per_step = (delta_rotvec / _OSC_ROT_OUTPUT_MAX) / replan_steps
    rot_actions = np.tile(rot_per_step, (replan_steps, 1))

    grip_vals = np.full((replan_steps, 1), grip_action)
    return np.concatenate([pos_actions, rot_actions, grip_vals], axis=1)

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)
