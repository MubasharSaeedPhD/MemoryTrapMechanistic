"""
Affordance Reasoning vs Memorized Pattern Test
==================================================
Tests whether grasp strategy (handle vs center) is triggered by:
  (a) LANGUAGE instruction content ("microwave" vs "plate" words)
  (b) SCENE/VISUAL context (what's actually in the scene)
  (c) Genuine affordance reasoning (adapts to actual destination)

Method: Keep Kitchen6 SCENE fixed (microwave physically present),
but swap the LANGUAGE INSTRUCTION to Living5's wording.
If grasp strategy follows LANGUAGE -> memorized language-task association.
If grasp strategy follows SCENE -> memorized visual-context association.
If grasp adapts intelligently -> genuine affordance reasoning (unlikely
given our memory trap findings, but must be tested).
"""
import sys, torch, pathlib, numpy as np, math, json, imageio, collections
sys.path.insert(0, '/mnt/home/mubashar/physicalintelligence/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from openpi_client import image_tools

BDDL_BASE = pathlib.Path(get_libero_path('bddl_files')) / 'libero_10'
BDDL_K6 = str(BDDL_BASE/'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl')
DUMMY = [0.0]*6+[-1.0]
N_EPS = 5
MUG_PREFIX = 'white_yellow_mug_1'
HANDLE_GEOM = 'white_yellow_mug_1_g1'

CONDITIONS = {
    'A_normal_microwave_lang': {
        'lang': 'put the yellow and white mug in the microwave and close it',
        'desc': 'Normal Kitchen6: microwave scene + microwave language',
    },
    'B_fake_plate_lang': {
        'lang': 'put the white mug on the left plate and put the yellow and white mug on the right plate',
        'desc': 'Kitchen6 SCENE + Living5 LANGUAGE (no plate exists in scene)',
    },
    'C_fake_basket_lang': {
        'lang': 'put both the alphabet soup and the cream cheese box in the basket',
        'desc': 'Kitchen6 SCENE + unrelated basket-task language',
    },
    'D_generic_pickup': {
        'lang': 'pick up the yellow and white mug',
        'desc': 'Kitchen6 SCENE + simple pickup instruction (no destination specified)',
    },
}

CKPT = pathlib.Path('/mnt/home/mubashar/.cache/openpi/openpi-assets/'
                    'checkpoints/pi05_libero_pytorch')
policy = _policy_config.create_trained_policy(
    _config.get_config('pi05_libero'), CKPT, pytorch_device='cuda')

def preprocess(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(img[::-1,::-1]), 224, 224))
def quat2aa(q):
    q=q.copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    return q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)

VID_DIR = pathlib.Path('results/videos/affordance_test')
VID_DIR.mkdir(parents=True, exist_ok=True)

def run_episode(lang, seed, ep_idx, vdir):
    env = OffScreenRenderEnv(bddl_file_name=BDDL_K6,
                             camera_heights=256, camera_widths=256)
    env.seed(seed); env.reset(); obs = env.reset()
    action_plan = collections.deque()
    t, success, replay = 0, False, []
    grasp_quat = None
    grasp_dist_handle = None
    grasp_dist_center = None
    was_open = True

    handle_geom_id = None
    for i in range(env.sim.model.ngeom):
        if env.sim.model.geom_id2name(i) == HANDLE_GEOM:
            handle_geom_id = i
            break

    while t < 250:
        if t < 10:
            obs,_,_,_ = env.step(DUMMY); t+=1; continue
        img = preprocess(obs['agentview_image'])
        wrist = preprocess(obs['robot0_eye_in_hand_image'])
        replay.append(img.copy())

        gripper_q = obs['robot0_gripper_qpos'][0]
        is_closed = gripper_q < 0.02
        if was_open and is_closed and grasp_quat is None:
            grasp_quat = obs['robot0_eef_quat'].copy()
            eef_pos = obs['robot0_eef_pos']
            center_pos = env.sim.data.get_body_xpos(f'{MUG_PREFIX}_main').copy()
            handle_pos = env.sim.data.geom_xpos[handle_geom_id].copy()
            grasp_dist_center = float(np.linalg.norm(eef_pos - center_pos))
            grasp_dist_handle = float(np.linalg.norm(eef_pos - handle_pos))
        was_open = not is_closed

        if not action_plan:
            state = np.concatenate((obs['robot0_eef_pos'],
                                    quat2aa(obs['robot0_eef_quat']),
                                    obs['robot0_gripper_qpos']))
            with torch.no_grad():
                chunk = policy.infer({'observation/image': img,
                                      'observation/wrist_image': wrist,
                                      'observation/state': state,
                                      'prompt': lang})['actions']
            action_plan.extend(chunk[:5])
        obs,_,done,_ = env.step(action_plan.popleft().tolist())
        if done: success=True; break
        t+=1

    env.close()
    sfx = 'success' if success else 'failure'
    label = f"w{grasp_quat[3]:.2f}" if grasp_quat is not None else "nograsp"
    imageio.mimwrite(str(vdir/f'ep{ep_idx+1:02d}_{label}_{sfx}.mp4'),
                    [np.asarray(x) for x in replay], fps=10)
    return {
        'success': success,
        'grasp_quat_w': float(grasp_quat[3]) if grasp_quat is not None else None,
        'dist_handle': grasp_dist_handle,
        'dist_center': grasp_dist_center,
        'closer_to': ('handle' if (grasp_dist_handle is not None and
                      grasp_dist_handle < grasp_dist_center) else 'center')
                      if grasp_quat is not None else 'no_grasp',
    }

all_results = {}
for cond_key, cond in CONDITIONS.items():
    print(f"\n{'='*65}")
    print(f"{cond_key}: {cond['desc']}")
    print(f"Language: \"{cond['lang']}\"")
    print(f"{'='*65}")

    vdir = VID_DIR/cond_key; vdir.mkdir(exist_ok=True)
    results = []
    for ep in range(N_EPS):
        r = run_episode(cond['lang'], 7+ep, ep, vdir)
        results.append(r)
        qw = r['grasp_quat_w']
        qw_str = f"{qw:.3f}" if qw is not None else "None"
        print(f"  Ep{ep+1}: success={r['success']}, quat_w={qw_str}, "
              f"closer_to={r['closer_to']}")

    n_handle = sum(1 for r in results if r['closer_to']=='handle')
    n_center = sum(1 for r in results if r['closer_to']=='center')
    qw_vals = [r['grasp_quat_w'] for r in results if r['grasp_quat_w'] is not None]
    mean_qw = np.mean(qw_vals) if qw_vals else None
    qw_summary = f"{mean_qw:.3f}" if mean_qw is not None else "N/A"
    print(f"\n  SUMMARY: handle={n_handle}/5, center={n_center}/5, mean_quat_w={qw_summary}")

    all_results[cond_key] = {
        'desc': cond['desc'], 'lang': cond['lang'],
        'episodes': results, 'n_handle': n_handle, 'n_center': n_center,
        'mean_quat_w': float(mean_qw) if mean_qw is not None else None,
    }

print(f"\n{'='*65}")
print("FINAL COMPARISON — What Triggers Grasp Strategy?")
print(f"{'='*65}")
print(f"{'Condition':<30} {'Handle':>8} {'Center':>8} {'Mean quat_w':>12}")
for k, v in all_results.items():
    qw_disp = f"{v['mean_quat_w']:.3f}" if v['mean_quat_w'] is not None else "N/A"
    print(f"{k:<30} {v['n_handle']:>6}/5  {v['n_center']:>6}/5  {qw_disp:>12}")

print(f"""
Reference values:
  Kitchen6 handle-grasp baseline: quat_w ~ 0.65-0.68
  Living5 center-grasp baseline:  quat_w ~ 0.98-0.99

INTERPRETATION GUIDE:
  If quat_w stays LOW (~0.65) across ALL conditions despite
  language change -> grasp triggered by SCENE/VISUAL context
  (microwave object presence), not language.

  If quat_w shifts HIGH (~0.98) when language changes to
  "plate" wording -> grasp triggered by LANGUAGE/instruction
  text matching a memorized task association.

  If quat_w varies unpredictably -> mixed or context-dependent
  trigger requiring further investigation.
""")

pathlib.Path('results/tables').mkdir(parents=True, exist_ok=True)
json.dump(all_results,
          open('results/tables/affordance_vs_memorization_test.json','w'),
          indent=2)
print("Saved -> results/tables/affordance_vs_memorization_test.json")
