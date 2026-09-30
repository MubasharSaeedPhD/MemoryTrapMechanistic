"""
Two experiments in one script:

EXP 1: v6 Ablation Study (Reviewer Concern #6)
  - Window ablation: t<50, t<100, t<150, t<200, t<300 (always-on)
  - Factor ablation: alpha=0.01, 0.05, 0.1, 0.5 (window fixed at t<150)
  - Head count ablation: top-20, top-42, top-84 heads by kNN score

EXP 2: v6 on Kitchen_Scene4 (Reviewer Q3)
  - Does the SAME intervention (84 heads, t<150, alpha=0.05)
    fix the swap failure on a completely different task?
  - Original: 5/5, Swap: 0/5 (confirmed baseline)
  - With v6: ?/5
"""
import sys, torch, pathlib, numpy as np, math, collections, json, imageio
sys.path.insert(0, '/mnt/home/mubashar/physicalintelligence/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

CKPT = pathlib.Path('/mnt/home/mubashar/.cache/openpi/openpi-assets/'
                    'checkpoints/pi05_libero_pytorch')
policy = _policy_config.create_trained_policy(
    _config.get_config('pi05_libero'), CKPT, pytorch_device='cuda')
model = policy._model
expert = model.paligemma_with_expert.gemma_expert
N_EXP_LAYERS = len(expert.model.layers)
N_HEADS, HEAD_DIM = 8, 128

# Load identified bad heads (84 Action Expert heads from multi-seed analysis)
exp_data = json.load(open('results/tables/expert_multiseed_swap_20260618_131948.json'))
ALL_84_HEADS = [(l, h) for l, h, r in exp_data['consistent_choc_favored']]
print(f"Loaded {len(ALL_84_HEADS)} biased heads")

# Load kNN scores for head count ablation
knn_data = json.load(open('results/tables/head_knn_scores.json'))
# top20 from pre-computed top20 list: format ['EXP_L17_H2', score]
def parse_head(name):
    # EXP_L17_H2 -> (17, 2)
    parts = name.replace('EXP_L','').replace('_H','_').split('_')
    return (int(parts[0]), int(parts[1]))
top20 = [parse_head(name) for name, _ in knn_data['top20'][:20]]
top42 = [parse_head(name) for name, _ in knn_data['top20'][:42]]
print(f"Head subsets: top20={len(top20)}, top42={len(top42)}, all84={len(ALL_84_HEADS)}")

DUMMY = [0.0]*6 + [-1.0]
N_EPS = 5
VID_BASE = pathlib.Path('results/videos/ablation')
VID_BASE.mkdir(parents=True, exist_ok=True)

def preprocess(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(img[::-1,::-1]), 224, 224))
def quat2aa(q):
    q=q.copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    return q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)

def make_suppression_hook(bad_heads, factor):
    """Returns pre-hook for o_proj that suppresses specified heads."""
    def hook(module, args):
        hidden = args[0].clone()
        shape = hidden.shape
        if shape[-1] != N_HEADS * HEAD_DIM:
            return args
        reshaped = hidden.view(*shape[:-1], N_HEADS, HEAD_DIM)
        for h in bad_heads:
            reshaped[..., h, :] *= factor
        return (reshaped.view(*shape),) + args[1:]
    return hook

def run_episode(bddl, lang, seed, bad_heads_by_layer,
                factor, window_end, vid_path=None):
    """Run one episode with suppression hooks active during [10, window_end]."""
    suppression_on = [False]

    handles = []
    for li in range(N_EXP_LAYERS):
        heads_this_layer = bad_heads_by_layer.get(li, [])
        if not heads_this_layer:
            continue
        def make_hook(heads):
            def hook(module, args):
                if not suppression_on[0]:
                    return args
                hidden = args[0].clone()
                shape = hidden.shape
                if shape[-1] != N_HEADS * HEAD_DIM:
                    return args
                reshaped = hidden.view(*shape[:-1], N_HEADS, HEAD_DIM)
                for h in heads:
                    reshaped[..., h, :] *= factor
                return (reshaped.view(*shape),) + args[1:]
            return hook
        h = expert.model.layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_hook(heads_this_layer))
        handles.append(h)

    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=256, camera_widths=256)
    env.seed(seed); env.reset(); obs = env.reset()
    action_plan = collections.deque()
    t, success, replay = 0, False, []

    while t < 530 + 10:
        if t < 10:
            obs,_,_,_ = env.step(DUMMY); t+=1; continue
        img   = preprocess(obs['agentview_image'])
        wrist = preprocess(obs['robot0_eye_in_hand_image'])
        replay.append(img.copy())
        suppression_on[0] = (10 <= t < window_end)
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
    for h in handles: h.remove()
    if vid_path and replay:
        imageio.mimwrite(vid_path, [np.asarray(x) for x in replay], fps=10)
    return success

def heads_by_layer(head_list):
    d = {}
    for l, h in head_list:
        d.setdefault(l, []).append(h)
    return d

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1A: Window ablation (alpha=0.05, heads=84)
# ─────────────────────────────────────────────────────────────
BDDL_SWAP = 'bddl_ood/LIVING_ROOM_SCENE6_OOD_swap.bddl'
LANG6 = ('put the white mug on the plate and '
         'put the chocolate pudding to the right of the plate')
bad84 = heads_by_layer(ALL_84_HEADS)

print("\n" + "="*60)
print("EXP 1A: Window Ablation (alpha=0.05, 84 heads)")
print("="*60)
window_results = {}
for window in [50, 100, 150, 200, 300, 9999]:
    label = 'always-on' if window == 9999 else f't<{window}'
    n_ok = 0
    for ep in range(N_EPS):
        ok = run_episode(BDDL_SWAP, LANG6, 7+ep, bad84, 0.05, window)
        n_ok += int(ok)
    window_results[label] = n_ok
    print(f"  window={label:10s}: {n_ok}/{N_EPS}")

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1B: Factor ablation (window=t<150, heads=84)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 1B: Factor Ablation (window=t<150, 84 heads)")
print("="*60)
factor_results = {}
for alpha in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    label = f'alpha={alpha}'
    n_ok = 0
    for ep in range(N_EPS):
        ok = run_episode(BDDL_SWAP, LANG6, 7+ep, bad84, alpha, 150)
        n_ok += int(ok)
    factor_results[label] = n_ok
    print(f"  {label:15s}: {n_ok}/{N_EPS}")

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1C: Head count ablation (window=t<150, alpha=0.05)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 1C: Head Count Ablation (window=t<150, alpha=0.05)")
print("="*60)
head_count_results = {}
for label, head_list in [('top20', top20), ('top42', top42),
                          ('all84', ALL_84_HEADS)]:
    bhl = heads_by_layer(head_list)
    n_ok = 0
    for ep in range(N_EPS):
        ok = run_episode(BDDL_SWAP, LANG6, 7+ep, bhl, 0.05, 150)
        n_ok += int(ok)
    head_count_results[label] = n_ok
    print(f"  {label:8s} ({len(head_list):3d} heads): {n_ok}/{N_EPS}")

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2: v6 on Kitchen_Scene4 swap
# ─────────────────────────────────────────────────────────────
BDDL_K4_SWAP = 'bddl_ood/kitchen_scene4/KITCHEN_SCENE4_swap.bddl'
LANG_K4 = 'put the black bowl in the bottom drawer of the cabinet and close it'
VID_K4 = VID_BASE / 'kitchen_scene4_v6'
VID_K4.mkdir(exist_ok=True)

print("\n" + "="*60)
print("EXP 2: v6 on Kitchen_Scene4 SWAP (cross-task transfer)")
print("="*60)
k4_results = {}

# Baseline (no suppression)
n_ok = 0
for ep in range(N_EPS):
    ok = run_episode(BDDL_K4_SWAP, LANG_K4, 7+ep,
                     {}, 1.0, 0)  # no suppression
    n_ok += int(ok)
k4_results['swap_no_intervention'] = n_ok
print(f"  Swap (no intervention):  {n_ok}/{N_EPS}")

# v6 suppression
n_ok = 0
for ep in range(N_EPS):
    vid = str(VID_K4 / f'ep{ep+1:02d}_PENDING.mp4')
    ok = run_episode(BDDL_K4_SWAP, LANG_K4, 7+ep,
                     bad84, 0.05, 150, vid_path=vid)
    sfx = 'success' if ok else 'failure'
    pathlib.Path(vid).rename(str(VID_K4 / f'ep{ep+1:02d}_{sfx}.mp4'))
    n_ok += int(ok)
    print(f"  Ep{ep+1}: {'SUCCESS' if ok else 'FAILURE'}")
k4_results['swap_v6'] = n_ok
print(f"  Swap (v6 intervention):  {n_ok}/{N_EPS}")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print("\n1A Window ablation:")
for k, v in window_results.items():
    bar = '█'*v + '░'*(N_EPS-v)
    print(f"  {k:12s}: {bar} {v}/{N_EPS}")

print("\n1B Factor ablation:")
for k, v in factor_results.items():
    bar = '█'*v + '░'*(N_EPS-v)
    print(f"  {k:15s}: {bar} {v}/{N_EPS}")

print("\n1C Head count ablation:")
for k, v in head_count_results.items():
    bar = '█'*v + '░'*(N_EPS-v)
    print(f"  {k:10s}: {bar} {v}/{N_EPS}")

print("\n2 Kitchen_Scene4 transfer:")
for k, v in k4_results.items():
    bar = '█'*v + '░'*(N_EPS-v)
    print(f"  {k:30s}: {bar} {v}/{N_EPS}")

all_results = {
    'window_ablation': window_results,
    'factor_ablation': factor_results,
    'head_count_ablation': head_count_results,
    'kitchen_scene4_transfer': k4_results,
}
json.dump(all_results,
          open('results/tables/ablation_and_kitchen_v6.json','w'), indent=2)
print("\nSaved -> results/tables/ablation_and_kitchen_v6.json")
