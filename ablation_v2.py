"""
Ablation Study v2 — using EXACT same hook mechanism as v6
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
N_HEADS, HEAD_DIM = 8, 256

exp_data = json.load(open('results/tables/expert_multiseed_swap_20260618_131948.json'))
ALL_84_HEADS = set((l, h) for l, h, r in exp_data['consistent_choc_favored'])

knn_data = json.load(open('results/tables/head_knn_scores.json'))
def parse_head(name):
    parts = name.replace('EXP_L','').replace('_H','_').split('_')
    return (int(parts[0]), int(parts[1]))
TOP20_SET = set(parse_head(name) for name, _ in knn_data['top20'][:20])
TOP42_SET = set(parse_head(name) for name, _ in knn_data['top20'][:42])

DUMMY = [0.0]*6 + [-1.0]
BDDL_SWAP = 'bddl_ood/LIVING_ROOM_SCENE6_OOD_swap.bddl'
LANG6 = ('put the white mug on the plate and '
         'put the chocolate pudding to the right of the plate')
BDDL_K4   = 'bddl_ood/kitchen_scene4/KITCHEN_SCENE4_swap.bddl'
LANG_K4   = 'put the black bowl in the bottom drawer of the cabinet and close it'
VID_BASE  = pathlib.Path('results/videos/ablation_v2')
VID_BASE.mkdir(parents=True, exist_ok=True)
MUG = 'porcelain_mug_1_main'
PLATE = 'plate_1_main'
BOWL = 'akita_black_bowl_1_main'

def preprocess(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(img[::-1,::-1]), 224, 224))
def quat2aa(q):
    q=q.copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    return q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)

def run_with_suppression(bddl, lang, seed, bad_heads_set,
                          factor, window_end, n_eps=5,
                          track_mug=False, vid_dir=None):
    """
    EXACT same pattern as head_suppression_v6_window150.py:
    - suppression_active = global flag list
    - hooks registered ONCE before episode loop
    - flag toggled inside timestep loop
    - hooks removed after all episodes
    """
    suppression_active = [False]

    # Register hooks exactly like v6
    def make_pre_hook(layer_idx):
        bad_heads_this_layer = [h for (l,h) in bad_heads_set if l == layer_idx]
        def hook(module, args):
            if not suppression_active[0] or not bad_heads_this_layer:
                return args
            hidden = args[0].clone()
            shape = hidden.shape
            if shape[-1] != N_HEADS * HEAD_DIM:
                return args
            hidden_reshaped = hidden.view(*shape[:-1], N_HEADS, HEAD_DIM)
            for h in bad_heads_this_layer:
                hidden_reshaped[..., h, :] *= factor
            return (hidden_reshaped.view(*shape),) + args[1:]
        return hook

    exp_handles = [
        expert.model.layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_pre_hook(li))
        for li in range(N_EXP_LAYERS)
        if any(l == li for l, h in bad_heads_set)
    ]

    results = []
    for ep in range(n_eps):
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=256, camera_widths=256)
        env.seed(seed + ep); env.reset(); obs = env.reset()
        action_plan = collections.deque()
        t, success, replay = 0, False, []
        touched, grasped = False, False

        while t < 530 + 10:
            if t < 10:
                obs,_,_,_ = env.step(DUMMY); t+=1; continue
            img   = preprocess(obs['agentview_image'])
            wrist = preprocess(obs['robot0_eye_in_hand_image'])
            replay.append(img.copy())

            # EXACT v6 pattern: set flag before infer
            suppression_active[0] = (t < window_end)

            if track_mug:
                mug_pos   = env.sim.data.get_body_xpos(MUG).copy()
                plate_pos = env.sim.data.get_body_xpos(PLATE).copy()
                eef_pos   = obs['robot0_eef_pos']
                d_mug = np.linalg.norm(eef_pos - mug_pos)
                if d_mug < 0.12: touched = True
                if d_mug < 0.05: grasped = True

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
        if vid_dir and replay:
            sfx = 'success' if success else 'failure'
            imageio.mimwrite(str(vid_dir/f'ep{ep+1:02d}_{sfx}.mp4'),
                           [np.asarray(x) for x in replay], fps=10)
        results.append({'success': success, 'touched': touched, 'grasped': grasped})

    for h in exp_handles: h.remove()
    n_success = sum(r['success'] for r in results)
    n_touched = sum(r['touched'] for r in results)
    n_grasped = sum(r['grasped'] for r in results)
    return n_success, n_touched, n_grasped

# ─────────────────────────────────────────────────────────────
# EXP 1A: Window ablation
# ─────────────────────────────────────────────────────────────
print("="*60)
print("EXP 1A: Window Ablation (alpha=0.05, 84 heads)")
print("="*60)
window_results = {}
for window in [50, 100, 150, 200, 300, 9999]:
    label = 'always-on' if window==9999 else f't<{window}'
    ns, nt, ng = run_with_suppression(
        BDDL_SWAP, LANG6, 7, ALL_84_HEADS, 0.05, window,
        n_eps=5, track_mug=True)
    window_results[label] = {'success':ns,'touched':nt,'grasped':ng}
    print(f"  {label:12s}: touched={nt}/5, grasped={ng}/5, full_success={ns}/5")

# ─────────────────────────────────────────────────────────────
# EXP 1B: Factor ablation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 1B: Factor Ablation (window=t<150, 84 heads)")
print("="*60)
factor_results = {}
for alpha in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    label = f'alpha={alpha}'
    ns, nt, ng = run_with_suppression(
        BDDL_SWAP, LANG6, 7, ALL_84_HEADS, alpha, 150,
        n_eps=5, track_mug=True)
    factor_results[label] = {'success':ns,'touched':nt,'grasped':ng}
    print(f"  {label:15s}: touched={nt}/5, grasped={ng}/5, full_success={ns}/5")

# ─────────────────────────────────────────────────────────────
# EXP 1C: Head count ablation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 1C: Head Count Ablation (window=t<150, alpha=0.05)")
print("="*60)
hc_results = {}
for label, hset in [('top20',TOP20_SET),('top42',TOP42_SET),('all84',ALL_84_HEADS)]:
    ns, nt, ng = run_with_suppression(
        BDDL_SWAP, LANG6, 7, hset, 0.05, 150,
        n_eps=5, track_mug=True)
    hc_results[label] = {'success':ns,'touched':nt,'grasped':ng,'n_heads':len(hset)}
    print(f"  {label:8s} ({len(hset):3d} heads): touched={nt}/5, grasped={ng}/5, full_success={ns}/5")

# ─────────────────────────────────────────────────────────────
# EXP 2: Kitchen_Scene4 v6 transfer
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 2: Kitchen_Scene4 v6 Transfer")
print("="*60)
vk4 = VID_BASE / 'kitchen_v6'; vk4.mkdir(exist_ok=True)
ns_base, _, _ = run_with_suppression(
    BDDL_K4, LANG_K4, 7, set(), 1.0, 0, n_eps=5)
print(f"  Swap baseline (no intervention): {ns_base}/5")

ns_v6, nt_v6, ng_v6 = run_with_suppression(
    BDDL_K4, LANG_K4, 7, ALL_84_HEADS, 0.05, 150,
    n_eps=5, vid_dir=vk4)
print(f"  Swap + v6 intervention:          {ns_v6}/5 (touched={nt_v6}, grasped={ng_v6})")

k4_results = {
    'swap_baseline': ns_base,
    'swap_v6': {'success':ns_v6,'touched':nt_v6,'grasped':ng_v6}
}

all_results = {
    'window_ablation': window_results,
    'factor_ablation': factor_results,
    'head_count_ablation': hc_results,
    'kitchen_scene4_transfer': k4_results,
}
json.dump(all_results,
          open('results/tables/ablation_v2.json','w'), indent=2)
print("\nSaved -> results/tables/ablation_v2.json")
