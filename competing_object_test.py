"""
Competing Object Test — Does chocolate's presence at the wrong
position SPECIFICALLY cause attention to collapse in the swap scene?

Three conditions compared, all using attention snapshots BEFORE and
AFTER mug-placement (same methodology as orig_attention_collapse_check):

A) ISOLATED swap-mug (MUG_ONLY_posA bddl) — mug at RIGHT (swap pos),
   NO chocolate present at all
B) COMBINED swap scene (Task6 OOD) — mug RIGHT, choc LEFT
   (choc at OLD mug position = "memory-trap conflict")
C) CONTROL: mug RIGHT, choc also RIGHT (non-competing position)
   -- to check if choc's presence alone matters or its SPECIFIC
   position at the "expected" mug location matters

Prediction if "competing object" hypothesis is correct:
  A > B in post-mug choc-attention (B collapses, A doesn't)
  C should behave closer to A than B (no positional conflict)
"""
import sys, torch, pathlib, numpy as np, math, collections, json
sys.path.insert(0, '/mnt/home/mubashar/physicalintelligence/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from openpi_client import image_tools
from robosuite.utils.camera_utils import (
    project_points_from_world_to_camera, get_camera_transform_matrix)
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models.model as _model_module

DUMMY = [0.0]*6 + [-1.0]
LANG_FULL = ('put the white mug on the plate and '
             'put the chocolate pudding to the right of the plate')
LANG_MUG  = 'put the white mug on the plate'
MUG, CHOC, PLATE = 'porcelain_mug_1_main', 'chocolate_pudding_1_main', 'plate_1_main'
N_IMG_TOKENS = 256
TARGET_LAYER = 16
SEEDS = [0, 1, 2, 7]

# Condition A — isolated swap-mug (no chocolate)
BDDL_A = pathlib.Path('bddl_simple/MUG_ONLY_posA.bddl')
LANG_A = LANG_MUG

# Condition B — combined swap (chocolate at OLD mug position = LEFT)
BDDL_B = pathlib.Path('bddl_ood/LIVING_ROOM_SCENE6_OOD_swap.bddl')
LANG_B = LANG_FULL

# Condition C — control: mug RIGHT, choc also RIGHT (non-conflicting)
# We create this bddl inline
BDDL_C_PATH = pathlib.Path('bddl_simple/MUG_RIGHT_CHOC_RIGHT_control.bddl')
BDDL_C_PATH.write_text('''(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the white mug on the plate and put the chocolate pudding to the right of the plate)
    (:regions
      (plate_right_region
          (:target living_room_table)
          (:ranges (
              (0.09999999999999999 0.05 0.2 0.15000000000000002)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (plate_init_region
          (:target living_room_table)
          (:ranges (
              (0.125 -0.025 0.175 0.025)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (porcelain_mug_init_region
          (:target living_room_table)
          (:ranges (
              (-0.07500000000000001 0.07500000000000001 -0.025 0.125)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (chocolate_pudding_init_region
          (:target living_room_table)
          (:ranges (
              (-0.07500000000000001 0.12500000000000001 -0.025 0.175)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
    )
  (:fixtures
    living_room_table - living_room_table
  )
  (:objects
    plate_1 - plate
    porcelain_mug_1 - porcelain_mug
    chocolate_pudding_1 - chocolate_pudding
  )
  (:obj_of_interest
    plate_1
    porcelain_mug_1
    chocolate_pudding_1
  )
  (:init
    (On plate_1 living_room_table_plate_init_region)
    (On porcelain_mug_1 living_room_table_porcelain_mug_init_region)
    (On chocolate_pudding_1 living_room_table_chocolate_pudding_init_region)
  )
  (:goal
    (And (On porcelain_mug_1 plate_1) (On chocolate_pudding_1 living_room_table_plate_right_region))
  )
)''')
LANG_C = LANG_FULL

CKPT = pathlib.Path('/mnt/home/mubashar/.cache/openpi/openpi-assets/'
                    'checkpoints/pi05_libero_pytorch')
policy = _policy_config.create_trained_policy(
    _config.get_config('pi05_libero'), CKPT, pytorch_device='cuda')
model = policy._model
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
expert = model.paligemma_with_expert.gemma_expert

def preprocess(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(img[::-1,::-1]), 224, 224))
def quat2aa(q):
    q=q.copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    return q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)
def bc(v, top=None, key=None):
    if isinstance(v, dict): return {kk: bc(vv,top=key,key=kk) for kk,vv in v.items()}
    if v is None: return None
    t = torch.from_numpy(v) if isinstance(v,np.ndarray) else \
        torch.tensor(bool(v)) if isinstance(v,(bool,np.bool_)) else torch.as_tensor(v)
    t = t.unsqueeze(0).to(device)
    if top=='image_masks': t=t.to(torch.bool)
    elif top=='images' or key=='state': t=t.to(dtype)
    return t

def get_attention_snapshot(img, wrist, state_np, mug_patch, choc_patch, lang):
    obs_dict = {'observation/image': img, 'observation/wrist_image': wrist,
                'observation/state': state_np, 'prompt': lang}
    obs_proc = policy._input_transform(obs_dict)
    obs_batched = {k: bc(v,top=k,key=k) for k,v in obs_proc.items()}
    observation = _model_module.Observation.from_dict(obs_batched)

    exp_attn_captures = {}
    def make_attn_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out)==2:
                exp_attn_captures[li] = out[1].detach().float().cpu().clone()
        return hook
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    h = expert.model.layers[TARGET_LAYER].self_attn.register_forward_hook(make_attn_hook(TARGET_LAYER))

    with torch.no_grad():
        bsize = observation.state.shape[0]
        num_steps = 10
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)
        images, img_masks, lang_tokens, lang_masks, state_t = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        dt = torch.tensor(-1.0/num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt/2:
            v_t = model.denoise_step(state_t, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time = time + dt
    h.remove()

    if TARGET_LAYER not in exp_attn_captures:
        return None, None
    w = exp_attn_captures[TARGET_LAYER]
    k_len = w.shape[3]
    n_img = min(N_IMG_TOKENS, k_len)
    # average over all action tokens and all heads
    avg_attn = w[0, :, :, :n_img].mean(dim=0).mean(dim=0).numpy()
    a_mug = float(avg_attn[mug_patch]) if mug_patch < n_img else None
    a_choc = float(avg_attn[choc_patch]) if choc_patch < n_img else None
    return a_mug, a_choc

def run_condition(cond_name, bddl, lang, seed, has_choc=True):
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed); env.reset(); obs = env.reset()
    action_plan = collections.deque()
    t = 0
    mug_placed, mug_placed_step = False, None
    snap_before, snap_after = None, None
    success = False

    while t < 300:
        if t < 10: obs,_,_,_ = env.step(DUMMY); t+=1; continue
        img = preprocess(obs['agentview_image'])
        wrist = preprocess(obs['robot0_eye_in_hand_image'])

        mug_pos = env.sim.data.get_body_xpos(MUG).copy()
        plate_pos = env.sim.data.get_body_xpos(PLATE).copy()
        choc_pos = env.sim.data.get_body_xpos(CHOC).copy() if has_choc else np.array([0,0,0])

        T_cam = get_camera_transform_matrix(env.sim, 'agentview', 224, 224)
        def to_patch(pos):
            px = project_points_from_world_to_camera(np.array([pos]), T_cam, 224, 224)[0]
            return (int(np.clip(px[1],0,223))//14)*16 + (int(np.clip(px[0],0,223))//14)
        mug_patch = to_patch(mug_pos)
        choc_patch = to_patch(choc_pos) if has_choc else 0

        mug_on_plate = (np.linalg.norm(mug_pos[:2]-plate_pos[:2]) < 0.07) and (abs(mug_pos[2]-plate_pos[2]) < 0.08)
        if mug_on_plate and not mug_placed:
            mug_placed = True; mug_placed_step = t

        if t == 20:
            state_ = np.concatenate((obs['robot0_eef_pos'], quat2aa(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
            snap_before = get_attention_snapshot(img, wrist, state_, mug_patch, choc_patch, lang)

        if mug_placed and snap_after is None and t > mug_placed_step + 15:
            state_ = np.concatenate((obs['robot0_eef_pos'], quat2aa(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
            snap_after = get_attention_snapshot(img, wrist, state_, mug_patch, choc_patch, lang)

        if not action_plan:
            state_ = np.concatenate((obs['robot0_eef_pos'], quat2aa(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
            with torch.no_grad():
                chunk = policy.infer({'observation/image': img, 'observation/wrist_image': wrist,
                                      'observation/state': state_, 'prompt': lang})['actions']
            action_plan.extend(chunk[:5])
        obs,_,done,_ = env.step(action_plan.popleft().tolist())
        if done: success=True; break
        t+=1

    env.close()
    return {
        'condition': cond_name, 'seed': seed,
        'mug_placed_step': mug_placed_step, 'success': success,
        'snap_before': snap_before, 'snap_after': snap_after
    }

CONDITIONS = [
    ('A_isolated_swap_mug',    BDDL_A, LANG_A, False),
    ('B_combined_swap_task6',  BDDL_B, LANG_B, True),
    ('C_control_both_right',   BDDL_C_PATH, LANG_C, True),
]

print("="*70)
print("COMPETING OBJECT TEST — 3 Conditions x 4 Seeds")
print("="*70)

all_results = []
for cond_name, bddl, lang, has_choc in CONDITIONS:
    print(f"\n--- Condition: {cond_name} ---")
    for seed in SEEDS:
        r = run_condition(cond_name, bddl, lang, seed, has_choc)
        all_results.append(r)
        print(f"  Seed {seed}: mug_placed_at={r['mug_placed_step']}, "
              f"success={r['success']}, "
              f"before={r['snap_before']}, "
              f"after={r['snap_after']}")

print("\n" + "="*70)
print("KEY COMPARISON (post-mug-placement choc-attention)")
print("="*70)
for cond_name,_,_,_ in CONDITIONS:
    vals = [r['snap_after'][1] for r in all_results
            if r['condition']==cond_name and r['snap_after'] and r['snap_after'][1] is not None]
    if vals:
        print(f"  {cond_name}: mean attn_choc AFTER = {np.mean(vals):.6f} (n={len(vals)})")

json.dump(all_results, open('results/tables/competing_object_test.json','w'), indent=2, default=str)
print("\nSaved -> results/tables/competing_object_test.json")
