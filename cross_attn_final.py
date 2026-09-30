import sys, torch, pathlib, numpy as np, math
sys.path.insert(0, '/mnt/home/mubashar/physicalintelligence/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from openpi_client import image_tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BDDL_BASE = pathlib.Path(get_libero_path('bddl_files')) / \
    'libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl'
BDDL_SWAP = pathlib.Path('bddl_ood/LIVING_ROOM_SCENE6_OOD_swap.bddl')
PLOTS_DIR = pathlib.Path('results/plots/cross_attention')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DUMMY = [0.0]*6 + [-1.0]

CKPT = pathlib.Path('/mnt/home/mubashar/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch')
policy = _policy_config.create_trained_policy(
    _config.get_config('pi05_libero'), CKPT, pytorch_device='cuda')

model  = policy._model
pg     = model.paligemma_with_expert
expert = pg.gemma_expert

dtype  = next(model.parameters()).dtype
device = next(model.parameters()).device

# ── Monkey patch denoise_step ──────────────────────────────
from openpi.models_pytorch import pi0_pytorch as pi0

captured_attn = {}
current_mode  = ['none']

orig_denoise = pi0.PI0Pytorch.denoise_step

def patched_denoise_step(self, state, prefix_pad_masks,
                          past_key_values, x_t, timestep):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    import openpi.models_pytorch.pi0_pytorch as _pi0

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = \
        self.embed_suffix(state, x_t, timestep)
    
    suffix_len   = suffix_pad_masks.shape[1]
    batch_size   = prefix_pad_masks.shape[0]
    prefix_len   = prefix_pad_masks.shape[1]
    
    prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len)
    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids   = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_4d    = self._prepare_attention_masks_4d(full_att_2d)
    
    # Force eager
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    
    # Hook Q,K on each expert layer
    layer_qk = {}
    handles   = []
    
    for i, layer in enumerate(expert.model.layers):
        def make_qk_hook(idx):
            def hook(module, inp, output):
                # inp[0] = hidden_states
                h = inp[0]
                input_shape = h.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                with torch.no_grad():
                    q = module.q_proj(h).view(hidden_shape).transpose(1,2)
                    k = module.k_proj(h).view(hidden_shape).transpose(1,2)
                layer_qk[idx] = (q.detach().cpu().float(),
                                  k.detach().cpu().float())
            return hook
        h = layer.self_attn.register_forward_hook(make_qk_hook(i))
        handles.append(h)
    
    # Run forward
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    
    for h in handles: h.remove()
    
    # Store Q,K for this denoising step
    mode = current_mode[0]
    if mode not in captured_attn:
        captured_attn[mode] = {}
    # Only store first denoising step (t=1.0)
    if 'step_0' not in captured_attn[mode]:
        captured_attn[mode]['step_0'] = layer_qk
        print(f"  [{mode}] Captured {len(layer_qk)} layers Q,K")
    
    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.action_horizon:]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)

# Apply monkey patch
pi0.PI0Pytorch.denoise_step = patched_denoise_step
print("Monkey patched denoise_step ✓")

def get_obs(bddl, t=150, seed=7):
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=256, camera_widths=256)
    env.seed(seed); env.reset(); obs = env.reset()
    for step in range(t+10):
        if step < 10: obs,_,_,_ = env.step(DUMMY); continue
        obs,_,done,_ = env.step(DUMMY)
        if done or step >= t: break
    img   = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
    eef = obs['robot0_eef_pos'].copy()
    q = obs['robot0_eef_quat'].copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    aa=q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)
    state = np.concatenate((eef, aa, obs['robot0_gripper_qpos']))
    env.close()
    return img, wrist, state

print("Getting observations...")
img_b, wrist_b, state_b = get_obs(BDDL_BASE, t=150)
img_s, wrist_s, state_s = get_obs(BDDL_SWAP,  t=150)

LANG = ('put the white mug on the plate and '
        'put the chocolate pudding to the right of the plate')

print("\nBaseline inference...")
current_mode[0] = 'baseline'
policy.infer({'observation/image': img_b, 'observation/wrist_image': wrist_b,
              'observation/state': state_b, 'prompt': LANG})

print("\nSwap inference...")
current_mode[0] = 'swap'
policy.infer({'observation/image': img_s, 'observation/wrist_image': wrist_s,
              'observation/state': state_s, 'prompt': LANG})

print(f"\nCaptured modes: {list(captured_attn.keys())}")

# Analyze
if 'baseline' in captured_attn and 'swap' in captured_attn:
    base_qk = captured_attn['baseline']['step_0']
    swap_qk = captured_attn['swap']['step_0']
    
    print(f"Layers: {sorted(base_qk.keys())}")
    if base_qk:
        l0 = list(base_qk.values())[0]
        print(f"Q shape: {l0[0].shape}")
        print(f"K shape: {l0[1].shape}")
    
    # Compute attention weights for L8, L10
    mug_p  = [r*16+c for r in range(9,12) for c in range(4,8)]
    choc_p = [r*16+c for r in range(8,12) for c in range(7,11)]
    
    print("\n=== CROSS ATTENTION: Action → Image Patches ===")
    
    for layer in [8, 10]:
        if layer not in base_qk or layer not in swap_qk:
            print(f"L{layer}: not captured")
            continue
        
        bq, bk = base_qk[layer]  # [batch, heads, seq, head_dim]
        sq, sk = swap_qk[layer]
        
        # Compute attention: Q @ K.T / sqrt(d)
        scale = bq.shape[-1] ** -0.5
        
        b_attn = torch.softmax(
            torch.matmul(bq, bk.transpose(-2,-1)) * scale, dim=-1)
        s_attn = torch.softmax(
            torch.matmul(sq, sk.transpose(-2,-1)) * scale, dim=-1)
        
        # b_attn: [1, heads, action_seq, full_seq]
        # full_seq = action tokens attending to themselves + past_kv
        # But here we only have suffix Q,K — need past_key context too
        
        print(f"\nL{layer}:")
        print(f"  b_attn shape: {b_attn.shape}")
        print(f"  s_attn shape: {s_attn.shape}")
        
        # Average over heads, all action tokens
        b_mean = b_attn[0].mean(dim=(0,1)).numpy()
        s_mean = s_attn[0].mean(dim=(0,1)).numpy()
        
        print(f"  Base self-attn mean: {b_mean.mean():.4f}")
        print(f"  Swap self-attn mean: {s_mean.mean():.4f}")
    
    # Restore original
    pi0.PI0Pytorch.denoise_step = orig_denoise
    print("\nRestored original denoise_step ✓")
