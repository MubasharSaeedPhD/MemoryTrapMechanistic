"""
Evaluate the LoRA-finetuned model on the SAME simple, single-object
tasks it was trained on -- to check if the LoRA weights genuinely
improved performance, BEFORE testing the harder, combined Task6.
"""
import sys, torch, torch.nn as nn, pathlib, numpy as np, math, collections
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
device = next(model.parameters()).device
expert = model.paligemma_with_expert.gemma_expert

class LoRALayer(nn.Module):
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=device, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=torch.float32))
        self.scale = alpha / rank
        for p in self.base.parameters(): p.requires_grad_(False)
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return getattr(self.base, 'bias', None)
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features
    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out.to(base_out.dtype)

TOP20_HEADS = [
    (17,2),(14,7),(17,5),(17,3),(16,3),(15,7),(17,4),(17,6),(17,7),(16,7),
    (17,0),(16,2),(14,3),(17,1),(16,5),(15,4),(15,2),(16,6),(16,0),(13,7),
]
target_layers = sorted(set(li for li, h in TOP20_HEADS))

lora_modules = []
for li in target_layers:
    layer = expert.model.layers[li]
    if hasattr(layer.self_attn, 'q_proj'):
        lora_q = LoRALayer(layer.self_attn.q_proj, rank=8, alpha=16)
        layer.self_attn.q_proj = lora_q
        lora_modules.append(lora_q)
    if hasattr(layer, 'mlp'):
        for sub in ['gate_proj', 'up_proj', 'down_proj']:
            sub_module = getattr(layer.mlp, sub, None)
            if sub_module is not None:
                lora_sub = LoRALayer(sub_module, rank=8, alpha=16)
                setattr(layer.mlp, sub, lora_sub)
                lora_modules.append(lora_sub)

# Load trained weights
state = torch.load('results/tables/surgical_lora_weights.pt', map_location=device)
all_params = []
for m in lora_modules:
    all_params += [m.lora_A, m.lora_B]
for i, p in enumerate(all_params):
    p.data.copy_(state[f'param_{i}'].to(device))
print(f"Loaded {len(all_params)} LoRA parameter tensors")

DUMMY = [0.0]*6 + [-1.0]
MAX_T = 300

def preprocess(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(img[::-1,::-1]), 224, 224))
def quat2aa(q):
    q=q.copy(); q[3]=max(-1,min(1,q[3]))
    den=np.sqrt(1-q[3]**2)
    return q[:3]*2*math.acos(q[3])/den if abs(den)>1e-6 else np.zeros(3)

def run_episode(bddl, lang, seed):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed); env.reset(); obs = env.reset()
    action_plan = collections.deque()
    t, success = 0, False
    while t < MAX_T + 10:
        if t < 10: obs,_,_,_ = env.step(DUMMY); t+=1; continue
        img = preprocess(obs['agentview_image'])
        wrist = preprocess(obs['robot0_eye_in_hand_image'])
        if not action_plan:
            state_ = np.concatenate((obs['robot0_eef_pos'], quat2aa(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
            with torch.no_grad():
                chunk = policy.infer({'observation/image': img, 'observation/wrist_image': wrist,
                                      'observation/state': state_, 'prompt': lang})['actions']
            action_plan.extend(chunk[:5])
        obs,_,done,_ = env.step(action_plan.popleft().tolist())
        if done: success=True; break
        t += 1
    env.close()
    return success

CONFIGS = {
    'MUG_ONLY_posA': ('bddl_simple/MUG_ONLY_posA.bddl', 'put the white mug on the plate'),
    'CHOC_ONLY_posA': ('bddl_simple/CHOC_ONLY_posA.bddl', 'put the chocolate pudding to the right of the plate'),
}

print("\n" + "="*60)
print("POST-LORA: Simple task re-evaluation (swap-position variants)")
print("="*60)
for name, (bddl, lang) in CONFIGS.items():
    n_ok = 0
    for ep in range(8):
        ok = run_episode(bddl, lang, ep)
        print(f"  [{name}] Ep{ep+1}: {'SUCCESS' if ok else 'fail'}")
        n_ok += int(ok)
    print(f"  -> {name}: {n_ok}/8\n")
