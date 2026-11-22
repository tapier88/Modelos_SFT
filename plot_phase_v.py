import json
from pathlib import Path
import matplotlib.pyplot as plt

data = {
    ("control_head_only", 271828): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.31106022745370865, 0.2248533945530653, 0.24212776497006416, 0.29373378679156303, 0.3758804202079773, 0.3370388150215149, 0.26951388269662857, 0.4344211667776108, 0.37924525141716003, 0.4219692125916481, 0.25371191650629044, 0.3244401589035988],
        "mae": 3.577933975566367,
    },
    ("control_head_only", 271829): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.35061343014240265, 0.4663415998220444, 0.33134786039590836, 0.3226590156555176, 0.4042651653289795, 0.38812917470932007, 0.39186060428619385, 0.3784077912569046, 0.33980434760451317, 0.39605261012911797, 0.3998464606702328, 0.3039631173014641],
        "mae": 3.5744016934631957,
    },
    ("control_head_only", 271830): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.33036210387945175, 0.25500990077853203, 0.37574969232082367, 0.26677224412560463, 0.2316267378628254, 0.29923243820667267, 0.33266030065715313, 0.3846811205148697, 0.5008336901664734, 0.3497340492904186, 0.25745857134461403, 0.3321397416293621],
        "mae": 3.5600991971513642,
    },
    ("joint_QV_head", 271828): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.3110510818660259, 0.22484635189175606, 0.24211306869983673, 0.2937285080552101, 0.375879380851984, 0.3369786813855171, 0.2694152630865574, 0.43437952548265457, 0.37907032668590546, 0.4216131344437599, 0.2535436190664768, 0.32407131791114807],
        "mae": 3.5784142975917796,
    },
    ("joint_QV_head", 271829): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.3506392613053322, 0.4663476286625862, 0.33134859427809715, 0.32264693826436996, 0.4042460769414902, 0.3881121724843979, 0.3918476551771164, 0.3783433437347412, 0.33974724635481834, 0.39597585424780846, 0.399779237806797, 0.30397987365722656],
        "mae": 3.5744321333948332,
    },
    ("joint_QV_head", 271830): {
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "loss": [None, 0.33036137372255325, 0.2550209313631058, 0.3757448196411133, 0.2667577788233757, 0.231705192476511, 0.2991788685321808, 0.33252717554569244, 0.3845812529325485, 0.5010087192058563, 0.3495531938970089, 0.2575686499476433, 0.33172735944390297],
        "mae": 3.5604065867629107,
    },
}

out_dir = Path("output/axxel/m10_phase_v_20261122/bootstrap_phase_v")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
arms = ["control_head_only", "joint_QV_head"]

for ax, arm in zip(axes, arms):
    for (a, seed), vals in data.items():
        if a != arm:
            continue
        valid = [(s, l) for s, l in zip(vals["steps"], vals["loss"]) if l is not None]
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=f"seed {seed}")
    ax.set_title(f"Loss por checkpoint — {arm}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

plt.tight_layout()
loss_path = out_dir / "loss_por_paso.png"
plt.savefig(loss_path, dpi=150)
plt.close()

seeds = [271828, 271829, 271830]
x = range(len(seeds))
width = 0.35

mae_ctrl = [data[("control_head_only", s)]["mae"] for s in seeds]
mae_qv = [data[("joint_QV_head", s)]["mae"] for s in seeds]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar([i - width / 2 for i in x], mae_ctrl, width, label="control_head_only")
bars2 = ax.bar([i + width / 2 for i in x], mae_qv, width, label="joint_QV_head")

ax.set_title("MAE final por semilla y brazo")
ax.set_xticks(list(x))
ax.set_xticklabels([str(s) for s in seeds])
ax.set_xlabel("Seed")
ax.set_ylabel("MAE")
ax.set_ylim(3.54, 3.59)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.legend()

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
mae_path = out_dir / "mae_por_semilla.png"
plt.savefig(mae_path, dpi=150)
plt.close()

print("Guardados:", loss_path, "y", mae_path)
