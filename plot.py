import matplotlib.pyplot as plt
import pandas

metrics_path = "/home/yuansui/tinker-examples/multi-agent-debate/Qwen/Qwen3-8B-debate-3agents-16groups-3e-05lr-2025-12-23-14-17/metrics.jsonl"
df = pandas.read_json(metrics_path, lines=True)
plt.plot(df["test/env/all/reward/total"], label="reward/total")
plt.legend()
plt.show()
plt.savefig("pairwise_reward.png")
