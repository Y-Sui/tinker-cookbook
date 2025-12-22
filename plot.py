import matplotlib.pyplot as plt
import pandas

metrics_path = "/tmp/tinker-examples/multi-agent-debate/Qwen/Qwen3-8B-debate-3agents-16groups-3e-05lr-2025-12-21-23-42/metrics.jsonl"
df = pandas.read_json(metrics_path, lines=True)
plt.plot(df["env/all/pairwise_reward"], label="reward/total")
plt.legend()
plt.show()
plt.savefig("Qwen3-8B-debate-3agents-16groups-3e-05lr-2025-12-21-23-42.pairwise_reward.png")
