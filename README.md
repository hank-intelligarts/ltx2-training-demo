# LTX-2 Training Demo

Demo repo 展示如何透過 Slurm GPU Cluster 提交 LTX-2 LoRA training job。

## 使用方式

### 透過 Dashboard

1. 打開 http://192.168.51.48:8765/
2. 填入：
   - Command: `python train.py configs/ltx2_lora_ryan.yaml`
   - Username: 你的名字
   - GPU: 1
   - Repo URL: `git@github.com:hank-intelligarts/ltx2-training-demo.git`
   - Venv: `ltx2`
3. Submit

### 透過 curl

```bash
curl -X POST http://192.168.51.48:8765/submit \
  -H "Content-Type: application/json" \
  -d '{
    "command": "python train.py configs/ltx2_lora_ryan.yaml",
    "username": "ryan",
    "gpu": 1,
    "job_name": "ltx2-lora-bones",
    "repo_url": "git@github.com:hank-intelligarts/ltx2-training-demo.git",
    "venv": "ltx2"
  }'
```

## 自訂訓練

1. 複製 `configs/ltx2_lora_ryan.yaml` 改成你的設定
2. 修改 `preprocessed_data_root` 指向你的資料
3. Commit & push
4. 提交 job
