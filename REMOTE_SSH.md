# 远程服务器

## 项目位置

- Host：`connect.westc.seetacloud.com`
- Port：`33893`
- User：`root`
- Project：`/root/autodl-tmp/vlm-qwen3vl`

## 连接

```bash
ssh -p 33893 root@connect.westc.seetacloud.com
cd /root/autodl-tmp/vlm-qwen3vl
```

## 当前数据

- SFT：`data/sft/train.jsonl`、`data/sft/valid.jsonl`
- Joint：`data/joint/train.jsonl`、`data/joint/valid.jsonl`
- Test：`data/test/test.jsonl`
- 图片：各数据集下的 `images/`

当前文档和数据以服务器为准；不要用本地副本覆盖服务器数据。不要在文档中记录密码或密钥。
