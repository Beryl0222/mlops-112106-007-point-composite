# 点投影的深度合成

samples/scene.json 是本地合成的几何片和训练像素交接。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 可微 alpha 合成算子

`alpha_compose/` 是 CPU 上的纯 Python/NumPy 实现，训练与交互预览共用同一份
可见性定义：

| 模块 | 职责 |
| --- | --- |
| `alpha_compose/load.py` | 交换资料加载与边界校验（坐标、深度、[0,1] 量值、分片归属） |
| `alpha_compose/composite.py` | 单像素前向合成、公开误差界证明、无除法反向梯度 |
| `alpha_compose/session.py` | 有状态会话：分片到达、区域读取、版本令牌、预算记账、梯度归并 |
| `alpha_compose/budget.py` | 临时区预算台账，禁止整幅图的点-像素相交矩阵 |
| `alpha_compose/timeline.py` | 按 `timeline` 回放 arrive/read/backward 事件 |

命令行回放：

```sh
python3 run_session.py samples/scene.json
python3 run_session.py samples/conflicting-overlap.json
```

### 关键语义

* **去重与冲突**：同版同一点在同一像素的重复报送只算一次；覆盖或深度冲突的
  像素标记为 `conflict`，不出合成结果也不接受梯度。
* **定序**：深度小者在前，同深度按稳定点号排序，一个点进入一个像素恰好一次。
* **有效不透明度**：`o = 覆盖系数 × 点透明度`；前向透射乘积合成，残余透射乘背景。
* **缺口与承诺**：尚未到达的几何分片构成深度缺口。缺口之后的颜色/透射每通道
  影响不超过缺口入口透射量；当残余透射上界已不超过 `error.forward` 时可在公开
  误差界内提前结束。几何已齐时必须走完全部候选，给出精确的零界结果。
* **可训练资格**：`forward_trainable` 按前向界与 `adjoint_bounds` 给出；
  `backward_ok_worst` 单独回答反向误差是否同时满足——颜色稳定时隐藏点的
  不透明度导数仍可能超界。
* **反向**：分别接收 RGB 与透射损失梯度，颜色梯度与点透明度梯度（含覆盖换算）
  按主记录求和；上游梯度超过 `adjoint_bounds` 的像素会被拒绝并要求重新取精度。
* **版本保护**：每次读取颁发版本令牌，冻结该次交付所依据的点顺序与覆盖事实；
  点主记录被替换后，旧令牌的归还要么按旧事实独立核算，要么直接报
  `StaleVersionError`，不会向已被取代的点套用新几何梯度。
* **预算**：像素分块处理，暂存超 `budget_bytes` 时区域标记 `needs_continue`，
  已承诺像素的训练承诺不受影响。
