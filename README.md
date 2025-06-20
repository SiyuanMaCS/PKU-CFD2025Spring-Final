

# 一维 Euler 方程 Sod 激波管求解器  


---

## 程序简介

本程序用于数值求解**一维无粘 Euler 方程**描述的 **Sod 激波管问题**。  
该问题是计算流体力学中的经典测试问题，包含激波、接触间断和膨胀波三种主要波结构。

我们使用以下方法进行求解：

### 🔍 支持的格式组合：
| 激波捕捉格式 | 通量分裂方法 |
|--------------|----------------|
| `tvd`        | `lax_friedrichs` |
| `gvc`        | `steger_warming` |
| `weno`       |                |

---

## 使用说明

### 运行命令示例：

```bash
python main.py --scheme weno --flux steger_warming --nx 200
```

### 📝 可用参数说明：

| 参数名         | 类型   | 默认值       | 描述                       |
|----------------|--------|--------------|----------------------------|
| `--scheme`     | 字符串 | `weno`       | 激波捕捉格式 (可选: tvd, gvc, weno) |
| `--flux`       | 字符串 | `steger_warming` | 通量分裂方法 (可选: lax_friedrichs, steger_warming) |
| `--nx`         | 整数   | `200`        | 网格点数                   |
| `--x_min`      | 浮点数 | `-5.0`       | 计算域左边界               |
| `--x_max`      | 浮点数 | `5.0`        | 计算域右边界               |
| `--t_end`      | 浮点数 | `2.0`        | 结束时间                   |
| `--gamma`      | 浮点数 | `1.4`        | 比热比                     |
| `--cfl`        | 浮点数 | `0.5`        | CFL 数（控制时间步长）     |
| `--bc_type`    | 字符串 | `non-reflective` | 边界条件类型（可选: non-reflective, periodic, fixed）|
| `--num_ghost`  | 整数   | `3`          | 虚单元层数                 |
| `--save_plots` | 标志   | 启用         | 是否保存图像               |
| `--plot_dir`   | 字符串 | `results`    | 图像保存路径               |

---

## 示例命令行运行

```bash
# 使用 WENO + Steger-Warming
python main.py --scheme weno --flux steger_warming --nx 200

# 使用 TVD + Lax-Friedrichs
python main.py --scheme tvd --flux lax_friedrichs --nx 400 --plot_dir results_tvd_lf

# 批量运行所有组合（需先创建 run_all.sh）
chmod +x run_all.sh
./run_all.sh
```

---

## 输出内容

程序输出如下信息：

- 控制台实时输出当前时间、步数、时间步长；
- 最终时刻误差统计（密度、速度、压强的 L2 误差）；
- 在指定目录下生成图像文件：
  - `FINAL_{scheme}_{flux}_nx{nx}.png`：最终时刻数值解与精确解对比图；
  - `tvd_steger_warming_t=*.png`：中间时间步结果。

图像中包含三子图：
- 密度（ρ）
- 速度（u）
- 压强（p）

虚线为精确解，实线为数值解。

---

## 实验结果分析

### 🕒 总体运行时间排序（从小到大）：

| 方法组合                  | 时间（秒） |
|---------------------------|------------|
| `gvc + lax_friedrichs`     | 12.56      |
| `tvd + steger_warming`     | 14.18      |
| `tvd + lax_friedrichs`     | 16.19      |
| `gvc + steger_warming`     | 18.42      |
| `weno + steger_warming`    | 27.73      |
| `weno + lax_friedrichs`    | 29.62      |

### 最终误差排序（从低到高）：

| 方法组合                  | 密度误差 | 速度误差 | 压强误差 |
|---------------------------|----------|----------|----------|
| `weno + lax_friedrichs`    | ★★★★★    | ★★★★★    | ★★★★★    |
| `gvc + lax_friedrichs`    | ★★★★☆    | ★★★★☆    | ★★★★☆    |
| `tvd + lax_friedrichs`    | ★★★☆☆    | ★★★☆☆    | ★★★☆☆    |
| `tvd + steger_warming`    | ★★★☆☆    | ★★★☆☆    | ★★★☆☆    |
| `gvc + steger_warming`    | ★★☆☆☆    | ★★☆☆☆    | ★★☆☆☆    |
| `weno + steger_warming`   | ★★☆☆☆    | ★★☆☆☆    | ★★☆☆☆    |

---

## 文件结构说明

- `main.py`: 主程序入口；
- `utils.py`: 包含边界处理、绘图、误差计算等功能；
- `initialize.py`: 初始化网格和物理参数；
- `time_integration.py`: 时间推进（Runge-Kutta）；
- `numerical_methods/fluxes.py`: 通量函数（Lax-Friedrichs 和 Steger-Warming）；
- `numerical_methods/schemes/gvc.py`: 激波捕捉格式（TVD、GVC、WENO）。

---
