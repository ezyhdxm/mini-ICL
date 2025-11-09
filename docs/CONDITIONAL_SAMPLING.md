# 条件概率采样方法详解

## 目录
1. [问题设定](#问题设定)
2. [数学基础](#数学基础)
3. [后向概率递推](#后向概率递推)
4. [条件采样公式](#条件采样公式)
5. [算法实现](#算法实现)
6. [多线程实现](#多线程实现)
7. [性能分析](#性能分析)

---

## 问题设定

我们考虑一个 **$k$ 阶马尔可夫链**，其中每个状态依赖于前 $k$ 个状态。给定：
- 序列长度：$L$
- 词汇表大小：$V$
- 马尔可夫阶数：$k$ (order)
- 转移矩阵：$\mathbf{P} \in \mathbb{R}^{V^k \times V}$，其中 $P_{s,v}$ 表示从状态历史 $s$（编码前 $k$ 个token）转移到token $v$ 的概率

**目标**：在给定位置 $t$ 的token必须为 $v_t$ 的条件下，采样完整的序列 $x_0, x_1, \ldots, x_{L-1}$。

即，我们需要从条件分布：
$$P(x_0, x_1, \ldots, x_{L-1} \mid x_t = v_t)$$

中采样，而不是使用拒绝采样（rejection sampling）方法。

---

## 数学基础

### 2.1 状态编码

对于 $k$ 阶马尔可夫链，在位置 $t$ 的状态历史 $s_t$ 是一个包含 $k$ 个token的窗口，包含位置 $t-k$ 到 $t-1$ 的token：$[x_{t-k}, x_{t-k+1}, \ldots, x_{t-1}]$。

我们使用以下编码方式将状态窗口编码为整数：

$$s_t = \sum_{i=0}^{k-1} x_{t-k+i} \cdot V^{k-1-i}$$

这样，每个长度为 $k$ 的token序列对应唯一的整数状态索引 $s_t \in \{0, 1, \ldots, V^k - 1\}$。

**示例**：对于 $k=2, V=10$，如果 $x_{t-2}=3, x_{t-1}=5$，则 $s_t = 3 \times 10^1 + 5 \times 10^0 = 35$。

### 2.2 转移概率

给定当前状态历史 $s$，下一个token $v$ 的概率为：
$$P(x_{t+1} = v \mid s_t = s) = P_{s,v}$$

其中 $s_t$ 表示时刻 $t$ 的状态历史。

### 2.3 状态更新

当生成新token $v$ 时，状态历史更新为：
$$s_{t+1} = \text{shift\_left}(s_t) \circ v$$

即：将当前状态历史左移一位，并在末尾添加新token $v$。

---

## 后向概率递推

### 3.1 定义

我们定义**后向概率（backward probability）** $\beta_t(s)$ 为：

$$\beta_t(s) = P(x_T = v_T \mid s_t = s)$$

其中：
- $t$ 是当前位置（$t < T$，$T$ 是目标位置）
- $s$ 是时刻 $t$ 的状态历史
- $v_T$ 是目标位置 $T$ 的给定token值
- $\beta_t(s)$ 表示：从状态 $s$ 开始，经过 $T-t$ 步后能够到达目标token $v_T$ 的概率

### 3.2 边界条件

在目标位置的前一个时刻 $T-1$：

$$\beta_{T-1}(s) = P(x_T = v_T \mid s_{T-1} = s) = P_{s, v_T}$$

这是从状态 $s$ 直接转移到目标token $v_T$ 的概率。

### 3.3 递推关系

对于 $t < T-1$，使用全概率公式：

$$\begin{align}
\beta_t(s) &= P(x_T = v_T \mid s_t = s) \\
&= \sum_{v=0}^{V-1} P(x_T = v_T, x_{t+1} = v \mid s_t = s) \\
&= \sum_{v=0}^{V-1} P(x_{t+1} = v \mid s_t = s) \cdot P(x_T = v_T \mid x_{t+1} = v, s_t = s)
\end{align}$$

由于马尔可夫性质，给定 $x_{t+1} = v$，$x_T$ 只依赖于 $s_{t+1}$（更新后的状态）：

$$P(x_T = v_T \mid x_{t+1} = v, s_t = s) = P(x_T = v_T \mid s_{t+1} = s') = \beta_{t+1}(s')$$

其中 $s' = \text{shift\_left}(s) \circ v$ 是更新后的状态。

因此，递推公式为：

$$\beta_t(s) = \sum_{v=0}^{V-1} P_{s,v} \cdot \beta_{t+1}(s')$$

其中 $s' = \text{shift\_left}(s) \circ v$。

### 3.4 递推步骤

我们使用**动态规划**从后向前计算所有 $\beta$ 值：

1. **初始化**（步骤 $d=0$，对应位置 $T-1$）：
   $$\beta_0(s) = P_{s, v_T}, \quad \forall s \in \{0, 1, \ldots, V^k - 1\}$$

2. **向后递推**（步骤 $d = 1, 2, \ldots, T-k$）：
   对于每个状态 $s$：
   $$\beta_d(s) = \sum_{v=0}^{V-1} P_{s,v} \cdot \beta_{d-1}(s')$$
   
   其中 $s' = \text{shift\_left}(s) \circ v$。

**注意**：这里 $d$ 表示距离目标的步数，$d = T - t - 1$。因此 $\beta_d(s)$ 实际上对应位置 $T-d-1$ 的状态 $s$。
- $d=0$ 对应位置 $T-1$（目标位置 $T$ 的前一个位置，即最后一个需要计算后向概率的位置）
- $d=T-k$ 对应位置 $k-1$（状态窗口为 $[x_0, \ldots, x_{k-1}]$，这是从位置 $k$ 开始采样时最后一个需要预计算后向概率的位置）

**说明**：从位置 $k$（即order）到位置 $T-1$，我们计算所有位置的 $\beta$ 值。其中位置 $T-1$ 的 $\beta$ 值（$d=0$）直接来自转移概率，位置 $k-1$ 的 $\beta$ 值（$d=T-k$）通过递推得到。

---

## 条件采样公式

### 4.1 条件概率

我们的目标是采样：
$$P(x_t = v \mid x_0, \ldots, x_{t-1}, x_T = v_T)$$

使用贝叶斯公式：

$$P(x_t = v \mid x_0, \ldots, x_{t-1}, x_T = v_T) = \frac{P(x_t = v, x_T = v_T \mid x_0, \ldots, x_{t-1})}{P(x_T = v_T \mid x_0, \ldots, x_{t-1})}$$

### 4.2 分子分解

$$P(x_t = v, x_T = v_T \mid x_0, \ldots, x_{t-1}) = P(x_t = v \mid x_0, \ldots, x_{t-1}) \cdot P(x_T = v_T \mid x_0, \ldots, x_{t-1}, x_t = v)$$

第一项是**前向概率**（无条件）：
$$P(x_t = v \mid x_0, \ldots, x_{t-1}) = P(x_t = v \mid s_{t-1}) = P_{s_{t-1}, v}$$

第二项是**后向概率**（已计算）：
$$P(x_T = v_T \mid x_0, \ldots, x_{t-1}, x_t = v) = P(x_T = v_T \mid s_t = s') = \beta_t(s') = \beta_{T-t-1}(s')$$

其中 $s' = \text{shift\_left}(s_{t-1}) \circ v$ 是选择token $v$ 后在位置 $t$ 的新状态。这里 $\beta_{T-t-1}(s')$ 使用了索引映射：$d = T-t-1$ 对应位置 $t$ 的状态 $s'$。

### 4.3 分母处理

分母 $P(x_T = v_T \mid x_0, \ldots, x_{t-1})$ 是所有可能 $v$ 的求和：

$$P(x_T = v_T \mid x_0, \ldots, x_{t-1}) = \sum_{v=0}^{V-1} P(x_t = v, x_T = v_T \mid x_0, \ldots, x_{t-1})$$

这正是我们在计算条件概率时需要归一化的项。

### 4.4 最终公式

综合以上，条件概率为：

$$P(x_t = v \mid x_0, \ldots, x_{t-1}, x_T = v_T) = \frac{P_{s_{t-1}, v} \cdot \beta_{T-t-1}(s')}{\sum_{u=0}^{V-1} P_{s_{t-1}, u} \cdot \beta_{T-t-1}(s'_u)}$$

其中：
- $s_{t-1}$ 是当前状态历史
- $s' = \text{shift\_left}(s_{t-1}) \circ v$ 是选择 $v$ 后的新状态
- $s'_u = \text{shift\_left}(s_{t-1}) \circ u$ 是选择 $u$ 后的新状态

### 4.5 特殊情况

当 $t < k$（在初始窗口内）：
- 状态历史 $s$ 可能不完全由之前的token决定
- 此时，我们直接设置 $x_t = v_T$（如果 $t$ 就是目标位置），或按均匀分布采样其他位置

当 $t = T$：
- 直接设置 $x_T = v_T$（确定值）

当 $t > T$：
- 使用标准的无条件前向采样：
  $$P(x_t = v \mid x_0, \ldots, x_{t-1}) = P_{s_{t-1}, v}$$

---

## 算法实现

### 5.1 算法流程

```
Algorithm: Conditional Sampling
Input: sampler, task_k, input_pos (目标位置，在padding格式中), target_token, num_samples
Output: samples (条件采样序列)

1. 获取转移矩阵 P = sampler.get_task_matrix(task_k)  # 形状: (V^k, V)
2. 计算 token_idx = input_pos // 2  (如果是padding格式，否则 token_idx = input_pos)
3. 设 T = token_idx（目标位置），k = order（马尔可夫阶数）
4. 
5. if token_idx < order:
6.     # 情况1: 目标位置在初始窗口内
7.     初始化状态窗口，确保 token_idx 位置为 target_token
8.     从 order 位置开始，正常前向采样到序列末尾
9. 
10. else:
11.    # 情况2: 目标位置在初始窗口之后
12.    初始化状态窗口（随机）
13.    
14.    # 步骤1: 计算后向概率 beta
15.    beta[0, :] = P[:, target_token]  # 边界条件：d=0对应位置T-1
16.    for step = 1 to token_idx - order:  # step对应d，范围是1到T-k
17.        for each state s:
18.            beta[step, s] = sum_{v} P[s, v] * beta[step-1, shift_left(s)∘v]
19.    
20.    # 步骤2: 从 order 到 token_idx-1 的条件采样
21.    for t = order to token_idx-1:
22.        for each candidate token v:
23.            s_new = shift_left(s_current) ∘ v  # 选择v后的新状态（在位置t）
24.            remaining = token_idx - t - 1  # 从位置t+1到token_idx-1的步数
25.            conditional_prob[v] = P[s_current, v] * beta[remaining, s_new]
26.        归一化 conditional_prob
27.        从 conditional_prob 采样 x_t
28.        更新状态窗口
29.    
30.    # 步骤3: 设置目标位置的token
31.    x[token_idx] = target_token
32.    更新状态窗口
33.    
34.    # 步骤4: 从 token_idx+1 到序列末尾，正常前向采样
35.    for t = token_idx+1 to L-1:
36.        x_t ~ P[s_{t-1}, :]
37.        更新状态窗口
38.
39. 返回 samples
```

### 5.2 复杂度分析

**时间复杂度**：
- Beta计算：$O((T-k) \cdot V^k \cdot V)$，其中 $T-k$ 是步数，$V^k$ 是状态数，$V$ 是词汇表大小
- 采样：$O((T-k) \cdot V)$ 每个位置需要计算 $V$ 个候选的概率
- **总复杂度**：$O((T-k) \cdot V^k \cdot V)$

**空间复杂度**：
- Beta存储：$O((T-k) \cdot V^k)$
- 序列存储：$O(L \cdot \text{num\_samples})$

**优势**：
- 相比拒绝采样，时间复杂度从 $O(\text{rejections} \cdot L)$ 降低到 $O((T-k) \cdot V^k \cdot V)$
- 对于稀有token（低概率），拒绝采样可能需要非常多次尝试，而我们的方法保证一次生成成功

---

## 多线程实现

### 6.1 并行化策略

我们的多线程实现在**词汇维度**进行并行化。对于每个目标token值 $v \in \{0, 1, \ldots, V-1\}$，我们需要生成 $B$ 个条件样本。这些任务是**相互独立**的，因此可以并行执行。

### 6.2 工作流程

```
主进程:
1. 创建任务列表: tasks = [(sampler, task_k, input_pos, v, Bmasked, B_pool) 
                            for v in range(V)]
2. 创建进程池: pool = Pool(num_workers)
3. 并行执行: results = pool.map(worker_function, tasks)
4. 收集结果: 将每个token的结果组装到输出张量

工作进程 (worker_function):
1. 接收参数: (sampler, task_k, input_pos, tok, Bmasked, B_pool)
2. 对于目标token tok，生成 Bmasked 个条件样本
3.    - 批量生成: batch_size = min(Bmasked, B_pool)
4.    - 循环直到收集到足够的样本
5. 返回 (tok, samples_cpu)  # 转换到CPU以跨进程传输
```

### 6.3 实现细节

#### 6.3.1 进程间通信

```python
def _generate_token_samples_for_vocab(args):
    """工作进程函数"""
    sampler, task_k, input_pos, tok, Bmasked, B_pool = args
    
    # 批量生成样本
    batch_size = max(1, min(Bmasked, B_pool))
    collected = []
    remaining = Bmasked
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        # 调用条件采样函数
        samples = generate_conditional_sample(
            sampler, task_k, input_pos, tok, current_batch
        )
        # 转换到CPU以便进程间传输
        collected.append(samples.cpu())
        remaining -= current_batch
    
    # 拼接所有批次
    result = torch.cat(collected, dim=0)[:Bmasked]
    return tok, result
```

#### 6.3.2 主进程调度

```python
def collect_tokenwise_batches(..., parallel_vocab=True, num_workers=None):
    if parallel_vocab and vocab_size_eff > 1:
        if num_workers is None:
            # 默认worker数: min(4, vocab_size, cpu_count)
            num_workers = min(4, vocab_size_eff, os.cpu_count() or 1)
        
        if num_workers > 1:
            # 使用spawn方法（Windows兼容）
            set_start_method("spawn", force=True)
            
            # 创建任务列表
            args_list = [(sampler, task_k, input_pos, tok, Bmasked, B_pool) 
                        for tok in range(vocab_size_eff)]
            
            # 并行执行
            with Pool(processes=num_workers) as pool:
                results = pool.map(_generate_token_samples_for_vocab, args_list)
            
            # 组装结果
            for tok, result in results:
                out[tok] = result.to(sampler.device)
```

#### 6.3.3 数据序列化

- **输入**：sampler对象需要可序列化（pickle）。PyTorch的multiprocessing使用spawn方法时，每个子进程会重新导入模块并重建对象。
- **输出**：将结果转换到CPU（`.cpu()`），因为GPU张量不能直接在进程间传输。

### 6.4 负载均衡

每个worker处理 $\lceil V / \text{num\_workers} \rceil$ 个token，任务大小相同，负载均衡。

### 6.5 性能提升

假设：
- 词汇表大小：$V = 20$
- Worker数量：$4$
- 每个token需要生成时间：$T_v$

**串行时间**：$\sum_{v=0}^{V-1} T_v$

**并行时间**（理想情况）：$\max(\text{worker负载}) \approx \frac{\sum_{v=0}^{V-1} T_v}{4}$

**加速比**：理论上可达 $4\times$（受限于CPU核心数和I/O开销）

---

## 性能分析

### 7.1 与拒绝采样对比

**拒绝采样方法**：
- 重复生成样本，直到找到 $x_t = v_T$ 的样本
- 对于稀有token（概率 $p \ll 1$），期望尝试次数：$1/p$
- 时间复杂度：$O(\frac{L}{p} \cdot \text{num\_samples})$

**条件采样方法**：
- 直接计算条件概率，一次性生成所需样本
- 时间复杂度：$O((T-k) \cdot V^k \cdot V + L \cdot \text{num\_samples})$

**对比**：
- 当 $p < \frac{1}{(T-k) \cdot V^k \cdot V}$ 时，条件采样更快
- 对于典型参数（$T-k=10$, $V^k=400$, $V=20$），阈值 $p \approx 0.00125$
- 对于稀有token（$p < 0.001$），条件采样显著更快

### 7.2 内存使用

**Beta矩阵**：
- 大小：$(T-k) \times V^k$
- 对于 $T-k=50$, $V^k=400$：约 80KB（float32）

**样本存储**：
- 大小：$V \times B \times L$
- 对于 $V=20$, $B=32$, $L=100$：约 256KB（int64）

**总内存**：相比拒绝采样（需要存储大量中间样本），内存使用更可控。

### 7.3 并行效率

**理论加速比**：$\min(\text{num\_workers}, V)$

**实际考虑**：
- 进程创建开销
- 数据序列化/反序列化开销
- I/O竞争（如果使用GPU）

**建议**：
- 当 $V \geq 8$ 时，使用并行化才有明显收益
- Worker数量建议：$2 \leq \text{num\_workers} \leq \min(8, V, \text{cpu\_count})$

---

## 总结

本文档详细介绍了基于条件概率的马尔可夫链采样方法：

1. **数学基础**：使用后向概率（beta）和贝叶斯公式，正确计算条件分布
2. **算法实现**：动态规划计算beta值，然后按条件概率采样
3. **多线程优化**：在词汇维度并行化，提升吞吐量

该方法相比拒绝采样具有以下优势：
- ✅ 对稀有token更高效（无浪费采样）
- ✅ 时间复杂度可控（不依赖于token概率）
- ✅ 支持多线程加速
- ✅ 内存使用合理

适用于需要大量条件样本的场景，特别是在分析模型对不同token类型的响应时。

