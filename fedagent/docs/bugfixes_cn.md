# Bug 修复记录

FedAgent verl-0.8 overlay 的重要正确性 / 健壮性修复日志,每条都讲清**为什么错**以及怎么修的。

---

## 2026-07-11 —— 环境服务池:`/create` 双重借用竞态 → 池耗尽 → rollout 挂死

- **服务:** `fedagent/envs/webshop/service/server.py`、`fedagent/envs/alfworld/service/server.py`
- **级别:** blocker —— 满批次训练可能永久卡死。**症状:** 训练静默停住,没有崩溃、没有 traceback。
- **来源:** 在抽取出的独立框架(AccelAgent)做 release 审计时发现,那边相同的继承服务已修复(commit
  `ac08200`);本分支为回移。

### 这段代码在干什么

每个 env 实例很贵(`gym.make` ~26 秒;ALFWorld 还要起一个 JVM),所以服务预热一个小**池子**
(`POOL_SIZE`,默认 4),每一集**借**一个:

```
/create      → env = await _pool.get()   # 借;池空时在这里挂起(await)
/reset,/step → 用它
/close       → _pool.put_nowait(env)     # 还回去
```

在 full-PPO 风暴下,两个条件**同时**成立(服务自己的注释都写了):

- **池子被打满** —— `train_batch_size × rollout.n`(几百)集共享 4 个 env,于是 `await _pool.get()`
  会阻塞。这是设计内的背压。
- **socket 中途 reset** —— HTTP 边界被打爆,客户端对传输错误**重试**,并且复用**同一个 `session_id`**。

### Bug 本身 —— check-then-borrow 的 TOCTOU 竞态

原始 `/create`:

```python
async def create(r: Sid):
    if r.session_id in _sessions:      # (1) 检查
        return {"ok": True}
    env = await _pool.get()            # (2) 借 —— 池空时在这里挂起
    _sessions[r.session_id] = _Session(env)   # (3) 登记
    return {"ok": True}
```

陷阱在检查 (1) 和登记 (3) 之间的那个 `await`:挂起期间,`session_id` 还**没进 `_sessions`**。
池耗尽下的时序(会话 `S`):

```
T0  客户端 /create(S) #0  → 协程 A:S 不在 → 卡在 await _pool.get()   (S 未登记)
T1  A 的 socket reset      → 客户端收到 TransportError → 重试
T2  客户端 /create(S) #1  → 协程 B:S 仍然不在 → B 也卡在 await _pool.get()
T3  池里还回来两个 env     → A 借到 env_a,登记 _sessions[S] = env_a
                            B 借到 env_b,登记 _sessions[S] = env_b   ← 覆盖
```

`env_a` 现在被**孤立(orphan)** —— 没有任何东西再引用它,也没有回收器 —— 于是池子的有效容量永久 −1。
一个长 run 里反复发生,池子干涸到 0,之后每个 `/create` 永远阻塞,**整个训练 run 挂死**(Layer A 的
`await env.reset()` 没有超时兜底)。

**关键、反直觉的事实:** 客户端连接 reset 时,服务端协程 A **不会被取消** —— Starlette/uvicorn 只置一个
"disconnected" 标志;正在跑的 ASGI 任务照样卡在 `await _pool.get()`。所以客户端的重试不是"替换"在飞的
请求,而是"又叠加"了一个并发的同会话 `/create`。

### 为什么原来的 guard 没挡住

handler 里本来就有 `if session_id in _sessions: return`,还配了注释 —— *"重试的 /create 绝不能借第二个
env —— 否则会 orphan 掉第一个、慢慢抽干池子。"* 注释里**描述的失效模式完全正确**,只是那个 guard 挡不住它。
它只覆盖了*完成之后丢响应*的情形(第一个 `/create` 已完成并登记,ok 响应丢了,重试发现 `S` 已在、直接短路)。
它**没**覆盖*在飞期间遇上池耗尽*的情形(第一个 `/create` 还 parked,`S` 不在,重试又借一次)—— 而这恰恰是池
耗尽制造出来的场景。经典的 check-then-act TOCTOU:检查和借+登记之间夹着一个 `await`,所以不原子。

### 修复

在阻塞借用**之前**先预留会话,并让重叠的调用者**等待**而不是去借:

```python
async with _create_lock:                       # 只护 O(1) 账本,绝不跨越借用
    if r.session_id in _sessions:
        return {"ok": True}
    ev = _pending.get(r.session_id)
    first = ev is None
    if first:
        _pending[r.session_id] = ev = asyncio.Event()   # 在 await 之前先占坑
if not first:
    await ev.wait()                            # 有并发的 create 正在借,等它
    return {"ok": r.session_id in _sessions}
env = await _pool.get()                        # 只有第一个调用者去借 —— 恰好一次
async with _create_lock:
    _sessions[r.session_id] = _Session(env)
    _pending.pop(r.session_id, None)
ev.set()                                       # 唤醒等待者
```

- 预留(`_create_lock` 下的 `_pending[S]`)在 `await _pool.get()` **之前**完成,所以重叠的重试能看见它,不会
  双重借。
- 重试**在 event 上等待**,只有 env 真正存在后才返回 ok —— 于是也不会"env 还没好就返回 ok"(那会让紧接着的
  `/reset` 404)。
- `/close` 现在在 `_create_lock` 下弹出会话,并在 `sess.lock` 下**归还**,所以 `/close` 撞上在飞的 `/step`
  不会把在用的 env 交给第二个会话。
- `/reset` 现在也持 `sess.lock`(之前没有),所以重试的 `/reset` 不会在这一集已经推进之后又重跑 `env.reset()`。

所有联邦异质性 / partition 代码**原样保留** —— 只动了池借用的并发。

### 为什么藏了这么久

这是一个负载 + 时序相关的竞态,表现为*挂起*(不崩溃、不出错数);它被一个"看起来正确"的 guard + 注释"防住"了;
而它只在满批次(池耗尽 + 频繁 reset)这个精确场景下才触发,恰恰是等价性 / 正确性验证不会碰的地方。已跑完的 run
的*结论*不受影响 —— 这个 bug 的风险是挂死,不是权重或 reward 被污染。

### 验证

`py_compile` 通过;幂等借用算法经过压力测试 —— 同会话风暴 → 恰好借一次;耗尽 + 重试 → 重试等待、单次借用;随机
create/step/close churn → 池计数守恒、零泄漏;锁序 `_create_lock → sess.lock` 从不嵌套(无死锁)。
