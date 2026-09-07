---
name: coev-oia-v1-implementation-audit
description: Fail-closed layer, data, gradient, mathematical, runtime, Git and foreground-training audit for COEV-OIA on the current tida_site_v20 worktree. Does not certify empirical improvement.
---

# COEV-OIA v1 严格实现审查 Skill

安装目标：当前worktree的`.codex/skills/coev-oia-v1-implementation-audit/SKILL.md`。本文件应与研究设计、实施合同、固定config一并读取。不得仅复制name后继续按旧TIDA Skill执行。

## 0. 最高优先级边界

用户本轮要求：当前worktree、tida_site_v20、不新增worktree；任务从头训练，只允许通用DINO预训练；原始RGB无派生cache；48GBGPU真实profile；每轮test、test选best；按`2026-09-07-coev-epoch-budget-amendment.md`前台监督18轮，不因指标弱随意终止。

旧Skill的“新worktree、冻结旧任务模型、885test、history_off复现旧image、10epoch+StageC”等条款**只适用于旧任务**。本次发现这些语义进入新factory或audit，直接FAIL。

禁止修改用户三份canonical MD旧历史；仅追加到`E:\sbw\FATE_Drive\task_plan.md/findings.md/progress.md`。计划和审查规范不是第二套状态日志，允许版本化保存。所有安全约束仍适用：不可把“不能随意停止”理解为必须执行NaN更新、无空间写盘、无限重试或绕过权限。

## 1. 审查只证明可审查的事实

将结论分成三层，禁止混写：

1. IMPLEMENTED：正式代码确实实现约定公式/形状/路径。
2. FUNCTIONAL：真实输入下计算被执行、参数按约定收到梯度并更新、干预/数学性质正确。
3. EMPIRICALLY_USEFUL：完整真实数据上的比较支持收益。

FULL_TRAIN_READY只涉及前两层和资源/身份条件，不保证第三层。不能把文件存在、577个旧tests通过、非零attention或单个正向case当泛化提升证明；也不能要求随机初始化模型在三步smoke就超过训练成熟的强基线。

## 2. 必读与设计复核

读canonical三MD的相关新增部分、当前AGENTS、01/02/04/05、YAML、新旧Skill、真实branch diff。记录已读文件sha256与章节，不抄写“已读”而不提取冲突。

至少做三轮不同问题的审查：
- 第1轮：任务语义/研究边界。逐项质疑交通量纲、样本标签输入、具名factor是否偷读全图、test选模口径。
- 第2轮：数学/实现。逐项检验actual dt、S2半项、面积符号、持续状态、缺失段、窗口边界、loss归约、梯度owner。
- 第3轮：正式运行。以实际launcher/factory验证真实DINO、当前worktree、no cache、GPUprofile、恢复、Git完整性、前台父子进程。

有独立review agent可让它只读复核；没有就明确是同agent的分阶段审查，不能虚称独立审核。问题必须逐条关闭并链接测试，而不是反复说APPROVED。若设计文件互相矛盾，先修订同一版本并重审，不能静默选择容易的实现。

## 3. 状态机

```
CONTEXT_LOCKED
→ DESIGN_REVIEW_PASS
→ IMPLEMENTATION_REVIEW_PASS
→ FUNCTIONAL_REVIEW_PASS
→ MEMORY_REVIEW_PASS
→ SOURCE_SYNC_VERIFIED
→ FULL_TRAIN_READY
→ TRAIN_RUNNING
→ TRAIN_COMPLETED
```

故障可转`REPAIRING`或`BLOCKED_EXTERNAL`；非自然完整结束不得写TRAIN_COMPLETED。METHOD_EFFECTIVE是完成后独立结论，不能由状态机自动产生。

审查产物`.review/coev_v1/*.json`至少含：spec/Skill/config/data/source hashes、当前HEAD/tree、required-source清单、环境、真实命令/退出码、每个R01–R40的symbol/test/artifact/数值/判定。源文件更改后旧PASS失效；文档/日志更改与训练计算源更改要按hash清单区分，不能随意修改旧审查metadata蒙混。

## 4. 静态审查不是全部，但必须先过

- 正式factory只实例化CoEVOIAModel；禁止旧AIE/VETRA/image_base/refiner/baseline_reuse/calibrator进入主路径。
- 只允许读取通用DINO预训练和本次run的resume checkpoint。torch.load路径记录；新run的任务参数随机初始化，不能strict=False吞掉未知任务权重。
- 输入CoEVInputs与CoEVTargets分离。模型参数不得含action/reason GT，clip_meta中也不得藏标注。
- 原始RGB路径可用；拒绝feature_cache/token_store/track_store/frame_store配置。审查运行I/O而不是grep单词cache。
- DINO1–8 frozen、9–12 task-trainable；4/8/12明确1-based；测量norm固定；一个DINO实例。
- 8个空间map是sigmoid不是空间softmax；事实缺失mask unknown；颜色/线型属性有独立来源。
- PathLift包含状态、精确M、S2反对称A与真实时钟；不允许以均值/协方差/abs(A)替代。
- 238坐标由两个共享basis处理，不生成210个独立专家或pair memory。
- 命名basis无全图CLS/第三观测输入。zV作为未命名项公开。
- 最终logits无cap、utility、no-harm许可和部署scale。
- 所有loss加入total且在autograd内；optimizer exact-cover。
- test0.5固定阈值，best视图唯一z_final；test_selected=true/publication=false。
- launcher仅前台；无Start-Process detached、TaskScheduler、Win32Process、nohup、daemon、shell后台操作。

## 5. 动态测试最低要求

所有02文档R01–R40均为必填；不允许skip/xfail关键测试。第三方旧测试不适配可单独记录，但新核心suite必须0skip、0xfail。

### 5.1 数学算子（独立oracle）

A. square forward/reverse有向面积+1/−1；末帧、每坐标值集合及均值相同。
B. 常量路径持续状态/M非零，S1/A为0。
C. 任意线段细分后的S1/S2/M一致。
D. 相同位移、不同dt，物理运动代理按1/dt变化；有时钟特征区分实际持续时间。
E. 相机整体平移与独立物体移动可区分；补偿两对照使用同定义。
F. 缺失中间点不跨gap积分；两条分段轨迹不能凭首尾连接生成耦合。
G. 所有invalid、单帧、无车辆有效0、对齐失败unknown皆finite且语义不同。
H. window边界仅在有效段插值；mean/M按观测时长归一化并报告coverage。
I. 第三观测改变不影响指定pair features/basis；Jacobian非所属坐标=0。
J. 读出贡献重构与节点删除精确；原始输入删除须重算不能复用states。

附带CPU参考代码只是上述算子的oracle，不能代替生产实现测试或真实GPUprobe。

### 5.2 梯度与更新

每个loss分别调用autograd.grad，生成owner×loss矩阵。允许梯度必须在对应有监督/有效输入的至少两个独立batch上finite且非零；禁止路径应为None或严格0。记录norm相对参数尺度，非零不等于正常，爆炸梯度也FAIL。

必须观测：DINO9/10/11/12分别更新；frozen prefix及测量norm不变；Action/Reason各自readout只收自己loss；共享query/basis收两个loss；predicate只收ground；matcher只收match。

证据basis普通非零初始化后，第一批有效证据应有梯度；不能全zero-init造成首步链路断开又写首步全部非零。缺失证据的batch允许该部分零，但不能用这作为所有真实batch均零的解释。

### 5.3 正式factory和输入依赖

装饰hook记录正式launcher实际到达的模块、调用数与shape。对真实train clip分别修改早期/中期/末期历史、标签、weak GT：
- 改RGB可改变历史表示；全部历史关掉走同一新模型。
- 改Action/Reason GT不改变相同输入的eval logits。
- 移除test侧弱标注读取权限不影响test forward。
- 改ground GT只改训练loss，不直接改预测。
- 不允许返回固定张量/读取旧saved logits。

## 6. 必须做的变异测试（防“审查程序也错了”）

在测试副本/内存monkeypatch中故意破坏，至少以下10项必须被捕获；不创建新worktree，不永久修改生产源码。

1. 给upper DINO整体套no_grad。
2. 把actual dt换成常量1。
3. 把A设为0或abs(A)。
4. 把空间sigmoid改成softmax。
5. 移除S2的0.5*d⊗d半项。
6. 跨invalid gap继续累计前缀。
7. 将历史真实输入替换末帧重复而保持valid全true。
8. 把Reason标签放进forward。
9. 只做编码后attention删除、不重算历史。
10. 将主输出偷换visual_only/旧baseline。
11. 从错误latest/EMA视图选best。
12. 在训练第3轮退出时写TRAIN_COMPLETED。
13. 漏掉predicate/matcher optimizer group。
14. 让测试标签影响下一轮lr或采样。

输出mutation_kill_report：每个变异触发哪个测试，未捕获即FAIL。不能只测试显而易见shape错误；至少6个变异应保持shape合法。

## 7. 真实RGB功能验证

使用训练集固定128条、真实DINO和配置规定的9帧，至少100次optimizer updates；所有任务和辅助loss均为正式配方，schedule分母仍是正式18轮、3600 total updates。probe模型/optimizer与正式模型分开，probe结束丢弃；正式训练重新按seed构造，不能偷偷从probe继承任务权重。

检查实测覆盖、更新、finite、输入依赖、干预重算。可以做32条小集合可拟合性测试，但不能要求无效/未知观察预测所有标签；对不充分证据项只验直接梯度和有监督的映射，不用全任务高分假门槛。

需抽查至少16个真实clip的帧、predicate map、对应/背景箭头、14条曲线和有效mask。人工/内置视觉检查记录具体ID与错误，不用大量OCR。若显著坐标翻转、颜色错判来自parser、时间不正确，修复后重跑。未训练map较弱不等于实现错误，不通过人为阈值制造“已经有高质量语义”。

## 8. 显存和效率

实际5880整卡上profile，不用CPU/mock推断45GiB。记录NVML/device used、allocated、reserved以及系统报的单位。目标43–45GiB是预算上限附近偏好，不是最低验收值。

每候选10 warm-up+至少30 measured optimizer updates做筛选；最终选中的配置200updates压力测试并包含一个完整尺寸test forward。全模块/真实RGB/正式反向图；peak reset前CUDA synchronize；记录吞吐中位数/P95、内存增长、解码比例。

允许调整microbatch与accum（有效batch32）、chunk、SDPA/checkpoint、workers。禁止为了过审减少帧数/分辨率/模块/上层梯度或改走feature cache。低于目标但吞吐更好选择它；不分配空张量填显存。

## 9. Git与前台绑定

检查required new sources全部`git ls-files`可见，注意已有.gitignore可能屏蔽新models/datasets文件。只提交本任务必要source/config/tests/docs/Skill；不stage数据、checkpoint、runtime tensors、凭据或用户无关dirty修改。

push同一tida_site_v20，读取实际remote URL决定使用origin或github，不硬编码错误remote。`git ls-remote <remote> refs/heads/tida_site_v20`必须等于本地提交。拒绝把push exit0等同remote identity而不读取。权限失败不绕过、不假称同步；按04记录外部阻塞。

source/config/spec/data/Skill hashes一致后才输出FULL_TRAIN_READY。正式引擎启动再次校验；任何关键计算源变化须重审受影响合同与identity，不手改PASS来迁移。

## 10. 训练监督与完成

保持前台附着父子进程，stdout/stderr持续消费；warnings不是失败，Python真实exitcode才是进程结果。每20updates/评估32batches输出heartbeat；监控data/fwd/backward/保存阶段，不能因评估沉默误杀。

18轮每轮test完整，昂贵交通干预每轮固定512条且最终best全量复测；检查模型、sampler、scheduler和source hash；不按test走势改结构/超参，不因早期弱结果擅停，不因已达目标提前结束。任务故障暂停更新、保全状态、修复最小实现错误并恢复；超过安全恢复范围记BLOCKED_EXTERNAL，而不是无限重启或空口“继续监督”。

TRAIN_COMPLETED要求：18/18有效epoch、18次完整test指标、best_joint严格可加载且视图一致、最终best完整交通诊断、所有阶段真实exitcode、最终Git与artifact检查。指标是否超过目标单独报告，绝不自动生成GOAL_ACHIEVED。

## 11. 输出JSON合同示意（字段说明，不是预置PASS）

```json
{
  "schema": "coev_audit_v1",
  "stage": "FUNCTIONAL_REVIEW",
  "status": "PASS_OR_FAIL_FROM_REAL_RUN",
  "source_commit": "actual",
  "required_source_hashes": {},
  "config_sha256": "actual",
  "spec_sha256": "actual",
  "skill_sha256": "actual",
  "data_identity_sha256": "actual",
  "requirements": {
    "R18": {"symbol": "actual", "test": "actual", "artifact": "actual", "observed": {}, "pass": false}
  },
  "mutation_kill_report": "actual_artifact_path",
  "empirical_improvement_claimed": false
}
```

不得复制示意中的actual字样生成正式记录；正式audit验证这些字段都是实际hash/已存在artifact/实际值。无证据就FAIL或NOT_RUN。
