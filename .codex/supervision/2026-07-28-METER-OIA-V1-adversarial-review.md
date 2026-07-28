# 鍙屼唬鐞嗙洃鐫ｆ棩蹇楋細METER-OIA V1 瀵规姉寮忓疄鐜板鏍?
**鏃ユ湡锛?* 2026-07-28
**鐘舵€侊細** 淇瀹屾垚锛岀瓑寰?clean-HEAD real-DINO profile 涓庢渶缁?audit
**涓绘墽琛岀锛?* 褰撳墠 Codex 涓讳細璇?
**鐩戠潱绔細** Helmholtz锛宎gent `019fa82b-3269-7c01-90fa-823900e1bf53`

## 1. 鐢ㄦ埛鍘熷瑕佹眰

鏍规嵁 `Codex_METER_OIA_V1_ImplementationPlan_20260728.md` 缁х画瀵规姉鎬у鏌ュ苟瀹屾暣瀹炵幇浠ｇ爜銆傞獙鏀舵爣鍑嗕笉鏄€滃彲浠ヨ缁冣€濓紝鑰屾槸璁″垝鍔熻兘鍏ㄩ儴杩涘叆姝ｅ紡 forward銆乴oss銆乷ptimizer銆乪valuation銆乤rtifact銆乺esume 鍜?supervisor 璋冪敤閾撅紱涓嶅緱閬楁紡銆佸崰浣嶃€侀敊璇疄鐜版垨鐣欎笅閫昏緫鍐茬獊銆傛湰杞姝㈠惎鍔?pilot/full train銆?
## 2. 鍥哄畾杈圭晫

- worktree锛歚E:\sbw\FATE_Drive\fate_oia_acpr_meter_oia_v1_worktree`
- branch锛歚acpr_meter_oia_v1_direct_image`
- base锛歚acpr_calalign_v1_2@373aa49feac17372574fd7fb056c1d79c7c848fe`
- 淇濈暀 frozen DINO銆乨irect image銆?60x640銆佸畬鏁?3600 patches銆?- 绂佹 cache銆乧ompression銆乸air memory銆乭ard pair銆乤ction-set final銆乼rainable threshold/calibration representation path銆乬raph/PMI/co-occurrence銆佸巻鍙?checkpoint distillation銆?- 鏈疆涓嶅惎鍔ㄨ缁冦€?
## 3. 鐩戠潱瀹℃煡缁撹

鐩戠潱绔粨璁轰负 `changes_required`锛屼笉鍏佽 pilot/full train銆傚叧閿彂鐜帮細

1. grounding evidence objective 浼氭帹鍔?support/counter score 鐩哥瓑锛屼笖 mirror loss 缂哄け锛?2. meta utility 灏嗗悓涓€涓叏灞€ utility 澶嶅埗缁欏洓涓?factor锛屾病鏈夐€?factor 铏氭嫙鏇存柊锛?3. calibration 鍙湁 prevalence matching锛屾病鏈?temperature銆乬lobal/group shrinkage銆佸洖閫€鍜?mAP/RMS 绾︽潫锛?4. full train 鍙彧鎸佹湁 PRE_PILOT_READY 鑰岀粫杩?pilot锛?5. PU audit 娌℃湁鍦?deliberately hidden positives 涓婅绠?eligibility锛孭U 鍙堣閲嶅涔?lambda锛?6. reason zero 鏉冮噸娌℃湁 observability锛?7. counterfactual 娌℃湁绱璐ㄩ噺閫夋嫨銆佷弗鏍?non-selected neighbor replacement 鍜?target-specific wrong factor锛?8. private reason decoder 缂?reason self-attention銆佸彲瀛︿範澶氬眰铻嶅悎鍜?step-0 candidate 鐩戠潱锛?9. optimizer/runtime 缂?no-decay銆乪ffective-batch LR scaling銆乀F32銆佺湡瀹?profiler 鍜岀湡瀹?owner delta锛?10. source hash銆乵icro-step resume銆乻tandalone evaluator銆乫ailure/evidence case artifacts 涓嶅畬鏁淬€?
## 4. 宸查噰绾冲苟瀹屾垚鐨勪慨澶?
- grounding 鏀逛负 source-confidence 鍔犳潈鐨?map NLL銆乸resence/signed-direction margin銆乵irror balance 鍜?compactness銆?- meta utility 鏀逛负閫?factor 琛?mask銆侀€?factor candidate銆侀€?factor utility/EMA/omega銆?- PU 浣跨敤瀹屽叏 detached 鐨?private decode锛泃rainer 涓嶅啀浜屾涔?lambda銆?- PU eligibility 鏀逛负 deliberately hidden positive 瀵?originally observed-zero 鐨?AUPRC 姣旇緝銆?- reason negative weight 浣跨敤 `0.1 + 0.3 * observability * (1-evidence)`銆?- private reason 鏂板涓€灞?reason self-attention銆佷笁灞傚彲瀛︿範 router锛屽苟璁?candidate 浠?step 0 鎺ュ彈涓?reason loss銆?- counterfactual 浣跨敤绱 evidence mass銆佸悓 sector/鐩歌繎绾靛悜/浣庡搷搴?control銆佹帓闄?selected patch 鐨?3x3/5x5 neighbor replacement銆乼arget-specific factor/wrong factor銆?- grounding 鍜?counterfactual 浣跨敤鐙珛 5%/10% update ramps銆?- calibration 鎼滅储 temperature銆乬lobal銆乬roup銆乬roup-shrinkage銆乸er-label锛涗繚鎸?mAP锛屼笉瓒呰繃 threshold RMS 闄愬埗锛岄€€鍖栨椂鍥為€€ group/global/raw銆?- metrics 澧炲姞 per-label AUC 鍜?mAUC锛沞valuator 澧炲姞璁″垝瑕佹眰鐨?branch aliases 涓?selector visual/semantic 闅旂銆?- optimizer 鎸?owner 鍒嗙粍锛宯orm/bias/embedding no decay锛孡R 鎸?effective batch / 32 缂╂斁锛屽惎鐢?TF32銆?- runtime log 澧炲姞鐪熷疄 parameter delta銆乷wner optimizer step count銆亃ero-gradient rate 鍜屽垎娈垫椂闂淬€?- profiler 鏀逛负鐪熷疄鍥惧儚銆佺湡瀹?DINO銆? warmup + 20 measured optimizer updates锛屽苟鍗曟祴 counterfactual/meta/calibration 浜嬩欢骞舵寜棰戠巼鎽婇攢銆?- standalone evaluator 鐪熷疄鍔犺浇 checkpoint/test dataset 骞跺啓 evaluation summary銆?- failure/evidence cases 鏀逛负鐪熷疄鏍锋湰璁板綍锛屼笉鍐嶅啓璇存槑鏂囧瓧鍗犱綅銆?- checkpoint source hash 鏀逛负褰撳墠瀹炵幇 HEAD锛涘鍔?deterministic epoch loader銆乵id-epoch checkpoint 鍜?micro-step resume銆?- pilot 鍥哄畾瑙勬ā鎭㈠涓?`4096/1024/512/512`銆?- trainer銆丳owerShell銆乻upervisor 涓夊眰閮借姹?full 妯″紡鎸佹湁 `METER_OIA_V1_FULL_TRAIN_READY.json`銆?- 3-epoch pilot 鍙湁閫氳繃 action/reason/meta/evidence/鏄惧瓨/GitHub HEAD 鍏ㄩ儴闂ㄦ鎵嶇敓鎴?FULL_TRAIN_READY銆?- audit 澧炲姞涓婅堪璇箟鐨勫姩鎬佸拰婧愮爜 hard checks锛宺eal-DINO 鍔ㄦ€佹鏌ヤ娇鐢ㄧ湡瀹?backbone銆?
## 5. 瀵规姉娴嬭瘯涓庡鏍歌疆娆?
| 杞 | 璇佹嵁 | 缁撹 |
| --- | --- | --- |
| RED 1 | 7 涓柊澧炶涔夋祴璇曞叏閮ㄥけ璐?| 璇佹槑 optimizer銆乨ecoder銆丳U銆乧alibration銆丄UC銆乫ull gate 鍧囦负鐪熷疄缂哄彛 |
| 淇鍥炲綊 1 | targeted `32 passed` | 绗竴鎵硅涔変慨澶嶆垚绔?|
| 鍏ㄩ噺鍥炲綊 1 | METER `64 passed` | METER 鏃ф祴璇曚笌鏂板娴嬭瘯鍏煎 |
| 鍏ㄩ噺鍥炲綊 2 | 鍏ㄤ粨 `154 passed` | ACPR/FATE 鏃ц矾寰勬湭琚牬鍧?|
| Resume 鍔犲浐鍚?| 鍏ㄤ粨 `155 passed` | deterministic next-update/micro-step 鍚堝悓鎴愮珛 |
| Mock dynamic audit | 鎵€鏈?functional/contract/dynamic checks 涓?true | 浠呭洜 real-DINO profile 灏氭湭鎵ц鑰屾纭嫆缁?PASS |

## 6. 璁″垝淇濈湡涓庡啿绐?
- 鏈浛鎹㈢爺绌朵富绾匡紝鎵€鏈変慨澶嶉兘灞炰簬鍘?METER 璁″垝鍚堝悓鐨勮惤瀹炪€?- calibration 鐨?temperature 浼氫娇 deploy 鍏紡鎴愪负 `logits / temperature - theta`锛涙棫娴嬭瘯鍙帴鍙?`logits - theta`锛屼笌鏈鍒掆€滄悳绱?threshold 鍜?temperature鈥濆啿绐併€傚凡鎸夊綋鍓?METER 璁″垝鏇存柊鏃ф祴璇曪紝鍚屾椂淇濈暀 raw/deploy 鍒嗙鍜?mAP 涓嶅彉鎬с€?- 鐩戠潱绔缓璁仮澶嶅畬鏁?pilot 鏍锋湰瑙勬ā锛屽凡閲囩撼銆?- 鏈噰绾充换浣曟斁瀹?gate銆佽烦杩?pilot 鎴栫洿鎺ュ惎鍔?full train 鐨勫仛娉曘€?
## 7. 鍓╀綑纭棬妲?
1. 鍦ㄦ彁浜ゅ悗鐨?clean HEAD 涓婇噸璺戝叏浠?tests銆?2. 鎵ц涓ユ牸 real-DINO/real-data runtime profile銆?3. 鐢?clean HEAD 鍜?profile 閲嶈窇 real-DINO implementation audit銆?4. REVIEW_PASS/PRE_PILOT_READY 蹇呴』缁戝畾褰撳墠 clean HEAD銆?5. 鏇存柊 canonical `task_plan.md`銆乣findings.md`銆乣progress.md`锛屾彁浜ゅ苟鎺ㄩ€侊紱鏍搁獙 GitHub HEAD銆?
## 8. 褰撳墠鍒ゅ畾

- **浠ｇ爜璇箟瀹℃煡锛?* 宸插畬鎴愮洃鐫ｇ鎻愬嚭鐨勪慨澶嶏紝mock dynamic audit 鍏ㄩ儴閫氳繃銆?- **鍙姤鍛婃渶缁堝畬鎴愶細** 灏氫笉鍙紱缂?clean-HEAD real-DINO profile/audit 璇佹嵁銆?- **鍏佽 pilot锛?* 灏氫笉鍙€?- **鍏佽 full train锛?* 涓嶅彲锛涘繀椤诲厛瀹屾垚涓ユ牸 3-epoch pilot 骞剁敓鎴?FULL_TRAIN_READY銆?
## 9. Final evidence update

- Final clean HEAD: `8e1c066bf026767bd83ee2210b69fa193f6fc966`; GitHub branch HEAD matches.
- Full repository verification: `156 passed`; one unrelated existing TypedStorage deprecation warning.
- Real-DINO runtime profile completed with 5 warmup and 20 measured optimizer updates per accepted comparison. Selected batch 6, accumulation 5, workers 4, prefetch 2; peak reserved 42.3652 GB; event-adjusted 10.2270 samples/s.
- Final real-DINO audit: `pass=true`, `missing_items=[]`, `warnings=[]`; all functional, contract, and dynamic checks passed.
- `REVIEW_PASS_METER_OIA_V1.txt` and `METER_OIA_V1_PRE_PILOT_READY.json` were generated.
- Supervisor verdict: implementation and pre-pilot audit closure are complete. Pilot/full training were not started. Full training still requires a strict pilot-generated `FULL_TRAIN_READY`.

## 10. Pilot 澶辨晥鍚庣殑鏍瑰洜淇澶嶅

**澶嶅鏃ユ湡锛?* 2026-07-29
**鐩戠潱绔細** `019fa909-9501-7a13-9abf-a37411fbf6ee`
**澶嶅鐘舵€侊細** 浠呮壒鍑嗚繘鍏ユ柊鐨?real-DINO smoke/pilot锛屼笉鎵瑰噯鐩存帴 full train銆?
涓ユ牸 pilot 璇佹槑瑙嗚涓诲共鑳藉瀛︿範锛屼絾鍒涙柊閾惧瓨鍦ㄧ湡瀹炲け娲伙細support/counter null 涓?0銆佷簩鑰?cosine 绾?0.99锛宻emantic contribution RMS 涓?0锛宻emantic AP 璺?epoch 鎭掑畾锛宻elector 鎺ヨ繎 0.99 visual锛宮eta utility/omega 涓?0锛宑ounterfactual 浠呰鐩?2 涓?action 鍜?1 涓?factor銆?
鎵ц绔寜 TDD 淇浠ヤ笅鏍瑰洜锛?
- signed factor 灏?3600 patch 鍒嗗竷涓庣嫭绔嬪彲瀛︿範 null mass 鍋氳仈鍚堝綊涓€锛岀户缁弧瓒?`patch_map.sum(-1) + null_mass = 1`銆?- support/counter query embedding 浣跨敤骞呭害鏇村己涓斾弗鏍肩浉鍙嶇殑鍒濆鍖栵紝淇濈暀 `q+=H+e+`銆乣q-=H+e-` 鍚堝悓銆?- semantic action 鐨?21-factor 鍒嗗竷涓庣嫭绔嬪彲瀛︿範 null mass 鑱斿悎褰掍竴锛岄伩鍏?entmax 璁粌鍚庨€€鍖栦负 null-only bias銆?- 淇濈暀 softmax 鍒?entmax 鐨勫墠 10% progress 杩囨浮銆佸畬鏁?additive semantic expert銆乻elector銆乵eta utility 鍜?counterfactual 鏁版嵁娴併€?
鐩戠潱绔唬鐮佺骇缁撹锛?
- 淇淇濇寔 METER 璁″垝鐨勫綊涓€鍖栥€佺█鐤忓寲鍜?additive contribution 璇箟銆?- 鏂?maps/null 鐪熷疄杩涘叆 reliability銆乻emantic action銆乺eason local銆乵eta utility 鍜?counterfactual銆?- 鏆備笉淇敼 counterfactual selection锛涘厛瑙傚療涓婃父璐＄尞鎭㈠鍚庢槸鍚﹁嚜鐒惰揪鍒?4 actions / 12 factors锛岄伩鍏嶄负杩?gate 浼€犺鐩栥€?- 鑻?null銆乵ap 鍒嗙銆乻emantic contribution 宸叉仮澶嶈€?CF 浠嶉暱鏈熶綆瑕嗙洊锛屾墠鍏佽灏嗗叾鍒や负鐙珛閲囨牱鍋忕疆骞舵寜 TDD 淇銆?
**鎵ц绔噰绾虫儏鍐碉細** 鍏ㄩ儴閲囩撼銆?
**鏂伴矞楠岃瘉锛?* 鍏ㄤ粨 `164 passed`锛屽彟鏈?1 涓笌鏈换鍔℃棤鍏崇殑 TypedStorage 寮冪敤璀﹀憡銆?
**涓嬩竴闂ㄦ锛?* clean commit/push -> real-DINO audit/readiness -> 鐪熷疄 smoke/pilot 鏈哄埗鏁板€奸獙璇?-> full train銆?
## 11. strict pilot 澶辫触鍚庣殑淇鏂规鎵ц鍓嶅鎵?
**澶嶅鏃ユ湡锛?* 2026-07-29
**缁戝畾 HEAD锛?* `e71340f22bae7525ecfd1115564eba367dc71135`
**澶嶅绫诲瀷锛?* 鍙鏂规瀹℃壒锛涙湭淇敼妯″瀷浠ｇ爜锛屾湭鍚姩璁粌
**缁撹锛?* `changes_required`銆傚厑璁稿湪涓嬪垪纭慨璁㈠畬鎴愬悗杩涘叆 RED tests/瀹炵幇/profile/audit/exact 3-epoch pilot锛涗笉鍏佽鐩存帴 full train銆?
### 11.1 淇濈湡缁撹

- 淇濈暀瀹屾暣 semantic expert `z_sem = bias + sum(c)`銆乣action_logits_semantic`銆乫actor contributions 涓?`0.40` direct ASL锛岀鍚堝師璁″垝鈥滃畬鏁翠笓瀹惰€岄潪 residual鈥濈殑杈圭晫銆?- 鏂板鐙珛 bounded semantic transport 鐢ㄤ簬 selector/final锛屽睘浜庝慨澶嶄笓瀹惰兘鍔涗笌浼犺緭棰勭畻娣风敤锛屼笉鏄墛寮?semantic expert锛屼篃涓嶆槸鏀惧 `[0.03,0.30]` gate銆?- selector regret 鏀逛负 detached experts銆乬ate-only gradient锛況eason mix 鏀逛负 detached experts銆乬ate-only gradient锛沜ounterfactual 鏀逛负 signed attribution锛沵eta 涓嶅己寮€ omega锛屽潎绗﹀悎 pilot 澶辫触鐨勬満鍒惰瘉鎹€?- 鍘熻鍒?gate 鏁板€间繚鎸佷笉鍙樸€傚彧鏈夊叏閮ㄥ師 gate 鍦?clean exact pilot 涓婇€氳繃锛屾墠鍏佽 full train銆?
### 11.2 鎵ц鍓嶅繀椤诲啓姝荤殑淇敼

1. **Transport 鍏紡涓庡懡鍚嶅繀椤诲垎绂汇€?* 淇濈暀 `action_logits_semantic=z_sem`銆傚彟杈撳嚭 `semantic_transport_delta` 涓?`semantic_transport_logits=z_vis+delta_transport`锛泂elector/final 鍙兘璇诲彇 transport logits锛宻emantic compatibility/mAP 浠嶅彧璇诲彇瀹屾暣 `z_sem`銆俛rtifact 鍚屾椂璁板綍 full-semantic/visual ratio銆乼ransport/visual ratio銆乻cale 涓?clamp saturation銆?2. **绂佹 batch-dependent 鎺ㄧ悊銆?* `(z_sem-z_vis)` 鐨?per-action detached RMS 涓嶅緱鐢卞綋鍓?eval batch鐩存帴鍐冲畾銆傚繀椤讳娇鐢ㄤ粎鐢?train 鏇存柊銆乧heckpoint 鎸佷箙鍖栥€乪val 鍐荤粨鐨?per-action running RMS锛屾垨绛変环鐨勬牱鏈棤鍏冲浐瀹氱粺璁°€倀est 涓嶅緱鏇存柊缁熻銆俁ED test 蹇呴』楠岃瘉鍚屼竴鏍锋湰鍗曠嫭鎺ㄧ悊涓庝笉鍚?batch 缁勫悎鎺ㄧ悊缁撴灉涓€鑷淬€?3. **Transport 鍙仛涔樻€ф湁鐣岀缉鏀俱€?* 浼樺厛瀵?delta 浣跨敤 detached per-action scale 骞堕檺鍒舵渶缁?RMS ratio锛涗笉寰楃敤閫愬厓绱?hard clamp 澶ч噺鎴柇鎺掑簭銆傚繀椤昏褰?scale 涓婁笅鐣屽懡涓巼锛岄伩鍏嶈〃闈㈣繃 gate銆佸疄闄?delta 鍏ㄩケ鍜屻€?4. **Selector regret 蹇呴』鐪熸 selector-only銆?* selector features銆乿isual logits銆乼ransport logits鍏ㄩ儴 detach锛屼粎 selector 鍙傛暟淇濇寔姊害锛涗娇鐢ㄤ笌姝ｅ紡 action ASL 鐩稿悓鐨勯€愭爣绛?positive/negative surrogate銆俁ED test 蹇呴』璇佹槑 regret 瀵?semantic/visual expert 姊害涓ユ牸涓?0銆佸 selector 姊害闈?0銆傚畬鏁?final/action direct losses浠嶆寜鍘熻鍒掕缁冨悇鑷垎鏀€?5. **Local 绛変环鍒濆鍖栭渶鏄庣‘鏄犲皠銆?* 褰撳墠 local 璺緞娌℃湁鐙珛 K/V锛屼笉鑳藉彧鍐欌€滃鍒?K/V/head鈥濄€傚疄鐜板繀椤讳簩閫変竴锛氭柊澧炶鍒掑厑璁哥殑 local K/V attention 骞朵粠 foundation K/V/head澶嶅埗锛涙垨鏄庣‘鎶?foundation value/head/norm 鏄犲皠鍒?`local_proj/local_head/local_norm`锛屽苟灏嗛澶?`factor_proj/action_proj` 鍒濆鍖栦负闆舵畫宸€傚繀椤荤敤 RED test璇佹槑 local 涓嶆槸闅忔満灏哄害鍚姩銆?6. **Reason mix regret 蹇呴』淇濇寔 PU 璇箟銆?* 浣跨敤涓庢寮?`weighted_reason_asl` 鐩稿悓鐨?confidence/observability/unknown-negative 鏉冮噸锛岃€屼笉鏄櫘閫?BCE/纭礋 ASL銆俫lobal/local logits涓?gate features detach锛屽彧鍏佽 mix gate鎺ユ敹璇?regret姊害銆傝褰?gate saturation銆乬lobal/local/mix per-label AP涓?mix regret銆?7. **Signed patch attribution 蹇呴』瀵瑰簲 score head銆?* 瀵?support/counter 鍒嗗埆璁＄畻鍚勫眰 `attention * value-to-score-head` 鐨?pre-softplus signed contribution锛屽啀鎸?layer weight鍚堝苟锛涗笉鑳界户缁妸绾?attention map褰?score attribution銆俷ull contribution鍗曠嫭璁板綍銆?8. **Support/counter 蹇呴』鍒嗗紑閫?factor銆?* support 鍙粠瀵圭洰鏍?action 涓烘璐＄尞涓?support valid 鐨?factor 涓€夛紱counter 鍙粠璐熻础鐚笖 counter valid 鐨?factor 涓€夈€傛棤鍚堟牸 factor 蹇呴』 skip锛岀姝㈠洖閫€鍒?`abs(contribution)` argmax銆俢overage balancing 鍙兘鍦ㄥ悎鏍?factor 闆嗗悎鍐呰繘琛屻€?9. **鏂瑰悜蹇呴』鍙岄噸楠岃瘉銆?* support 鍒犻櫎搴斾娇 support score涓嬮檷涓旂洰鏍?action logit涓嬮檷锛沜ounter 鍒犻櫎搴斾娇 counter score涓嬮檷涓旂洰鏍?action logit涓婂崌銆俿elected/control 浣跨敤鍚?patch 鏁板拰鍚屾浛鎹㈣鍒欙紝artifact 鍒嗗埆璁板綍鍥涗釜鏂瑰悜鍙?skip reason銆?10. **Meta readiness 鏀逛负 robust 璇佹嵁銆?* `meta_positive_utility` 涓嶅緱鍐嶇敤浠绘剰 `utility_ema>0`锛涘繀椤昏姹?admission score>0銆丩CB瓒呰繃 matched-null q99涓旀棤 resolution failure銆俙meta_high_omega_4/meta_share_rate` 鍘熼棬妲涗繚鎸併€備粎鏈?EMA 姝ｅ彿鏃?omega 蹇呴』缁х画涓洪浂銆?
### 11.3 蹇呴』鏂板鐨?RED/楠岃瘉璇佹嵁

- 瀹屾暣 semantic expert forward/mAP 涓?transport forward涓ユ牸鍒嗙銆?- transport batch-invariance銆乼rain-only running-stat銆乺esume涓€鑷存€ф祴璇曘€?- selector-only 涓?mix-gate-only gradient ownership娴嬭瘯銆?- local initialization闈為殢鏈哄昂搴︿笌棣栨鏈夐檺姊害娴嬭瘯銆?- synthetic signed attribution娴嬭瘯锛氬凡鐭ユ/璐?patch琚纭€夋嫨锛屽垹闄ゆ柟鍚戠鍚堝畾涔夈€?- support/counter 鐙珛 factor selection銆乪ligible-only coverage涓庢棤鍚堟牸椤?skip娴嬭瘯銆?- meta EMA 寰浣?robust admission澶辫触鏃?readiness浠嶅け璐ョ殑娴嬭瘯銆?- 鍏ㄦ祴銆乺eal-DINO profile銆乮mplementation audit銆乧lean-HEAD缁戝畾鍚庯紝浠庡ご杩愯 exact 3-epoch pilot锛涚姝?resume 鏃?pilot鍐掑厖鏂扮粨鏋溿€?
### 11.4 鎵ц璁稿彲

- **鍏佽鎵ц锛?* RED tests -> 浠ｇ爜淇 -> targeted/full tests -> real-DINO profile/audit -> clean exact 3-epoch pilot銆?- **涓嶅厑璁告墽琛岋細** 鐩存帴 full train銆侀檷浣庢垨鏀瑰啓鍘?gate 鏁板€笺€佸己鍒?omega銆佺敤 test 鏇存柊 RMS/meta/mix/calibration銆佷负 coverage 杞鏃犳晥 factor銆?- **full train 鏉′欢锛?* 鏂?clean HEAD 涓婂叏閮ㄥ師璁″垝 gate閫氳繃骞剁敓鎴愬搴?FULL_TRAIN_READY锛涘惁鍒欑户缁繚鎸侀樆鏂€?
