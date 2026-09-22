"""
AI_design_v1_8.py — Order Loss Dilution Fix + LSE Temperature Recalibration (v1.8)
─────────────────────────────────────
[v1.8 변경, docs/command/command_v1_8.md 실행]
  §1 Order 손실 Top-K 풀링(uni_section_v19.py) + w_order 동반 재조정(1.0→0.02): 4개 버전
     내내 원인 불명이던 Order 손실 무한 증가의 실제 원인(불변 파일 내부 mean 희석)을
     최초로 발견·수정. Top-K 분모 축소(~50배)를 상쇄하도록 가중치도 반드시 함께 조정.
  §2 smoothness LSE 온도 물리 단위 재보정(0.01→1.5mm): v1.7의 float32 오버플로우 위험
     경로를 원천 차단.
  구현: /synod idea 세션(Gemini flash conf95→90, OpenAI gpt4o conf75→85)에서 Gemini가
  Critic 라운드에서 자신의 1라운드 제안("w_order 재조정은 후순위 안전장치")을
  "Top-K와 반드시 동시 적용해야 하는 필수 선행 조건"으로 스스로 정정했다(N/K≈50배
  스케일 변화를 수식으로 도출). OpenAI가 제안한 손실 통합은 review_v1_7.md의 원인 분리
  실패 전례를 근거로 변수 분리 원칙에 따라 기각, 후속 버전으로 보류. 구현 세션(/synod
  design 재검증)에서 OpenAI가 "K/N=0.02는 정확한 보정이 아니라 근사치"임을 통계적으로
  재검증해 W_ORDER_V18을 "시작값, 스모크 테스트 후 재보정 필수"로 격하했고, Gemini가
  지적한 k 경계값·제곱 순서 문제는 코드로 이미 안전함을 확인했다. OpenAI의 pickle/
  sys.modules 우려는 이 프로젝트가 state_dict 텐서만 저장하고 체크포인트 재개가
  없어(v1.6 구현 세션에서 이미 확인) 무관하다고 판단해 기각했다.
그 외 로직은 AI_design_v1_7.py와 동일. uni_section_v19.py는 compute_mesh_order_loss()
1개 함수만 수정, 나머지는 uni_section_v18.py와 동일.
─────────────────────────────────────
AI_design_v1_7.py — Node-Level Spike Fix + Continuity Simplification (v1.7)
─────────────────────────────────────
[v1.7 변경, docs/command/command_v1_7.md 실행]
  §1 정적 이웃 맵 + LogSumExp 보강 스무스니스 손실: compute_smoothness_loss_angle()(불변)의
     mean 희석·x좌표 검사 회피 결함을 우회하는 신규 보강 항 추가. 초기 좌표 기준으로 좌/우
     이웃을 1회만 고정, LogSumExp로 집계(그래디언트 [0,1] 유계 — 수식으로 검증됨). Stage4
     호출부에도 반드시 배선(누락 시 크래시).
  §2 연속성 가중치 단순화: v1.6의 폐루프 제어(get_adaptive_continuity_weight, 상태 저장,
     EMA)를 래칫 버그(review_v1_6.md 확인)를 이유로 완전 제거, 단순 하한선(max(sigmoid,
     0.3))으로 대체.
  구현: /synod idea 세션(Gemini flash conf98, OpenAI gpt4o conf75)에서 OpenAI가 제기한
  DELETED 파트 마스킹 제안은 기존 베이스라인과의 일관성을 이유로 기각, Stage4 호출부 누락
  위험과 신규 항의 독립 가중치 필요성은 채택했다. 구현 세션(fairy_tail /synod design 재검증)
  에서 Gemini(conf95)·OpenAI(conf71) 둘 다 독립적으로 "정적 이웃 맵을 edge_type 필터 없이
  구축하면 join_pair 등 비물리 엣지가 섞여 들어간다"는 동일 버그를 지적해 반영(원본
  compute_smoothness_loss_angle()도 edge_type==0으로 필터링하던 것과 동일하게 맞춤). 빈
  텐서 가드의 디바이스 불일치 위험(Gemini)도 반영. OpenAI가 주장한 "다른 호출부
  (test_opt_flow.py 등)가 깨진다"는 근거는 해당 파일이 실제로 존재하지 않아(코드베이스 검색
  확인) 기각했다.
그 외 로직은 AI_design_v1_6.py와 동일.
─────────────────────────────────────
AI_design_v1_6.py — Structural Collapse Fix (v1.6)
─────────────────────────────────────
[v1.6 변경, docs/command/command_v1_6.md 실행]
  §1 연속성 가중치 폐루프 하한: continuity_weight_schedule()의 무하한 감쇠(w_min=0.05) 위에
     실측 l_continuity 기반 폐루프 제어를 추가, 절대 하한 0.3을 보장.
  §2 Stage4 사전 게이트 첨예화: CGDN17.compute_gates_multi()에서 epoch 200~350 구간에
     log_alpha_candidates를 forward 시점에만 최대 3배 증폭해 게이트가 Stage4(epoch 500)
     진입 전 이미 0/1 근처에 도달하도록 유도. §4 엔트로피 스케줄(epoch 400 시작)과의 충돌을
     피하기 위해 원안(종료 400)보다 앞당겨 350에서 종료·고정.
  §3(P1, 미적용): 파트 간 관통 페널티의 valid_mask 공백 — §1·§2로 해결 시 불필요(이번 버전
     에서는 미적용, §6 검증 계획에서 재실행 결과를 보고 후속 세션에서 판단).
  구현: /synod design 세션(Gemini flash conf98, OpenAI o3 conf77)에서 OpenAI가 제기한
  "collision_spec이 인접 셀 전제라 파트 간 교차를 못 잡는다"는 주장을 Claude가
  uni_section_v18.build_collision_spec() 코드 확인으로 반증(이미 전 파트쌍 커버) —
  §3을 새 메커니즘이 아닌 기존 메커니즘의 valid_mask 공백 보강으로 재정의했다. 또한
  OpenAI가 발견한 게이트 첨예화-엔트로피 스케줄 충돌(둘 다 epoch 400 근방에서 활성)을
  반영해 §2 종료 시점을 400→350으로 앞당겼다. 구현 세션(fairy_tail /synod design 재검증)
  에서 Gemini(conf98)가 get_adaptive_continuity_weight() 호출부의 기본 인자 누락으로 인한
  TypeError 위험을 지적해 반영했고, OpenAI(conf78)가 제기한 DDP/체크포인트 재개 관련 우려는
  이 코드베이스에 해당 메커니즘 자체가 없어(단일 프로세스, 재개 없음) 스코프 밖으로 기각했다.
그 외 로직은 AI_design_v1_5.py와 동일.
─────────────────────────────────────
AI_design_v1_5.py — Collision Penalty Explosion Fix (v1.5)
─────────────────────────────────────
[v1.5 변경, docs/command/command_v1_5.md 실행]
  §2 Huber형 충돌 페널티: compute_collision_penalty_unclamped()의 무클램프 제곱 항을
     δ=1.0mm(런타임 조정 가능) 기준 Huber loss로 교체 — 관통이 깊어져도 그래디언트가
     2δ로 상한이 걸려 폭발하지 않는다.
  §3 W_COL_AUX 코사인 웜업: epoch 11(게이트 활성화)~80(실측 loss 폭증 시점 이전) 구간에
     걸쳐 절대 floor 0.5에서 전역 상수 5.0까지 증가. §4(지연 엔트로피, epoch 400~)와는
     독립적 스케줄 — idea_v1_5.md 원안의 "엔트로피 구간과 동기화" 가정은 이번 세션에서
     실측 오류로 정정됨(alpha_ent 시작은 실제로 epoch 400).
  §4 로깅: l_col_aux, 현재 W_COL_AUX(t)를 콘솔 출력에 추가.
  구현: /synod design 세션(Gemini flash conf 98/95, OpenAI o3 conf 92/79)에서 두 모델이
  제안한 웜업 floor 공식이 실제 W_COL_AUX=5.0 상수를 반영하지 못해 5배 어긋난 오류를
  Claude가 발견·정정했고(Judge 최종 판정), 웜업 종료 시점도 실측 로그 기반으로 100→80
  으로 앞당겼다. 구현 세션(fairy_tail /synod design 재검증)에서 Gemini(conf98)가 Huber
  C1 연속성을 수식으로 재확인했고, OpenAI(conf83)가 지적한 Stage4 호출부(L1586 부근)의
  w_col_aux_val 미전달은 Stage4가 웜업 종료(epoch 80) 이후에만 시작되므로 무해함을
  코드 확인으로 검증했다(§3.2 주석 참고). w_col_aux_val 기본값은 sentinel 패턴(None ->
  런타임에 W_COL_AUX 참조)으로 구현해 정의 시점 바인딩 위험을 제거했다(Gemini 제안 채택).
그 외 로직은 AI_design_v1_4.py와 동일.
─────────────────────────────────────
AI_design_v1_4.py — Collision Penetration / Zero-Deletion fix (v1.4)
─────────────────────────────────────
[v1.4 변경, docs/command/command_v1_4.md 실행]
  §2 보조 무클램프 충돌 페널티: uni_section_v18의 compute_collision_loss_v5()는 관통 1mm에서
     .clamp(max=1.0)로 그래디언트가 소멸한다. 동일 collision_spec을 재사용하되 클램프 없는
     이차 페널티(compute_collision_penalty_unclamped())를 보강으로 추가.
  §3 면적 하한 가드레일: usv18.compute_alda_loss()의 f=area/target_area가 relu 없는 상시
     하강압이라 목표 초과 달성을 막지 못함 — l_area_floor로 반대 방향 벌점 추가.
  §4 중립 초기화 + 지연·어닐링 엔트로피: CGDN17.__init__의 죽어있던 init_log_alpha 파라미터를
     되살려 z_init≈0.5로 시작, ASWC_STAGE2_END까지 alpha_ent=0으로 지연 후 선형 어닐링.
  §5 R4: log_alpha_candidates<0 ⟺ z_gate<0.5라는 사실을 이용해 삭제-부재를 저비용으로 감시
     (진단 전용, raise 하지 않음).
  구현: /synod design 세션(Gemini flash conf 98/98, OpenAI o3 conf 93/93)에서 collision_spec이
  좌표를 캐시하지 않는다는 점, CGDN17의 init_log_alpha가 죽은 파라미터였다는 점을 검증받았다.
그 외 로직은 AI_design_v1_3.py와 동일.
─────────────────────────────────────
AI_design_v1_3.py — Mass Paradox / Gate Convergence / Death Zone fix (v1.3)
─────────────────────────────────────
[v1.3 변경, docs/command/command_v1_3.md 실행]
  §2 질량 역설 제거: l_sparse 상수 페널티 무력화(W_SPARSE_V13=0.0), 이진화 압력을 엔트로피로 이관
     (ALPHA_ENT_V13=0.05, alpha_ent를 train_step_multi 인자로 주입). l_mass를 예산 제약으로
     loss에 편입(target_area = 초기 단면적의 95%).
  §3 두께 [0.7, 2.3] 전 구간 도달성 확보: T_MIN_V13/T_MAX_V13/DELTA_SCALE_V13을 CGDN17 클래스
     속성으로 오버라이드, frac/logit 계산을 _t_init_logit() 헬퍼로 단일화(forward()·
     report_design_comparison() 양쪽 공유).
  §4 이진화 임계 0.3으로 정렬(BINARIZE_THRESHOLD, >= 비교로 통일).
  §5 Stage 4: 본 학습 루프 종료 후 애매 게이트가 남아 있으면 게이트를 강제 이진화(±5.0)하고
     좌표+두께만 재수렴시키는 독립 구간. DELETED는 부활시키지 않는다.
  §7 전역 사망 원장(death_log) + 롤백 규칙 R1~R3(SUSPECT 사망/연쇄 붕괴/질량 역설 재발 감시,
     전 섹션 공통, 특정 섹션 하드코딩 금지).
  구현: /synod design 세션(Gemini flash conf 95, OpenAI o3 conf 83)에서 §2.6 l_mass 섀도잉,
  Stage4 옵티마이저의 log_alpha_candidates 격리 순서, frac 헬퍼 이중 존재 위험을 검증받았다.
그 외 로직은 AI_design_v1_2.py와 동일.
─────────────────────────────────────
AI_design_v1_2.py — Dynamic Part Splitting + Design Property Report (v1.2)
─────────────────────────────────────
[v1.2 변경, 2026-08-07 /synod debug 세션(synod-20260807-163922-79efc2) 합의]
  1. 학습 종료 후 initial vs final design property 비교 리포트(report_design_comparison):
     섹션별 Mp(initial/target/final-soft/final-hard/달성률), 총 단면적, 파트·run별 두께,
     candidate 존재 맵. soft(학습 상태)와 hard(제조 상태: 게이트 0.5 임계 + DELETED AND 조건,
     결정론적 eval 게이트) Mp를 병기하고 괴리 >2%면 경고 — Gemini(conf 95): "하드 이진화는
     제조 관점에서 필수, 단 soft와 병기하고 미달 시 fine-tuning 필요를 명시".
     콘솔 표 + reports/AI_design_v1_2_report.md 저장(o3: 텐서 스팸 방지).
  2. epoch 로그의 candidate_gate_mean(패치 평균)을 파트별 개별 출력으로 교체 —
     생존(non-DELETED) 섹션만 평균 + 생존 수 표기: g3=0.973(17/17) g4=0.541(12/17)
     (o3 지적: 삭제 섹션을 평균에 넣으면 0으로 끌려 추세 왜곡).
  o3의 "run 복제로 t_raw 희석" 우려는 기각 — 본 구조는 노드 복제가 없고 run별 그룹이
  분리 pooling되므로 해당 없음.
그 외 로직은 AI_design_v1_1.py와 동일.
─────────────────────────────────────
docs/command/command_v2.md 실행 (2026-08-05 /synod idea+design 세션 합의안). AI_design_v1.py를
기반으로 candidate 파트(3, 4)의 두께 pooling을 3-Phase 스케줄로 교체한다:

  Phase A/B (분할 전, DELETED 없음): candidate도 **파트당 1 두께** (전 섹션 글로벌 pooling —
    "하나의 패치 = 하나의 두께"). v1의 "섹션별 완전 독립 pooling"(두께가 섹션마다 진동 → 제조
    불가)은 제거되었다.
  Phase C (분할 후): pruning_state에서 DELETED가 확정되는 순간(newly_deleted 이벤트),
    DELETED 섹션을 절단점으로 1D CCL(compute_segment_ids)을 수행해 **연결 run당 1 두께**로
    pooling을 세분화한다. DELETED는 비가역이므로 그룹은 단조 세분화만 된다(재병합 없음).

/synod design 세션(2026-08-05, synod-20260805-235100-38ce43) 반영 사항:
  - dry-run은 model.eval()로 게이트 난수를 차단해야 candidate 균일 두께 assert가 성립
    (OpenAI o3 지적: HardConcrete train 모드 난수가 uniform 가정을 깨뜨림).
  - run_key 계산은 seg.clamp(min=0)로 음수 산술을 먼저 제거한 뒤 where 마스킹 — seg=-1이
    산술식에 들어가 인접 키와 충돌하는 것을 방지 (Gemini/OpenAI 공통 지적).
  - curriculum/온도/gate 스케줄은 **원래 max_epochs로 동결**하고, adaptive extension은 루프
    상한만 연장한다 (스케줄 불연속 방지, Gemini conf 95 검증).
  - segment_ids는 newly_deleted가 나올 때마다 매번 재계산 (OpenAI 지적: 분할이 여러 번
    일어날 수 있음).
원본 v1의 나머지 로직(게이팅, loss, 시각화 골격)은 그대로 유지. uni_section_v18.py 불변.

**[정정 이력]** 최초 버전은 part 삭제/재등장을 `configs/part_schedule_v1.json`(사람이 미리 짜는 존재
스케줄)과 9번째 노드 컬럼(`logical_part_id`)으로 구현했으나, 사용자가 "삭제/재등장은 학습 중 목표 Mp를
맞추다 보니 특정 섹션에서 자연히 켜지거나 꺼지는 것"이라고 정정함. 이번 버전은:
  - 노드 feature 8열 그대로 유지(신규 컬럼 없음), 17섹션 모두 처음부터 5-part 전부를 노드로 포함
    (사전 필터링 없음).
  - `log_alpha`를 파트당 1개(원본, shape (5,))에서 **(section, candidate_part)당 1개** (shape
    (17, len(CANDIDATE_PARTS))=(17,2))로 확장 — 원본의 HardConcrete 게이팅(`compute_gates`)을 그대로
    재사용하되 flatten(-1)→호출→unflatten으로 감싼다(브로드캐스트 버그 방지, /synod debug 세션에서
    Gemini/OpenAI 둘 다 독립적으로 제안한 안전한 패턴).
  - `pruning_state`를 (17, len(CANDIDATE_PARTS)) 텐서로 벡터화(ALIVE/PENDING_DELETE/DELETED
    히스테리시스, 원본 update_pruning_state의 for-loop 로직을 벡터 연산으로 이식).
  - 두께 pooling은 물리 part_id만으로 분기(연속 파트=전 섹션 공통, candidate 파트=섹션별 독립) —
    이미 섹션별 독립 pooling이므로 "재등장 시 새 두께"가 재넘버링 없이 자동 성립.
  - z_gate 히트맵(17×2) 시각화 추가 — 어느 섹션에서 patch가 학습을 통해 켜지고 꺼졌는지 확인.

/synod debug 세션(2026-07-27): Gemini(flash, conf 95)와 OpenAI(o3, conf 85) 모두 flatten/unflatten
wrapper와 (17,5) 전체 z_gate 조립 방식을 독립적으로 제안해 수렴. OpenAI가 지적한 실무 함정(Stage 0에서
"5파트 전부 동일 두께" 식으로 검사하면 실패 — 연속 파트만 검사해야 함, 로그 포맷 문자열에 텐서를 그대로
넣으면 콘솔 폭주 등)을 반영했다.
"""

import copy
import json
import os
import math

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Gulim'

from torch_geometric.data import Data

import sys
# __file__은 일반 스크립트 실행 시에는 존재하지만 Jupyter 노트북 셀에서 직접 실행할 때는 정의되지
# 않는다(NameError) — 이 경우 현재 작업 디렉터리(os.getcwd())를 기준으로 삼는다. 노트북은 보통
# 이 파일이 있는 CGNN 디렉터리에서 실행되므로 getcwd()가 안전한 대체값이다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(_THIS_DIR, 'uni-section', 'code'))
import uni_section_v19 as usv18   # [v1.8] compute_mesh_order_loss() Top-K 풀링 적용판


# ══════════════════════════════════════════════════════════════════
# [v1.3] 로컬 오버라이드 상수 (uni_section_v18.py 불변, 여기서만 덮는다)
# 근거: docs/command/command_v1_3.md
# ★ 배치: import 직후 / class CGDN17 정의보다 앞 (§1) — import 시 NameError 방지
# ══════════════════════════════════════════════════════════════════

# §2. 질량 역설 — 상수 페널티 제거, 이진화 압력을 엔트로피로 이관
W_SPARSE_V13  = 0.0     # usv18.W_SPARSE(0.5) 대체. 0.0 = 제안 A안(권장)
ALPHA_ENT_V13 = 0.05    # usv18.ALPHA_ENT(0.01) 대체. 대칭 이진화 압력
USE_AREA_WEIGHTED_SPARSE = False   # §2.2 B안 스위치. 기본 False

# §2.6 질량 감량 목표 — 초기 단면적의 95%
AREA_TARGET_RATIO = 0.95
W_MASS_V13        = 10.0   # l_mass를 loss에 편입할 때의 가중치. 0.0이면 예산 제약 없음

# §3. 두께 한계 (제조 제약)
T_MIN_V13 = 0.7         # usv18.CGDN.T_MIN(0.1) 대체
T_MAX_V13 = 2.3         # usv18.CGDN.T_MAX(2.5) 대체
FRAC_CLAMP_HI = 0.99    # 앵커 logit 상한 (+4.60)
FRAC_CLAMP_LO = 0.01    # 앵커 logit 하한 (-4.60)
DELTA_SCALE_V13 = 10.0  # usv18.CGDN.DELTA_SCALE(1.35) 대체. 전 구간 도달의 핵심 노브

# §4. 이진화 임계 — 프루닝 임계(0.08)와의 "조용한 죽음의 구간" 축소
BINARIZE_THRESHOLD = 0.3

# §5. Stage 4 — 존재 확정 후 형상·두께 재수렴
ENABLE_STAGE4   = None  # None = §5.1 자동 판정 / True = 강제 진입 / False = 강제 비활성
STAGE4_EPOCHS   = 200
STAGE4_LR_SCALE = 0.1

# §7. 전역 게이트 사망 감시 — 특정 섹션 하드코딩 금지, 전 섹션 × 전 후보에 적용
DEATH_MP_THRESHOLD   = 0.98   # 사망 시점 해당 섹션 hard/target 이 이 값 미만이면 SUSPECT
DEATH_BURST_WINDOW   = 100    # R2: 연쇄 붕괴 감시 창 (epoch)
DEATH_BURST_RATIO    = 0.20   # R2: 창 내 신규 0-포화가 ALIVE 대비 이 비율 이상이면 중단
AREA_REGRESS_WINDOW  = 200    # R3: 질량 역설 감시 창 (epoch)

# ★ §2.2 B안 가드 — 스위치만 켜고 가중치를 0으로 두면 침묵 no-op이 된다
assert not (USE_AREA_WEIGHTED_SPARSE and W_SPARSE_V13 == 0.0), \
    "B안(면적 가중 l_sparse) 활성 시 W_SPARSE_V13는 0.02~0.05여야 한다 (command_v1_3.md §2.2)"


# ══════════════════════════════════════════════════════════════════
# [v1.4] 로컬 오버라이드 상수 — 관통 방지 + 삭제 유도
# 근거: docs/idea/idea_v1_4.md, docs/review/review_v1_3.md, docs/command/command_v1_4.md
# ══════════════════════════════════════════════════════════════════

# §2. 보조 무클램프 충돌 페널티
PENETRATION_FLOOR = 0.8      # mm. 이 깊이부터 무클램프 페널티가 발동
W_COL_AUX = 5.0               # 보조 충돌 페널티 가중치

# §3. 면적 하한 가드레일 — f(=area/target_area)의 상시 하강압을 상쇄
SAFETY_RATIO = 0.90           # 목표 면적의 90% 밑으로는 반대 방향 벌점 발동
W_AREA_FLOOR = 10.0            # W_MASS_V13과 대칭적인 크기로 시작

# §4. 중립 초기화 + 지연·어닐링 엔트로피
INIT_LOG_ALPHA_V14 = 0.0      # usv18.CGDN 기본 동작(2.0) 대체. z_init ≈ 0.5
ALPHA_ENT_START_EPOCH = None   # None = ASWC_STAGE2_END로 자동 설정. 그 전까지 alpha_ent=0
ALPHA_ENT_ANNEAL_EPOCHS = 50   # START_EPOCH부터 이 기간에 걸쳐 0 → ALPHA_ENT_V13으로 선형 어닐링

# §5. R4 — 삭제-부재 감시 (진단 전용, raise 하지 않음)
R4_GRACE_EPOCHS = 200          # ASWC_STAGE2_END + 이 값까지 어떤 게이트도 0.5 밑으로 안 내려가면 경고


# ══════════════════════════════════════════════════════════════════
# [v1.5] 로컬 오버라이드 상수 — 충돌 페널티 상한 + 웜업
# 근거: docs/idea/idea_v1_5.md, docs/review/review_v1_4.md, docs/command/command_v1_5.md
# ══════════════════════════════════════════════════════════════════

# §2. Huber형 충돌 페널티 상한 — δ는 실제 메시/격자 해상도 상수를 코드에서 찾지 못해(design 세션
# 확인) 확정값 대신 런타임 인자화한다. 기본값은 review_v1_4.md 권장 범위(1.0~1.2mm)의 보수적
# 하한(더 이른 지점에서 선형 전환 = 더 안전)을 사용한다.
HUBER_DELTA_MM = 1.0         # mm. 이 값 이하는 제곱, 초과는 선형 페널티(C1 연속)로 전환

# §3. W_COL_AUX 코사인 웜업 — §4(지연 엔트로피, epoch 400~450)와 무관하게 독립 설계됨에 주의.
# 실측 로그(epoch 90~100 사이 loss 23배 폭증) 기준으로 웜업이 그 이전에 완료되도록 구간을 잡는다.
W_COL_AUX_WARMUP_START = 11   # usv18.GATE_ACTIVE_EPOCH와 동일값(게이트 학습 시작 시점)
W_COL_AUX_WARMUP_END   = 80   # 실측 폭증 시점(epoch 90) 이전에 전체 가중치(W_COL_AUX)에 도달
W_COL_AUX_FLOOR         = 0.5  # 절대값. W_COL_AUX(5.0)의 비율이 아님 — §7.2 참고
# §7.2 floor 설계 근거(design 세션 Judge 판정): W_COL_AUX_FLOOR는 웜업 시작 시점(epoch 11)의
# 초기 가중치를 정하는 파라미터일 뿐이며, 웜업 종료 후(epoch 80 이후) 값은 floor 선택과 무관하게
# 항상 전역 상수 W_COL_AUX=5.0으로 수렴한다. floor=0.5*W_COL_AUX(=2.5)는 "후반부 억제력 보강"이라는
# 잘못된 근거로 초기 억제력을 5배 과도하게 키우는 것이므로 채택하지 않는다 — 반드시 절대값 0.5.


# ══════════════════════════════════════════════════════════════════
# [v1.6] 로컬 오버라이드 상수 — 연속성 폐루프 제어 + 게이트 사전 첨예화
# 근거: docs/idea/idea_v1_6.md, docs/review/review_v1_5.md, docs/command/command_v1_6.md
# ══════════════════════════════════════════════════════════════════

# §1. 연속성 가중치 — 기존 continuity_weight_schedule()의 w_min=0.05 무하한 감쇠가 구조적
# 붕괴의 1차 원인(review_v1_5.md)이었다. v1.6은 폐루프 제어로 대응했으나 래칫 버그(review_v1_6.md)
# 로 회귀했다 — v1.7은 단순 하한선만 남긴다(command_v1_7.md §2).
CONTINUITY_W_FLOOR = 0.3    # 절대 하한 — sigmoid가 이 값 밑으로는 못 내려간다

# §2. 게이트 사전 첨예화 — candidate 게이트(z_gate)가 z≈0.5에서 300+ epoch 방치되다 Stage4
# (epoch 500)에서 한 번에 ±5.0으로 강제 이진화되는 충격을 완화한다. §4 엔트로피 활성화(epoch
# 400, ALPHA_ENT_START_EPOCH 기본값=ASWC_STAGE2_END)와의 직접 충돌을 피하기 위해 첨예화는
# 반드시 entropy 활성화 이전에 완료하고 그 이후로는 beta_steep을 고정한다.
GATE_STEEP_START = 200      # 첨예화 시작 epoch
GATE_STEEP_END   = 350      # 종료 시점 — entropy 시작(400)과 50epoch 여유
GATE_STEEP_MAX   = 3.0      # 최대 배율(fp 정밀도 여유를 위해 보수적으로 설정)


# ══════════════════════════════════════════════════════════════════
# [v1.7] 로컬 오버라이드 상수 — 노드 단위 스파이크 보강 손실
# 근거: docs/idea/idea_v1_7.md, docs/review/review_v1_6.md, docs/command/command_v1_7.md
# ══════════════════════════════════════════════════════════════════
W_SMOOTH_LSE = 0.25   # 보강 손실 가중치 — 기존 weights['w_smooth']와 독립(커리큘럼 미상속)
SMOOTH_LSE_TEMPERATURE = 1.5   # [v1.8] mm 단위. 판금 제조 공차 스케일에 맞춘 물리적 재보정.
                                 # T=0.01은 exp(deviation²/T)가 10mm급 편차에서 exp(1000) 수준의
                                 # float32 오버플로우를 사실상 확정적으로 유발함(design 세션 검증,
                                 # review_v1_7.md의 "near-hard-max" 진단을 수식으로 재확인) —
                                 # 물리적으로 의미 있는 스케일(1.5mm)로 교체해 정상 수치 범위에서
                                 # "부드러운 최댓값"으로 동작하도록 한다.


# ══════════════════════════════════════════════════════════════════
# [v1.8] 로컬 오버라이드 상수 — Order 손실 Top-K 풀링 동반 가중치 재조정
# 근거: docs/idea/idea_v1_8.md, docs/review/review_v1_7.md, docs/command/command_v1_8.md
# ══════════════════════════════════════════════════════════════════
W_ORDER_V18 = 0.02   # uni_section_v19.compute_mesh_order_loss()의 top_k_fraction(0.02)과 동일값
                      # (시작값 — K/N 근사는 부정확할 수 있어 §6 검증 계획에서 실측 재보정 필수).
                      # Top-K 전환으로 분모가 2992→60(~50배 축소)되므로, 원래 w_order=1.0의
                      # 그래디언트 크기를 유지하려면 이 배율만큼 가중치를 낮춰야 한다
                      # (idea_v1_8.md §1, Critic 라운드에서 Gemini가 자기 수정으로 도출).


def _t_init_logit(t_init, t_min, t_max):
    """[v1.3 §3.3] frac/logit 계산의 단일 소스. CGDN17.forward()와 report_design_comparison()
    양쪽이 이 헬퍼만 호출해야 두 곳이 갈라지는 사고를 구조적으로 막는다."""
    frac = torch.clamp((t_init - t_min) / (t_max - t_min), FRAC_CLAMP_LO, FRAC_CLAMP_HI)
    return torch.logit(frac)


@torch.no_grad()
def assert_thickness_reachability(x, t_min=T_MIN_V13, t_max=T_MAX_V13,
                                   delta_scale=DELTA_SCALE_V13, tol=0.01):
    """[v1.3 §3.4] 모든 노드가 [T_MIN+tol, T_MAX-tol] 전 구간에 도달 가능한지 검증한다(정적 상한,
    thickness_gate 승수·그룹 평균 pooling 미반영). 단순 범위 검사가 아니라 delta=±delta_scale에서의
    도달 범위를 확인한다 — t_init == T_MAX인 노드가 logit 포화로 동결되는 사고를 막기 위함."""
    t_init = x[:, 6]
    assert (t_init >= t_min - 1e-6).all() and (t_init <= t_max + 1e-6).all(), \
        f"초기 두께가 [{t_min}, {t_max}] 밖: min={t_init.min():.3f} max={t_init.max():.3f}"

    logit = _t_init_logit(t_init, t_min, t_max)
    lo = t_min + (t_max - t_min) * torch.sigmoid(logit - delta_scale)
    hi = t_min + (t_max - t_min) * torch.sigmoid(logit + delta_scale)

    bad_lo = lo > t_min + tol
    bad_hi = hi < t_max - tol
    assert not bad_lo.any(), (
        f"[Reachability] 하한 미달 노드 {int(bad_lo.sum())}개. 최악 lo={float(lo.max()):.4f}mm > "
        f"{t_min + tol:.2f}mm. DELTA_SCALE({delta_scale})을 올릴 것 (command_v1_3.md §3.2)")
    assert not bad_hi.any(), (
        f"[Reachability] 상한 미달 노드 {int(bad_hi.sum())}개. 최악 hi={float(hi.min()):.4f}mm < "
        f"{t_max - tol:.2f}mm. DELTA_SCALE({delta_scale})을 올릴 것 (command_v1_3.md §3.2)")
    return lo, hi


# ══════════════════════════════════════════════════════════════════
# SECTION 0: 17-section 데이터 생성 (command_v1.md §2 — 사전 필터링 없음)
# ══════════════════════════════════════════════════════════════════

NUM_SECTIONS = 17
CONTINUOUS_PARTS = usv18.S_PROTECT        # [0, 1, 2] — Outer Hat, Inner Plate, Inner Hat
CANDIDATE_PARTS = usv18.CANDIDATE_PARTS   # [3, 4] — Patch 1, Patch 2


def scale_factor(k: int) -> float:
    """initial_section.md §1: s(k) = 1.6 - 0.6*(k-1)/16, k=1..17(1-indexed, k=17이 원본/최상단).
    이 파일 내부에서는 0-indexed section_id(0=최상단 원본 ~ 16=최하단)를 쓰므로 k=section_id+1."""
    k1 = k + 1
    return 1.6 - 0.6 * (k1 - 1) / 16.0


def build_bpillar_17section():
    """build_bpillar_section()(uni_section_v18.py line 805)을 17번 호출해 스케일링 후 병합.
    [정정] 모든 섹션에 5-part 전부를 처음부터 노드로 포함한다 — 존재/삭제는 사전 필터링이 아니라
    학습된 게이트(§1 CGDN17)로 결정되므로 여기서는 어떤 노드도 걸러내지 않는다.
    반환: Data(x[N,8], edge_index, edge_attr, join_pairs)."""
    all_x, all_edge_index, all_edge_attr = [], [], []
    node_offset = 0

    for k in range(NUM_SECTIONS):
        base_data, _ = usv18.build_bpillar_section()
        x_k = base_data.x.clone()
        s = scale_factor(k)
        x_k[:, 0] *= s
        x_k[:, 1] *= s
        x_k[:, 5] = float(k)  # section_id

        edge_index_k = base_data.edge_index + node_offset
        edge_attr_k = base_data.edge_attr.clone()

        all_x.append(x_k)
        all_edge_index.append(edge_index_k)
        all_edge_attr.append(edge_attr_k)
        node_offset += x_k.shape[0]

    x = torch.cat(all_x, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1)
    edge_attr = torch.cat(all_edge_attr, dim=0)
    join_pairs = torch.zeros((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, join_pairs=join_pairs)


# ══════════════════════════════════════════════════════════════════
# SECTION 1: CGDN17 — 두께 pooling 분기 + Section-Aware 학습된 게이팅 (command_v1.md §3)
# ══════════════════════════════════════════════════════════════════

class CGDN17(usv18.CGDN):
    """forward()는 uni_section_v18.CGDN.forward()(line 327-391)의 완전한 복제본이며:
      §3.1 두께 pooling: composite_key를 물리 part_id만으로 분기(연속=전 섹션 공통, candidate=섹션별 독립).
      §3.2 존재 게이트: log_alpha_candidates(17,len(CANDIDATE_PARTS)) 기반 section-aware HardConcrete.
    나머지 로직(좌표 예측, join_pairs)은 원본과 100% 동일하다."""

    T_MIN       = T_MIN_V13        # [v1.3] 0.7   (usv18.CGDN.T_MIN = 0.1)
    T_MAX       = T_MAX_V13        # [v1.3] 2.3   (usv18.CGDN.T_MAX = 2.5)
    DELTA_SCALE = DELTA_SCALE_V13  # [v1.3] 10.0  (usv18.CGDN.DELTA_SCALE = 1.35) — §3.2 필수

    def __init__(self, *args, init_log_alpha=2.0, **kwargs):
        super().__init__(*args, init_log_alpha=init_log_alpha, **kwargs)
        # 원본 self.log_alpha(shape (5,))는 그대로 두되(원본 코드 하위호환), 실제 게이팅에는
        # 아래 candidate 전용 파라미터를 사용한다.
        del self.log_alpha
        # [v1.4 §4.1] v1.2/v1.3은 이 값을 리터럴 2.0으로 하드코딩해 init_log_alpha 인자가 죽어
        # 있었다(부모 클래스가 삭제되는 self.log_alpha에만 영향). 파라미터를 실제로 사용하도록 수정.
        self.log_alpha_candidates = torch.nn.Parameter(
            torch.full((NUM_SECTIONS, len(CANDIDATE_PARTS)), float(init_log_alpha))
        )

    def compute_gates_multi(self, training, temperature):
        """원본 compute_gates()(line 188)는 1D 텐서만 받으므로, (17,len(CANDIDATE_PARTS))를
        flatten(-1)해서 그대로 통과시킨 뒤 동일 shape으로 unflatten한다 — 이 순서를 지켜야
        (17,2,17) 같은 유령 브로드캐스트 차원이 생기지 않는다(/synod debug 세션 확인).
        [v1.6 §2] epoch 200~350 구간에 log_alpha_candidates를 GATE_STEEP_MAX까지 선형 증폭해
        시그모이드를 미리 첨예화한다 — 실제 파라미터 값은 건드리지 않고(forward 시점의
        스케일링만) Stage4 진입 시점(epoch 500)에는 이미 게이트가 0/1 근처에 있어 강제
        이진화가 작은 보정에 그치도록 만든다. self.current_epoch은 run_training_multi가 매
        epoch 설정한다. R4/death ledger는 원본 self.log_alpha_candidates(스케일 전)를 직접
        참조하므로 영향받지 않는다 — beta_steep > 0이므로 부호는 배율에 무관하게 보존된다."""
        epoch = getattr(self, 'current_epoch', 0)
        if epoch < GATE_STEEP_START:
            beta_steep = 1.0
        elif epoch < GATE_STEEP_END:
            progress = (epoch - GATE_STEEP_START) / (GATE_STEEP_END - GATE_STEEP_START)
            beta_steep = 1.0 + progress * (GATE_STEEP_MAX - 1.0)
        else:
            beta_steep = GATE_STEEP_MAX   # epoch 350 이후 고정 — §4 entropy와의 충돌 방지
        flat_log_alpha = (self.log_alpha_candidates * beta_steep).reshape(-1)   # [v1.6 §2]
        z_gate_flat, z_open_flat = usv18.compute_gates(
            flat_log_alpha, training=training, temperature=temperature, s_protect=None
        )
        z_gate_cand = z_gate_flat.view_as(self.log_alpha_candidates)           # (17, len(CANDIDATE_PARTS))
        z_open_cand = z_open_flat.view_as(self.log_alpha_candidates)

        device = flat_log_alpha.device
        z_gate = torch.ones(NUM_SECTIONS, self.num_parts, device=device)       # 연속 파트=1.0 고정(마스크 방식)
        z_open = torch.ones(NUM_SECTIONS, self.num_parts, device=device)
        cand_idx = torch.tensor(CANDIDATE_PARTS, device=device)
        z_gate[:, cand_idx] = z_gate_cand
        z_open[:, cand_idx] = z_open_cand
        return z_gate, z_open   # 각각 (17, 5)

    def forward(self, x, edge_index, edge_attr, target_mp,
                fix_x_mask, fix_y_mask, join_pairs=None, thickness_gate=1.0,
                gate_active=True, temperature=0.5, segment_ids=None):
        h = self.node_encoder(x)

        for i, block in enumerate(self.blocks):
            gamma, beta = self.film_generators[i](target_mp)
            h = block(h, edge_index, edge_attr, gamma, beta)

        delta_coords = self.coord_decoder(h)
        delta_coords = torch.clamp(delta_coords, -self.max_displacement, self.max_displacement)
        delta_x = delta_coords[:, 0:1] * (~fix_x_mask).float()
        delta_y = delta_coords[:, 1:2] * (~fix_y_mask).float()
        delta_coords = torch.cat([delta_x, delta_y], dim=1)
        new_coords = x[:, :2] + delta_coords

        if join_pairs is not None and join_pairs.shape[0] > 0:
            u_idx = join_pairs[:, 0]
            v_idx = join_pairs[:, 1]
            mid = (new_coords[u_idx] + new_coords[v_idx]) * 0.5
            new_coords = new_coords.clone()
            new_coords[u_idx] = mid
            new_coords[v_idx] = mid

        delta_t_raw = self.thickness_decoder(h)
        delta_t_raw = self.leaky_tanh(delta_t_raw) * self.DELTA_SCALE

        part_ids_local    = x[:, 4].long()
        section_ids_local = x[:, 5].long()
        t_initial         = x[:, 6].unsqueeze(1)

        # ── §3.1 [v1.1] 두께 pooling: 3-Phase 동적 파트 분할 (command_v2.md §2) ──
        continuous_ids = torch.tensor(CONTINUOUS_PARTS, device=x.device)
        is_continuous  = torch.isin(part_ids_local, continuous_ids)

        if segment_ids is None:
            # Phase A/B: candidate도 파트 단위 글로벌 pooling ("하나의 패치 = 하나의 두께").
            # v1의 섹션별 독립 pooling(10_000 + section*max_parts + part)은 제거 — 이어진 구간의
            # 두께가 섹션마다 진동하는(제조 불가) 원인이었다.
            composite_key = part_ids_local
        else:
            # Phase C: DELETED 절단점 기준 run 단위 pooling. segment_ids: (17, n_cand) long, x와 동일 device.
            # 연속 파트 노드도 cand_col 산술을 거치지만(clamp로 음수 인덱스 방지) 최종 where에서 무시된다.
            cand_col = (part_ids_local - CANDIDATE_PARTS[0]).clamp(min=0, max=len(CANDIDATE_PARTS) - 1)
            seg_of_node = segment_ids[section_ids_local, cand_col]           # (N,) long, DELETED 칸은 -1
            # seg=-1이 산술식에 들어가 인접 키와 충돌하지 않도록 clamp 후 별도 dummy 키(90000대)로 격리
            # (/synod design 세션 Gemini+OpenAI 공통 지적). dummy 그룹 노드는 z_gate≈0이라 t_final≈0.
            run_key = 10_000 + part_ids_local * (NUM_SECTIONS + 1) + seg_of_node.clamp(min=0)
            run_key = torch.where(seg_of_node < 0,
                                  90_000 + part_ids_local,
                                  run_key)
            composite_key = torch.where(is_continuous, part_ids_local, run_key)
        _, inverse = torch.unique(composite_key, return_inverse=True)
        num_groups = int(inverse.max().item()) + 1

        delta_t_1d  = delta_t_raw.squeeze(-1)
        group_sum   = torch.zeros(num_groups, device=x.device).scatter_add_(0, inverse, delta_t_1d)
        group_count = torch.zeros(num_groups, device=x.device).scatter_add_(0, inverse, torch.ones_like(delta_t_1d))
        group_mean  = group_sum / group_count.clamp(min=1)
        delta_t_part = group_mean[inverse].unsqueeze(-1)

        delta_t_part = delta_t_part * thickness_gate

        t_min, t_max = self.T_MIN, self.T_MAX
        t_initial_logit = _t_init_logit(t_initial, t_min, t_max)   # [v1.3 §3.3] 단일 소스 헬퍼
        t_raw = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)

        # ── §3.2 존재 게이트: section-aware — 원본 z_gate[part_ids_local] 대신 섹션까지 인덱싱 ──
        if gate_active:
            z_gate, z_open = self.compute_gates_multi(training=self.training, temperature=temperature)  # (17,5)
        else:
            z_gate = torch.ones(NUM_SECTIONS, self.num_parts, device=x.device)
            z_open = torch.ones(NUM_SECTIONS, self.num_parts, device=x.device)

        part_gate_node = z_gate[section_ids_local, part_ids_local].unsqueeze(-1)
        t_final = t_raw * part_gate_node

        return new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open


# ══════════════════════════════════════════════════════════════════
# SECTION 2: pruning_state 벡터화 (command_v1.md §3.3)
# ══════════════════════════════════════════════════════════════════

STATE_ALIVE, STATE_PENDING, STATE_DELETED = 0, 1, 2


def make_pruning_state_multi():
    n_cand = len(CANDIDATE_PARTS)
    return {
        'ewma_z':            torch.ones(NUM_SECTIONS, n_cand),
        'state':             torch.zeros(NUM_SECTIONS, n_cand, dtype=torch.long),
        'pending_duration':  torch.zeros(NUM_SECTIONS, n_cand, dtype=torch.long),
        'ewma_alpha': 0.15, 'thresh_low': 0.08, 'thresh_high': 0.15, 'confirm_epochs': 20,
    }


@torch.no_grad()
def update_pruning_state_multi(pruning_state, z_gate_candidates):
    """원본 update_pruning_state()(line 238-264)의 for-loop 히스테리시스를 (17,len(CANDIDATE_PARTS))
    전체에 대한 벡터 연산으로 이식. z_gate_candidates: (17, len(CANDIDATE_PARTS))."""
    a = pruning_state['ewma_alpha']
    ewma_z = pruning_state['ewma_z']
    state = pruning_state['state']
    dur = pruning_state['pending_duration']

    ewma_z.mul_(1 - a).add_(z_gate_candidates.detach().cpu(), alpha=a)

    thresh_low, thresh_high, confirm_epochs = (
        pruning_state['thresh_low'], pruning_state['thresh_high'], pruning_state['confirm_epochs'])

    # ALIVE -> PENDING_DELETE
    to_pending = (state == STATE_ALIVE) & (ewma_z < thresh_low)
    state[to_pending] = STATE_PENDING
    dur[to_pending] = 0

    # PENDING_DELETE -> ALIVE (회복)
    mask_pending = state == STATE_PENDING
    back_to_alive = mask_pending & (ewma_z > thresh_high)
    state[back_to_alive] = STATE_ALIVE
    dur[back_to_alive] = 0

    # PENDING_DELETE 유지 -> duration 증가
    still_pending = mask_pending & ~back_to_alive
    dur[still_pending] += 1

    # confirm_epochs 초과 -> DELETED
    newly_deleted = still_pending & (dur >= confirm_epochs)
    state[newly_deleted] = STATE_DELETED

    return newly_deleted  # (17, len(CANDIDATE_PARTS)) bool — 이번 스텝에 새로 확정된 (section, part)


@torch.no_grad()
def compute_segment_ids(pruning_state, hard_exists=None):
    """[v1.1, command_v2.md §1] candidate 파트별 1D Connected-Component Labeling.
    반환: seg_id (NUM_SECTIONS, len(CANDIDATE_PARTS)) long, CPU —
          생존(non-DELETED) 연속 run마다 0부터 시작하는 조밀한 run 번호, DELETED 칸은 -1.
    PENDING_DELETE는 생존으로 취급한다(절단점은 비가역 확정인 DELETED만).
    전부 정수/불리언 연산이라 autograd 그래프에 들어가지 않는다.
    [v1.3 §5.3] hard_exists가 주어지면(Stage 4) 그것을 존재맵으로 삼아 run을 자른다.
    None이면 기존 동작(STATE_DELETED 기준) — 본 학습 루프 호출부는 수정 불필요."""
    if hard_exists is None:
        deleted = (pruning_state['state'] == STATE_DELETED)          # (17, n_cand) bool
    else:
        deleted = ~hard_exists.cpu()
    alive = ~deleted
    # run 시작점: 살아있으면서 (첫 섹션이거나 바로 위 섹션이 DELETED)
    prev_deleted = torch.cat(
        [torch.ones(1, deleted.shape[1], dtype=torch.bool), deleted[:-1]], dim=0)
    run_start = alive & prev_deleted
    run_id = torch.cumsum(run_start.long(), dim=0) - 1               # 생존 칸에서 0,1,2,...
    seg_id = torch.where(alive, run_id, torch.full_like(run_id, -1))
    return seg_id


def count_thickness_groups(segment_ids):
    """[v1.1] 현재 pooling 그룹 수 = 연속 파트 3 + candidate run 수(+ dummy 그룹은 제외하고
    '실제 두께 변수' 개수만 센다). segment_ids=None(Phase A/B)이면 3 + 파트 수."""
    if segment_ids is None:
        return len(CONTINUOUS_PARTS) + len(CANDIDATE_PARTS)
    n = len(CONTINUOUS_PARTS)
    for c in range(segment_ids.shape[1]):
        col = segment_ids[:, c]
        n += int(col.max().item()) + 1 if (col >= 0).any() else 0
    return n


def plot_gate_heatmap(z_gate, epoch, save_dir="reports/figures", pruning_state=None):
    """z_gate: (17, 5) 텐서. candidate 파트(3,4)만 시각화 — 연속 파트는 항상 1.0이라 무의미.
    [v1.1] pruning_state를 주면 DELETED 확정 칸에 'X' 마커를 오버레이한다."""
    os.makedirs(save_dir, exist_ok=True)
    cand = z_gate[:, CANDIDATE_PARTS].detach().cpu().numpy().T  # (len(CANDIDATE_PARTS), 17)
    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow(cand, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    if pruning_state is not None:
        deleted_np = (pruning_state['state'] == STATE_DELETED).cpu().numpy()  # (17, n_cand)
        for sec in range(NUM_SECTIONS):
            for c in range(deleted_np.shape[1]):
                if deleted_np[sec, c]:
                    ax.text(sec, c, 'X', ha='center', va='center',
                            fontsize=11, fontweight='bold', color='black')
    ax.set_yticks(range(len(CANDIDATE_PARTS)))
    ax.set_yticklabels([f'part-{p}' for p in CANDIDATE_PARTS])
    ax.set_xticks(range(NUM_SECTIONS))
    ax.set_xlabel('section_id (0=top/original ~ 16=bottom)')
    ax.set_title(f'z_gate heatmap @ epoch {epoch}' + (' (X=DELETED)' if pruning_state is not None else ''))
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'gate_heatmap_v12_e{epoch:04d}.png'), dpi=150)
    plt.close(fig)


PART_COLORS = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
PART_NAMES  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
REP_SECTIONS = [0, 8, 16]   # 대표 섹션(top/mid/bottom) — /synod design 세션 Gemini+OpenAI 수렴안


def visualize_training_multi(history, base_coords, final_coords, part_ids, section_ids,
                              target_mps, save_path="reports/v1_1/AI_design_v1_1_result.png",
                              final_seg_ids=None, split_epochs=None):
    """uni_section_v18.visualize_training()(line 1501-1627) 8-panel 스타일을 17섹션 joint 모델에
    맞게 재현. 17섹션×5파트를 그대로 그리면 범례가 터지므로(/synod design 세션 Gemini conf 100,
    OpenAI conf 82 공통 지적), panel 2(단면 형상)는 대표 섹션[0,8,16]만, panel 7/8(게이트/두께)은
    mean±min-max band로 압축한다."""
    part_ids_np = part_ids.cpu().numpy().astype(int)
    section_ids_np = section_ids.cpu().numpy().astype(int)
    base_np = base_coords.cpu().numpy()
    final_np = final_coords.cpu().numpy()
    epochs = list(range(len(history['loss'])))

    fig, axes = plt.subplots(2, 4, figsize=(26, 9))
    axes = axes.flatten()

    def add_stage_lines(ax):
        ax.axvline(ASWC_STAGE1_END, color='gray', linestyle='--', linewidth=1.0, label=f'Stage1 ({ASWC_STAGE1_END})')
        ax.axvline(ASWC_STAGE2_END, color='#E91E63', linestyle=':', linewidth=1.2, label=f'Stage2 ({ASWC_STAGE2_END})')

    ax = axes[0]
    ax.plot(epochs, history['loss'], color='#2196F3', linewidth=1.2, label='Total Loss')
    add_stage_lines(ax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_yscale('log')
    ax.set_title('Total Loss 수렴', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for sec in REP_SECTIONS:
        smask = section_ids_np == sec
        for pid in range(5):
            mask = smask & (part_ids_np == pid)
            if not mask.any():
                continue
            c = PART_COLORS[pid]
            ax.plot(base_np[mask, 0], base_np[mask, 1], 'o--', color=c, alpha=0.12, linewidth=1.0)
            ax.plot(final_np[mask, 0], final_np[mask, 1], 's-', color=c, alpha=0.9, linewidth=1.6,
                    label=f'sec{sec} {PART_NAMES[pid]}' if sec == REP_SECTIONS[0] else None)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'단면 형상: Base(옅음) vs Final — 대표 섹션 {REP_SECTIONS}', fontweight='bold')
    ax.legend(loc='best', fontsize=6, ncol=2); ax.grid(True, alpha=0.3); ax.axis('equal')

    ax = axes[2]
    ax.plot(epochs, [e * 100 for e in history['mp_rel_err']], color='#FF5722', linewidth=1.2, label='Mp rel err (%)')
    ax.axhline(2.0, color='gray', linestyle=':', linewidth=1.0, label='Feasibility 임계 (2%)')
    add_stage_lines(ax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mp 상대오차 (%)'); ax.set_yscale('log')
    ax.set_title('Mp 오차 추이 (17섹션 합산 기준)', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[3]
    for key, label in [('l_smooth', 'Smooth'), ('l_mass', 'Mass'), ('l_collision', 'Collision'),
                        ('l_order', 'Order'), ('l_sparse', 'Sparse'), ('l_anchor', 'Anchor'), ('l_sat', 'Sat')]:
        ax.plot(epochs, history[key], label=label, linewidth=1.0)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss term'); ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_title('보조 손실 항 추이', fontweight='bold'); ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(epochs, history['area'], color='#4CAF50', linewidth=1.5, label='Area (17섹션 합산, mm²)')
    add_stage_lines(ax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Area (mm²)')
    ax.set_title('단면적(질량) 추이', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(epochs, history['l_continuity'], color='#00BCD4', linewidth=1.3, label='Shape continuity loss')
    add_stage_lines(ax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('l_continuity')
    ax.set_title('인접 섹션 형상 연속성 loss', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[6]
    z_mean = np.array(history['z_gate_cand_mean'])   # (T, len(CANDIDATE_PARTS))
    z_min  = np.array(history['z_gate_cand_min'])
    z_max  = np.array(history['z_gate_cand_max'])
    for i, pid in enumerate(CANDIDATE_PARTS):
        c = PART_COLORS[pid]
        ax.plot(epochs, z_mean[:, i], color=c, linewidth=1.3, label=f'{PART_NAMES[pid]} mean')
        ax.fill_between(epochs, z_min[:, i], z_max[:, i], color=c, alpha=0.15, label=f'{PART_NAMES[pid]} min-max(17섹션)')
    ax.axvline(usv18.GATE_ACTIVE_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'GateActive ({usv18.GATE_ACTIVE_EPOCH})')
    ax.axhline(0.08, color='red', linestyle=':', linewidth=1.0, label='prune thresh (0.08)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('z_gate')
    ax.set_title('Candidate 파트 존재 게이트(z_gate) — 섹션 mean±min/max', fontweight='bold')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[7]
    t_mean = np.array(history['t_mean'])   # (T, 5)
    for pid in CONTINUOUS_PARTS:
        c = PART_COLORS[pid]
        ax.plot(epochs, t_mean[:, pid], color=c, linewidth=1.3, label=PART_NAMES[pid])
    # [v1.1] candidate 파트: 최종 seg_ids 기준 run별 두께 라인 — 분할 epoch에서 갈라지는
    # 형태가 동적 분할 성공의 직접 증거 (command_v2.md §5)
    t_full = np.array(history['t_full'])   # (T, 17, 5)
    linestyles = ['-', '--', '-.', ':']
    for i, pid in enumerate(CANDIDATE_PARTS):
        c = PART_COLORS[pid]
        if final_seg_ids is not None and (final_seg_ids[:, i] >= 0).any() \
                and int(final_seg_ids[:, i].max().item()) > 0:
            for r in range(int(final_seg_ids[:, i].max().item()) + 1):
                sec_mask = (final_seg_ids[:, i] == r).cpu().numpy()
                if not sec_mask.any():
                    continue
                ax.plot(epochs, t_full[:, sec_mask, pid].mean(axis=1), color=c,
                        linewidth=1.3, linestyle=linestyles[r % len(linestyles)],
                        label=f'{PART_NAMES[pid]}-run{r}')
        else:
            ax.plot(epochs, t_mean[:, pid], color=c, linewidth=1.3, label=PART_NAMES[pid])
    if split_epochs:
        for se in split_epochs:
            ax.axvline(se, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    add_stage_lines(ax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Thickness t (mm)')
    ax.set_title('파트별 두께 — candidate는 run별 라인(검은 점선=분할 이벤트)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [v1.1] 두께 그룹 수 추이를 panel 5(area)에 보조축으로 겹쳐 그림
    ax2 = axes[4].twinx()
    ax2.step(epochs, history['num_thickness_groups'], color='#9E9E9E', linewidth=1.0, where='post')
    ax2.set_ylabel('# thickness groups', color='#9E9E9E', fontsize=8)
    ax2.tick_params(axis='y', labelsize=7, colors='#9E9E9E')

    plt.suptitle('AI_design_v1_1 — 17-section joint training + Dynamic Splitting 결과',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] 학습 결과 dashboard 저장: {save_path}")


def plot_sections_3d_plotly_multi(base_coords, final_coords, part_ids, section_ids, final_z_gate,
                                   save_path="reports/figures/AI_design_v1_1_3d.html", z_coef=15.0,
                                   final_seg_ids=None):
    """initial_section.py의 plot_sections_3d_plotly()(line 253-282) 스타일 인터랙티브 3D HTML.
    [/synod design 세션 수렴안] Initial/Final 상태를 버튼으로 토글하는 단일 HTML로 만들고, candidate
    파트는 최종 z_gate 값에 비례해 투명도/선굵기를 조절해 "꺼진 섹션"을 시각적으로 표현한다."""
    import plotly.graph_objects as go

    part_ids_np = part_ids.cpu().numpy().astype(int)
    section_ids_np = section_ids.cpu().numpy().astype(int)
    base_np = base_coords.cpu().numpy()
    final_np = final_coords.cpu().numpy()
    z_gate_np = final_z_gate.cpu().numpy()   # (17, 5)

    fig = go.Figure()
    n_initial, n_final = 0, 0

    # ── Initial 상태 traces ──
    shown = set()
    for sec in range(NUM_SECTIONS):
        z = sec * z_coef
        for pid in range(5):
            mask = (section_ids_np == sec) & (part_ids_np == pid)
            if not mask.any():
                continue
            pts = base_np[mask]
            show_legend = pid not in shown
            shown.add(pid)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=PART_COLORS[pid], width=3),
                marker=dict(size=2, color=PART_COLORS[pid]),
                name=PART_NAMES[pid], legendgroup=PART_NAMES[pid], showlegend=show_legend,
                hovertemplate=f'[Initial] Section {sec}, {PART_NAMES[pid]}<br>x=%{{x:.1f}}, y=%{{y:.1f}}<extra></extra>',
                visible=True,
            ))
            n_initial += 1

    # ── Final 상태 traces (candidate 파트는 z_gate에 비례해 투명도/굵기 조절) ──
    shown = set()
    for sec in range(NUM_SECTIONS):
        z = sec * z_coef
        for pid in range(5):
            mask = (section_ids_np == sec) & (part_ids_np == pid)
            if not mask.any():
                continue
            pts = final_np[mask]
            g = float(z_gate_np[sec, pid]) if pid in CANDIDATE_PARTS else 1.0
            opacity = 0.15 + 0.85 * g
            width = 1.0 + 4.0 * g
            # [v1.1] candidate 파트는 run별 legendgroup 분리 — 분할된 조각을 범례에서 개별 토글 가능
            trace_name = PART_NAMES[pid]
            if final_seg_ids is not None and pid in CANDIDATE_PARTS:
                col = CANDIDATE_PARTS.index(pid)
                seg = int(final_seg_ids[sec, col].item())
                trace_name = f'{PART_NAMES[pid]}-run{seg}' if seg >= 0 else f'{PART_NAMES[pid]}-deleted'
            show_legend = trace_name not in shown
            shown.add(trace_name)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=PART_COLORS[pid], width=width),
                marker=dict(size=2, color=PART_COLORS[pid]),
                name=trace_name, legendgroup=trace_name, showlegend=show_legend,
                opacity=opacity,
                hovertemplate=(f'[Final] Section {sec}, {trace_name}<br>x=%{{x:.1f}}, y=%{{y:.1f}}'
                               + (f'<br>z_gate={g:.2f}' if pid in CANDIDATE_PARTS else '') + '<extra></extra>'),
                visible=False,
            ))
            n_final += 1

    total = n_initial + n_final
    show_initial = [True] * n_initial + [False] * n_final
    show_final = [False] * n_initial + [True] * n_final

    fig.update_layout(
        title='AI_design_v1 — 17-Section B-Pillar (Drag to rotate, scroll to zoom)',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title='Section (0=top/original ~ 16=bottom)', aspectmode='data'),
        updatemenus=[dict(
            type='buttons', direction='right', x=0.5, y=1.08, xanchor='center', yanchor='top', showactive=True,
            buttons=[
                dict(label='Initial State', method='update',
                     args=[{'visible': show_initial}, {'title': 'AI_design_v1 — Initial (base) shape'}]),
                dict(label='Final State', method='update',
                     args=[{'visible': show_final}, {'title': 'AI_design_v1 — Final (trained) shape, candidate 파트는 z_gate 비례 투명도'}]),
            ],
        )],
        width=1100, height=850,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"[viz] 인터랙티브 3D 저장: {save_path}")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: 형상 연속성 loss (command_v1.md §4.1 — 물리 part_id 사용으로 정정)
# ══════════════════════════════════════════════════════════════════

def compute_shape_continuity_loss(new_coords, section_ids, part_ids, threshold=2.0):
    """인접 섹션(k, k+1) 간, 같은 물리 part_id를 가진 노드끼리 스케일 정규화 후 hinge loss.
    threshold(mm) 이내 변화는 무패널티. candidate 파트가 z_gate≈0인 섹션에서는 두께가 이미 0에
    가까우므로, 이 loss가 걸려도 실제 구조에는 영향이 없다(command_v1.md §4.1 정정 사항)."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0

    unique_secs = torch.unique(section_ids).long().sort()[0]
    for i in range(len(unique_secs) - 1):
        k_a, k_b = int(unique_secs[i].item()), int(unique_secs[i + 1].item())
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)

        mask_a = section_ids == k_a
        mask_b = section_ids == k_b

        for pid in torch.unique(part_ids):
            pid_int = int(pid.item())
            ca = new_coords[mask_a & (part_ids == pid_int)] / s_a
            cb = new_coords[mask_b & (part_ids == pid_int)] / s_b
            if ca.shape[0] == 0 or cb.shape[0] == 0:
                continue
            dist = torch.cdist(ca, cb, p=2)
            min_dist, _ = torch.min(dist, dim=1)
            violation = torch.clamp(min_dist - threshold, min=0.0)
            total = total + torch.mean(violation ** 2)
            n_terms += 1

    if n_terms > 0:
        total = total / n_terms
    return total


def continuity_weight_schedule(epoch, stage2_end, w_max=1.0, w_min=0.05, beta=0.1,
                                w_floor=CONTINUITY_W_FLOOR):
    """[v1.7 §2] v1.5 sigmoid 그대로 + 하한선(review_v1_5.md 1차 원인 대응)만 추가.
    v1.6의 폐루프 제어(get_adaptive_continuity_weight, 상태 저장, EMA, eta 튜닝)는 래칫
    버그(review_v1_6.md에서 로그로 확인 — w_continuity가 전체 620 epoch 동안 1.000에 고정)
    를 이유로 완전히 제거한다."""
    w_base = w_min + (w_max - w_min) / (1.0 + math.exp(beta * (epoch - stage2_end)))
    return max(w_base, w_floor)


def build_static_neighbor_map(initial_coords, edge_index, edge_attr):
    """[v1.7 §1] compute_smoothness_loss_angle()(불변)의 두 결함 — (a) mean 희석,
    (b) 매 스텝 x좌표 재비교로 인한 검사 회피 — 을 근본적으로 우회한다. 초기(미변형)
    좌표 기준으로 좌/우 이웃을 딱 한 번만 결정해 영구히 고정한다 — 이후 좌표가 아무리
    뒤틀려도 이 노드가 "검사 대상에서 빠지는" 일이 없다.
    [v1.7 구현 세션 정정] 원본 함수와 동일하게 edge_type==0(물리 엣지)만 사용한다 —
    Gemini/OpenAI가 독립적으로 지적한 대로, 필터링 없이 전체 edge_index를 쓰면 join_pair
    등 비물리 엣지가 섞여 들어와 이웃 판정이 오염된다.
    정확히 2개의 물리 이웃을 가지며 초기 좌표에서 좌/우가 명확히 갈리는 노드만 포함한다
    (파트 접합부처럼 이웃이 3개 이상인 노드는 원본 함수와 동일하게 제외 — 범위 확장 없음)."""
    edge_type = edge_attr[:, 3]
    physical_mask = torch.isclose(edge_type, torch.zeros_like(edge_type))
    physical_edge_index = edge_index[:, physical_mask].cpu()   # [v1.7] CUDA tolist() 방지

    N = initial_coords.shape[0]
    adj = {i: [] for i in range(N)}
    for u, v in physical_edge_index.t().tolist():
        if u != v:
            adj[u].append(v)

    coords_cpu = initial_coords.detach().cpu()
    node_idx_list, left_idx_list, right_idx_list = [], [], []
    for i in range(N):
        neighbors = list(set(adj[i]))
        if len(neighbors) != 2:
            continue
        n1, n2 = neighbors
        x_i = coords_cpu[i, 0].item()
        x_n1, x_n2 = coords_cpu[n1, 0].item(), coords_cpu[n2, 0].item()
        if x_n1 < x_i and x_n2 > x_i:
            node_idx_list.append(i); left_idx_list.append(n1); right_idx_list.append(n2)
        elif x_n2 < x_i and x_n1 > x_i:
            node_idx_list.append(i); left_idx_list.append(n2); right_idx_list.append(n1)

    device = initial_coords.device
    return {
        'node_idx': torch.tensor(node_idx_list, dtype=torch.long, device=device),
        'left_idx': torch.tensor(left_idx_list, dtype=torch.long, device=device),
        'right_idx': torch.tensor(right_idx_list, dtype=torch.long, device=device),
    }


def compute_smoothness_lse(new_coords, neighbor_map, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.7 §1] 정적 이웃 쌍에 대한 Laplacian형 편차를 LogSumExp로 집계.
    mean()이 아니므로 소수의 심한 스파이크가 희석되지 않고(§0(a) 해결), 이웃 판정이
    고정돼 있으므로 스파이크가 생겨도 검사가 무효화되지 않는다(§0(b) 해결). LogSumExp의
    개별 노드 그래디언트는 softmax 출력이라 [0,1] 유계 — hard max와 달리 그래디언트
    폭발 메커니즘이 없다(design 세션에서 수식으로 증명, idea_v1_7.md §1.2 참고).
    DELETED 파트에 속한 노드도 그대로 포함한다 — compute_smoothness_loss_angle()(불변)도
    DELETED를 구분하지 않으므로 기존 베이스라인과 동일하게 동작시킨다(OpenAI가 마스킹을
    제안했으나 범위 확장으로 판단해 기각 — idea_v1_7.md 참고)."""
    node_idx, left_idx, right_idx = (neighbor_map['node_idx'], neighbor_map['left_idx'],
                                      neighbor_map['right_idx'])
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0   # [v1.7] 디바이스/dtype/autograd 그래프 자동 일치(Gemini 제안)
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)
    return temperature * torch.logsumexp(sq_deviation / temperature, dim=0)


def _huber_collision_term(violation: torch.Tensor, delta: float = HUBER_DELTA_MM) -> torch.Tensor:
    """[v1.5 §2] violation(mm, >=0)에 대한 Huber-squared 페널티.
    delta 이하: violation**2 (원본과 동일한 제곱 페널티, 얕은 관통에서 정밀한 그래디언트).
    delta 초과: delta*(2*violation - delta) (선형 성장 — 1차 미분이 delta에서 연속(C1),
    관통이 깊어져도 그래디언트가 2*delta로 상한이 걸려 폭발하지 않는다)."""
    return torch.where(
        violation <= delta,
        violation ** 2,
        delta * (2.0 * violation - delta),
    )


def compute_collision_penalty_unclamped(new_coords, t_final, part_ids, section_ids,
                                         collision_spec, penetration_floor=PENETRATION_FLOOR):
    """[v1.4 §2] usv18.compute_collision_loss_v5()와 동일한 시그니처·순회 구조를 쓰되, 위반량에
    .clamp(max=1.0)을 적용하지 않는다 — 원본은 관통 1mm 이상에서 그래디언트가 정확히 0이 되어
    되돌아올 유인을 잃는다(review_v1_3.md [ERROR]2). new_coords/t_final은 매 스텝 살아있는 텐서를
    받는다 — collision_spec은 sigma/clearance 메타데이터만 담고 좌표는 담지 않으므로, 캐시된
    좌표를 재사용하면 autograd 그래프가 끊겨 그래디언트가 전혀 흐르지 않는다.
    [v1.5 §2] 무클램프 "제곱" 자체는 유지하되(얕은 관통의 정밀한 그래디언트는 그대로 보존), 깊은
    관통에서의 폭주만 _huber_collision_term()으로 상한을 건다 — W_COL_AUX=5.0이 상시 곱해지는
    게이트 요동 구간(epoch 11~100)에서의 그래디언트 폭발이 근본 크래시 원인이었다(review_v1_4.md)."""
    total = torch.tensor(0.0, device=new_coords.device, requires_grad=True)
    n_dirs = 0
    for (sec_int, a, b), directions in collision_spec.items():
        sec_mask = (section_ids == sec_int)
        part_coords = {
            a: new_coords[sec_mask & (part_ids == a)],
            b: new_coords[sec_mask & (part_ids == b)],
        }
        part_t = {
            a: t_final[sec_mask & (part_ids == a)].mean(),
            b: t_final[sec_mask & (part_ids == b)].mean(),
        }
        for d in directions:
            c_seg = part_coords[d['seg_part']]
            c_pt  = part_coords[d['pt_part']]
            if c_seg.shape[0] < 2 or c_pt.shape[0] == 0:
                continue
            proj, valid = usv18._signed_projections(c_seg, c_pt)
            if valid.sum() == 0:
                continue
            t_sum_half = (part_t[d['seg_part']] + part_t[d['pt_part']]) / 2.0
            gap = d['sigma'] * proj - t_sum_half - d['clearance']
            # ★ 핵심 차이: .clamp(max=1.0) 없음 — 관통이 깊을수록 계속 커진다(단 [v1.5] Huber
            # 상한으로 그래디언트 자체는 2*HUBER_DELTA_MM에서 폭주를 멈춘다)
            violation = torch.relu(-gap - penetration_floor) * valid.float()
            loss_dir = _huber_collision_term(violation).sum() / valid.float().sum()   # [v1.5 §2]
            total = total + loss_dir
            n_dirs += 1
    if n_dirs > 0:
        total = total / n_dirs
    return total


# ══════════════════════════════════════════════════════════════════
# SECTION 4: GP 기반 목표 Mp (command_v1.md §5, 변경 없음)
# ══════════════════════════════════════════════════════════════════

INITIAL_MP_TABLE = [
    21_637_199, 23_286_535, 24_996_460, 26_766_975, 28_598_079,
    30_489_775, 32_442_063, 34_454_943, 36_528_416, 38_662_482,
    40_857_143, 43_112_398, 45_428_249, 47_804_695, 50_241_738,
    52_739_377, 55_297_613,
]


def generate_gp_target_mp(seed=None, sigma=0.25, ell=3.0):
    """initial_section.md §2 표 기준, 공간 상관 있는 GP 스무딩 +/-50% 목표 Mp 생성.
    반환: {section_id(int): target_mp(float)}."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    initial_mp = np.array(INITIAL_MP_TABLE, dtype=np.float64)
    sections = np.arange(NUM_SECTIONS, dtype=np.float64)
    K = sigma ** 2 * np.exp(-(sections[:, None] - sections[None, :]) ** 2 / (2 * ell ** 2))
    K += 1e-6 * np.eye(NUM_SECTIONS)
    alpha = rng.multivariate_normal(np.zeros(NUM_SECTIONS), K)
    alpha = np.clip(alpha, -0.5, 0.5)

    target_mp = initial_mp * (1.0 + alpha)
    return {k: float(target_mp[k]) for k in range(NUM_SECTIONS)}


# ══════════════════════════════════════════════════════════════════
# SECTION 4.5: [v1.2] Initial vs Final Design Property 비교 리포트
# ══════════════════════════════════════════════════════════════════

def _per_section_mpl(coords, t, fy, edge_index, edge_attr, section_ids, device):
    """섹션별 Mp 계산 — train_step_multi의 물리 엣지 필터링 + 로컬 재인덱싱과 동일 패턴.
    (섹션 간 엣지는 build 단계에서 생기지 않지만, 물리 엣지(edge_type==0)만 남기는 필터는 필요)"""
    out = {}
    src, dst = edge_index
    edge_type = edge_attr[:, 3]
    for k in range(NUM_SECTIONS):
        m = section_ids == k
        emask = m[src] & m[dst] & torch.isclose(edge_type, torch.zeros_like(edge_type))
        local = torch.full((coords.shape[0],), -1, dtype=torch.long, device=device)
        local[m] = torch.arange(int(m.sum()), device=device)
        e_sec = local[edge_index[:, emask]]
        out[k] = float(usv18.calculate_mpl(coords[m], t[m], fy[m], e_sec).item())
    return out


@torch.no_grad()
def report_design_comparison(model, data, target_mps, pruning_state, final_seg_ids, device,
                              temperature=None, save_path="reports/AI_design_v1_8_report.md",
                              death_log=None, target_area=None):
    """[v1.3] initial vs final design property 비교 리포트.
    - Mp: soft(학습 상태, t_final=t_raw*z_gate)와 hard(제조 상태) 병기. hard 존재 판정은
      (state != DELETED) AND (z_gate >= BINARIZE_THRESHOLD) — [v1.3 §4] 0.5 → 0.3, 비교연산자 >=.
    - 두께: 제조 관점 값은 pooled t_raw(soft 게이트가 곱해진 t_final이 아님). 삭제 칸은 0.
    - 괴리 |hard-soft| > 2%인 섹션은 경고 표시(fine-tuning 필요 신호, Gemini 지적).
    [v1.2 수정] temperature를 하드코딩(1.0)하지 않고 학습 종료 시점의 usv18.compute_gate_temperature
    값을 호출부에서 전달받는다 — 온도가 다르면 같은 log_alpha라도 HardConcrete 하드 샘플이 학습
    당시와 달라져(예: 250 epoch 분석에서 Patch1이 state=ALIVE인데도 리포트만 X로 표시된 원인 중 하나),
    리포트가 실제 학습 종료 상태를 반영하지 못하는 문제가 있었다. temperature=None이면 이전 동작과
    호환되도록 1.0을 fallback으로 사용한다."""
    if temperature is None:
        temperature = 1.0
    model.eval()
    x = data.x
    section_ids = x[:, 5].long()
    part_ids = x[:, 4].long()
    fy = x[:, 7].unsqueeze(1)
    t_init = x[:, 6].unsqueeze(1)
    base_coords = x[:, :2]

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=device)
    for k in range(NUM_SECTIONS):
        target_mp_node[section_ids == k] = target_mps[k]

    fix_x = x[:, 2].bool().unsqueeze(1)
    fix_y = x[:, 3].bool().unsqueeze(1)

    new_coords, _, t_soft, delta_t_part, z_gate, _ = model(
        x, data.edge_index, data.edge_attr, target_mp_node, fix_x, fix_y, data.join_pairs,
        thickness_gate=1.0, gate_active=True, temperature=temperature, segment_ids=final_seg_ids)

    # t_raw 복원: forward의 두께 변환식 재현 (t_final은 soft 게이트가 곱해져 제조값이 아님)
    t_min, t_max = model.T_MIN, model.T_MAX
    t_raw = t_min + (t_max - t_min) * torch.sigmoid(_t_init_logit(t_init, t_min, t_max) + delta_t_part)

    # hard 존재 판정 (17, n_cand): 상태기계 AND 결정론적 게이트
    # [v1.3 §4] 0.5 → BINARIZE_THRESHOLD(0.3), 비교연산자 >= 로 통일(§5.2와 판정 일치)
    state = pruning_state['state']                                    # CPU (17, n_cand)
    cand_idx = torch.tensor(CANDIDATE_PARTS, device=device)
    exists_cand = (state.to(device) != STATE_DELETED) & (z_gate[:, cand_idx] >= BINARIZE_THRESHOLD)
    exists_full = torch.ones(NUM_SECTIONS, model.num_parts, dtype=torch.bool, device=device)
    exists_full[:, cand_idx] = exists_cand
    exists_node = exists_full[section_ids, part_ids].unsqueeze(1)
    t_hard = t_raw * exists_node.float()

    # 섹션별 Mp: initial / final-soft / final-hard
    mp_init = _per_section_mpl(base_coords, t_init, fy, data.edge_index, data.edge_attr, section_ids, device)
    mp_soft = _per_section_mpl(new_coords, t_soft, fy, data.edge_index, data.edge_attr, section_ids, device)
    mp_hard = _per_section_mpl(new_coords, t_hard, fy, data.edge_index, data.edge_attr, section_ids, device)

    area_init, _ = usv18.compute_section_area(base_coords.cpu(), t_init.cpu(), data.edge_index.cpu())
    area_soft, _ = usv18.compute_section_area(new_coords.cpu(), t_soft.cpu(), data.edge_index.cpu())
    area_hard, _ = usv18.compute_section_area(new_coords.cpu(), t_hard.cpu(), data.edge_index.cpu())

    lines = []
    lines.append("# AI_design_v1_2 — Initial vs Final Design Property Report\n")
    lines.append(f"(hard existence 판정 temperature={temperature:.4f} — 학습 종료 시점 어닐링 온도와 일치)\n")
    lines.append("## 1. Section-wise Mp (N*mm)\n")
    lines.append("| sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |")
    lines.append("|----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|")
    n_warn = 0
    for k in range(NUM_SECTIONS):
        tgt = target_mps[k]
        ach = mp_hard[k] / tgt * 100.0
        gap = abs(mp_hard[k] - mp_soft[k]) / max(mp_soft[k], 1e-9)
        flag = ""
        if gap > 0.02:
            flag += f"soft-hard gap {gap*100:.1f}% "
            n_warn += 1
        if abs(ach - 100.0) > 2.0:
            flag += "target miss"
        lines.append(f"| {k} | {mp_init[k]:,.0f} | {tgt:,.0f} | {mp_soft[k]:,.0f} | "
                     f"{mp_hard[k]:,.0f} | {ach:.1f} | {flag} |")
    tot_i, tot_t = sum(mp_init.values()), sum(target_mps.values())
    tot_s, tot_h = sum(mp_soft.values()), sum(mp_hard.values())
    lines.append(f"| **sum** | {tot_i:,.0f} | {tot_t:,.0f} | {tot_s:,.0f} | {tot_h:,.0f} | "
                 f"{tot_h/tot_t*100:.1f} | |")

    lines.append("\n## 2. Total cross-section area (mm^2, 17-section sum)\n")
    lines.append("| initial | final(soft) | final(hard) | delta(hard-init) |")
    lines.append("|--------:|------------:|------------:|-----------------:|")
    lines.append(f"| {float(area_init):,.1f} | {float(area_soft):,.1f} | {float(area_hard):,.1f} | "
                 f"{float(area_hard) - float(area_init):+,.1f} ({(float(area_hard)/float(area_init)-1)*100:+.1f}%) |")
    if target_area is not None:
        # [v1.3 §2.6.4] baseline(area_init, t_init 기준)은 target_area와 독립적으로 재계산되므로
        # target_area를 낮춰도 initial vs final 비교의 기준선은 그대로 실제 설계값이다.
        lines.append(f"\n| 감량 목표 | {target_area:,.1f} mm² (초기 대비 {target_area/float(area_init)*100:.1f}%) |")
        lines.append(f"| 달성률 | {float(area_hard)/target_area*100:.1f}% (100% 이하면 목표 달성) |")

    # [v1.3 §4.2] 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트 — §4 효과 측정 지표
    zone = (state.to(device) != STATE_DELETED) & (z_gate[:, cand_idx] > 0.08) \
                                               & (z_gate[:, cand_idx] < 0.5)
    lines.append(f"\n- 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **{int(zone.sum())}개** "
                 f"(임계 {BINARIZE_THRESHOLD} 적용 시 구제: "
                 f"{int(((z_gate[:, cand_idx] >= BINARIZE_THRESHOLD) & zone).sum())}개)")

    if death_log:
        n_suspect = sum(1 for d in death_log if d["verdict"] == "SUSPECT")
        lines.append(f"\n## 4. 전역 게이트 사망 원장 (§2.4.1) — SUSPECT {n_suspect}건 / 전체 {len(death_log)}건\n")
        lines.append("| epoch | sec | cand | Mp@사망 | area@사망 | verdict |")
        lines.append("|------:|----:|-----:|--------:|----------:|:--------|")
        for d in death_log:
            lines.append(f"| {d['epoch']} | {d['sec']} | {d['cand']} | {d['mp_at_death']:.1%} | "
                         f"{d['area']:,.0f} | {d['verdict']} |")

    lines.append("\n## 3. Part thickness (mm, 제조 기준 = pooled t_raw)\n")
    lines.append("| part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |")
    lines.append("|:-----|----------:|:------------------|:--------------------------------|")
    for pid in range(5):
        pm = part_ids == pid
        ti = float(t_init[pm].mean())
        if pid in CONTINUOUS_PARTS:
            tf = float(t_raw[pm].mean())
            lines.append(f"| {PART_NAMES[pid]} | {ti:.3f} | {tf:.3f} (전 섹션 공통) | 항상 존재 |")
        else:
            c = CANDIDATE_PARTS.index(pid)
            exist_map = ''.join('O' if bool(exists_cand[k, c]) else 'X' for k in range(NUM_SECTIONS))
            if final_seg_ids is not None and (final_seg_ids[:, c] >= 0).any() \
                    and int(final_seg_ids[:, c].max().item()) > 0:
                runs = []
                for r in range(int(final_seg_ids[:, c].max().item()) + 1):
                    secs = (final_seg_ids[:, c] == r).nonzero().flatten().tolist()
                    if not secs:
                        continue
                    rm = pm & torch.isin(section_ids, torch.tensor(secs, device=device))
                    runs.append(f"run{r}(sec{secs[0]}-{secs[-1]})={float(t_raw[rm].mean()):.3f}")
                tf_str = ', '.join(runs)
            else:
                tf_str = f"{float(t_raw[pm].mean()):.3f} (run 1개)"
            n_alive = int(exists_cand[:, c].sum())
            lines.append(f"| {PART_NAMES[pid]} | {ti:.3f} | {tf_str} | {exist_map} ({n_alive}/17 alive) |")

    if n_warn:
        lines.append(f"\n> [WARNING] soft-hard Mp 괴리 >2%인 섹션 {n_warn}개 — 게이트가 아직 0/1로 "
                     f"수렴하지 않았습니다. 추가 fine-tuning epoch를 권장합니다.")

    text = '\n'.join(lines)
    print('\n' + text + '\n')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"[report] 비교 리포트 저장: {save_path}")
    return {'mp_init': mp_init, 'mp_soft': mp_soft, 'mp_hard': mp_hard,
            'area': (float(area_init), float(area_soft), float(area_hard)),
            'exists_cand': exists_cand.cpu()}


# ══════════════════════════════════════════════════════════════════
# SECTION 5: Stage 0 — Static Shape/Index Dry-Run (command_v1.md §6.1, 정정 반영)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_static_dry_run(data, model, target_mps, device):
    """학습 없는(zero-grad) 정적 검증. [v1.1 변경] Phase A/B가 candidate도 글로벌 pooling이므로
    candidate 파트에도 두께 불변 검사를 적용한다(v1의 "candidate는 검사 제외" 정정을 다시 뒤집음).
    추가로 인위 DELETED 주입 → run별 독립 pooling 검증을 수행한다(command_v2.md §4).
    model.eval() 필수 — HardConcrete train 모드 난수가 균일 두께 가정을 깨뜨림(/synod design, o3)."""
    print("[Stage 0] Static shape/index dry-run 시작...")
    was_training = model.training
    model.eval()
    x = data.x

    assert x.shape[1] == 8, f"노드 feature가 8열이어야 함(정정: logical_part_id 없음) (실제: {x.shape[1]})"
    section_ids = x[:, 5].long()
    assert section_ids.min().item() == 0 and section_ids.max().item() == NUM_SECTIONS - 1, \
        f"section_id 범위가 [0,{NUM_SECTIONS-1}]이어야 함"
    part_ids = x[:, 4].long()
    for k in range(NUM_SECTIONS):
        parts_in_section = set(int(p.item()) for p in torch.unique(part_ids[section_ids == k]))
        assert parts_in_section == {0, 1, 2, 3, 4}, \
            f"섹션 {k}에 5-part 전부가 없음(사전 필터링 금지 위반): {parts_in_section}"
    assert data.edge_index.max().item() < x.shape[0], "edge_index가 노드 수를 벗어남"

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=device)
    for section in torch.unique(section_ids):
        target_mp_node[section_ids == section] = target_mps[int(section.item())]

    fix_x_mask = x[:, 2].bool().unsqueeze(1)
    fix_y_mask = x[:, 3].bool().unsqueeze(1)

    # [v1.1] thickness_gate=1.0 — delta_t를 살려야 글로벌 pooling 균일성 검사가 실질적이다
    # (v1은 0.0이었으나 그 경우 delta=0이라 어떤 pooling 구조든 검사를 통과해버림)
    new_coords, _, t_final, _, z_gate, z_open = model(
        x, data.edge_index, data.edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, data.join_pairs,
        thickness_gate=1.0, gate_active=True, temperature=1.0,
    )
    assert not torch.isnan(new_coords).any(), "forward() 결과 new_coords에 NaN"
    assert not torch.isnan(t_final).any(), "forward() 결과 t_final에 NaN"

    assert z_gate.shape == (NUM_SECTIONS, model.num_parts), f"z_gate shape 오류: {z_gate.shape}"
    assert torch.allclose(z_gate[:, CONTINUOUS_PARTS], torch.ones_like(z_gate[:, CONTINUOUS_PARTS])), \
        "연속 파트(S_PROTECT)의 z_gate가 1.0으로 고정되지 않음"

    # [v1.1] Phase A/B(segment_ids=None)에서는 연속+candidate 5파트 전부 파트당 두께 균일해야 함
    for pid in range(5):
        mask = (part_ids == pid)
        t_vals = t_final[mask].detach()
        spread = (t_vals.max() - t_vals.min()).item()
        assert spread < 1e-4, (
            f"part_id={pid}의 두께가 파트 내에서 다름(spread={spread:.6f}) - "
            f"Phase A/B 글로벌 pooling(composite_key) 오류"
        )

    # ── [v1.1] 인위 분할 주입 테스트: 섹션 5~7의 Patch1(part 3)을 DELETED로 가정 ──
    fake_state = make_pruning_state_multi()
    fake_state['state'][5:8, 0] = STATE_DELETED
    seg_ids = compute_segment_ids(fake_state).to(device)
    assert seg_ids.shape == (NUM_SECTIONS, len(CANDIDATE_PARTS)), f"seg_ids shape 오류: {seg_ids.shape}"
    assert (seg_ids[5:8, 0] == -1).all() and seg_ids[0, 0] == 0 and seg_ids[16, 0] == 1, \
        f"CCL 결과 오류: {seg_ids[:, 0].tolist()}"

    # thickness_gate=1.0로 호출해야 delta_t가 살아있어 pooling 그룹 구조를 실제로 검증할 수 있다
    # (0.0이면 delta가 전부 0이라 어떤 pooling이든 균일해져 검사가 무의미해짐)
    _, _, t_split, _, _, _ = model(
        x, data.edge_index, data.edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, data.join_pairs,
        thickness_gate=1.0, gate_active=True, temperature=1.0, segment_ids=seg_ids,
    )
    part3 = (part_ids == 3)
    top_mask = part3 & (section_ids < 5)          # run 0
    bot_mask = part3 & (section_ids > 7)          # run 1
    for name, m in [("run0(top)", top_mask), ("run1(bottom)", bot_mask)]:
        tv = t_split[m].detach()
        spread = (tv.max() - tv.min()).item()
        assert spread < 1e-4, f"분할 주입 테스트: part3 {name} 내부 두께가 균일하지 않음(spread={spread:.6f})"
    # 미분할 파트(Patch2)는 여전히 run 1개 = 파트 전체 균일
    part4 = (part_ids == 4)
    tv4 = t_split[part4].detach()
    assert (tv4.max() - tv4.min()).item() < 1e-4, "분할 주입 테스트: part4(미분할)가 균일하지 않음"
    print(f"[Stage 0] 분할 주입 테스트 통과 - part3이 run 2개로 독립 pooling됨 "
          f"(그룹 수: {count_thickness_groups(seg_ids)})")

    if was_training:
        model.train()

    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"[Stage 0] GPU 메모리 사용량: {mem_mb:.1f} MB")

    print(f"[Stage 0] 통과 - 노드 {x.shape[0]}개, 엣지 {data.edge_index.shape[1]}개, "
          f"z_gate shape {tuple(z_gate.shape)}, 5파트 글로벌 두께 불변 + 분할 주입 검증 완료.")
    return True


# ══════════════════════════════════════════════════════════════════
# SECTION 6: train_step_multi / run_training_multi (command_v1.md §4)
# ══════════════════════════════════════════════════════════════════

ASWC_STAGE1_END = 100   # Global Coarse Fit: 좌표 freeze, thickness만 학습
ASWC_STAGE2_END = 400   # Sequential Refinement: 좌표+두께 동시 학습, 종료 시 연속 파트 두께 detach
ASWC_STAGE3_END = 500   # Global Relaxation: 좌표만 재학습, 두께는 계속 detach
# [v1.2 수정] 250 epoch 실행 결과, g3/g4가 250 epoch 내내 0.74~0.96 사이에 머물러 DELETED
# 임계값(thresh_low=0.08)에 전혀 도달하지 못했고(분할 이벤트 0회), 상단 섹션(0~8)의 Mp가
# 목표 대비 최대 147%로 수렴하지 못했다(reports/v1_2/AI_design_v1_2.md 분석). 각 스테이지를
# 2배로 늘려 게이트 수렴과 좌표 재분배에 더 긴 시간을 준다. GATE_ACTIVE_EPOCH/TEMP_WARMUP_EPOCHS
# 등 uni_section_v18의 절대 epoch 상수는 그대로이므로 게이트 활성화 자체는 기존과 동일 시점에
# 시작하고, 늘어난 구간은 순수하게 추가 수렴 시간으로 쓰인다.
GATE_VIZ_STRIDE = 20    # z_gate 히트맵 저장 주기


def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None, neighbor_map=None):
    # [v1.5 §3.2] sentinel 패턴 — 기본값을 모듈 정의 시점에 W_COL_AUX로 정적 바인딩하지 않고,
    # 호출 시점에 None이면 전역 상수를 참조한다(Stage4 호출부처럼 웜업 인자를 안 넘기는 경우의
    # 안전한 폴백; design 세션 재검증에서 Gemini가 제안).
    if w_col_aux_val is None:
        w_col_aux_val = W_COL_AUX
    model.train()
    optimizer.zero_grad()

    x          = data.x
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    join_pairs = data.join_pairs
    base_coords = x[:, :2].detach()

    fix_x_mask  = x[:, 2].bool().unsqueeze(1)
    fix_y_mask  = x[:, 3].bool().unsqueeze(1)
    part_ids    = x[:, 4]
    section_ids = x[:, 5]
    fy          = x[:, 7].unsqueeze(1)

    unique_sections = torch.unique(section_ids)

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
    for section in unique_sections:
        section_mask = (section_ids == section)
        target_mp_node[section_mask] = target_mps[int(section.item())]

    gate = usv18.thickness_gate_value(epoch)
    gate_active = epoch >= usv18.GATE_ACTIVE_EPOCH
    current_temp = usv18.compute_gate_temperature(epoch)

    new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open = model(
        x, edge_index, edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, join_pairs, thickness_gate=gate,
        gate_active=gate_active, temperature=current_temp, segment_ids=segment_ids
    )

    if epoch < ASWC_STAGE1_END:
        new_coords = base_coords.detach() + 0.0 * (new_coords - base_coords.detach())

    if epoch >= ASWC_STAGE2_END and not stage4_active:      # [v1.3 §5.4] Stage 4는 두께 재학습 허용
        continuous_mask = torch.isin(part_ids.long(),
                                      torch.tensor(CONTINUOUS_PARTS, device=x.device))
        t_final = torch.where(continuous_mask.unsqueeze(1), t_final.detach(), t_final)

    l_phys_terms, pred_mp_tensors, pred_mp_sections = [], [], []
    for section in unique_sections:
        section_mask = (section_ids == section)
        coords_section = new_coords[section_mask]
        t_section = t_final[section_mask]
        fy_section = fy[section_mask]

        src, dst = edge_index
        edge_mask = section_mask[src] & section_mask[dst]
        edge_type = edge_attr[:, 3]
        physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))
        edge_index_section = edge_index[:, physical_mask]

        local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
        edge_index_section = local_index[edge_index_section]

        pred_mp_section = usv18.calculate_mpl(coords_section, t_section, fy_section, edge_index_section)

        section_int = int(section.item())
        target_mp_section = torch.tensor(target_mps[section_int], dtype=torch.float32, device=x.device)

        l_phys_terms.append(usv18.asymmetric_huber_phys(pred_mp_section, target_mp_section))
        pred_mp_tensors.append(pred_mp_section)
        pred_mp_sections.append(pred_mp_section.item())

    l_phys_total = torch.stack(l_phys_terms).mean()
    pred_mp_total = torch.stack(pred_mp_tensors).sum()
    target_mp_total = torch.tensor(sum(target_mps.values()), dtype=torch.float32, device=x.device)

    pred_mp_sections = np.array(pred_mp_sections)
    mp_rel_err = float(np.abs(np.sum(pred_mp_sections) - sum(target_mps.values())) / sum(target_mps.values()))

    newly_deleted = torch.zeros(NUM_SECTIONS, len(CANDIDATE_PARTS), dtype=torch.bool)
    if pruning_state is not None:
        cand_idx = torch.tensor(CANDIDATE_PARTS, device=x.device)
        z_open_cand = z_open[:, cand_idx]                 # (17, len(CANDIDATE_PARTS))
        # [v1.1] 반환값을 버리지 않는다 — 동적 분할(Phase C 전환)의 트리거 (command_v2.md §3)
        newly_deleted = update_pruning_state_multi(pruning_state, z_open_cand)
        # 적응형 TAU_GATE는 원본과 동일하게 mp_rel_err 기반(섹션 게이트와 무관, 전역 스칼라 유지)
        current_tau_gate = usv18.TAU_GATE
    else:
        current_tau_gate = usv18.TAU_GATE

    gate_multiplier = float(torch.sigmoid(torch.tensor(usv18.SPARSE_K * (current_tau_gate - mp_rel_err))).item())
    w_sparse_effective = weights['w_sparse'] * gate_multiplier
    l_sparse = z_open[:, CANDIDATE_PARTS].mean()   # (17, len(CANDIDATE_PARTS)) 전체 평균
    contrib_sparse = w_sparse_effective * l_sparse

    contrib_entropy = torch.tensor(0.0, device=x.device)
    if gate_active:
        z_cand = z_open[:, CANDIDATE_PARTS]
        entropy = -torch.mean(z_cand * torch.log(z_cand + usv18.ENTROPY_EPS)
                               + (1.0 - z_cand) * torch.log(1.0 - z_cand + usv18.ENTROPY_EPS))
        contrib_entropy = (alpha_ent * gate_multiplier) * entropy   # [v1.3 §2.3] usv18.ALPHA_ENT(0.01) → 인자화

    if frozen_curriculum is not None:
        s_phys, s_smooth = frozen_curriculum   # [v1.3 §5.6] Stage4: 루프 탈출 시점 값으로 고정, 외삽 금지
    else:
        s_phys, s_smooth = 1.0, 1.0
        if curriculum:
            s_phys, s_smooth = usv18.get_curriculum_weights_v10(epoch, max_epochs, curriculum_ratio)

    l_smooth    = usv18.compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
    l_smooth_lse = compute_smoothness_lse(new_coords, neighbor_map)   # [v1.7 §1]
    # [v1.3 §2.1 정정] usv18.compute_mass_loss()가 반환하는 l_mass는 로깅 전용이며 loss에 더해지지
    # 않는다(실제 질량 항은 아래 §2.6 l_mass_budget). area만 재사용하고 l_mass_legacy는 로깅에만 쓴다.
    area, l_mass_legacy = usv18.compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)
    l_mass = torch.relu(area / target_area - 1.0) ** 2   # [v1.3 §2.6] 목표 초과분만 이차 벌점(예산 제약)
    l_area_floor = torch.relu(1.0 - area / (SAFETY_RATIO * target_area + 1e-12)) ** 2  # [v1.4 §3]
    # z_gate는 collision loss에서 part별 스칼라를 기대(원본 line 632 z_gate[part]) — 섹션마다 다를 수
    # 있으므로 섹션 평균을 사용(연속 파트는 항상 1이라 영향 없음, candidate 파트만 근사가 생김을 인지).
    z_gate_part_avg = z_gate.mean(dim=0)  # (5,)
    l_collision = usv18.compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                                   collision_spec, z_gate=z_gate_part_avg)
    l_col_aux = compute_collision_penalty_unclamped(new_coords, t_final, part_ids, section_ids,
                                                     collision_spec)   # [v1.4 §2] 무클램프 보강
    l_order     = usv18.compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr)
    l_anchor    = usv18.compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask)
    l_sat       = usv18.compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE)

    if w_continuity_val is None:                                          # [v1.6 §1]
        w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
    else:
        w_continuity = w_continuity_val
    l_continuity = compute_shape_continuity_loss(new_coords, section_ids, part_ids, threshold=2.0)
    contrib_continuity = w_continuity * l_continuity

    L_alda, g_Mp_val, g_col_val = usv18.compute_alda_loss(
        area, target_area, pred_mp_total, target_mp_total, l_collision, alda_state)
    L_alda_effective = L_alda * max(gate, 0.05)

    contrib_phys   = weights['w_phys']   * l_phys_total * s_phys
    contrib_smooth = weights['w_smooth'] * l_smooth     * s_smooth
    contrib_order  = weights['w_order']  * l_order
    contrib_anchor = weights['w_anchor'] * l_anchor
    contrib_sat    = weights['w_sat']    * l_sat
    contrib_mass   = weights['w_mass']   * l_mass        # [v1.3 §2.6.2-a] stage 무관, 항상 계산·적용
    contrib_col_aux    = w_col_aux_val * l_col_aux                    # [v1.5 §3] 웜업 스케줄 값 사용
    contrib_area_floor = weights['w_area_floor'] * l_area_floor       # [v1.4 §3]
    contrib_smooth_lse = W_SMOOTH_LSE * l_smooth_lse                  # [v1.7 §1]

    loss = (contrib_phys + contrib_smooth + L_alda_effective
            + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparse + contrib_entropy + contrib_continuity
            + contrib_mass + contrib_col_aux + contrib_area_floor
            + contrib_smooth_lse)                                     # [v1.7 §1]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    # ── 시각화용 (section, part) 평균 두께 — group_by 없이 per-part 마스크 평균으로 계산 ──
    with torch.no_grad():
        t_per_part_section = torch.zeros(NUM_SECTIONS, 5, device=x.device)
        for pid in range(5):
            pmask = part_ids.long() == pid
            for sec in range(NUM_SECTIONS):
                smask = pmask & (section_ids.long() == sec)
                if smask.any():
                    t_per_part_section[sec, pid] = t_final[smask].mean()

    return {
        "loss": loss.item(), "mp_rel_err": mp_rel_err,
        "l_continuity": l_continuity.item(), "w_continuity": w_continuity,
        "area": area.item(), "z_gate": z_gate.detach(),
        "gate_multiplier": gate_multiplier, "thickness_gate": gate,
        "l_smooth": l_smooth.item(), "l_smooth_lse": l_smooth_lse.item(),   # [v1.7 §1]
        "l_mass": l_mass_legacy.item(), "contrib_mass": contrib_mass.item(),
        "l_col_aux": l_col_aux.item(), "contrib_col_aux": contrib_col_aux.item(),        # [v1.4 §2]
        "l_area_floor": l_area_floor.item(), "contrib_area_floor": contrib_area_floor.item(),  # [v1.4 §3]
        "l_collision": l_collision.item(), "l_order": l_order.item(),
        "l_anchor": l_anchor.item(), "l_sat": l_sat.item(), "l_sparse": l_sparse.item(),
        "t_per_part_section": t_per_part_section.detach(),  # (17, 5)
        "new_coords": new_coords.detach(),
        "newly_deleted": newly_deleted,                     # (17, n_cand) bool, CPU — [v1.1] 분할 트리거
        "mp_per_section": pred_mp_sections,                 # [v1.3] 사망 원장 SUSPECT/JUSTIFIED 판정용
        "target_mp_per_section": np.array([target_mps[int(s.item())] for s in unique_sections]),
    }


def run_training_multi(data, target_mps, target_area=None, max_epochs=ASWC_STAGE3_END,
                        lr=1e-3, weights=None, curriculum=True, curriculum_ratio=(0.2, 0.7),
                        area_target_ratio=AREA_TARGET_RATIO, w_mass=W_MASS_V13,
                        w_area_floor=W_AREA_FLOOR):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = CGDN17(
        in_channels=8, hidden_channels=128, num_layers=4, heads=4, edge_dim=4,
        max_displacement=50.0, num_parts=5, init_log_alpha=INIT_LOG_ALPHA_V14,  # [v1.4] §4.1
    ).to(device)

    run_static_dry_run(data, model, target_mps, device)

    thick_param_ids = {id(p) for p in model.thickness_decoder.parameters()}
    gate_param_ids  = {id(model.log_alpha_candidates)}
    main_params  = [p for p in model.parameters() if id(p) not in thick_param_ids and id(p) not in gate_param_ids]
    thick_params = list(model.thickness_decoder.parameters())
    gate_params  = [model.log_alpha_candidates]
    optimizer = optim.AdamW(
        [{'params': main_params, 'name': 'main'},
         {'params': thick_params, 'name': 'thickness_decoder'},
         {'params': gate_params, 'name': 'gate_params', 'lr': 0.0}],
        lr=lr, weight_decay=1e-4)

    if weights is None:
        weights = {'w_phys': 10.0, 'w_order': W_ORDER_V18, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,   # [v1.8 §1]
                   'w_sparse': W_SPARSE_V13,      # [v1.3 §2.1] usv18.W_SPARSE(0.5) → 0.0
                   'w_mass':   w_mass,            # [v1.3 §2.6] l_mass 예산 제약 가중치
                   'w_area_floor': w_area_floor}  # [v1.4 §3.4] 면적 하한 가드레일 가중치

    alda_state = usv18.make_alda_state()
    pruning_state = make_pruning_state_multi()

    x = data.x
    part_ids = x[:, 4]
    section_ids = x[:, 5]

    usv18.verify_thickness_gradient(x[:, :2], x[:, 6:7], x[:, 7:8], data.edge_index)

    area_init_ref, _ = usv18.compute_section_area(x[:, :2].cpu(), x[:, 6:7].cpu(), data.edge_index.cpu())
    if target_area is None:
        target_area = area_target_ratio * float(area_init_ref)     # [v1.3 §2.6.1] 감량 목표
    print(f"[mass] area_init = {float(area_init_ref):.1f} mm^2 | target_area = {target_area:.1f} mm^2 "
          f"(감량 목표 {area_target_ratio:.0%})")

    collision_spec = usv18.build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids)
    print(f"[collision v5] {len(collision_spec)}쌍(섹션x파트쌍) 부호 앵커 산정 완료")

    # [v1.7 §1] 정적 이웃 맵 — 초기(미변형) 좌표 기준 1회 구축, collision_spec과 동일 패턴.
    initial_coords = x[:, :2].clone().detach()
    neighbor_map = build_static_neighbor_map(initial_coords, data.edge_index, data.edge_attr)
    print(f"[smooth-lse] 정적 이웃 맵 {neighbor_map['node_idx'].numel()}개 노드"
          f"(전체 {x.shape[0]}개 중) 구축 완료")

    history = {
        'loss': [], 'mp_rel_err': [], 'l_continuity': [], 'area': [],
        'l_smooth': [], 'l_mass': [], 'l_collision': [], 'l_order': [],
        'l_anchor': [], 'l_sat': [], 'l_sparse': [],
        # 원본 배열은 저장하지 않고(메모리), 매 epoch 요약치만 저장 — /synod design 세션(Gemini/OpenAI
        # 수렴안): mean + min/max band로 17섹션 x 2 candidate 게이트, 17섹션 x 5 파트 두께를 압축.
        'z_gate_cand_mean': [], 'z_gate_cand_min': [], 'z_gate_cand_max': [],   # 각 (2,)
        't_mean': [], 't_min': [], 't_max': [],                                 # 각 (5,)
        # [v1.1] 동적 분할 추적: pooling 그룹 수(계단식 증가가 분할 성공의 1차 증거) +
        # run별 두께 시각화용 (17,5) 원본 스냅샷(250~370 epoch x 85 float — 메모리 무시 가능)
        'num_thickness_groups': [], 't_full': [],
    }
    base_coords_snapshot = x[:, :2].detach().clone()

    print(f"\n{'='*78}\n[ AI_design_v1_1 ] ASWC Training + Dynamic Splitting | 17 sections | "
          f"Section-aware gating ({NUM_SECTIONS}x{len(CANDIDATE_PARTS)}) | Stage1<{ASWC_STAGE1_END} "
          f"Stage2<{ASWC_STAGE2_END} Stage3<{ASWC_STAGE3_END}\n{'='*78}")

    # [v1.1] curriculum/온도/gate 스케줄 계산용 max_epochs는 초기값으로 동결하고(스케줄 불연속 방지,
    # /synod design 세션 검증), adaptive extension은 루프 상한(loop_max)만 늘린다.
    curriculum_max_epochs = max_epochs
    loop_max = max_epochs
    extension_cap = max_epochs + 120     # 무한 연장 방지 상한 (command_v2.md §3.3)
    seg_ids = None                       # Phase A/B: None → candidate 글로벌 pooling
    split_epochs = []

    gate_stage_done = False
    epoch = 0
    last_epoch = -1
    death_log = []                                              # [v1.3 §2.4.1] 전역 사망 원장
    dead0_prev = torch.zeros(NUM_SECTIONS, len(CANDIDATE_PARTS), dtype=torch.bool)
    area_hist = {}                                              # [v1.3 §7.0 R3] epoch -> area

    # [v1.4 §4.2] 엔트로피 지연·어닐링 — ASWC_STAGE2_END 전까지 0, 이후 ALPHA_ENT_ANNEAL_EPOCHS에
    # 걸쳐 0 → ALPHA_ENT_V13으로 선형 상승. Stage4 호출부는 별도(alpha_ent_active=0.0 고정)라 무관.
    alpha_ent_start = ALPHA_ENT_START_EPOCH if ALPHA_ENT_START_EPOCH is not None else ASWC_STAGE2_END

    def alpha_ent_schedule(ep):
        if ep < alpha_ent_start:
            return 0.0
        t = min(1.0, (ep - alpha_ent_start) / ALPHA_ENT_ANNEAL_EPOCHS)
        return ALPHA_ENT_V13 * t

    # [v1.5 §3] W_COL_AUX 코사인 웜업 — §4 엔트로피 스케줄(alpha_ent_start=400)과 독립적으로 설계됨.
    # floor는 절대값(design 세션 Judge 판정, §7.2). 종료 시점(80)은 실측 폭증 시점(epoch 90) 이전.
    def w_col_aux_schedule(ep):
        if ep < W_COL_AUX_WARMUP_START:
            return W_COL_AUX_FLOOR
        if ep >= W_COL_AUX_WARMUP_END:
            return W_COL_AUX
        span = W_COL_AUX_WARMUP_END - W_COL_AUX_WARMUP_START
        progress = (ep - W_COL_AUX_WARMUP_START) / span
        cos_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
        return W_COL_AUX_FLOOR + (W_COL_AUX - W_COL_AUX_FLOOR) * cos_factor

    any_gate_crossed_half = False   # [v1.4 §5] R4 상태 — Stage4 루프까지 이어지므로 루프 밖에서 선언

    while epoch < loop_max:
        model.current_epoch = epoch    # [v1.6 §2] compute_gates_multi()의 첨예화 스케줄용
        # ── [원본 uni_section_v18.run_training() line 1303-1310과 동일] ──
        # gate_params(log_alpha_candidates)는 optimizer 생성 시 lr=0.0으로 묶여 있다가
        # GATE_ACTIVE_EPOCH 시점에 별도로 lr=1e-2로 풀린다. 이 활성화를 빠뜨리면 34개 게이트가
        # 전혀 학습되지 않고 초기값(sigmoid(2.0)≈0.88 부근)에 고정된다 — 스모크 테스트에서
        # candidate_gate_mean이 205 epoch 내내 요지부동인 것으로 실제 확인된 버그.
        if (not gate_stage_done) and epoch == usv18.GATE_ACTIVE_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'gate_params':
                    group['lr'] = 1e-2
            gate_stage_done = True
            print(f"[Gate Active] epoch {epoch}: gate_params lr -> 1e-2 "
                  f"(section-aware 게이트 학습 시작)")

        curr_w_col_aux = w_col_aux_schedule(epoch)                        # [v1.5 §3]
        curr_w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)   # [v1.7 §2] 단순화
        info = train_step_multi(model, data, optimizer, target_mps, target_area,
                                 epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                                 collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                                 alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                                 w_col_aux_val=curr_w_col_aux,
                                 w_continuity_val=curr_w_continuity,
                                 neighbor_map=neighbor_map)  # [v1.4 §4.2][v1.5 §3][v1.7 §1][v1.7 §2]
        last_epoch = epoch                                      # [v1.3 §5.1] 루프 탈출 시점 기록
        area_hist[epoch] = info['area']

        # [v1.4 §5] R4 — 삭제-부재 감시(진단 전용). log_alpha<0 ⟺ z_gate<0.5 (stretched sigmoid
        # γ=-0.1,ζ=1.1에서 la=0일 때 정확히 z=0.5이므로 sigmoid 재계산 없이 부호만 보면 된다).
        if not any_gate_crossed_half and bool((model.log_alpha_candidates < 0.0).any()):
            any_gate_crossed_half = True
            print(f"[R4] epoch {epoch}: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)")
        if epoch == ASWC_STAGE2_END + R4_GRACE_EPOCHS and not any_gate_crossed_half:
            print(f"  !! [R4 WARNING] epoch {epoch} 시점까지 게이트가 단 하나도 0.5 밑으로 "
                  f"내려가지 않았습니다 — 삭제 유도력이 여전히 부족할 수 있습니다 "
                  f"(command_v1_4.md §4 파라미터 재검토 권장).")

        # [v1.3 §2.4.1] 전역 사망 원장 — 전 섹션 × 전 후보, 특정 섹션 하드코딩 금지
        la = model.log_alpha_candidates.detach().cpu()
        alive_ledger = (pruning_state['state'] != STATE_DELETED)
        dead0 = (la < -2.4) & alive_ledger
        newly_dead_ledger = dead0 & (~dead0_prev)
        if newly_dead_ledger.any():
            mp_arr = info['mp_per_section']
            tgt_arr = info['target_mp_per_section']
            for k, c in newly_dead_ledger.nonzero().tolist():
                mp_k = float(mp_arr[k] / tgt_arr[k]) if tgt_arr[k] else 0.0
                verdict = "JUSTIFIED" if mp_k >= DEATH_MP_THRESHOLD else "SUSPECT"
                death_log.append({"epoch": epoch, "sec": k, "cand": CANDIDATE_PARTS[c],
                                   "mp_at_death": mp_k, "area": info['area'], "verdict": verdict})
                if verdict == "SUSPECT":
                    print(f"  !! SUSPECT DEATH sec{k} cand{CANDIDATE_PARTS[c]} @ep{epoch} "
                          f"(Mp={mp_k:.1%} < {DEATH_MP_THRESHOLD:.0%}, area={info['area']:,.0f})")
        dead0_prev = dead0.clone()

        # [v1.3 §7.0] 롤백 규칙 R1~R3 — 전 섹션 무관, 전 섹션 × 전 후보 공통 적용
        if epoch >= ASWC_STAGE2_END and any(d["verdict"] == "SUSPECT" for d in death_log):
            raise RuntimeError(f"[R1] SUSPECT death detected — {death_log[-1]}. "
                                f"ALPHA_ENT_V13를 0.02로 낮춰 재시도할 것 (command_v1_3.md §7.0).")
        recent_deaths = [d for d in death_log if d["epoch"] > epoch - DEATH_BURST_WINDOW]
        n_alive_now = int(alive_ledger.sum())
        if n_alive_now > 0 and len(recent_deaths) >= DEATH_BURST_RATIO * n_alive_now:
            raise RuntimeError(f"[R2] 연쇄 붕괴: {DEATH_BURST_WINDOW}ep 내 {len(recent_deaths)}개 사망. "
                                f"W_MASS_V13를 5.0으로 낮춰 재시도할 것 (command_v1_3.md §7.0).")
        area_ref = area_hist.get(epoch - AREA_REGRESS_WINDOW)
        if area_ref is not None and info['area'] > area_ref \
                and any(d["epoch"] > epoch - AREA_REGRESS_WINDOW for d in death_log):
            raise RuntimeError(f"[R3] 질량 역설: 게이트가 죽는 중 area 증가 "
                                f"({area_ref:,.0f} -> {info['area']:,.0f}). "
                                f"ALPHA_ENT_V13를 0.02로 낮춰 재시도할 것 (command_v1_3.md §7.0).")

        # ── [v1.1] 동적 분할 이벤트 (command_v2.md §3.2): DELETED 확정마다 seg_ids 재계산 ──
        if info['newly_deleted'].any():
            seg_ids = compute_segment_ids(pruning_state).to(device)
            split_epochs.append(epoch)
            n_runs = [(int(seg_ids[:, c].max().item()) + 1 if (seg_ids[:, c] >= 0).any() else 0)
                      for c in range(len(CANDIDATE_PARTS))]
            print(f"[Dynamic Split] epoch {epoch}: DELETED 확정 "
                  f"{info['newly_deleted'].nonzero().tolist()} -> candidate run 수 {n_runs} "
                  f"(두께 그룹 {count_thickness_groups(seg_ids)}개로 재구성)")
            # Adaptive Epoch Extension: 분할 후 최소 60 epoch의 독립 최적화 시간 보장
            needed = min(epoch + 60, extension_cap)
            if needed > loop_max:
                loop_max = needed
                print(f"[Adaptive Extension] loop_max -> {loop_max} "
                      f"(curriculum 스케줄은 {curriculum_max_epochs} 기준으로 동결 유지)")
        history['loss'].append(info['loss'])
        history['mp_rel_err'].append(info['mp_rel_err'])
        history['l_continuity'].append(info['l_continuity'])
        history['area'].append(info['area'])
        history['l_smooth'].append(info['l_smooth'])
        history['l_mass'].append(info['l_mass'])
        history['l_collision'].append(info['l_collision'])
        history['l_order'].append(info['l_order'])
        history['l_anchor'].append(info['l_anchor'])
        history['l_sat'].append(info['l_sat'])
        history['l_sparse'].append(info['l_sparse'])

        z_cand = info['z_gate'][:, CANDIDATE_PARTS]     # (17, len(CANDIDATE_PARTS))
        history['z_gate_cand_mean'].append(z_cand.mean(dim=0).cpu().numpy())
        history['z_gate_cand_min'].append(z_cand.min(dim=0).values.cpu().numpy())
        history['z_gate_cand_max'].append(z_cand.max(dim=0).values.cpu().numpy())

        t_sec_part = info['t_per_part_section']   # (17, 5)
        history['t_mean'].append(t_sec_part.mean(dim=0).cpu().numpy())
        history['t_min'].append(t_sec_part.min(dim=0).values.cpu().numpy())
        history['t_max'].append(t_sec_part.max(dim=0).values.cpu().numpy())
        history['num_thickness_groups'].append(count_thickness_groups(seg_ids))
        history['t_full'].append(t_sec_part.cpu().numpy())

        if epoch % 10 == 0 or epoch == loop_max - 1:
            # [v1.2] candidate 게이트를 파트별 개별 출력 — 생존(non-DELETED) 섹션만 평균
            # (o3 지적: 삭제 섹션의 0을 평균에 넣으면 추세가 왜곡됨), 생존 수 병기.
            gate_strs = []
            for c, pid in enumerate(CANDIDATE_PARTS):
                alive = pruning_state['state'][:, c] != STATE_DELETED          # (17,) CPU
                zc = info['z_gate'][:, pid].cpu()
                g = zc[alive].mean().item() if alive.any() else 0.0
                gate_strs.append(f"g{pid}={g:.3f}({int(alive.sum())}/{NUM_SECTIONS})")
            sat_ratio = float((la.abs() > 2.4).float().mean())   # [v1.3 §2.4]
            print(f"Epoch {epoch:4d} || loss={info['loss']:.4f} | MpErr={info['mp_rel_err']*100:.2f}% "
                  f"| l_continuity={info['l_continuity']:.4f}(w={info['w_continuity']:.3f}) "
                  f"| area={info['area']:.1f} | {' '.join(gate_strs)} "
                  f"| groups={history['num_thickness_groups'][-1]} "
                  f"| log_alpha_sat={sat_ratio:.2%} | dead0={int(dead0.sum())}/{int(alive_ledger.sum())} "
                  f"| suspect={sum(1 for d in death_log if d['verdict']=='SUSPECT')} "
                  f"| alpha_ent={alpha_ent_schedule(epoch):.4f} "
                  f"| l_col_aux={info['l_col_aux']:.4f}(w={curr_w_col_aux:.2f}) "
                  f"| w_continuity={curr_w_continuity:.3f} "
                  f"| l_smooth_lse={info['l_smooth_lse']:.4f}")   # [v1.4 §4.3][v1.5 §4][v1.6 §1][v1.7 §1]

        if epoch % GATE_VIZ_STRIDE == 0 or epoch == loop_max - 1:
            plot_gate_heatmap(info['z_gate'], epoch, pruning_state=pruning_state)

        if epoch == loop_max - 1:
            final_coords = info['new_coords'].detach().clone()
            final_z_gate = info['z_gate'].detach().clone()
            final_epoch = epoch   # [v1.2 수정] 리포트의 온도를 학습 종료 시점과 맞추기 위해 기록

        epoch += 1

    if split_epochs:
        print(f"\n[Dynamic Split] 분할 이벤트 epoch: {split_epochs}, 최종 두께 그룹 {count_thickness_groups(seg_ids)}개")
    else:
        print("\n[Dynamic Split] 분할 이벤트 없음 - candidate는 끝까지 파트당 1 두께 유지 (Phase A/B)")

    # ══════════════════════════════════════════════════════════════
    # [v1.3 §5] Stage 4 — 존재 확정 후 형상·두께 재수렴
    # ══════════════════════════════════════════════════════════════
    assert last_epoch >= 0, "본 학습 루프가 한 번도 실행되지 않았다"
    la = model.log_alpha_candidates.detach().cpu()
    alive_final = (pruning_state['state'] != STATE_DELETED)
    ambiguous = int((alive_final & (la.abs() <= 2.4)).sum())

    if ENABLE_STAGE4 is None:
        enter_stage4 = (ambiguous > 0)
    else:
        enter_stage4 = bool(ENABLE_STAGE4)

    stage4_start_epoch = last_epoch + 1     # [v1.3 §5.1] final_epoch(리포트용)와는 별개 변수
    assert stage4_start_epoch >= curriculum_max_epochs, \
        f"Stage4 시작점({stage4_start_epoch})이 커리큘럼 종료({curriculum_max_epochs})보다 앞선다"

    if enter_stage4:
        stage4_start = stage4_start_epoch
        stage4_end = stage4_start_epoch + STAGE4_EPOCHS
        print(f">>> Stage 4: 존재 확정 + 형상/두께 재수렴 ({stage4_start} -> {stage4_end}), "
              f"애매 게이트 {ambiguous}개")

        with torch.no_grad():
            model.eval()                                        # HardConcrete 난수 차단 (필수)
            z_gate_final, _ = model.compute_gates_multi(
                training=False, temperature=usv18.compute_gate_temperature(stage4_start))
            model.train()

            cand_idx_t = torch.tensor(CANDIDATE_PARTS, device=device)
            zc = z_gate_final[:, cand_idx_t].detach().cpu()      # (17, n_cand)
            alive_sc = (pruning_state['state'] != STATE_DELETED)

            # ★ DELETED 부활 없음 (사용자 확정) — state와 임계값을 AND로 결합
            hard_exists = alive_sc & (zc >= BINARIZE_THRESHOLD)

            for k in range(NUM_SECTIONS):
                for c in range(len(CANDIDATE_PARTS)):
                    model.log_alpha_candidates[k, c] = 5.0 if hard_exists[k, c] else -5.0

        model.log_alpha_candidates.requires_grad_(False)

        seg_ids = compute_segment_ids(pruning_state, hard_exists=hard_exists).to(device)

        # 옵티마이저 재빌드 — 실제 변수명(main_params/thick_params) 사용, gate_params 그룹 제외.
        # log_alpha_candidates를 requires_grad_(False)로 만든 뒤에 재빌드해 잔여 캡처를 막는다.
        optimizer = optim.AdamW(
            [{'params': main_params,  'name': 'main',              'lr': lr * STAGE4_LR_SCALE},
             {'params': thick_params, 'name': 'thickness_decoder', 'lr': lr * STAGE4_LR_SCALE}],
            lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)

        weights['w_sparse'] = 0.0
        alpha_ent_active = 0.0
        # [v1.3 §5.6] 커리큘럼은 루프 탈출 시점 값으로 1회 계산 후 고정 재사용(외삽 방지)
        frozen_s_phys, frozen_s_smooth = usv18.get_curriculum_weights_v10(
            last_epoch, curriculum_max_epochs, curriculum_ratio)

        for ep in range(stage4_start, stage4_end):
            # [v1.5] w_col_aux_val 미전달 -> sentinel이 전역 W_COL_AUX(5.0)로 폴백. Stage4는
            # 항상 웜업 종료 시점(epoch 80) 이후에만 시작하므로(stage4_start > ASWC_STAGE3_END
            # 근방) 무해함 — design 세션 재검증에서 OpenAI 지적 사항을 코드로 확인.
            # [v1.7 §1] neighbor_map은 sentinel 기본값이 없다 — Stage4는 최종 좌표 재수렴 구간
            # 이라 이 보강 손실이 가장 필요한 지점이므로 반드시 전달한다(command_v1_7.md §1.4.4,
            # 누락 시 compute_smoothness_lse() 내부에서 NoneType 인덱싱 에러로 즉시 크래시).
            info = train_step_multi(model, data, optimizer, target_mps, target_area,
                                     ep, curriculum_max_epochs, weights, curriculum=curriculum,
                                     curriculum_ratio=curriculum_ratio, collision_spec=collision_spec,
                                     alda_state=alda_state, pruning_state=pruning_state,
                                     segment_ids=seg_ids, alpha_ent=alpha_ent_active, stage4_active=True,
                                     frozen_curriculum=(frozen_s_phys, frozen_s_smooth),
                                     neighbor_map=neighbor_map)
            if ep % 20 == 0 or ep == stage4_end - 1:
                print(f"[Stage4] epoch {ep:4d} || loss={info['loss']:.4f} "
                      f"| MpErr={info['mp_rel_err']*100:.2f}% | area={info['area']:.1f}")

        final_coords = info['new_coords'].detach().clone()
        final_z_gate = info['z_gate'].detach().clone()

        # §5.7 종료 assert — |log_alpha|=5로 완전 포화했으므로 z_gate는 정확히 0.0/1.0이어야 하고,
        # 이는 곧 t_soft(=t_raw*z_gate)와 t_hard(=t_raw*hard_exists)가 부동소수점 오차 내로 같다는 뜻이다.
        with torch.no_grad():
            model.eval()
            z_gate_chk, _ = model.compute_gates_multi(training=False, temperature=1.0)
            model.train()
            zc_chk = z_gate_chk[:, cand_idx_t].cpu()
        assert torch.allclose(zc_chk, hard_exists.float(), atol=1e-6), \
            (f"Stage 4 종료 후에도 z_gate가 0/1로 포화하지 않음 "
             f"(max diff={float((zc_chk - hard_exists.float()).abs().max()):.3e}) — "
             f"§4/§5.2 비교 연산자(>=) 불일치부터 확인할 것 (command_v1_3.md §5.7)")
    else:
        print(">>> Stage 4: 진입 조건 미충족(애매 게이트 0개) — 3단계 결과를 최종으로 사용")

    # [v1.2] pruning_state + final_epoch 반환 — 비교 리포트의 hard 존재 판정(state != DELETED)과
    # 온도(current_temp) 일치를 위해 필요 (하드코딩된 temperature=1.0 사용 시 학습 종료 시점의
    # 어닐링된 온도와 어긋나 z_gate 하드 샘플이 실제 학습 상태와 달라지는 문제 수정)
    # [v1.3 §5.9] death_log·enter_stage4 추가 — 9-tuple → 11-tuple
    return model, history, base_coords_snapshot, final_coords, final_z_gate, seg_ids, split_epochs, \
        pruning_state, final_epoch, death_log, enter_stage4


# ══════════════════════════════════════════════════════════════════
# [v1.5 §2/§3 검증] Huber 연속성 + 웜업 경계값 스모크 테스트
# w_col_aux_schedule은 run_training_multi() 내부 클로저(§3.1 설계 그대로)라 여기서 직접 단위
# 테스트할 수 없다 — 경계값 검증은 §7.3 스모크 테스트(20 epoch 실행, 콘솔 로그의 w= 값 확인)로
# 대체한다. Huber 헬퍼는 모듈 레벨 함수이므로 여기서 바로 검증 가능.
# ══════════════════════════════════════════════════════════════════
_d = HUBER_DELTA_MM
assert torch.isclose(_huber_collision_term(torch.tensor(_d)), torch.tensor(_d ** 2))
assert torch.isclose(_huber_collision_term(torch.tensor(2 * _d)), torch.tensor(_d * (2 * 2 * _d - _d)))
_eps = 1e-4
_lo = _huber_collision_term(torch.tensor(_d - _eps))
_hi = _huber_collision_term(torch.tensor(_d + _eps))
assert abs((_hi - _lo).item() / (2 * _eps) - 2 * _d) < 0.5   # 경계에서 기울기가 2*delta로 수렴
del _d, _eps, _lo, _hi


# ══════════════════════════════════════════════════════════════════
# [v1.7 §2 검증] 단순화된 연속성 하한 경계값 스모크 테스트
# ══════════════════════════════════════════════════════════════════
assert abs(continuity_weight_schedule(0, 400) - 1.0) < 1e-6         # epoch 0: sigmoid ~ w_max
assert continuity_weight_schedule(1000, 400) >= CONTINUITY_W_FLOOR - 1e-9   # 절대 하한 준수
assert abs(continuity_weight_schedule(1000, 400) - CONTINUITY_W_FLOOR) < 1e-6  # 충분히 지난 epoch -> floor 도달


# ══════════════════════════════════════════════════════════════════
# SECTION 7: __main__
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ASWC_STAGE3_END)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-only", action="store_true",
                         help="Stage 0 static dry-run만 수행하고 종료 (학습 없음)")
    parser.add_argument("--area-target-ratio", type=float, default=AREA_TARGET_RATIO,
                         help="감량 목표 비율. 1.0 = 초기 단면적 유지, 0.95 = 5%% 감량 (기본)")
    parser.add_argument("--w-mass", type=float, default=W_MASS_V13,
                         help="l_mass 가중치. 0.0이면 예산 제약 비활성")
    parser.add_argument("--w-area-floor", type=float, default=W_AREA_FLOOR,
                         help="l_area_floor(면적 하한 가드레일) 가중치. 0.0이면 비활성")
    # Jupyter(ipykernel_launcher)로 실행하면 sys.argv에 "--f=...kernel-....json" 같은 커널 인자가
    # 섞여 들어와 parse_args()가 SystemExit(2)로 죽는다 — parse_known_args()로 낯선 인자는 무시한다.
    args, _unknown = parser.parse_known_args()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    data = build_bpillar_17section()
    target_mps = generate_gp_target_mp(seed=args.seed)

    print(f"[data] 17-section 그래프(사전 필터링 없음): 노드 {data.x.shape[0]}개, 엣지 {data.edge_index.shape[1]}개")
    print(f"[target_mp] {[f'{v:,.0f}' for v in target_mps.values()]}")

    if args.dry_run_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGDN17(in_channels=8, hidden_channels=128, num_layers=4, heads=4, edge_dim=4,
                        max_displacement=50.0, num_parts=5,
                        init_log_alpha=INIT_LOG_ALPHA_V14).to(device)  # [v1.4] §4.1
        assert_thickness_reachability(data.x)          # [v1.3 §3.4]
        run_static_dry_run(data.to(device), model, target_mps, device)
    else:
        assert_thickness_reachability(data.x)          # [v1.3 §3.4] 전 구간 도달성 사전 검증

        model, history, base_coords, final_coords, final_z_gate, final_seg_ids, split_epochs, \
            pruning_state, final_epoch, death_log, enter_stage4 = run_training_multi(
                data, target_mps, max_epochs=args.epochs,
                area_target_ratio=args.area_target_ratio, w_mass=args.w_mass,
                w_area_floor=args.w_area_floor)   # [v1.4] w_area_floor 추가

        torch.save(model.state_dict(), "weights/bpillar_17sec_v1_8.pt")   # [v1.8] v1_7 가중치와 분리

        part_ids = data.x[:, 4]
        section_ids = data.x[:, 5]
        seg_cpu = final_seg_ids.cpu() if final_seg_ids is not None else None
        visualize_training_multi(history, base_coords, final_coords, part_ids, section_ids, target_mps,
                                 save_path="reports/figures/AI_design_v1_8_result.png",
                                 final_seg_ids=seg_cpu, split_epochs=split_epochs)
        plot_sections_3d_plotly_multi(base_coords, final_coords, part_ids, section_ids, final_z_gate,
                                      save_path="reports/figures/AI_design_v1_8_3d.html",
                                      final_seg_ids=seg_cpu)

        # ── initial vs final design property 비교 리포트 ──
        # 온도를 1.0으로 하드코딩하지 않고 학습 종료 시점(final_epoch)의 어닐링 온도를 그대로 사용
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_temp = usv18.compute_gate_temperature(final_epoch)
        print(f"[report] hard existence 판정용 temperature={final_temp:.4f} (epoch {final_epoch} 기준)")
        # [v1.3 §2.6.1] run_training_multi()와 동일한 결정론적 계산으로 target_area 재현
        area_init_for_report, _ = usv18.compute_section_area(
            data.x[:, :2].cpu(), data.x[:, 6:7].cpu(), data.edge_index.cpu())
        target_area_for_report = args.area_target_ratio * float(area_init_for_report)
        report_design_comparison(model, data.to(device), target_mps, pruning_state,
                                 final_seg_ids, device, temperature=final_temp,
                                 death_log=death_log, target_area=target_area_for_report)

        print(f">>> Stage 4 진입 여부: {enter_stage4}")
        print("[done] weights/bpillar_17sec_v1_8.pt, reports/figures/AI_design_v1_8_result.png, "
              "reports/figures/AI_design_v1_8_3d.html, reports/AI_design_v1_8_report.md 저장 완료")
