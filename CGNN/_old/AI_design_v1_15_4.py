"""
AI_design_v1_15_4.py — target_mp_v1_15_4 적용 + My 목표 없이 결과만 리포트 (v1.15.4)
───────────────────────────────────
[v1.15.4 변경, 사용자 직접 요청]
  base는 AI_design_v1_15_3.py이며 물리/학습 로직·리포트 구조는 전혀 건드리지 않았다.
  변경은 목표값 CSV 교체와 산출물 파일명 분리 두 가지뿐이다.

  §1 target_mp 입력을 target_mp_v1_15_3.csv -> **target_mp_v1_15_4.csv**로 교체
     (--target-csv 기본값). target_mp_v1_15_4.csv는 섹션별 초기 전소성 모멘트(Mp initial)의
     **1.3배**로 설정된 mp_target을 담고 있으며, 컬럼은 v1.15.3과 동일하게 index,mp_target
     두 개뿐이다(my_target 없음 — 선택 컬럼 처리는 v1.15.2 §1 그대로 유지).
     v1.15.3의 1.1배보다 목표를 더 **올린** 설정이므로 학습은 두께/단면적을 더 크게 키우는
     방향으로 작동한다 — 1.5배(v1_15_1) / 0.7배(v1_15_2) / 1.1배(v1_15_3)와 같은 코드에
     CSV만 바꿔 끼우는 실험 계열의 네 번째 조건이다.
     (수치 생성 근거: target_mp_v1_15_3.csv의 각 값 = 초기 Mp x 1.1이므로,
      이번 CSV는 동일한 초기 Mp에 1.3을 곱한 값 — 즉 v1_15_3 값 / 1.1 * 1.3을 반올림했다.
      index 0=최하단, 16=최상단 대응은 v1.15.3과 동일하게 유지.)
  §2 산출물 파일명 v1_15_3 -> v1_15_4 (가중치/그림/3D/리포트/final_section CSV, gate heatmap)
     — v1.15.3 결과를 덮어쓰지 않도록 분리. 이력 주석의 [v1.15.x §y] 태그는 출처 표기이므로
     그대로 남겼다.


[v1.15.3 변경, 사용자 직접 요청]
  base는 AI_design_v1_15_2.py이며, target_mp_v1_15_3.csv(초기 Mp의 1.1배) 교체와 산출물
  파일명 v1_15_2 -> v1_15_3 분리 두 가지뿐이었다(물리/학습 로직 무변경).


[v1.15.2 변경, 사용자 직접 요청]
  base는 AI_design_v1_15.py이며 물리/학습 로직은 전혀 건드리지 않았다. 변경은 목표값 입력과
  리포트 출력 두 곳뿐이다.

  §1 target_mp 입력을 target_mp.csv -> **target_mp_v1_15_2.csv**로 교체(--target-csv 기본값).
     target_mp_v1_15_2.csv는 섹션별 초기 전소성 모멘트(Mp initial)의 0.7배로 설정된 mp_target을
     담고 있으며, 컬럼은 index,mp_target 두 개뿐이다. 초기값보다 목표를 **낮춘** 설정이므로
     학습은 두께/단면적을 줄이는 방향으로 작동한다(1.5배로 올려둔 target_mp_v1_15_1.csv와
     대조되는 조건 — 같은 코드에 CSV만 바꿔 끼우는 실험 구성이다).
     load_target_mp_my_from_csv()에서 my_target을 **선택 컬럼**으로 완화했다 — 컬럼이 없으면
     target_my=None을 반환한다. 항복 모멘트의 목표값은 모르기 때문에 애초에 목표를 세우지
     않는다는 사용자 결정이고, 없는 목표를 임의로 추정해 채우지 않는다.
     (my_target 컬럼이 있는 CSV를 주면 v1.15와 동일하게 목표 대비로 병기한다 — 하위 호환.)
  §2 report_design_comparison(): target_mys=None일 때 v1.15는 My 표 자체를 생략했으나,
     v1.15.2은 **final_my를 반드시 출력**한다. 목표를 모르는 것과 결과를 안 보는 것은 별개다.
     target_mys=None이면 My target / hard-target 달성률 컬럼만 빼고
     initial / final(soft) / final(hard) + final(hard)/initial 비율을 출력한다.
     My는 v1.12부터 계속 리포트 전용이다 — 단면 생성/학습/손실에 전혀 주입되지 않는다.
  §3 산출물 파일명 v1_15 -> v1_15_2 (가중치/그림/3D/리포트/final_section CSV) — v1.15 결과를
     덮어쓰지 않도록 분리. 이력 주석의 [v1.15 §x] 태그는 출처 표기이므로 그대로 남겼다.

[v1.15 변경, docs/command/command_v1_15.md 실행]
  v1.14 잔여 겹침(Plate-Inner 0.4119mm, Outer-Plate 0.0620mm)은 설계 결함이 아니라
  모델링 아티팩트임이 실측으로 확증됐다(idea_v1_15.md §0): 초기 형상의 일부 고정
  노드쌍은 중심선 거리가 (t_a+t_b)/2와 1e-6mm 이내로 일치하는 **표면 접촉** 상태로
  설계된 용접 플랜지이고, 두께 증가분으로 예측한 겹침(+0.0620/+0.4119mm)이 실측과
  소수점 4자리까지 정확히 일치한다. 양쪽 노드가 모두 BC 고정이라 좌표 자유도가 0이므로
  collision loss는 기하학적으로 만족 불가능한 제약을 강요해 두께를 부당하게 깎았다
  (Mp 달성률 78.8% 정체).

  §1 import uni_section_v22 -> uni_section_v23(별칭 usv18 유지). v23은 fixed-fixed 항목을
     collision loss에서 제외한다(ff_mask, AND 조건, 로컬 인덱스 공간).
  §3 identify_mating_flange_pairs(): 런타임 임계값(x_tol/gap0_max)으로 탐지하지 않고,
     초기 형상을 직접 분석해 확정한 FLANGE_PAIR_SPEC(20쌍, (part,point_idx) 명시적 대응표)을
     모든 floor에 적용해 인덱스/법선/gap0/테이퍼 버퍼를 생성한다 — nearest-x 방식은 같은
     x좌표에 다른 y밴드의 고정 노드가 겹칠 때 오탐(Outer-Plate gap0=33.17mm)을 피할 수 없어
     폐기했다(2026-08-17 세션에서 밴드 분석으로 재도출, FLANGE_PAIR_SPEC 정의부 주석 참고).
  §4 CGDN17.forward()에 두께 연동 오프셋을 t_raw 직후 삽입. 구동 두께는 t_raw
     (제조 두께)이며 t_final(=t_raw*z_gate)이 아니다 — z_gate는 학습 중 0~1 연속
     완화값(존재 확률)이라 "두께가 절반인 판재" 같은 비물리 기하가 만들어진다.
     hard 삭제된 파트가 낌 pair는 pair 단위로 비활성화한다.
  §5 Stage-1(epoch<100)은 좌표를 base로 덮어쓰는데, 그 구간에도 두께는 자라므로
     오프셋까지 지우면 Stage-1 내내 간섭이 발생한다. 동결 대상을 GNN 변위로만
     한정하고 오프셋은 보존한다.
  §6 플랜지 노드만 밀면 폴리라인에 단차가 생겨 v1.7~v1.10에서 싸운 "국소 꺾임"이
     재발한다 — 이웃 ±1, ±2까지 가중치 [1, 2/3, 1/3]로 테이퍼링한다.
  §9 진단 전용 비마스킹 관통 지표(역전파 없음) + export 후 하드 assert.

  [구현 세션 pre-flight 검증에서 지시서 대비 수정된 사항 3건 — Gemini conf98 /
   OpenAI conf95가 독립적으로 동일하게 판정]
   (a) command_v1_15.md §4는 forward()가 오프셋을 **7번째 반환값**으로 내놓으라고 했으나,
       실제로는 model(...) 언패킹 호출부가 AI_design에 5개(L1732/1816/2007/2039/2132),
       **물리 계층 uni_section에 2개**(compute_rho_adaptive._fresh_forward, train_step)로 총 7개라
       반환 개수를 바꾸면 물리 계층까지 깨진다. 이 저장소가 이미 쓰는 관례
       (`model.current_epoch = epoch` → `getattr(self,'current_epoch',0)`)를 따라
       **self.last_flange_offset 속성**으로 전달한다(반환 개수 6개 유지).
   (b) §8("clearance 커리큘럼에서 접합 쌍 제외")은 **미적용**한다. §7이 이미 해당 항목을
       valid에서 빼버려 커리큘럼의 clearance 값이 그 항목에 닿지 않으므로 이중 규제가
       성립하지 않고, ff_mask는 (n_pts,n_segs) 행렬이라 direction 전체를 제외하면 같은
       direction 안의 **일반(비플랜지) 항목까지 clearance가 동결되는 회귀**가 된다.
   (c) 테이퍼 가중치는 (K,) 공유 벡터가 아니라 **(P,K) 쌍별 행렬**이어야 한다. 이웃 인덱스가
       (part,floor) 그룹 밖으로 나가 자기 자신으로 클랡프될 때 공유 벡터를 쓰면 그 가중치가
       플랜지 노드 자신에 더해져 최대 2배로 과도 이동한다. 범위 밖은 가중치 0으로 죽인다.
   또한 index_add_(인플레이스) 대신 **아웃오브플레이스 index_add**를 써 autograd 그래프
   구성을 확실히 한다.
  그 외 로직(target_mp/My, 손실 함수, 학습 루프)은 AI_design_v1_14.py와 동일하다.
───────────────────────────────────
AI_design_v1_14.py — 파트 간 겹침(overlap) 해소 3종 세트 (v1.14)
─────────────────────────────────────
[v1.14 변경, docs/command/command_v1_14.md 실행]
  v1.13 학습 결과(reports/v1_13/final_section_v1_13.csv)에서 입력 형상(initial_section_v2.csv,
  겹침 0으로 사전 검증됨)과 달리 Part0↔Part1(최대 0.35mm)/Part1↔Part2(최대 0.70mm)/
  Part1↔Part4(최대 0.65mm) 실질 겹침이 발생했고, 전 파트 두께가 T_MAX_V13(2.3mm)로 균일
  saturate됐다. 근본 원인 진단은 docs/review/review_v1_13.md, 개선안은 docs/idea/idea_v1_14.md.

  §1 [필수] direction-레벨 Top-K 풀링: import uni_section_v21 -> uni_section_v22(별칭 usv18 유지).
     v22의 compute_collision_loss_v5()가 153개 direction 전체 평균 대신 상위 COL_TOPK_DIR_FRACTION
     (0.25, 최소 COL_MIN_DIRS_TO_KEEP=5개)만 평균한다. 이 파일에서는 호출부에 top_k_dir_fraction을
     명시적으로 전달하는 것만 바뀐다(review_v1_13.md 근본 원인 1위).
  §2 [필수] clearance epoch 커리큘럼: build_collision_spec()이 학습 시작 시 초기(near-zero-slack)
     형상으로 1회 산정하고 700epoch 내내 고정하던 clearance(≈-0.05mm)를, epoch 진행에 따라
     CLEARANCE_TARGET_MM(0.5mm)까지 선형 강화한다(apply_clearance_curriculum()).
     ★ collision_spec은 {(sec,a,b): [direction_dict, ...]} 형태의 중첩 dict-of-list-of-dict이며
     최상위 키가 튜플이다 — /synod design 세션 Solver 라운드에서 Gemini/OpenAI 두 모델의 코드
     스케치가 모두 이 구조를 잘못 다뤄(각각 KeyError/TypeError) Critic 라운드에서 공통 발견·정정
     됐다. 반드시 .items() -> 리스트 -> 각 dict의 'clearance'를 갱신하는 2중 루프여야 한다.
     또한 커리큘럼이 누적 덮어쓰기가 되지 않도록 direction마다 원본값을 '_init_clearance'로 1회
     백업해 두고 매번 그 원본에서 보간한다(direction별 초기 slack이 서로 다르므로 전역 상수 하나로
     덮으면 안 된다).
  §3 [기본 OFF] ALDA dual-ascent 복구: l_collision은 loss에 직접 가중합되지 않고 compute_alda_loss()
     의 term_col을 통해서만 반영되는데, alda_state['mu_col']/['rho_col']이 학습 내내 한 번도
     갱신되지 않는 죽은 경로였다(usv18.compute_rho_adaptive()도 호출된 적 없음 — 이번 세션에서
     전체 파일 grep으로 확인). update_alda_dual_ascent()로 복구하되 ENABLE_DYNAMIC_ALDA=False가
     기본값이다: Solver 라운드에서 Gemini는 상시 활성화를 주장했으나 Critic 라운드에서 스스로
     정정해, 검증되지 않은 §1(새 손실 역학)과 한 번도 구동된 적 없는 dual-ascent(rho_col 최대 10배
     에스컬레이션)를 동시에 켜면 두 비선형 메커니즘이 상호작용해 발산 위험이 크다는 OpenAI의
     기본 OFF 안으로 두 모델이 수렴했다. §1·§2 결과를 먼저 보고 켤 것.
  §4(미적용) 경계 접합부 clearance 마진 보상, §5 Out of Scope는 command_v1_14.md 참고.
  그 외 로직(target_mp/My, CGDN17, 나머지 손실 함수)은 AI_design_v1_13.py와 완전히 동일하다.
─────────────────────────────────────
AI_design_v1_13.py — CSV 기반 initial section 로더 적용 (v1.13)
─────────────────────────────────────
[v1.13 변경, 사용자 직접 요청 — /synod debug 세션에서 이전 시도(v1_10 기반) 오류 수정]
  base를 AI_design_v1_10.py가 아니라 AI_design_v1_12.py로 바로잡았다 — v1.11(target_mp 단조감소)과
  v1.12(target_mp.csv 기반 로딩 + My 항복모멘트 리포트, load_target_mp_my_from_csv/
  compute_edge_my_elastic/calculate_myl/_per_section_myl 등)를 건너뛰고 v1.10을 base로 삼았던 게
  "My 부분이 없어짐, target_mp가 달라짐" 버그의 원인이었다.
  이번 버전에서 바꾸는 것은 오직 initial section 좌표 뿐이다:
    - import uni_section_v20 as usv18 -> import uni_section_v21 as usv18(별칭 유지)
    - build_bpillar_17section()이 usv18.build_bpillar_section() 17회 호출 + scale_factor(k) 등방
      스케일링 대신, usv18.build_bpillar_from_csv(INITIAL_SECTION_CSV)로
      initial_section/initial_section.csv를 그대로 로드한다.
    - scale_factor() 함수 자체는 compute_shape_continuity_loss_v2()/compute_boundary_continuity_loss()
      정규화 용도로 계속 쓰이므로 삭제하지 않는다.
  target_mp/target_my 로딩(target_mp.csv, load_target_mp_my_from_csv), My 리포트, 학습 루프,
  손실 함수, CGDN17 모델 등 나머지는 AI_design_v1_12.py와 완전히 동일 — 다른 부분은 손대지 않았다.
─────────────────────────────────────
AI_design_v1_12.py — CSV-Sourced Target Mp + My Reporting (v1.12)
─────────────────────────────────────
[v1.12 변경, /synod design 세션(Gemini flash conf98, OpenAI o3 conf93) — 조기 합의로 Critic/Defense
 라운드 생략. 두 모델 모두 병렬(concurrent Bash) 실행 시 서로 무관한 과거 프롬프트(rdo_gpce 프로젝트
 GPCE/RDO 응답)가 섞여 나오는 환경 결함을 발견해, 순차(단독) 실행으로 재시도해 정상 응답을 받았다
 — 이 프로젝트의 Bash 백그라운드 병렬 실행 방식이 두 개의 `cmd //c` 하위 프로세스의 표준입력을
 뒤섞는 것으로 추정되며, 향후 /synod 세션에서도 병렬 실행이 이상하면 순차 실행으로 재시도할 것.]
  §1 load_target_mp_my_from_csv(): target_mp.csv(UTF-8, 콤마 구분, 17행: index,mp_target,my_target)를
     읽어 target_mp를 절차적 GP 생성(generate_gp_target_mp, v1.11까지 사용) 대신 사용한다.
     [최초 구현 이력] 처음에는 CSV의 순번(1~17)과 섹션 인덱스(0~16) 대응 방향이 불확실하다고 보고
     추세 부호 비교 기반 자동판정(order="auto"/"as_is"/"reverse")을 두었다(/synod design 세션,
     Gemini conf98·OpenAI conf93 공통 제안). [이번 변경] target_mp.csv의 index 컬럼이 섹션
     인덱스(0~16)를 그대로 담도록 정리되어 그 추측 로직 자체가 불필요해졌다 — index 컬럼 값을
     곧바로 딕셔너리 키로 써서 "그대로 매칭"한다(order 인자·자동판정·reverse 로직 전부 제거).
     index 집합이 {0,...,16}과 정확히 일치하지 않으면 fail-loud로 즉시 중단하는 안전장치만 유지.
  §2 [수정] 최초 구현은 target_my를 CSV에서 그대로 옮겨 표시만 했으나, 사용자가 "항복 모멘트도
     계산해서 결과로 출력"의 의미를 확인해와 실제 계산으로 보완했다. compute_edge_my_elastic()/
     calculate_myl()/_per_section_myl()을 신규 로컬 구현(uni_section_v20.py는 프로젝트 관례상
     불변이므로 그 안의 compute_edge_mp_pna()와 동일한 "얇은 판 → y방향 등가 스트립" 기하 모델만
     재사용하고 계산 자체는 별도 함수로 새로 작성) — My_init/soft/hard를 Mp와 동일한 논리로 실제
     생성된 단면 형상에서 계산해 "## 2. Section-wise My" 표에 target(CSV)과 나란히 출력한다.
     여전히 학습 루프·손실 함수 어디에도 주입하지 않는다(사용자 요구사항: 단면 생성에 미반영,
     리포트에만 결과값 추가) — train_step_multi 등 학습 경로는 완전히 그대로.
  §3 generate_gp_target_mp()는 삭제하지 않고 그대로 보존한다(ad-hoc 실험용, OpenAI 제안 채택) —
     기본 실행 경로에서는 더 이상 호출되지 않을 뿐이다.
  구현 세션에서 두 모델의 신뢰도가 이미 1라운드에서 각각 98/93으로 90 이상, can_exit=true로
  수렴했으므로 Synod 조기 종료 조건에 따라 Critic/Defense 라운드 없이 바로 합성했다.
─────────────────────────────────────
AI_design_v1_11.py — Monotonic Target Mp Suite (v1.11)
─────────────────────────────────────
[v1.11 변경, /synod 없이 사용자 직접 요청으로 구현 — docs/idea, docs/command 문서 없음]
  §1 generate_gp_target_mp(): 기존에는 초기 Mp(INITIAL_MP_TABLE, sec0=하단이 최대값) 대비
     공간 상관 GP 노이즈(+/-50%)를 곱해 target_mp를 만들었기 때문에, 이 노이즈가 구간별로
     독립적으로 튀면서 target_mp가 섹션 인덱스 증가(하단→상단) 방향으로 항상 감소한다는
     보장이 없었다(예: v1.10 리포트 실측값도 sec9(27.9M)→sec11(39.7M)→sec13(47.5M)처럼
     상단으로 갈수록 오히려 증가하는 구간이 존재). 사용자 요청: "가장 하단 섹션부터 상단
     섹션으로 갈수록 target_mp가 줄어들도록" — GP 스무딩 자체는 유지하되(공간 상관 있는
     완만한 형상), 계산된 target_mp 배열에 대해 sec0(최하단)부터 sec16(최상단) 방향으로
     누적 최솟값(cumulative min)을 적용해 강제로 단조 비증가(그리고 부동소수 tie를 피하기
     위해 매 섹션마다 미세 상대 감소분 1e-4를 추가로 곱해 엄격한 단조 감소)로 만든다.
     호출부(main()의 `generate_gp_target_mp(seed=args.seed)`)는 그대로 — 함수 시그니처와
     반환 형식({section_id: target_mp}) 불변이라 train_step_multi 등 하류 코드 수정 불필요.
  주의: 이 변경은 /synod design 세션을 거치지 않았다(CLAUDE.md의 /synod 강제 규정은 사용자가
  `/synod` 슬래시 커맨드를 실행했을 때만 적용되며, 이번 요청은 직접 코드 작성 요청이었다).
  따라서 이전 버전들과 달리 Gemini/OpenAI 교차검증 기록이 없다 — 필요 시 `/synod review`로
  사후 검토 권장.
─────────────────────────────────────
AI_design_v1_10.py — Residual Spike (Boundary Kink) Mitigation Suite (v1.10)
─────────────────────────────────────
[v1.10 변경, docs/command/command_v1_10.md 실행]
  §1 compute_smoothness_lse_v3(): 하드 eps 컷오프 -> sigmoid soft-threshold(_soft_active_weight),
     섹션 내부 logsumexp(사실상 max) -> Top-K(SMOOTH_TOPK_FRACTION=0.3) 평균. review_v1_9.md
     ①(두께 소멸 우회)·③(섹션당 1곳만 남는 패턴) 동시 해소. compute_smoothness_lse_v2(v1.9)는
     보존(미사용, ablation 참조용).
  §2 compute_boundary_continuity_loss(): candidate 파트 생존/삭제 경계 섹션에서만 연속 파트
     (0/1/2)에 더 엄격한 threshold(1.0mm)로 추가 완충 손실. 기존 compute_shape_continuity_loss_v2
     (전 구간 2.0mm)는 그대로 유지 — 대체 아닌 추가. review_v1_9.md ②(Stage4 보간이 연속 파트
     재형성을 방치) 해소.
  §3 w_smooth_lse_schedule(): 기존 continuity_weight_schedule()과 동일한 sigmoid 성장 패턴
     재사용. W_SMOOTH_LSE_FLOOR(0.5)에서 시작해 ASWC_STAGE2_END 근방부터 목표치(W_SMOOTH_LSE=
     3.0)로 서서히 상승 — review_v1_9.md ①(처음부터 세게 누르는 부작용) 완화.
  구현: /synod design 세션(Gemini flash conf98, OpenAI o3 conf92)에서 §1·§3을 동일 커밋으로
  묶어야 한다는 데 두 모델이 수렴했다(§3의 목표치가 §1의 새 집계 방식을 전제로 재보정된 값이므로).
  OpenAI가 제시한 검증 계획(unit_test_loss_grad.py 등 존재하지 않는 테스트 스크립트, "offline
  실험에서 <1% loss 폭주"라는 근거 없는 인용)은 이 프로젝트에 그런 테스트 인프라나 실행 로그가
  없음을 Judge가 확인해 기각하고, 이 프로젝트가 실제로 써 온 검증 방식(콘솔 로그+스모크 실행+
  전체 학습 후 3D 결과 육안 확인)으로 대체했다. 두 모델 모두 train_step_multi()의 실제 시그니처를
  단순화해 부정확하게 제시했으나, Judge가 실제 코드(AI_design_v1_9.py)를 직접 대조해 정확한
  파라미터 목록과 w_continuity_val sentinel의 정확한 동작 방식(정적 상수 폴백이 아니라 스케줄
  함수를 그 자리에서 호출하는 방식)을 확인하고 w_smooth_lse_val도 동일 패턴으로 구현했다. 구현
  세션(pre-flight 검증)에서 Gemini/OpenAI 둘 다 "함수 기본 인자값은 지연 바인딩이라 상수 정의
  순서와 무관하게 안전하다"고 답했으나, 이는 함수 본문 내부 이름 조회에만 해당하는 설명이며 기본
  인자값(default parameter value)은 def 시점에 즉시 평가된다 — Judge가 이 차이를 확인해 신규
  상수 블록(§4)을 이를 참조하는 모든 신규 함수 정의보다 앞선 위치에 배치했다.
그 외 로직은 AI_design_v1_9.py와 동일. uni_section_v20.py는 이번 버전에서 수정하지 않는다.
─────────────────────────────────────
AI_design_v1_9.py — Structural Collapse (Spike) Mitigation Suite (v1.9)
─────────────────────────────────────
[v1.9 변경, docs/command/command_v1_9.md 실행]
  §1 compute_smoothness_lse_v2(): 17섹션 전역 단일 스칼라 LSE → 섹션별 분리 LSE +
     t_final 노드 단위 마스킹. W_SMOOTH_LSE 0.25→3.0. review_v1_8.md ①②③(가중치 불균형/
     whack-a-mole/Ghost Gradient Hijacking) 동시 해소.
  §2 Stage4 게이트 강제 이진화(±5.0 단일 스텝) → STAGE4_GATE_WARMUP_EPOCHS(50)에 걸친 선형
     보간 + optimizer 모멘텀 이전(모델 파라미터가 재생성되지 않으므로 파라미터 객체를 직접
     dict key로 재사용). review_v1_8.md ⑤ 해소.
  §3 compute_shape_continuity_loss_v2(): Chamfer-style 최근접-이웃 매칭 → 인덱스 1:1 대응
     거리 비교(17섹션 동일 토폴로지 전제). review_v1_8.md ⑥ 해소.
  §4 l_phys_total 및 collision 손실(uni_section_v20.compute_collision_loss_v5,
     AI_design_v1_9.compute_collision_penalty_unclamped) 평균 → Top-K(PHYS_TOPK_FRACTION=0.3)
     풀링. review_v1_8.md ⑦⑨ 해소. 기존 §1(v1.8)의 Order 손실 Top-K 패턴을 재사용(새 메커니즘
     도입 없음).
  §5 z_gate_part_avg: 17섹션 전체 평균 → pruning_state 기반 생존 섹션만의 평균. review_v1_8.md
     ⑧ 해소.
  구현: /synod design 세션(Gemini flash conf98→100, OpenAI o3 conf92→84→71)에서 두 모델 모두
  Solver/Critic/Defense 각 라운드에서 실제 코드에 없는 구체적 수치(가짜 실험 로그, 잘못된
  기존 상수값, 존재하지 않는 PyTorch 실패 시나리오)를 반복적으로 인용했다 — Claude가 매 라운드
  실제 소스 코드 대조로 반증했다(SMOOTH_LSE_TEMPERATURE 실제값 1.5, top_k_fraction 실제값 0.02,
  Stage4는 모델을 재생성하지 않으므로 파라미터 객체 직접 매칭이면 충분함 등). OpenAI가 제안한
  compute_smoothness_lse() legacy wrapper 유지 방식은 이 프로젝트에 전례가 없어 기각, Gemini의
  완전 교체 방식을 채택했다. 파일 버전 관리도 OpenAI가 제안한 "기존 파일 그대로 사용"을 프로젝트
  관례 위반으로 기각하고 Gemini의 신규 파일 생성안을 채택했다. optimizer state 이전은 두 모델의
  제안(id() 매핑, state_dict() 재직렬화 기반 매칭) 모두 이 코드베이스의 실제 상황(모델 미재생성)
  에는 불필요하게 복잡하거나 실제로는 키 타입 불일치로 동작하지 않는 코드였음을 Judge가 직접
  확인해, 파라미터 객체를 그대로 dict key로 쓰는 가장 단순한 방식으로 대체했다. 구현 세션(자체
  pre-flight 검증)에서 Gemini/OpenAI 둘 다 uni_section_v20.compute_collision_loss_v5()와
  AI_design_v1_9.compute_collision_penalty_unclamped()가 거의 동일하지만 Huber 적용 여부가
  다르다는 점을 독립적으로 지적해, Top-K 교체 시 이 차이를 유지하도록 반영했다.
그 외 로직은 AI_design_v1_8.py와 동일. uni_section_v20.py는 compute_collision_loss_v5() 1개
함수만 수정, 나머지는 uni_section_v19.py와 동일.
─────────────────────────────────────
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
import pandas as pd   # [v1.12] load_target_mp_my_from_csv()
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
import uni_section_v23 as usv18   # [v1.15 §1] fixed-fixed 항목 collision 제외 적용판
                                   # (v21의 build_bpillar_from_csv 등 나머지 함수는 v22에 그대로 있음)

# [v1.13] initial section 좌표 CSV — initial_section.csv(원본, Plate x Patch1/Patch2 간섭 있음)
# 대신 initial_section_v2.csv(Outer/Plate 고정, Inner Hat/Patch1/Patch2만 국소 이동해 간섭 0으로
# 검증된 버전)를 사용한다. 그 외 로직(target_mp/My 등)은 v1.12와 완전히 동일.
INITIAL_SECTION_CSV = os.path.join(
    _THIS_DIR, 'initial_section', 'initial_section_v2.csv')


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
W_SMOOTH_LSE = 3.0    # [v1.9 §1] 0.25 → 3.0. Prosecution 라운드에서 5~10 상향안이 w_mass/
                       # w_area_floor(10.0)와 같은 자릿수가 되어 Mp 수렴을 방해할 위험이 있다는
                       # 반론으로 채택되지 않고, 물리 손실의 약 30% 수준으로 절충됐다(idea_v1_9.md).
                       # §7 검증 계획에서 실측 후 재보정 필수인 시작값이다. 보강 손실 가중치 — 기존
                       # weights['w_smooth']와 독립(커리큘럼 미상속)
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


# ══════════════════════════════════════════════════════════════════
# [v1.9] 로컬 오버라이드 상수 — 구조적 붕괴(스파이크) 해소 5종 세트
# 근거: docs/idea/idea_v1_9.md, docs/review/review_v1_8.md, docs/command/command_v1_9.md
# ══════════════════════════════════════════════════════════════════
STAGE4_GATE_WARMUP_EPOCHS = 50   # [v1.9 §2] Stage4 hard-flip을 이 기간에 걸쳐 선형 보간으로 완화
PHYS_TOPK_FRACTION = 0.3         # [v1.9 §4] l_phys_total/collision 손실 Top-K 비율. 17섹션(또는
                                  # 방향별 위반 노드) 중 최악 30%에 그래디언트 집중.
                                  # compute_mesh_order_loss()의 top_k_fraction(0.02, 물리 엣지
                                  # ~2992개 기준)과는 절대 개수 스케일이 다르므로(섹션 17개,
                                  # collision 방향은 노드 수가 훨씬 적음) 의도적으로 더 큰 비율을
                                  # 시작값으로 잡는다 — §7에서 재보정.

assert STAGE4_EPOCHS >= STAGE4_GATE_WARMUP_EPOCHS, \
    "STAGE4_EPOCHS가 STAGE4_GATE_WARMUP_EPOCHS보다 작으면 보간이 끝나기 전에 Stage4 루프가 종료된다"

# 주의: SMOOTH_LSE_TEMPERATURE(1.5)와 uni_section_v20.compute_mesh_order_loss()의
# top_k_fraction(0.02)은 이번 버전 범위 밖이므로 변경하지 않는다.


# ══════════════════════════════════════════════════════════════════
# [v1.10] 로컬 오버라이드 상수 — 잔존 스파이크(경계 국소 꺾임) 해소 3종 세트
# 근거: docs/idea/idea_v1_10.md, docs/review/review_v1_9.md, docs/command/command_v1_10.md
# ★ 배치 주의: 아래 상수들은 §1~3의 신규 함수 정의에서 "기본 인자값(default parameter value)"으로
# 참조된다 — 기본 인자값은 함수 본문과 달리 def 시점에 즉시 평가되므로, 반드시 해당 함수 정의보다
# 앞에 위치해야 한다(구현 세션에서 Gemini/OpenAI가 "지연 바인딩이라 순서 무관"이라 답했으나 이는
# 기본 인자값에는 적용되지 않는 설명임을 Judge가 확인).
# ══════════════════════════════════════════════════════════════════
SMOOTH_TOPK_FRACTION = 0.3            # §1. 섹션 내 활성 노드 중 상위 30%(최소 1개) 평균
SMOOTH_EPS_TRANSITION_WIDTH = 0.02    # §1. eps 근방 soft-threshold 전이폭(mm)
BOUNDARY_CONTINUITY_THRESHOLD = 1.0   # §2. mm. 일반 continuity threshold(2.0mm)보다 엄격
W_BOUNDARY_CONTINUITY = 0.5           # §2. 시작값 — §7에서 재보정
W_SMOOTH_LSE_FLOOR = 0.5              # §3. 학습 초반 최소값(v1.9의 3.0 고정값보다 낮게 시작)
# W_SMOOTH_LSE(기존 상수, 값은 3.0 그대로 유지)는 이제 w_smooth_lse_schedule()의 "도달 목표치"로
# 의미가 바뀐다 — 상수 정의 자체는 변경하지 않는다.


# ══════════════════════════════════════════════════════════════════
# [v1.14] 로컬 오버라이드 상수 — 파트 간 겹침 해소 3종 세트
# 근거: docs/review/review_v1_13.md, docs/idea/idea_v1_14.md, docs/command/command_v1_14.md
# ★ 배치 주의: v1.10 블록과 동일하게, 아래 상수들도 신규 함수의 기본 인자값으로 참조되므로
# 해당 함수 정의보다 앞에 있어야 한다.
# ══════════════════════════════════════════════════════════════════

# §1. direction-레벨 Top-K — 실제 값은 uni_section_v22.COL_TOPK_DIR_FRACTION(0.25)과 동일하게
# 시작한다. 여기서 별도 상수로 두는 이유는 이 파일이 호출부에서 명시적으로 넘겨(=v22 기본값에
# 의존하지 않고) 향후 ablation 시 이 파일만 고쳐도 되게 하기 위함이다.
# 0.0 또는 1.0으로 두면 v22가 v21과 동일한 전체 평균으로 폴백한다(대조군 실험용).
COL_TOPK_DIR_FRACTION = 0.25

# §2. clearance epoch 커리큘럼
CLEARANCE_TARGET_MM = 0.5             # 도달 목표 clearance(mm). usv18.build_collision_spec()의
                                       # clearance_default(0.5)와 같은 값 — 초기 형상의 slack이
                                       # 거의 0이라 실제로는 min()에 걸려 ≈-0.05mm로 시작한다.
CLEARANCE_CURRICULUM_END = 400        # 이 epoch에 target에 도달하고 이후 고정. ASWC_STAGE2_END(400)와
                                       # 같은 값이지만 그 상수는 SECTION 6에서야 정의되므로(정의 순서)
                                       # 여기서는 리터럴로 두고, 정합성은 아래 assert로 강제한다.

# §3. ALDA dual-ascent 복구 — 기본 OFF(design 세션 Critic 라운드 합의)
ENABLE_DYNAMIC_ALDA = False           # True로 바꾸면 mu_col/rho_col이 실제로 갱신된다.
                                       # False면 collision 쪽 ALDA 경로는 v1.13과 완전히 동일
                                       # (단 §1·§2는 이 플래그와 무관하게 항상 적용됨에 주의).
ALDA_RHO_COL_GROWTH = 1.2             # g_col > slack_threshold 지속 시 rho_col 증가 배율
                                       # (rho_max=2000 상한은 alda_state가 이미 갖고 있다)


# ══════════════════════════════════════════════════════════════════
# [v1.15] 로컬 오버라이드 상수 — 두께 연동 플랜지 오프셋
# 근거: docs/idea/idea_v1_15.md, docs/command/command_v1_15.md §2
# ★ 배치 주의: v1.10/v1.14 블록과 동일하게, 아래 상수들은 신규 함수의 기본 인자값으로 참조되므로
# 해당 함수 정의보다 앞에 있어야 한다(기본 인자값은 def 시점에 즉시 평가됨).
# ══════════════════════════════════════════════════════════════════
FLANGE_TAPER_N       = 2      # 플랜지 노드에서 자유 영역 쪽으로 오프셋을 감쇄시킬 이웃 개수
FLANGE_TAPER_WEIGHTS = (1.0, 2.0 / 3.0, 1.0 / 3.0)   # [본인, ±1, ±2] 감쇄 가중치
FLANGE_ASSERT_TOL_MM = 0.01   # export 후 간섭 하드 assert 허용 오차

# [v1.15 재작성] 접합 플랜지 쌍을 런타임 임계값(x_tol/gap0_max)으로 "탐지"하는 대신, 초기 형상
# (initial_section_v2.csv, floor0)을 직접 분석해 얻은 **명시적 (part_id, point_idx) 대응표**를
# 그대로 하드코딩한다. point_idx 대응은 모든 floor에서 동일하다(각 (part,floor) 블록이 항상
# 같은 점 개수·순서를 갖는 template 규격이므로).
#
# 왜 임계값 방식을 버렸나: nearest-x + gap0 필터는 "같은 x좌표에 서로 다른 y레벨의 고정 노드가
# 겹쳐 있는" 경우를 구분하지 못했다. 예를 들어 Plate는 point_idx 4~10 구간이 Outer(y=48)가 아니라
# Inner(y=12.88, 완전히 다른 밴드)와 마주보는 자리인데, x좌표만 보면 Outer의 point_idx 4,5와
# 정확히 일치해 "Outer-Plate 접합"으로 오판됐다(실측 gap0=33.17mm — 명백히 비물리적). 이런
# 오탐은 gap0_max를 아무리 조정해도 근본적으로 못 없앤다(파트마다 실제 접합 간극 자체가 0.05~
# 2.61mm로 다양해 안전한 전역 임계값이 없다) — 그래서 아예 필터를 없애고 대응표를 확정했다.
#
# 도출 방법(2026-08-17 세션): 각 파트의 BC 고정 노드를 point_idx 연속 + y값 동일(0.05mm 이내)
# 기준으로 밴드로 나누고, 물리적으로 인접한 파트쌍(Outer-Plate, Plate-Inner, Plate-Patch1,
# Plate-Patch2, Inner-Patch2)에 대해 밴드끼리 x범위가 겹치는 구간만 후보로 삼았다. 그 결과
# 24쌍이 나왔으나, Outer-Plate 쌍 중 4개(point_idx 4,5,26,27)는 gap0=33.17mm로 다른 20개
# (0.05~2.61mm)와 자릿수가 다른 명백한 이상치라 제외했다 — Outer band(y=48, pt1~5)가 Plate의
# 두 밴드(outer band pt1~3/28~30, y=44.88 및 inner band pt4~10/21~27, y=12.88)에 걸쳐 x범위가
# 겹치는데, pt4,5는 Plate의 inner band와 우연히 x가 같아 잘못 걸린 것이다.
#
# (part_a, point_idx_a, part_b, point_idx_b) — part_id: 0=Outer,1=Plate,2=Inner,3=Patch1,4=Patch2
FLANGE_PAIR_SPEC = [
    # Outer(#00) <-> Plate(#03) — 외곽 플랜지 양 끝(6쌍). gap0(floor0)=1.17mm
    (0, 1, 1, 1), (0, 2, 1, 2), (0, 3, 1, 3),
    (0, 28, 1, 28), (0, 29, 1, 29), (0, 30, 1, 30),
    # Plate(#03) <-> Inner(#06) — 내측 플랜지 양 끝(6쌍). gap0(floor0)=0.96mm
    (1, 7, 2, 1), (1, 8, 2, 2), (1, 9, 2, 3),
    (1, 22, 2, 16), (1, 23, 2, 17), (1, 24, 2, 18),
    # Plate(#03) <-> Patch1(#07) — 양 끝점(2쌍). gap0(floor0)=0.99mm
    (1, 10, 3, 1), (1, 21, 3, 12),
    # Plate(#03) <-> Patch2(#08) — 3쌍. gap0(floor0)=0.05mm(거의 접촉)
    (1, 22, 4, 1), (1, 23, 4, 2), (1, 24, 4, 3),
    # Inner(#06) <-> Patch2(#08) — 3쌍. gap0(floor0)=2.61mm
    (2, 16, 4, 1), (2, 17, 4, 2), (2, 18, 4, 3),
]   # 총 20쌍 x 17 floor = 340개 (동일 노드가 여러 쌍에 관여 가능 — 예: Plate pt22/23/24는
    # Inner와 Patch2 양쪽에 동시에 접합, 3-way 스택 지점이라 물리적으로 정상)


@torch.no_grad()
def identify_mating_flange_pairs(x, spec=FLANGE_PAIR_SPEC,
                                  taper_n=FLANGE_TAPER_N, taper_weights=FLANGE_TAPER_WEIGHTS):
    """[v1.15 §3, 재작성] FLANGE_PAIR_SPEC(명시적 (part,point_idx) 대응표)을 모든 floor에 적용해
    오프셋 레이어용 버퍼를 만든다. 더 이상 런타임에 "접합 쌍인지" 탐지하지 않는다 — 위 표가 곧
    사양이다. gap0(초기 표면 간극)은 그 확정된 쌍에 대해서만 실제 좌표로 계산한다.

    단위 법선 n_ij = (p_j - p_i)/dist0.

    테이퍼 인덱스는 "같은 (part, floor) 그룹 안에서 노드 인덱스가 연속이고 point_idx 순서"라는
    build_bpillar_from_csv()의 성질을 그대로 이용한다 — k번째 이웃은 인덱스 ±k이며 그래프 순회가
    필요 없다. 그룹 경계를 넘는 이웃은 자기 자신으로 클램프하되 **해당 가중치를 0으로 죽인다**
    (공유 가중치 벡터를 쓰면 그 몫이 플랜지 노드 자신에게 더해져 최대 2배로 과이동 — pre-flight
    검증에서 양 모델이 공통 지적).

    반환: dict of tensors (P = 채택된 쌍 수, floor마다 spec 그대로 적용되므로 P = len(spec) * 17).
    """
    coords = x[:, :2]
    part_ids = x[:, 4].long()
    sec_ids = x[:, 5].long()
    t_init = x[:, 6]
    device = x.device
    N = x.shape[0]

    # (part, floor) 그룹의 인덱스 범위 — 연속성 전제를 이용해 경계를 구한다(테이퍼용)
    group_lo = torch.full((N,), -1, dtype=torch.long, device=device)
    group_hi = torch.full((N,), -1, dtype=torch.long, device=device)
    # point_idx -> 전역 노드 인덱스 조회용: (part_id, floor, point_idx) -> node index
    node_of = {}
    for pid in torch.unique(part_ids):
        for sec in torch.unique(sec_ids):
            m = (part_ids == pid) & (sec_ids == sec)
            if not bool(m.any()):
                continue
            idxs = torch.nonzero(m, as_tuple=False).squeeze(-1)
            assert int(idxs.max() - idxs.min()) == idxs.numel() - 1, \
                "(part, floor) 그룹의 노드 인덱스가 연속이 아니다 — 테이퍼 전제가 깨졌다"
            group_lo[idxs] = int(idxs.min())
            group_hi[idxs] = int(idxs.max())
            base = int(idxs.min())
            for k, ni in enumerate(idxs.tolist(), start=1):   # point_idx는 1부터
                node_of[(int(pid), int(sec), k)] = ni

    idx_a, idx_b, normals, gap0s = [], [], [], []
    parts_a, parts_b, secs = [], [], []

    for sec in [int(s) for s in torch.unique(sec_ids)]:
        for a, pa, b, pb in spec:
            key_a, key_b = (a, sec, pa), (b, sec, pb)
            if key_a not in node_of or key_b not in node_of:
                continue   # 해당 floor에 그 part/point_idx가 없는 경우(현재 spec 범위에선 없음)
            i, j = node_of[key_a], node_of[key_b]
            d = coords[j] - coords[i]
            dist0 = float(torch.linalg.norm(d))
            need = float((t_init[i] + t_init[j]) / 2.0)
            gap0 = dist0 - need
            idx_a.append(i); idx_b.append(j)
            normals.append((d / dist0) if dist0 > 1e-9 else torch.zeros(2, device=device))
            gap0s.append(gap0)
            parts_a.append(a); parts_b.append(b); secs.append(sec)

    P = len(idx_a)
    if P == 0:
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        K = 1 + 2 * taper_n
        return {
            'flange_idx_a': empty_l, 'flange_idx_b': empty_l,
            'flange_normal': torch.zeros(0, 2, device=device),
            'flange_gap0': torch.zeros(0, device=device),
            'flange_part_a': empty_l, 'flange_part_b': empty_l, 'flange_sec': empty_l,
            'taper_idx_a': torch.zeros(0, K, dtype=torch.long, device=device),
            'taper_idx_b': torch.zeros(0, K, dtype=torch.long, device=device),
            'taper_w_a': torch.zeros(0, K, device=device),
            'taper_w_b': torch.zeros(0, K, device=device),
        }

    idx_a_t = torch.tensor(idx_a, dtype=torch.long, device=device)
    idx_b_t = torch.tensor(idx_b, dtype=torch.long, device=device)

    def _taper(idx_t):
        """(P, K) 이웃 인덱스와 (P, K) 가중치. 그룹 경계를 넘는 이웃은 자기 자신으로 클램프하고
        가중치를 0으로 만든다."""
        offsets = [0]
        for k in range(1, taper_n + 1):
            offsets += [-k, k]
        cols_i, cols_w = [], []
        lo, hi = group_lo[idx_t], group_hi[idx_t]
        for off in offsets:
            cand = idx_t + off
            inside = (cand >= lo) & (cand <= hi)
            cols_i.append(torch.where(inside, cand, idx_t))
            w = float(taper_weights[abs(off)])
            cols_w.append(torch.where(inside,
                                      torch.full_like(idx_t, 1, dtype=torch.float32) * w,
                                      torch.zeros(idx_t.shape, dtype=torch.float32, device=device)))
        return torch.stack(cols_i, dim=1), torch.stack(cols_w, dim=1)

    ti_a, tw_a = _taper(idx_a_t)
    ti_b, tw_b = _taper(idx_b_t)

    return {
        'flange_idx_a': idx_a_t,
        'flange_idx_b': idx_b_t,
        'flange_normal': torch.stack(normals).to(device).float(),
        'flange_gap0': torch.tensor(gap0s, dtype=torch.float32, device=device),
        'flange_part_a': torch.tensor(parts_a, dtype=torch.long, device=device),
        'flange_part_b': torch.tensor(parts_b, dtype=torch.long, device=device),
        'flange_sec': torch.tensor(secs, dtype=torch.long, device=device),
        'taper_idx_a': ti_a, 'taper_idx_b': ti_b,
        'taper_w_a': tw_a, 'taper_w_b': tw_b,
    }


# ══════════════════════════════════════════════════════════════════
# [v1.14 §2] clearance epoch 커리큘럼
# ══════════════════════════════════════════════════════════════════

def backup_initial_clearance(collision_spec):
    """[v1.14 §2] 각 direction dict의 최초 clearance를 '_init_clearance'로 1회 백업한다.

    커리큘럼은 매 epoch `clearance = lerp(_init_clearance, target, progress)`로 "원본에서 다시"
    보간한다 — 직전 값에서 누적 보간하면 매 epoch 조금씩 target 쪽으로 끌려가 스케줄과 무관하게
    조기 도달하는 누적 덮어쓰기 버그가 된다.

    ★ collision_spec 구조: {(sec_int, a, b): [direction_dict, ...]} — 최상위 키가 튜플이고 값이
    dict의 '리스트'다. `for spec in collision_spec:`는 튜플 키를 순회하므로 TypeError,
    `collision_spec['clearance']`는 KeyError가 난다(/synod design 세션에서 Gemini/OpenAI 두
    모델의 코드 스케치가 모두 이 실수를 했고 Critic 라운드에서 공통 발견·정정됨). 반드시
    .values() -> 리스트 -> 각 dict 순으로 2중 순회할 것.
    """
    n_dirs = 0
    for dir_list in collision_spec.values():
        for d in dir_list:
            d['_init_clearance'] = d['clearance']
            n_dirs += 1
    return n_dirs


def apply_clearance_curriculum(collision_spec, epoch,
                                target_clearance=CLEARANCE_TARGET_MM,
                                end_epoch=CLEARANCE_CURRICULUM_END):
    """[v1.14 §2] epoch에 따라 각 direction의 clearance를 _init_clearance -> target으로 선형 강화.

    반환: (progress, mean_clearance) — 로깅용.

    ★ 주의(설계 트레이드오프, 실측 재보정 필수): initial_section_v2.csv는 접합부가 거의 맞닿아
    있어(slack≈0) build_collision_spec()이 산정하는 초기 clearance가 ≈-0.05mm다. 예를 들어
    Part1(Plate)-Part2(Inner) 접합부는 초기 proj≈1.6mm, t_sum_half=1.6mm이므로
    gap = 1.6 - 1.6 - clearance 가 되어, clearance=-0.05일 때는 위반이 없지만(+0.05)
    clearance=0.5까지 조이면 gap=-0.5로 0.5mm 위반이 된다. 이 접합부 노드는 fix_x/fix_y로 좌표가
    고정돼 proj를 키울 수 없으므로(review_v1_13.md 근본 원인 2위), 이 제약을 만족시키는 유일한
    경로는 두께를 t_sum_half<=1.1mm까지 줄이는 것이다(T_MIN_V13=0.7이라 도달 자체는 가능).
    즉 target 0.5mm는 "겹침 제거"뿐 아니라 "두께 하향" 압력으로도 강하게 작용한다 — Mp 달성률이
    과도하게 희생되면 command_v1_14.md §6 검증 계획에 따라 target을 0.2~0.3mm로 낮출 것.
    """
    if end_epoch <= 0:
        progress = 1.0
    else:
        progress = min(max(epoch / float(end_epoch), 0.0), 1.0)

    total, n = 0.0, 0
    for dir_list in collision_spec.values():
        for d in dir_list:
            init_cl = d['_init_clearance']
            d['clearance'] = (1.0 - progress) * init_cl + progress * target_clearance
            total += d['clearance']
            n += 1
    return progress, (total / n if n else 0.0)


# ══════════════════════════════════════════════════════════════════
# [v1.14 §3] ALDA dual-ascent 복구 (기본 OFF)
# ══════════════════════════════════════════════════════════════════

def update_alda_dual_ascent(alda_state, g_col_val, epoch, enable=None):
    """[v1.14 §3] collision 제약의 mu_col/rho_col을 표준 Augmented Lagrangian 방식으로 갱신한다.

    v1.13까지 alda_state는 update_every/rho_max/slack_threshold 등 dual-ascent용 필드를 갖고
    있었지만 실제로는 어디에서도 갱신되지 않는 죽은 경로였다(usv18.compute_rho_adaptive()도
    호출된 적 없음). 그 결과 rho_col=200, mu_col=1.0이 700epoch 내내 고정돼 충돌 제약 압력이
    전혀 에스컬레이션되지 않았다.

    기본 OFF인 이유(design 세션 Critic 라운드 합의): §1이 이미 검증되지 않은 새 손실 역학을
    도입하는데, 한 번도 구동된 적 없는 rho 에스컬레이션(최대 10배)을 동시에 켜면 두 비선형
    메커니즘이 상호작용해 발산할 위험이 있다. §1·§2 결과를 먼저 확인한 뒤 켤 것.

    enable=None이면 모듈 상수 ENABLE_DYNAMIC_ALDA를 그 자리에서 참조한다(v1.5의 w_col_aux_val
    sentinel과 동일 패턴 — 기본 인자값 정적 바인딩 회피).
    반환: 실제로 갱신했으면 True.
    """
    if enable is None:
        enable = ENABLE_DYNAMIC_ALDA
    if not enable:
        return False
    if epoch < alda_state['rho_freeze_epochs']:
        return False
    if alda_state['update_every'] <= 0 or epoch % alda_state['update_every'] != 0:
        return False
    if g_col_val <= alda_state['slack_threshold']:
        return False

    alda_state['mu_col'] = min(
        alda_state['max_mu'],
        alda_state['mu_col'] + alda_state['rho_col'] * g_col_val)
    alda_state['rho_col'] = min(
        alda_state['rho_col'] * ALDA_RHO_COL_GROWTH,
        alda_state['rho_max'])
    return True


def check_part_overlap(csv_path, tol_mm=FLANGE_ASSERT_TOL_MM, x_tol_mm=1.0):
    """[v1.15 §9] 저장된 template 규격 CSV를 다시 읽어 파트 간 겹침을 전수 검사한다.

    검사 방식은 review_v1_13.md에서 v1.13/v1.14 결과를 분석할 때 쓴 것과 동일하다: 각 노드가
    중심선 ±t/2 밴드를 차지한다고 보고, 같은 floor에서 x가 x_tol_mm 이내로 대응되는 다른 파트
    노드와 y밴드가 교차하는지 본다.

    [v1.15 개정] 이전 버전은 위반 시 AssertionError를 던져 학습 파이프라인 전체(리포트/CSV/가중치
    저장까지 끝난 뒤의 마지막 단계)를 중단시켰다 — 노트북에서 실행하면 커널이 트레이스백으로
    끝나 앞서 저장된 산출물을 확인하기도 번거로웠다. 이제는 예외를 던지지 않고 겹침 상세를
    반환만 한다. 호출부(main())가 이 결과를 리포트 .md 파일에 섹션으로 append하고, 콘솔에는
    경고만 출력한다 — 실행은 항상 정상 종료된다.

    반환: {'n_bad': int, 'worst_mm': float, 'worst_at': (part_i, part_j, floor, x) or None,
           'pairs': [(part_i, part_j, floors:list, max_depth_mm, n_points), ...]} — floors/pairs는
    겹침이 있는 파트쌍별 요약(리포트 표 작성용), depth 내림차순 정렬.
    """
    import csv as _csv
    from collections import defaultdict as _dd

    rows = list(_csv.reader(open(csv_path, encoding='utf-8')))[1:]
    parts, cur = [], []
    for r in rows:
        f, p, xx, yy, rr, tt = map(float, r)
        if p == 0.0:
            parts.append({'d': cur, 't': tt}); cur = []
        else:
            cur.append((int(f), xx, yy))
    per = []
    for pt in parts:
        d = _dd(list)
        for f, xx, yy in pt['d']:
            d[f].append((xx, yy))
        per.append(d)

    worst, worst_at, n_bad = 0.0, None, 0
    pair_stats = {}   # (i,j) -> {'floors': set, 'max': float, 'n': int}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            hi_, hj = parts[i]['t'] / 2.0, parts[j]['t'] / 2.0
            for fl in sorted(set(per[i]) & set(per[j])):
                bj = [(x, y - hj, y + hj) for x, y in per[j][fl]]
                for x, y in per[i][fl]:
                    lo_i, up_i = y - hi_, y + hi_
                    xj, lo_j, up_j = min(bj, key=lambda b: abs(b[0] - x))
                    if abs(xj - x) >= x_tol_mm:
                        continue
                    depth = min(up_i, up_j) - max(lo_i, lo_j)
                    if depth > tol_mm:
                        n_bad += 1
                        if depth > worst:
                            worst, worst_at = depth, (i, j, fl, x)
                        e = pair_stats.setdefault((i, j), {'floors': set(), 'max': 0.0, 'n': 0})
                        e['floors'].add(fl); e['n'] += 1
                        e['max'] = max(e['max'], depth)

    pairs = [(i, j, sorted(e['floors']), e['max'], e['n'])
             for (i, j), e in sorted(pair_stats.items(), key=lambda kv: -kv[1]['max'])]

    if n_bad:
        i, j, fl, x = worst_at
        print(f"[v1.15 §9][WARNING] 최종 형상에 파트 간 간섭 {n_bad}개 지점, "
              f"최대 침투 {worst:.4f}mm (> 허용 {tol_mm}mm) @ part{i}-part{j} floor{fl} x={x:.2f}. "
              f"상세는 리포트 파일 참고: {csv_path}")
    else:
        print(f"[v1.15 §9] 간섭 검사 통과 — 허용 오차 {tol_mm}mm 초과 겹침 없음 ({csv_path})")

    return {'n_bad': n_bad, 'worst_mm': worst, 'worst_at': worst_at, 'pairs': pairs}


def append_overlap_section_to_report(report_path, overlap_result, part_names=None):
    """[v1.15 §9] check_part_overlap() 결과를 기존 리포트 .md 파일 끝에 섹션으로 append한다.
    report_design_comparison()이 이미 저장한 "## 4. Part thickness" 뒤에 이어 붙는 형태다."""
    if part_names is None:
        part_names = {0: "#00(Outer)", 1: "#03(Plate)", 2: "#06(Inner)",
                      3: "#07(Patch1)", 4: "#08(Patch2)"}

    lines = ["\n## 5. 파트 간 겹침 검사 (§9, 노드 ±t/2 밴드 기준)\n"]
    n_bad = overlap_result['n_bad']
    if n_bad == 0:
        lines.append("겹침 없음 — 모든 파트쌍이 허용 오차 이내.\n")
    else:
        worst = overlap_result['worst_mm']
        wi, wj, wfl, wx = overlap_result['worst_at']
        lines.append(f"**겹침 {n_bad}개 지점 발견** — 최대 침투 {worst:.4f}mm "
                     f"@ {part_names.get(wi, wi)}-{part_names.get(wj, wj)} "
                     f"floor{wfl} x={wx:.2f}\n")
        lines.append("| 파트쌍 | floors | 지점 수 | 최대 침투(mm) |")
        lines.append("|:---|:---|---:|---:|")
        for i, j, floors, depth, n in overlap_result['pairs']:
            fl_str = ",".join(str(f) for f in floors) if len(floors) <= 8 else \
                f"{floors[0]}~{floors[-1]} ({len(floors)}개)"
            lines.append(f"| {part_names.get(i, i)}-{part_names.get(j, j)} | {fl_str} "
                         f"| {n} | {depth:.4f} |")
        lines.append("")

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[v1.15 §9] 겹침 검사 결과를 리포트에 추가: {report_path}")


@torch.no_grad()
def compute_flange_penetration_diag(model, new_coords, t_final):
    """[v1.15 §9] 접합 쌍의 실제 관통량을 **마스킹 없이** 계산하는 진단 전용 지표.

    collision loss는 §7에서 이 쌍들을 제외하므로, 오프셋 레이어에 버그가 있어도 손실만 봐서는
    알 수 없다. 여기서 계산한 값은 `detach()` 상태이며 loss/역전파에 절대 들어가지 않는다.

    관통량 = relu( (t_a+t_b)/2 + gap0 - dist )  — 목표는 0이다.
    두께는 t_final(게이팅 반영)이 아니라 실제 판재 두께를 봐야 하므로, 게이트가 닫힌 쌍은
    애초에 오프셋 대상이 아니었으니 t_final 평균으로 근사해도 무방하다(진단용).
    """
    zero = {'l_flange_pen_max': 0.0, 'l_flange_pen_mean': 0.0, 'n_flange_pen': 0}
    if not getattr(model, '_flange_ready', False):
        return zero
    ia, ib = model.flange_idx_a, model.flange_idx_b
    if ia.numel() == 0:
        return zero
    dist = torch.linalg.norm(new_coords[ib] - new_coords[ia], dim=1)          # (P,)
    t_half = 0.5 * (t_final.squeeze(-1)[ia] + t_final.squeeze(-1)[ib])
    pen = torch.relu(t_half + model.flange_gap0 - dist)
    return {
        'l_flange_pen_max': float(pen.max()),
        'l_flange_pen_mean': float(pen.mean()),
        'n_flange_pen': int((pen > 1e-6).sum()),
    }


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
# [v1.13] initial_section.csv를 그대로 로드한다 — 그 외(target_mp/My/학습 루프 등)는 v1.12와
# 완전히 동일. scale_factor()는 compute_shape_continuity_loss_v2()/compute_boundary_continuity_loss()
# 정규화 용도로 계속 쓰이므로 삭제하지 않는다(§0 참고).
# ══════════════════════════════════════════════════════════════════

NUM_SECTIONS = 17
CONTINUOUS_PARTS = usv18.S_PROTECT        # [0, 1, 2] — Outer Hat, Inner Plate, Inner Hat
CANDIDATE_PARTS = usv18.CANDIDATE_PARTS   # [3, 4] — Patch 1, Patch 2


def scale_factor(k: int) -> float:
    """[v1.13] build_bpillar_17section()에는 더 이상 쓰이지 않는다(CSV가 이미 최종 실좌표를
    담고 있음) — compute_shape_continuity_loss_v2()/compute_boundary_continuity_loss()의 섹션 간
    정규화 용도로는 계속 필요해 그대로 유지한다.
    initial_section.md §1: s(k) = 1.6 - 0.6*(k-1)/16, k=1..17(1-indexed, k=17이 원본/최상단).
    이 파일 내부에서는 0-indexed section_id(0=최상단 원본 ~ 16=최하단)를 쓰므로 k=section_id+1."""
    k1 = k + 1
    return 1.6 - 0.6 * (k1 - 1) / 16.0


def build_bpillar_17section():
    """[v1.13] usv18.build_bpillar_from_csv(INITIAL_SECTION_CSV)로 17-section 전체를 한 번에
    로드한다 — 모든 섹션에 5-part 전부를 처음부터 노드로 포함한다(존재/삭제는 사전 필터링이 아니라
    학습된 게이트(§1 CGDN17)로 결정). 반환: Data(x[N,8], edge_index, edge_attr, join_pairs)."""
    return usv18.build_bpillar_from_csv(INITIAL_SECTION_CSV, num_sections=NUM_SECTIONS)


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
        # [v1.15 §4] 플랜지 오프셋 버퍼는 set_flange_pairs()로 나중에 등록된다.
        # 등록 전에는 오프셋이 0이 되도록 forward()가 방어한다(dry-run 등에서 안전).
        self._flange_ready = False
        # [v1.15 §4(a)] forward()의 반환 개수를 6개로 유지하고 오프셋은 속성으로 전달한다 —
        # model(...) 언패킹 호출부가 AI_design 5곳 + uni_section(물리 계층) 2곳으로 총 7곳이라
        # 7번째 반환값을 추가하면 물리 계층까지 깨진다. 이 저장소가 이미 쓰는
        # `model.current_epoch = epoch` 패턴과 동일한 상태 전달 방식이다.
        self.last_flange_offset = None

    def set_flange_pairs(self, pairs: dict):
        """[v1.15 §3] identify_mating_flange_pairs() 결과를 버퍼로 등록한다.
        모델이 이미 원하는 device에 올라간 뒤에 호출해도 되도록, 등록 시 파라미터 device로 옮긴다."""
        dev = self.log_alpha_candidates.device
        for k, v in pairs.items():
            self.register_buffer(k, v.to(dev), persistent=False)
        self._flange_ready = bool(self.flange_idx_a.numel() > 0)
        return self._flange_ready

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

        # ── [v1.15 §4] 두께 연동 플랜지 오프셋 ──────────────────────────────
        # 초기 표면 간극 gap0 = dist0 - (t_a0+t_b0)/2 를 학습 내내 보존한다:
        # 중심선 거리가 (Δt_a + Δt_b)/2 만큼 벌어지면 표면 간극은 gap0 로 유지된다.
        # 구동 두께는 t_raw(제조 두께)다 — t_final(=t_raw*z_gate)을 쓰면 z_gate 가 0~1 연속
        # 완화값(존재 '확률')이라 "두께가 절반인 판재" 같은 비물리 기하가 만들어진다. 삭제된
        # 파트는 두께를 흐리는 대신 pair 자체를 끈다(pair_alive).
        # ★ 삽입 위치가 t_raw(위)와 z_gate(위) 이후인 이유: 오프셋은 두 값을 모두 필요로 한다.
        flange_offset = torch.zeros_like(new_coords)
        if getattr(self, '_flange_ready', False):
            # ★ Δt 의 기준선은 x[:,6](=t_init)이 아니라 "delta_t_part=0 일 때의 t_raw"다.
            # _t_init_logit()이 frac 을 [FRAC_CLAMP_LO, FRAC_CLAMP_HI]=[0.01,0.99]로 clamp 하므로,
            # t_init 이 T_MAX(2.3)인 #00(Outer)는 sigmoid(logit(0.99)) 경로에서 t_raw0=2.2840 이 되어
            # t_init 과 0.016mm 어긋난다. t_init 을 기준으로 삼으면 학습 전(Δ=0)부터 -0.008mm 씩
            # 밀려나는 상수 오프셋이 생긴다(구현 스모크 테스트에서 실측·발견).
            t_raw0 = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit)
            dt = (t_raw - t_raw0).squeeze(-1)                         # (N,) Δt (파트 내 균일)
            dt_a = dt[self.flange_idx_a]                              # (P,)
            dt_b = dt[self.flange_idx_b]

            # hard 삭제된 파트가 낀 pair 는 비활성화(연속 파트는 z_gate=1.0 고정이라 항상 살아있다)
            pair_alive = (z_gate[self.flange_sec, self.flange_part_a]
                          * z_gate[self.flange_sec, self.flange_part_b])
            pair_alive = (pair_alive >= BINARIZE_THRESHOLD).float()   # (P,) 0/1, 그래디언트 없음

            # i 는 -n 방향, j 는 +n 방향으로 각자 Δt/2 만큼 → 거리 증가 = (Δt_a+Δt_b)/2
            move_a = (-0.5 * dt_a * pair_alive).unsqueeze(-1) * self.flange_normal   # (P,2)
            move_b = ( 0.5 * dt_b * pair_alive).unsqueeze(-1) * self.flange_normal

            # §6 테이퍼: 본인 + 이웃 ±1, ±2 로 [1, 2/3, 1/3] 감쇄 분배. 가중치는 (P,K) 쌍별
            # 행렬이라 그룹 경계를 넘어 클램프된 이웃은 0으로 죽는다(과이동 방지).
            # ★ 아웃오브플레이스 index_add 사용 — 인플레이스(index_add_)를 grad 소스와 함께
            # 반복 호출하는 것보다 autograd 그래프 구성이 확실하다(pre-flight 검증 공통 권고).
            for k in range(self.taper_idx_a.shape[1]):
                flange_offset = flange_offset.index_add(
                    0, self.taper_idx_a[:, k], self.taper_w_a[:, k].unsqueeze(-1) * move_a)
                flange_offset = flange_offset.index_add(
                    0, self.taper_idx_b[:, k], self.taper_w_b[:, k].unsqueeze(-1) * move_b)

            new_coords = new_coords + flange_offset

        # [v1.15 §4(a)] 반환 개수를 6개로 유지하고 오프셋은 속성으로 전달한다(§5 Stage-1에서 사용).
        # 매 forward 마다 덮어쓰므로 누적되지 않는다.
        self.last_flange_offset = flange_offset
        # ────────────────────────────────────────────────────────────────────

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
    fig.savefig(os.path.join(save_dir, f'gate_heatmap_v1_15_4_e{epoch:04d}.png'), dpi=150) # v1_ver.no
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
                                   final_seg_ids=None, fix_mask=None):
    """initial_section.py의 plot_sections_3d_plotly()(line 253-282) 스타일 인터랙티브 3D HTML.
    [/synod design 세션 수렴안] Initial/Final 상태를 버튼으로 토글하는 단일 HTML로 만들고, candidate
    파트는 최종 z_gate 값에 비례해 투명도/선굵기를 조절해 "꺼진 섹션"을 시각적으로 표현한다.

    [v1.14] fix_mask: (N,) bool — True인 노드(BC 고정, fix_x=fix_y=1)를 일반 노드(작은 원)와
    구분되도록 **네모 마커**로 덧그린다. 겹침이 남은 접합부가 좌표 이동이 불가능한 고정 노드인지
    (review_v1_13.md 근본 원인 2위) 육안으로 바로 확인하기 위한 용도다. None이면 기존 동작 그대로.
    build_bpillar_from_csv()가 fix_x/fix_y에 같은 값을 넣으므로 둘을 구분하지 않는다."""
    import plotly.graph_objects as go

    part_ids_np = part_ids.cpu().numpy().astype(int)
    section_ids_np = section_ids.cpu().numpy().astype(int)
    base_np = base_coords.cpu().numpy()
    final_np = final_coords.cpu().numpy()
    z_gate_np = final_z_gate.cpu().numpy()   # (17, 5)

    if fix_mask is not None:
        fix_np = (fix_mask.cpu().numpy() if hasattr(fix_mask, 'cpu') else np.asarray(fix_mask))
        fix_np = fix_np.astype(bool).reshape(-1)
    else:
        fix_np = None

    def _fix_overlay(coords_np, label, visible):
        """고정 노드만 모아 네모 마커 1개 trace로 덧그린다(파트 색상은 점별로 지정)."""
        idx = np.nonzero(fix_np)[0]
        zs = section_ids_np[idx] * z_coef
        return go.Scatter3d(
            x=coords_np[idx, 0], y=coords_np[idx, 1], z=zs,
            mode='markers',
            marker=dict(size=5, symbol='square',
                        color=[PART_COLORS[p] for p in part_ids_np[idx]],
                        opacity=1.0),
            name=f'BC 고정 노드 ({label})', legendgroup='fixed_nodes',
            showlegend=True, visible=visible,
            customdata=np.stack([section_ids_np[idx], part_ids_np[idx]], axis=1),
            hovertemplate=(f'[{label}] BC 고정 노드<br>Section %{{customdata[0]}}, '
                           'part_id=%{customdata[1]}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>'),
        )

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

    # [v1.14] Initial 고정 노드 오버레이(네모) — Initial 그룹 마지막에 붙인다
    n_fix_i = 0
    if fix_np is not None and fix_np.any():
        fig.add_trace(_fix_overlay(base_np, 'Initial', True))
        n_fix_i = 1

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

    # [v1.14] Final 고정 노드 오버레이(네모)
    n_fix_f = 0
    if fix_np is not None and fix_np.any():
        fig.add_trace(_fix_overlay(final_np, 'Final', False))
        n_fix_f = 1

    # trace 순서: [Initial 본체][Initial 고정][Final 본체][Final 고정]
    show_initial = ([True] * n_initial + [True] * n_fix_i
                    + [False] * n_final + [False] * n_fix_f)
    show_final = ([False] * n_initial + [False] * n_fix_i
                  + [True] * n_final + [True] * n_fix_f)

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

def compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0):
    """[v1.9 §3] compute_shape_continuity_loss()(Chamfer-style 최근접-이웃 매칭)를 대체한다.
    17섹션 모두 build_bpillar_section()의 동일 토폴로지를 스케일링만 다르게 사용하므로
    (build_bpillar_17section() 확인 — 노드를 이어붙이기만 하고 재정렬하지 않음), 인접 섹션 간
    같은 로컬 순번은 항상 같은 물리적 위치를 가리킨다 — cdist+min 없이 인덱스로 직접 대응시킨다.
    review_v1_8.md §6: 기존 방식은 스파이크가 우연히 다른 노드 근처면 위반을 놓칠 수 있었다."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    unique_secs = torch.unique(section_ids).long().sort()[0]

    for i in range(len(unique_secs) - 1):
        k_a, k_b = int(unique_secs[i].item()), int(unique_secs[i + 1].item())
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)
        for pid in torch.unique(part_ids):
            pid_int = int(pid.item())
            idx_a = ((section_ids == k_a) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            idx_b = ((section_ids == k_b) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            if idx_a.numel() == 0 or idx_b.numel() == 0:
                continue
            if idx_a.numel() != idx_b.numel():
                # [v1.9, 안전장치] 이 코드베이스에서는 발생하지 않지만(모든 섹션이 동일 노드 수),
                # 향후 섹션별 노드 수가 달라지는 변경이 생기면 조용히 스킵한다.
                continue
            ca = new_coords[idx_a] / s_a
            cb = new_coords[idx_b] / s_b
            dist = torch.norm(ca - cb, dim=-1)
            violation = torch.clamp(dist - threshold, min=0.0)
            total = total + torch.mean(violation ** 2)
            n_terms += 1
    if n_terms > 0:
        total = total / n_terms
    return total


def find_candidate_boundary_sections(pruning_state):
    """[v1.10 §2] pruning_state['state']((17, n_cand), ALIVE=0/PENDING=1/DELETED=2 카테고리 값)
    에서 인접 섹션 간 상태가 달라지는 (k, k+1) 쌍을 찾는다. candidate 파트가 살아있다가 죽거나,
    죽어있다가 살아나는 경계를 모두 포함한다. pruning_state가 None이면(Stage 0 dry-run 등)
    빈 리스트를 반환한다."""
    if pruning_state is None:
        return []
    state = pruning_state['state']    # (17, n_cand), CPU
    boundary_pairs = []
    for k in range(NUM_SECTIONS - 1):
        if not torch.equal(state[k], state[k + 1]):
            boundary_pairs.append((k, k + 1))
    return boundary_pairs


def compute_boundary_continuity_loss(new_coords, section_ids, part_ids, boundary_pairs,
                                      threshold=BOUNDARY_CONTINUITY_THRESHOLD):
    """[v1.10 §2] candidate 생존 경계 섹션에서 연속 파트(CONTINUOUS_PARTS=[0,1,2])에 한해
    compute_shape_continuity_loss_v2()와 동일한 인덱스 1:1 대응 방식을 재사용하되(새 메커니즘
    도입 금지), 경계 섹션·연속 파트로만 범위를 좁히고 threshold를 더 엄격하게(1.0mm) 적용한다.
    review_v1_9.md §2: Stage4 게이트 보간은 candidate 파트만 다루고 인접 연속 파트의 재형성은
    별도 완충 장치 없이 방치돼 경계 섹션에서 국소 꺾임이 남았다."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    for k_a, k_b in boundary_pairs:
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)
        for pid_int in CONTINUOUS_PARTS:   # [0, 1, 2]만 — candidate 파트 자체는 이미 §1에서 다룸
            idx_a = ((section_ids == k_a) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            idx_b = ((section_ids == k_b) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            if idx_a.numel() == 0 or idx_b.numel() == 0 or idx_a.numel() != idx_b.numel():
                continue
            ca = new_coords[idx_a] / s_a
            cb = new_coords[idx_b] / s_b
            dist = torch.norm(ca - cb, dim=-1)
            violation = torch.clamp(dist - threshold, min=0.0)
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


def w_smooth_lse_schedule(epoch, stage2_end, w_max=W_SMOOTH_LSE, w_min=W_SMOOTH_LSE_FLOOR,
                           beta=0.05):
    """[v1.10 §3] continuity_weight_schedule()과 동일한 sigmoid 성장 패턴(새 메커니즘 도입 금지).
    stage2_end(ASWC_STAGE2_END=400) 근방부터 목표치(W_SMOOTH_LSE=3.0)로 서서히 올라간다 — Stage4
    게이트 확정 이전에는 형상 탐색이 Mp/면적 목표를 우선하도록 완화하고(review_v1_9.md §1 부작용
    대응), 게이트가 정해지는 시점부터 smoothness 억제력을 강화한다."""
    return w_min + (w_max - w_min) / (1.0 + math.exp(-beta * (epoch - stage2_end)))


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


def compute_smoothness_lse_v2(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.9 §1] compute_smoothness_lse()(v1.7)를 대체한다. 두 가지를 동시에 고친다:
    (1) 17섹션 전체를 하나의 스칼라로 묶던 것을 섹션별로 분리해 whack-a-mole 방지
        (review_v1_8.md §2).
    (2) t_final 기준 노드 단위(섹션 단위 아님) 마스킹으로 DELETED 노드의 Ghost Gradient
        차단(review_v1_8.md §3). 연속 파트(0/1/2)는 항상 alive이므로 섹션 전체를 죽은
        것으로 마스킹하면 안 된다 — 반드시 t_final 기준 노드별 판정.
    neighbor_map은 build_static_neighbor_map()이 반환하는 그대로(node_idx/left_idx/right_idx,
    이미 정확히 2개의 물리 이웃을 가진 노드만 필터링된 상태 — 패딩/-1 처리 불필요)."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    active = (t_flat[node_idx] > eps) & (t_flat[left_idx] > eps) & (t_flat[right_idx] > eps)
    if not active.any():
        return new_coords.sum() * 0.0

    node_idx, left_idx, right_idx = node_idx[active], left_idx[active], right_idx[active]
    section_ids = x[node_idx, 5].long()

    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)

    section_lse = []
    for sec in range(num_sections):
        sec_mask = section_ids == sec
        if not sec_mask.any():
            continue
        d = sq_deviation[sec_mask]
        section_lse.append(temperature * torch.logsumexp(d / temperature, dim=0))
    if not section_lse:
        return new_coords.sum() * 0.0
    return torch.stack(section_lse).mean()


def _soft_active_weight(t_flat, eps=0.1, transition_width=SMOOTH_EPS_TRANSITION_WIDTH):
    """[v1.10 §1.1] compute_smoothness_lse_v2()의 하드 eps 컷오프(`t_flat > eps` 불리언)를
    대체하는 sigmoid soft-threshold. t_flat=0(진짜 DELETED, z_gate=0으로 t_final이 정확히 0)이면
    sigmoid((0-0.1)/0.02)=sigmoid(-5)≈0.0067로 사실상 0 — review_v1_8.md §3 Ghost Gradient
    Hijacking 방지 효과는 그대로 유지된다. Stage4 워밍업 중 t_flat이 eps 근방을 서서히 통과하는
    노드는 가중치가 매끄럽게 0→1로 바뀌어, '살짝만 깎아도 손실 항 전체가 증발'하는 계단 효과
    (review_v1_9.md §1의 우회 경로)를 없앤다."""
    return torch.sigmoid((t_flat - eps) / transition_width)


def compute_smoothness_lse_v3(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, transition_width=SMOOTH_EPS_TRANSITION_WIDTH,
                               top_k_fraction=SMOOTH_TOPK_FRACTION,
                               temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.10 §1] compute_smoothness_lse_v2()(v1.9)를 대체한다.
    (1) 하드 eps 컷오프 -> sigmoid soft-threshold(_soft_active_weight): DELETED 노드는 여전히
        가중치≈0으로 배제되지만, 전이 구간의 그래디언트 계단 현상을 없앤다.
    (2) 섹션 내부 logsumexp(사실상 max) -> Top-K 평균: review_v1_9.md §3에서 3D 결과를 직접
        디코딩해 확인한 "섹션당 정확히 1곳만 남는" 패턴의 직접 원인을 제거한다."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    w_i = _soft_active_weight(t_flat[node_idx],  eps, transition_width)
    w_l = _soft_active_weight(t_flat[left_idx],  eps, transition_width)
    w_r = _soft_active_weight(t_flat[right_idx], eps, transition_width)
    active_weight = w_i * w_l * w_r          # (M,) — 셋 다 살아있어야 1에 가까움

    section_ids = x[node_idx, 5].long()
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1) * active_weight

    section_losses = []
    for sec in range(num_sections):
        sec_mask = section_ids == sec
        if not sec_mask.any():
            continue
        d = sq_deviation[sec_mask]
        n = d.numel()
        k = max(1, int(n * top_k_fraction))
        top_d, _ = torch.topk(d, k=k, largest=True)
        section_losses.append(top_d.mean())
    if not section_losses:
        return new_coords.sum() * 0.0
    return torch.stack(section_losses).mean()


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
            # [v1.9 §4-B] mean → Top-K 풀링(review_v1_8.md §9). Huber는 그대로 유지한다(이 함수의
            # 존재 이유 자체가 v1.3의 클램프형 제곱 페널티 그래디언트 소멸을 무클램프+Huber로
            # 보강하는 것이므로 — uni_section_v20.compute_collision_loss_v5()와는 역할이 다르다).
            huber_v = _huber_collision_term(violation)[valid.bool()]
            if huber_v.numel() > 0:
                k_col = max(1, int(huber_v.numel() * PHYS_TOPK_FRACTION))
                top_huber, _ = torch.topk(huber_v, k=k_col, largest=True)
                loss_dir = top_huber.mean()
            else:
                loss_dir = violation.sum() * 0.0
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


def generate_gp_target_mp(seed=None, sigma=0.25, ell=3.0, monotonic_decay=1e-4):
    """initial_section.md §2 표 기준, 공간 상관 있는 GP 스무딩 +/-50% 목표 Mp 생성.
    [v1.11] sec0(최하단)→sec{NUM_SECTIONS-1}(최상단) 방향으로 target_mp가 항상 줄어들도록
    누적 최솟값을 적용해 강제 단조 감소시킨다(GP 노이즈만으로는 이 방향성이 보장되지 않았음).
    monotonic_decay: 매 섹션 스텝마다 곱해지는 추가 감소 비율 — 인접 섹션이 GP 노이즈로
    우연히 같은 값이 되는 것을 막아 '엄격히' 감소하도록 한다(0이면 비증가만 보장).
    반환: {section_id(int): target_mp(float)}."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    initial_mp = np.array(INITIAL_MP_TABLE, dtype=np.float64)
    sections = np.arange(NUM_SECTIONS, dtype=np.float64)
    K = sigma ** 2 * np.exp(-(sections[:, None] - sections[None, :]) ** 2 / (2 * ell ** 2))
    K += 1e-6 * np.eye(NUM_SECTIONS)
    alpha = rng.multivariate_normal(np.zeros(NUM_SECTIONS), K)
    alpha = np.clip(alpha, -0.5, 0.5)

    target_mp = initial_mp * (1.0 + alpha)

    # [v1.11] sec0(최하단) 기준 누적 최솟값 강제 — 상단으로 갈수록 target_mp가 줄어들게 함
    for k in range(1, NUM_SECTIONS):
        cap = target_mp[k - 1] * (1.0 - monotonic_decay)
        target_mp[k] = min(target_mp[k], cap)

    return {k: float(target_mp[k]) for k in range(NUM_SECTIONS)}


def load_target_mp_my_from_csv(path):
    """[v1.12 §1, index 0~16 직접 매칭으로 갱신] target_mp CSV(UTF-8, 콤마 구분, 헤더 +
    17행: index,mp_target[,my_target] — index는 섹션 인덱스(0=최하단~16=최상단)와 동일한 값)를
    읽어 (target_mp, target_my) 딕셔너리 쌍을 반환한다 — 둘 다 generate_gp_target_mp()와 동일한
    {section_id(int): value(float)} 형식.

    [v1.15.2 §1] my_target 컬럼을 **선택 사항**으로 바꿨다. v1.15.4의 기본 입력인
    target_mp_v1_15_4.csv는 index,mp_target 두 컬럼뿐이다 — 항복 모멘트의 목표값을 모르기 때문에
    애초에 목표를 세우지 않는다는 사용자 결정이다. 컬럼이 없으면 target_my=None을 반환하고,
    리포트는 target/달성률 없이 계산된 My(initial/soft/hard)만 출력한다(§2).
    my_target 컬럼이 있으면 v1.15와 동일하게 읽어 리포트에 목표 대비로 병기한다 —
    target_mp.csv(구 입력)와의 호환성 유지.

    [v1.12 수정] 이전 버전은 CSV 행 순서(1~17)와 섹션 인덱스(0~16) 간 대응 방향이 불확실하다는
    전제로 추세 부호 비교 기반 자동판정(order="auto"/"as_is"/"reverse")을 두었으나, target_mp.csv의
    index 컬럼이 이제 섹션 인덱스(0~16)를 그대로 담도록 정리되어 그 추측이 더 이상 필요 없다.
    index 컬럼 값을 곧바로 딕셔너리 키로 써서 그대로 매칭한다 — 행 순서에 의존하지 않으므로 CSV
    행이 섞여 있어도 무관하다. index 집합이 {0,...,NUM_SECTIONS-1}과 정확히 일치하지 않으면
    fail-loud로 즉시 중단한다(silent mis-map 방지).
    항복 모멘트(my_target)는 이 함수가 반환할 뿐 어디에도 학습/손실에 주입하지 않는다 —
    호출부에서 리포트 출력에만 사용한다."""
    df = pd.read_csv(path, encoding="utf-8")
    if len(df) != NUM_SECTIONS:
        raise RuntimeError(f"{path}: {NUM_SECTIONS}행이어야 하는데 {len(df)}행입니다.")

    for col in ("index", "mp_target"):
        if col not in df.columns:
            raise RuntimeError(f"{path}: 필수 컬럼 '{col}'이 없습니다 (컬럼: {list(df.columns)}).")

    idx = df["index"].to_numpy(dtype=np.int64)
    expected = set(range(NUM_SECTIONS))
    if set(idx.tolist()) != expected:
        raise RuntimeError(
            f"{path}: index 컬럼이 {{0,...,{NUM_SECTIONS-1}}}과 정확히 일치해야 하는데 "
            f"{sorted(idx.tolist())}입니다.")

    target_mp = {int(row["index"]): float(row["mp_target"]) for _, row in df.iterrows()}
    # [v1.15.2 §1] my_target은 선택 컬럼 — 없으면 None(리포트에서 목표 없이 My만 출력)
    if "my_target" in df.columns:
        target_my = {int(row["index"]): float(row["my_target"]) for _, row in df.iterrows()}
    else:
        target_my = None
    return target_mp, target_my


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


def compute_edge_my_elastic(coords, t, fy, edge_index):
    """[v1.12 §2] 탄성 항복모멘트(My) 계산 — uni_section_v20.compute_edge_mp_pna()의 "얇은 판을
    y방향으로 투영한 등가 스트립(area=L*t_e, 높이 H=y_top-y_bot에 균등 분포)" 기하 모델을 그대로
    재사용하되, 소성(PNA 이분탐색) 대신 탄성(도심축, 면적 1차/2차 모멘트)으로 대체한다.
    uni_section_v20.py는 프로젝트 관례상(파일 상단 주석) 별도 수정 대상이 아니므로 이 계산은
    AI_design_v1_12.py 안에 로컬로 새로 구현한다 — 기존 소성(Mp) 계산 경로는 전혀 건드리지 않는다.

    - 탄성 중립축(도심) y_na: PNA와 달리 fy에 무관하게 면적 1차모멘트만으로 닫힌 형태로 구해진다.
    - 단면 2차모멘트 I: 각 엣지를 [y_bot,y_top] 구간에 면적이 균등 분포한 스트립으로 보고
      자체관성(A*H^2/12) + 평행축(A*(centroid-y_na)^2)을 합산.
    - My: "가장 먼저 항복하는 위치가 그 섹션 전체의 항복모멘트를 지배한다"는 first-yield 기준으로,
      엣지별 M_e = fy_e * I / max(|y_top_e-y_na|, |y_bot_e-y_na|) 중 최솟값을 취한다(어느 한 곳이라도
      fy_e에 도달하면 그 단면은 항복 시작으로 간주 — Mp의 PNA 완전소성 가정과 대비되는 탄성 한계).
    Returns: (my_total, y_na)."""
    mask = edge_index[0] < edge_index[1]
    u, v = edge_index[0][mask], edge_index[1][mask]

    y_u, y_v = coords[u, 1], coords[v, 1]
    x_u, x_v = coords[u, 0], coords[v, 0]
    L = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
    t_e = t[u].squeeze(-1)
    fy_e = fy[u].squeeze(-1)

    dx = torch.abs(x_u - x_v)
    t_y = t_e * (dx / (L + 1e-12))
    y_max = torch.maximum(y_u, y_v)
    y_min = torch.minimum(y_u, y_v)
    y_top = y_max + t_y / 2.0
    y_bot = y_min - t_y / 2.0
    H = torch.clamp(y_top - y_bot, min=1e-12)

    area_e = L * t_e
    centroid_e = (y_top + y_bot) / 2.0

    y_na = torch.sum(area_e * centroid_e) / torch.clamp(torch.sum(area_e), min=1e-12)

    I_self = area_e * H ** 2 / 12.0
    I_parallel = area_e * (centroid_e - y_na) ** 2
    I_total = torch.sum(I_self + I_parallel)

    c_e = torch.maximum(torch.abs(y_top - y_na), torch.abs(y_bot - y_na))
    m_e = fy_e * I_total / torch.clamp(c_e, min=1e-9)
    my_total = torch.min(m_e)

    return my_total, y_na


def calculate_myl(coords, t, fy, edge_index):
    my_total, _ = compute_edge_my_elastic(coords, t, fy, edge_index)
    return my_total


def _per_section_myl(coords, t, fy, edge_index, edge_attr, section_ids, device):
    """섹션별 My 계산 — _per_section_mpl()과 동일한 물리 엣지 필터링/로컬 재인덱싱 패턴."""
    out = {}
    src, dst = edge_index
    edge_type = edge_attr[:, 3]
    for k in range(NUM_SECTIONS):
        m = section_ids == k
        emask = m[src] & m[dst] & torch.isclose(edge_type, torch.zeros_like(edge_type))
        local = torch.full((coords.shape[0],), -1, dtype=torch.long, device=device)
        local[m] = torch.arange(int(m.sum()), device=device)
        e_sec = local[edge_index[:, emask]]
        out[k] = float(calculate_myl(coords[m], t[m], fy[m], e_sec).item())
    return out


@torch.no_grad()
def export_final_section_csv(model, data, target_mps, pruning_state, final_seg_ids, device,
                              temperature=None, save_path="reports/final_section_v1_15_4.csv"):
    """[v1.14 §6] 학습 결과 단면을 floor_profiles_template.md 규격 CSV로 내보낸다.

    이 함수는 command_v1_14.md §1~§4(손실/스케줄 수정)에는 없지만 §6 검증 계획이 요구하는
    산출물이다 — v1.13까지 이 저장소에는 결과 단면을 template 규격으로 내보내는 코드가 없었고
    (reports/v1_13/final_section_v1_13.csv는 코드 산출물이 아니다), 그 CSV가 없으면 v1.14의
    1차 성공 기준인 "파트 간 겹침 해소"를 review_v1_13.md와 동일한 방법(노드±t/2 밴드 겹침
    전수 검사)으로 검증할 수 없다. 학습·손실에는 전혀 관여하지 않는 순수 출력 함수다.

    규격(floor_profiles_template.md):
      - 헤더: floor_idx,point_idx,x_mm,y_mm,r_mm,t_mm
      - 파트 블록마다 [좌표 데이터 행들] 다음에 [구분 행 0.0,0.0,0.0,0.0,0.0,<그 파트 두께>]
      - 데이터 행의 r_mm/t_mm은 항상 0.0, 두께는 구분 행에서만 파트 단위로 선언
      - 두께는 제조 기준값인 pooled t_raw를 쓴다(soft 게이트가 곱해진 t_final이 아님 —
        report_design_comparison()의 "## 4. Part thickness" 표와 동일한 정의)
      - hard 삭제된 (섹션, candidate 파트) 조합은 해당 floor를 통째로 생략한다
        (v1.13 결과에서 #08(Patch2)이 floor 0~12만 남았던 것과 동일한 규칙)
    """
    if temperature is None:
        temperature = 1.0
    model.eval()
    x = data.x
    section_ids = x[:, 5].long()
    part_ids = x[:, 4].long()
    t_init = x[:, 6].unsqueeze(1)

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=device)
    for k in range(NUM_SECTIONS):
        target_mp_node[section_ids == k] = target_mps[k]
    fix_x = x[:, 2].bool().unsqueeze(1)
    fix_y = x[:, 3].bool().unsqueeze(1)

    # report_design_comparison()과 동일한 eval forward + t_raw 복원 경로를 그대로 재현한다.
    new_coords, _, _, delta_t_part, z_gate, _ = model(
        x, data.edge_index, data.edge_attr, target_mp_node, fix_x, fix_y, data.join_pairs,
        thickness_gate=1.0, gate_active=True, temperature=temperature, segment_ids=final_seg_ids)
    t_min, t_max = model.T_MIN, model.T_MAX
    t_raw = t_min + (t_max - t_min) * torch.sigmoid(_t_init_logit(t_init, t_min, t_max) + delta_t_part)

    cand_idx = torch.tensor(CANDIDATE_PARTS, device=device)
    exists_cand = (pruning_state['state'].to(device) != STATE_DELETED) & \
                  (z_gate[:, cand_idx] >= BINARIZE_THRESHOLD)
    exists_full = torch.ones(NUM_SECTIONS, model.num_parts, dtype=torch.bool, device=device)
    exists_full[:, cand_idx] = exists_cand

    coords_cpu = new_coords.detach().cpu()
    t_raw_cpu = t_raw.detach().cpu().squeeze(-1)
    sec_cpu = section_ids.cpu()
    part_cpu = part_ids.cpu()
    exists_cpu = exists_full.cpu()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    n_rows, skipped = 0, []
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        f.write("floor_idx,point_idx,x_mm,y_mm,r_mm,t_mm\n")
        for pid in range(model.num_parts):
            pmask_all = (part_cpu == pid)
            if not bool(pmask_all.any()):
                continue
            for sec in range(NUM_SECTIONS):
                if not bool(exists_cpu[sec, pid]):
                    skipped.append((pid, sec))
                    continue
                node_mask = pmask_all & (sec_cpu == sec)
                if not bool(node_mask.any()):
                    continue
                # build_bpillar_from_csv()가 (floor, part, point) 순으로 노드를 쌓으므로
                # 마스크 순서가 곧 point_idx 순서다(1부터 시작).
                idxs = torch.nonzero(node_mask, as_tuple=False).squeeze(-1)
                for point_idx, ni in enumerate(idxs.tolist(), start=1):
                    f.write(f"{float(sec)},{float(point_idx)},"
                            f"{float(coords_cpu[ni, 0])},{float(coords_cpu[ni, 1])},0.0,0.0\n")
                    n_rows += 1
            # 파트 블록 종료 → 구분 행으로 그 파트의 두께를 확정(좌표 뒤에 온다)
            t_part = float(t_raw_cpu[pmask_all].mean())
            f.write(f"0.0,0.0,0.0,0.0,0.0,{t_part}\n")

    print(f"[export] {save_path} 저장 완료 — 데이터 행 {n_rows}개, 파트 {model.num_parts}개"
          + (f", hard 삭제로 생략된 (part,floor) {len(skipped)}개" if skipped else ""))
    return save_path


@torch.no_grad()
def report_design_comparison(model, data, target_mps, pruning_state, final_seg_ids, device,
                              temperature=None, save_path="reports/AI_design_v1_15_4_report.md",
                              death_log=None, target_area=None, target_mys=None):
    """[v1.3] initial vs final design property 비교 리포트.
    - Mp: soft(학습 상태, t_final=t_raw*z_gate)와 hard(제조 상태) 병기. hard 존재 판정은
      (state != DELETED) AND (z_gate >= BINARIZE_THRESHOLD) — [v1.3 §4] 0.5 → 0.3, 비교연산자 >=.
    - 두께: 제조 관점 값은 pooled t_raw(soft 게이트가 곱해진 t_final이 아님). 삭제 칸은 0.
    - 괴리 |hard-soft| > 2%인 섹션은 경고 표시(fine-tuning 필요 신호, Gemini 지적).
    [v1.2 수정] temperature를 하드코딩(1.0)하지 않고 학습 종료 시점의 usv18.compute_gate_temperature
    값을 호출부에서 전달받는다 — 온도가 다르면 같은 log_alpha라도 HardConcrete 하드 샘플이 학습
    당시와 달라져(예: 250 epoch 분석에서 Patch1이 state=ALIVE인데도 리포트만 X로 표시된 원인 중 하나),
    리포트가 실제 학습 종료 상태를 반영하지 못하는 문제가 있었다. temperature=None이면 이전 동작과
    호환되도록 1.0을 fallback으로 사용한다.
    [v1.12 §2] target_mys: {section_id: target_my(float)} 선택 인자. My_init/soft/hard는
    calculate_myl()(탄성 first-yield 기준)로 실제 생성된 단면 형상에서 계산하며, target_mys는
    CSV에서 읽은 목표값을 나란히 표시하는 참고용일 뿐이다 — 어느 쪽도 단면 생성/학습/손실에
    관여하지 않는다(사용자 요구사항: My는 리포트 전용).
    [v1.15.2 §2 변경] target_mys=None이어도 My 표를 **생략하지 않는다**. v1.15까지는 None이면
    표 자체를 빼버렸으나, v1.15.4의 기본 입력(target_mp_v1_15_4.csv)에는 my_target이 없으므로 그러면
    My가 리포트에서 완전히 사라진다. 목표값을 모르는 것과 결과값을 안 보는 것은 별개다 —
    None이면 target/달성률 컬럼만 빼고 계산된 My(initial/soft/hard=final_my)를 그대로 출력한다."""
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

    # [v1.12 §2] 섹션별 My(탄성 항복모멘트): initial / final-soft / final-hard — 리포트 전용.
    # [v1.15.2 §2] target_mys 유무와 무관하게 **항상** 계산한다 — 목표가 없어도 결과(final_my)는
    # 출력해야 하므로.
    my_init = _per_section_myl(base_coords, t_init, fy, data.edge_index, data.edge_attr, section_ids, device)
    my_soft = _per_section_myl(new_coords, t_soft, fy, data.edge_index, data.edge_attr, section_ids, device)
    my_hard = _per_section_myl(new_coords, t_hard, fy, data.edge_index, data.edge_attr, section_ids, device)

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

    # [v1.12 §2] 항복모멘트(My) — Mp와 병행 표기하되 완전히 별도 표(단면 생성에 영향 없음,
    # 순수 참고 지표). calculate_myl()로 실제 형상에서 계산한 init/soft/hard를 출력한다.
    # [v1.15.2 §2] target_mys가 있으면 목표/달성률까지, 없으면 계산값만.
    n_warn_my = 0
    if target_mys is not None:
        lines.append("\n## 2. Section-wise My (N*mm, 탄성 first-yield 기준, 리포트 전용 — 단면 생성에 미반영)\n")
        lines.append("| sec | My initial | My target | My final(soft) | My final(hard) | hard/target(%) | flag |")
        lines.append("|----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|")
        for k in range(NUM_SECTIONS):
            tgt_y = target_mys[k]
            ach_y = my_hard[k] / tgt_y * 100.0
            gap_y = abs(my_hard[k] - my_soft[k]) / max(my_soft[k], 1e-9)
            flag_y = ""
            if gap_y > 0.02:
                flag_y += f"soft-hard gap {gap_y*100:.1f}% "
                n_warn_my += 1
            if abs(ach_y - 100.0) > 2.0:
                flag_y += "target miss"
            lines.append(f"| {k} | {my_init[k]:,.0f} | {tgt_y:,.0f} | {my_soft[k]:,.0f} | "
                         f"{my_hard[k]:,.0f} | {ach_y:.1f} | {flag_y} |")
        tot_iy, tot_ty = sum(my_init.values()), sum(target_mys.values())
        tot_sy, tot_hy = sum(my_soft.values()), sum(my_hard.values())
        lines.append(f"| **sum** | {tot_iy:,.0f} | {tot_ty:,.0f} | {tot_sy:,.0f} | {tot_hy:,.0f} | "
                     f"{tot_hy/tot_ty*100:.1f} | |")
    else:
        # [v1.15.2 §2] target_my를 모르는 경우 — 목표/달성률 없이 계산된 My만 출력.
        # final(hard)가 제조 상태의 My이므로 "final_my"로 읽으면 된다.
        lines.append("\n## 2. Section-wise My (N*mm, 탄성 first-yield 기준, 리포트 전용 — 단면 생성에 미반영)\n")
        lines.append("(target_my 미설정 — 목표값 없이 계산 결과만 출력. final(hard)=제조 상태 My)\n")
        lines.append("| sec | My initial | My final(soft) | My final(hard) | final(hard)/initial(%) | flag |")
        lines.append("|----:|-----------:|---------------:|---------------:|-----------------------:|:----|")
        for k in range(NUM_SECTIONS):
            ratio_y = my_hard[k] / max(my_init[k], 1e-9) * 100.0
            gap_y = abs(my_hard[k] - my_soft[k]) / max(my_soft[k], 1e-9)
            flag_y = ""
            if gap_y > 0.02:
                flag_y += f"soft-hard gap {gap_y*100:.1f}%"
                n_warn_my += 1
            lines.append(f"| {k} | {my_init[k]:,.0f} | {my_soft[k]:,.0f} | {my_hard[k]:,.0f} | "
                         f"{ratio_y:.1f} | {flag_y} |")
        tot_iy = sum(my_init.values())
        tot_sy, tot_hy = sum(my_soft.values()), sum(my_hard.values())
        lines.append(f"| **sum** | {tot_iy:,.0f} | {tot_sy:,.0f} | {tot_hy:,.0f} | "
                     f"{tot_hy/max(tot_iy,1e-9)*100:.1f} | |")
    if n_warn_my:
        lines.append(f"\n> [WARNING] soft-hard My 괴리 >2%인 섹션 {n_warn_my}개.")

    lines.append("\n## 3. Total cross-section area (mm^2, 17-section sum)\n")
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
        lines.append(f"\n## 5. 전역 게이트 사망 원장 (§2.4.1) — SUSPECT {n_suspect}건 / 전체 {len(death_log)}건\n")
        lines.append("| epoch | sec | cand | Mp@사망 | area@사망 | verdict |")
        lines.append("|------:|----:|-----:|--------:|----------:|:--------|")
        for d in death_log:
            lines.append(f"| {d['epoch']} | {d['sec']} | {d['cand']} | {d['mp_at_death']:.1%} | "
                         f"{d['area']:,.0f} | {d['verdict']} |")

    lines.append("\n## 4. Part thickness (mm, 제조 기준 = pooled t_raw)\n")
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

    lines.append("\n### 4.1 Part별 물성치 (항복응력, FY_BY_PART 고정값)\n")
    lines.append("| part | fy (MPa) |")
    lines.append("|:-----|---------:|")
    for pid in range(5):
        pm = part_ids == pid
        fy_part = float(fy[pm].mean())
        lines.append(f"| {PART_NAMES[pid]} | {fy_part:.1f} |")

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
            'my_init': my_init, 'my_soft': my_soft, 'my_hard': my_hard,
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

# [v1.14 §2] CLEARANCE_CURRICULUM_END는 상수 정의 순서 때문에 위쪽 v1.14 블록에서 리터럴(400)로
# 선언됐다 — ASWC_STAGE2_END와 갈라지면 "연속 파트 두께가 detach되는 시점"과 "clearance가 target에
# 도달하는 시점"이 어긋나 의도치 않은 스케줄 불연속이 생기므로 여기서 정합성을 강제한다.
# 의도적으로 다르게 두려면 이 assert를 지우고 그 근거를 command 문서에 남길 것.
assert CLEARANCE_CURRICULUM_END == ASWC_STAGE2_END, (
    f"CLEARANCE_CURRICULUM_END({CLEARANCE_CURRICULUM_END}) != ASWC_STAGE2_END({ASWC_STAGE2_END}) "
    f"— command_v1_14.md §2 참고")


def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None, neighbor_map=None,
                      w_smooth_lse_val=None):   # [v1.10 §3]
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
        # [v1.15 §5] 동결 대상은 GNN이 예측한 변위뿐이다. 두께의 결정론적 함수인 플랜지 오프셋까지
        # 지우면, 두께가 계속 자라는 Stage 1(=두께 전용 학습 구간) 내내 접합부가 간섭 상태가 되고
        # Stage 2 진입 시 좌표가 튄다. 오프셋은 model.last_flange_offset 속성으로 전달된다
        # (반환 개수를 바꾸면 uni_section 물리 계층의 언패킹까지 깨지므로 — §4(a) 참고).
        _fo = getattr(model, 'last_flange_offset', None)
        if _fo is None:
            _fo = torch.zeros_like(new_coords)
        new_coords = base_coords.detach() + _fo

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

    # [v1.9 §4-A] mean → Top-K 풀링(review_v1_8.md §7). 17섹션 중 1~2개의 국소 붕괴가
    # 1/17로 희석되던 문제를 compute_mesh_order_loss()(v1.8 §1)와 동일한 패턴으로 해소.
    l_phys_stack = torch.stack(l_phys_terms)                      # (17,)
    k_phys = max(1, int(l_phys_stack.numel() * PHYS_TOPK_FRACTION))
    l_phys_topk, _ = torch.topk(l_phys_stack, k=k_phys, largest=True)
    l_phys_total = torch.mean(l_phys_topk)
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
    l_smooth_lse = compute_smoothness_lse_v3(x, new_coords, neighbor_map, t_final)   # [v1.10 §1]
    # [v1.3 §2.1 정정] usv18.compute_mass_loss()가 반환하는 l_mass는 로깅 전용이며 loss에 더해지지
    # 않는다(실제 질량 항은 아래 §2.6 l_mass_budget). area만 재사용하고 l_mass_legacy는 로깅에만 쓴다.
    area, l_mass_legacy = usv18.compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)
    l_mass = torch.relu(area / target_area - 1.0) ** 2   # [v1.3 §2.6] 목표 초과분만 이차 벌점(예산 제약)
    l_area_floor = torch.relu(1.0 - area / (SAFETY_RATIO * target_area + 1e-12)) ** 2  # [v1.4 §3]
    # z_gate는 collision loss에서 part별 스칼라를 기대(원본 line 632 z_gate[part]) — 섹션마다 다를 수
    # 있으므로 섹션 평균을 사용(연속 파트는 항상 1이라 영향 없음, candidate 파트만 근사가 생김을 인지).
    # [v1.9 §5] 17섹션 전체 평균 → 생존 섹션만의 평균(review_v1_8.md §8). DELETED 확정 섹션이
    # 분모에 남아 있으면, 생존 섹션에서도 candidate 파트의 collision 억제력이 부당하게 희석된다.
    if pruning_state is not None:
        alive_mask = torch.ones_like(z_gate, dtype=torch.bool)   # 연속 파트는 항상 alive 취급
        cand_idx_t = torch.tensor(CANDIDATE_PARTS, device=z_gate.device)
        alive_cand = (pruning_state['state'] != STATE_DELETED).to(z_gate.device)  # (17, n_cand)
        alive_mask[:, cand_idx_t] = alive_cand
        z_gate_part_avg = (z_gate * alive_mask).sum(dim=0) / alive_mask.sum(dim=0).clamp(min=1)
    else:
        z_gate_part_avg = z_gate.mean(dim=0)  # (5,) — pruning_state가 없는 호출 경로(Stage0 등) 폴백
    # [v1.9 §4-C] top_k_fraction = 각 direction '내부' 노드 Top-K
    # [v1.14 §1] top_k_dir_fraction = direction '사이' Top-K(신규) — 153개 전체 평균으로 위반
    #            소수 파트쌍이 1/153로 희석되던 문제 해소(review_v1_13.md 근본 원인 1위).
    #            두 인자는 서로 다른 계층에 적용되는 별개 파라미터다.
    l_collision = usv18.compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                                   collision_spec, z_gate=z_gate_part_avg,
                                                   top_k_fraction=PHYS_TOPK_FRACTION,
                                                   top_k_dir_fraction=COL_TOPK_DIR_FRACTION)
    l_col_aux = compute_collision_penalty_unclamped(new_coords, t_final, part_ids, section_ids,
                                                     collision_spec)   # [v1.4 §2] 무클램프 보강
    l_order     = usv18.compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr)
    l_anchor    = usv18.compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask)
    l_sat       = usv18.compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE)

    if w_continuity_val is None:                                          # [v1.6 §1]
        w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
    else:
        w_continuity = w_continuity_val
    l_continuity = compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0)
    contrib_continuity = w_continuity * l_continuity

    # [v1.10 §2] candidate 생존/삭제 경계 섹션에서만 연속 파트에 추가 완충 손실
    boundary_pairs = find_candidate_boundary_sections(pruning_state)
    l_boundary_continuity = compute_boundary_continuity_loss(new_coords, section_ids, part_ids,
                                                               boundary_pairs)
    contrib_boundary_continuity = W_BOUNDARY_CONTINUITY * l_boundary_continuity

    if w_smooth_lse_val is None:                                          # [v1.10 §3]
        w_smooth_lse = w_smooth_lse_schedule(epoch, ASWC_STAGE2_END)
    else:
        w_smooth_lse = w_smooth_lse_val

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
    contrib_smooth_lse = w_smooth_lse * l_smooth_lse                  # [v1.10 §3]

    loss = (contrib_phys + contrib_smooth + L_alda_effective
            + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparse + contrib_entropy + contrib_continuity
            + contrib_boundary_continuity                              # [v1.10 §2]
            + contrib_mass + contrib_col_aux + contrib_area_floor
            + contrib_smooth_lse)                                     # [v1.10 §3]

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
        # [v1.14 §6] 검증 계획용 — g_col_val은 §3 dual-ascent의 입력이기도 하다(그 전까지는
        # compute_alda_loss()가 계산만 하고 호출부가 버리던 값이라 로그에 남지 않았다).
        "g_col_val": g_col_val, "g_Mp_val": g_Mp_val,
        # [v1.15 §9] 진단 전용 비마스킹 관통 지표 — collision loss에서 제외한 접합 쌍이 실제로
        # 간섭하고 있지 않은지 감시한다. detach 상태이며 역전파에 절대 들어가지 않는다.
        **compute_flange_penetration_diag(model, new_coords, t_final),
        "contrib_phys": contrib_phys.item(),
        "L_alda_effective": L_alda_effective.item(),
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

    # [v1.15 §3] 접합 플랜지 쌍 1회 등록(FLANGE_PAIR_SPEC 명시적 테이블) — dry-run 전에 해야
    # dry-run도 실제 경로를 탄다.
    _fp = identify_mating_flange_pairs(data.x)
    model.set_flange_pairs(_fp)
    _P = int(_fp['flange_idx_a'].numel())
    print(f"[v1.15 §3] 접합 플랜지 쌍 {_P}개 등록 "
          f"(FLANGE_PAIR_SPEC {len(FLANGE_PAIR_SPEC)}쌍 x {NUM_SECTIONS}floor, 테이퍼 ±{FLANGE_TAPER_N})")
    if _P > 0:
        _g = _fp['flange_gap0']
        print(f"[v1.15 §3] gap0 min={float(_g.min()):.4f} / mean={float(_g.mean()):.4f} / "
              f"max={float(_g.max()):.4f} mm, 관여 노드 "
              f"{int(torch.unique(torch.cat([_fp['flange_idx_a'], _fp['flange_idx_b']])).numel())}개")
    else:
        print("  !! [v1.15 경고] 접합 쌍이 0개다 — 오프셋 레이어가 아무것도 하지 않는다. "
              "FLANGE_PAIR_SPEC이 비어있거나 point_idx가 CSV와 어긋나지 않았는지 확인할 것.")

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

    # [v1.15 §7] fix_mask 를 넘겨 fixed-fixed 항목(ff_mask)을 direction마다 미리 조립한다.
    collision_spec = usv18.build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids,
                                                 fix_mask=x[:, 2].bool())
    print(f"[collision v5] {len(collision_spec)}쌍(섹션x파트쌍) 부호 앵커 산정 완료")

    # [v1.14 §2] clearance 커리큘럼 준비 — 원본 clearance를 direction마다 백업해 둔다.
    # (build_collision_spec()은 초기 형상 기준 1회만 산정하므로, 이 값이 커리큘럼의 시작점이 된다)
    n_col_dirs = backup_initial_clearance(collision_spec)
    _init_cls = [d['_init_clearance'] for dl in collision_spec.values() for d in dl]
    print(f"[v1.14 §2] clearance 커리큘럼 준비: direction {n_col_dirs}개, "
          f"초기 clearance min={min(_init_cls):.4f} / mean={sum(_init_cls)/len(_init_cls):.4f} / "
          f"max={max(_init_cls):.4f} mm -> target {CLEARANCE_TARGET_MM}mm (epoch {CLEARANCE_CURRICULUM_END}까지)")
    print(f"[v1.14 §1] direction-레벨 Top-K: fraction={COL_TOPK_DIR_FRACTION} "
          f"(전체 {n_col_dirs}개 중 상위 약 {max(usv18.COL_MIN_DIRS_TO_KEEP, int(n_col_dirs * COL_TOPK_DIR_FRACTION))}개 사용)")
    print(f"[v1.14 §3] ALDA dual-ascent: {'ON' if ENABLE_DYNAMIC_ALDA else 'OFF(기본값)'} "
          f"— rho_col={alda_state['rho_col']:.1f}, mu_col={alda_state['mu_col']:.3f}")

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
        curr_w_smooth_lse = w_smooth_lse_schedule(epoch, ASWC_STAGE2_END)        # [v1.10 §3]
        # [v1.14 §2] collision_spec의 clearance를 이번 epoch 값으로 갱신(in-place).
        # train_step_multi()는 collision_spec을 그대로 넘겨받으므로 호출 '전에' 적용해야 한다.
        cl_progress, cl_mean = apply_clearance_curriculum(collision_spec, epoch)
        info = train_step_multi(model, data, optimizer, target_mps, target_area,
                                 epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                                 collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                                 alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                                 w_col_aux_val=curr_w_col_aux,
                                 w_continuity_val=curr_w_continuity,
                                 w_smooth_lse_val=curr_w_smooth_lse,      # [v1.10 §3]
                                 neighbor_map=neighbor_map)  # [v1.4 §4.2][v1.5 §3][v1.7 §1][v1.7 §2][v1.10 §3]
        last_epoch = epoch                                      # [v1.3 §5.1] 루프 탈출 시점 기록
        area_hist[epoch] = info['area']

        # [v1.14 §3] ALDA dual-ascent — 기본 OFF. 켜져 있을 때만 mu_col/rho_col이 갱신된다.
        if update_alda_dual_ascent(alda_state, info['g_col_val'], epoch):
            print(f"  [ALDA] epoch {epoch}: g_col={info['g_col_val']:.5f} > "
                  f"slack({alda_state['slack_threshold']}) -> mu_col={alda_state['mu_col']:.3f}, "
                  f"rho_col={alda_state['rho_col']:.1f}")

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
                  f"| l_smooth_lse={info['l_smooth_lse']:.4f}"     # [v1.4 §4.3][v1.5 §4][v1.6 §1][v1.7 §1]
                  # [v1.14 §6] 검증 계획 필드 — l_collision/g_col은 Top-K(§1) 효과를, clearance는
                  # 커리큘럼(§2) 진행을, contrib_phys 대비 L_alda는 물리 손실과의 스케일 경쟁을 본다.
                  # [v1.15 §9] flange_pen 은 진단 전용(역전파 없음) — Stage 1부터 ~0 이어야 한다
                  f" | flange_pen={info['l_flange_pen_max']:.4f}({info['n_flange_pen']}) "
                  f"| l_col={info['l_collision']:.4f} g_col={info['g_col_val']:+.4f} "
                  f"| clr={cl_mean:+.3f}mm({cl_progress:.0%}) "
                  f"| rho_col={alda_state['rho_col']:.0f} "
                  f"| phys={info['contrib_phys']:.3f} alda={info['L_alda_effective']:.3f}")

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

        # [v1.9 §2] 단일 스텝 강제 이진화 대신 목표값(±5.0)만 정하고, 실제 대입은 Stage4 루프 안에서
        # STAGE4_GATE_WARMUP_EPOCHS에 걸쳐 매 epoch 선형 보간한다. Stage4 optimizer의 param_groups
        # 에는 애초에 log_alpha_candidates가 없으므로(main_params/thick_params만 포함) 이 파라미터
        # 자체가 그래디언트로 갱신되지는 않는다 — 보간의 목적은 log_alpha가 아니라 z_gate/t_final이
        # 매 epoch 부드럽게 이동하도록 만들어, collision/continuity 등 다른 손실이 보는 형상이
        # 한 스텝에 불연속으로 튀지 않게 하는 것이다. requires_grad는 보간이 끝난 뒤에만 끈다.
        # compute_gates_multi()는 log_alpha 값을 그대로 sigmoid에 넣으므로 보간 중에도 항상 미분
        # 가능한 soft 값을 계산한다(Straight-Through Estimator 불필요 — idea_v1_9.md 부록 참고).
        la_init_v19 = model.log_alpha_candidates.detach().clone()
        la_target_v19 = torch.where(hard_exists.to(device), torch.tensor(5.0, device=device),
                                     torch.tensor(-5.0, device=device))

        seg_ids = compute_segment_ids(pruning_state, hard_exists=hard_exists).to(device)

        # [v1.9 §2] optimizer state 이전: 이 시점까지 model의 파라미터 객체(main_params/thick_params)
        # 는 한 번도 재생성되지 않았다(Stage4가 모델을 다시 만들지 않고 optimizer만 새로 만듦) —
        # 따라서 기존 optimizer.state는 여전히 "같은 Parameter 객체"를 키로 갖고 있다. id()나 이름
        # 매칭 같은 간접 수단이 필요 없다: 파라미터 객체 자체를 dict key로 재사용할 수 있다.
        old_state_by_param = {p: optimizer.state[p] for p in main_params + thick_params
                               if p in optimizer.state}

        # 옵티마이저 재빌드 — 실제 변수명(main_params/thick_params) 사용, gate_params 그룹 제외.
        optimizer = optim.AdamW(
            [{'params': main_params,  'name': 'main',              'lr': lr * STAGE4_LR_SCALE},
             {'params': thick_params, 'name': 'thickness_decoder', 'lr': lr * STAGE4_LR_SCALE}],
            lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)

        try:
            for p, state in old_state_by_param.items():
                optimizer.state[p] = state
            print(f"[Stage4] optimizer 모멘텀 이전 완료: {len(old_state_by_param)}개 파라미터")
        except Exception as e:   # [v1.9 §2] 안전장치 — 실패해도 학습이 죽지 않고 모멘텀 없이 계속
            print(f"[Stage4][WARNING] optimizer state 이전 실패({e}) — 모멘텀 없이 새로 시작합니다")

        weights['w_sparse'] = 0.0
        alpha_ent_active = 0.0
        # [v1.3 §5.6] 커리큘럼은 루프 탈출 시점 값으로 1회 계산 후 고정 재사용(외삽 방지)
        frozen_s_phys, frozen_s_smooth = usv18.get_curriculum_weights_v10(
            last_epoch, curriculum_max_epochs, curriculum_ratio)

        for ep in range(stage4_start, stage4_end):
            # [v1.9 §2] 게이트 선형 보간 — matching hard-flip을 STAGE4_GATE_WARMUP_EPOCHS에 걸쳐
            # 완화한다. progress>=1.0 시점(=WARMUP 종료)에만 동결해 §5.7 종료 assert와 호환된다
            # (STAGE4_EPOCHS >= STAGE4_GATE_WARMUP_EPOCHS를 상수 정의부에서 assert로 보장).
            progress = min(1.0, (ep - stage4_start) / STAGE4_GATE_WARMUP_EPOCHS)
            with torch.no_grad():
                model.log_alpha_candidates.copy_(
                    (1.0 - progress) * la_init_v19 + progress * la_target_v19)
            if progress >= 1.0 and model.log_alpha_candidates.requires_grad:
                model.log_alpha_candidates.requires_grad_(False)

            # [v1.5] w_col_aux_val 미전달 -> sentinel이 전역 W_COL_AUX(5.0)로 폴백. Stage4는
            # 항상 웜업 종료 시점(epoch 80) 이후에만 시작하므로(stage4_start > ASWC_STAGE3_END
            # 근방) 무해함 — design 세션 재검증에서 OpenAI 지적 사항을 코드로 확인.
            # [v1.7 §1] neighbor_map은 sentinel 기본값이 없다 — Stage4는 최종 좌표 재수렴 구간
            # 이라 이 보강 손실이 가장 필요한 지점이므로 반드시 전달한다(command_v1_7.md §1.4.4,
            # 누락 시 compute_smoothness_lse() 내부에서 NoneType 인덱싱 에러로 즉시 크래시).
            # [v1.14 §2] Stage4(ep >= 500)는 CLEARANCE_CURRICULUM_END(400)를 이미 지났으므로
            # progress=1.0, 즉 target clearance로 고정된다. 본 루프에서 이미 target에 도달한 값을
            # 그대로 물려받지만, Stage4만 단독 실행/재개하는 경우에도 동일하게 성립하도록 명시 호출한다.
            apply_clearance_curriculum(collision_spec, ep)
            info = train_step_multi(model, data, optimizer, target_mps, target_area,
                                     ep, curriculum_max_epochs, weights, curriculum=curriculum,
                                     curriculum_ratio=curriculum_ratio, collision_spec=collision_spec,
                                     alda_state=alda_state, pruning_state=pruning_state,
                                     segment_ids=seg_ids, alpha_ent=alpha_ent_active, stage4_active=True,
                                     frozen_curriculum=(frozen_s_phys, frozen_s_smooth),
                                     neighbor_map=neighbor_map)
            # [v1.14 §3] Stage4에서도 dual-ascent 유지(기본 OFF)
            update_alda_dual_ascent(alda_state, info['g_col_val'], ep)
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
    parser.add_argument("--target-csv", type=str,
                         default=r"C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\target_mp_v1_15_4.csv",
                         help="[v1.15.2] target_mp를 읽어올 CSV 경로 "
                              "(index,mp_target 2열 필수, UTF-8, index=섹션 인덱스 0~16 직접 매칭). "
                              "my_target 컬럼은 선택 — 없으면 리포트에 계산된 My만 출력한다.")
    # Jupyter(ipykernel_launcher)로 실행하면 sys.argv에 "--f=...kernel-....json" 같은 커널 인자가
    # 섞여 들어와 parse_args()가 SystemExit(2)로 죽는다 — parse_known_args()로 낯선 인자는 무시한다.
    args, _unknown = parser.parse_known_args()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    data = build_bpillar_17section()
    # [v1.12] target_mp: 절차적 GP 생성(generate_gp_target_mp, v1.11까지) 대신 CSV를 사용.
    # index 컬럼(0~16)이 섹션 인덱스와 직접 매칭된다 — 순서 추론/반전 없음.
    # [v1.15.2 §1] 기본 입력은 target_mp_v1_15_4.csv(index,mp_target) — target_my는 값을 모르므로
    # 설정하지 않는다. 그 경우 target_mys=None이 되고 리포트는 final_my만 출력한다(§2).
    target_mps, target_mys = load_target_mp_my_from_csv(args.target_csv)

    print(f"[data] 17-section 그래프(사전 필터링 없음): 노드 {data.x.shape[0]}개, 엣지 {data.edge_index.shape[1]}개")
    print(f"[target_mp] {[f'{v:,.0f}' for v in target_mps.values()]}  (from {args.target_csv})")
    if target_mys is None:
        print("[target_my] 미설정 (CSV에 my_target 컬럼 없음) — 리포트에 계산된 My만 출력합니다.")
    else:
        print(f"[target_my] {[f'{v:,.0f}' for v in target_mys.values()]} (리포트 전용, 학습에 미반영)")

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

        torch.save(model.state_dict(), "weights/bpillar_17sec_v1_15_4.pt")   # [v1.8] v1_7 가중치와 분리

        part_ids = data.x[:, 4]
        section_ids = data.x[:, 5]
        seg_cpu = final_seg_ids.cpu() if final_seg_ids is not None else None
        visualize_training_multi(history, base_coords, final_coords, part_ids, section_ids, target_mps,
                                 save_path="reports/figures/AI_design_v1_15_4_result.png",
                                 final_seg_ids=seg_cpu, split_epochs=split_epochs)
        plot_sections_3d_plotly_multi(base_coords, final_coords, part_ids, section_ids, final_z_gate,
                                      save_path="reports/figures/AI_design_v1_15_4_3d.html",
                                      final_seg_ids=seg_cpu,
                                      fix_mask=data.x[:, 2].bool().cpu())   # [v1.14] BC 고정 노드 = 네모

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
                                 death_log=death_log, target_area=target_area_for_report,
                                 target_mys=target_mys)

        # [v1.14 §6] 결과 단면을 template 규격 CSV로 내보낸다 — 이 파일이 있어야 v1.14의 1차
        # 성공 기준(파트 간 겹침 해소)을 review_v1_13.md와 동일한 방법으로 검증할 수 있다.
        _csv_path = export_final_section_csv(model, data.to(device), target_mps, pruning_state,
                                              final_seg_ids, device, temperature=final_temp,
                                              save_path="reports/final_section_v1_15_4.csv")
        # [v1.15 §9] collision loss에서 접합 쌍을 제외했으므로, 최종 형상이 실제로 간섭이 없는지는
        # 반드시 별도로 확인해야 한다. 오프셋 레이어가 빠지거나 잘못 설정돼도 조용히 통과하지
        # 못하게 막는 마지막 관문 — 단, 여기서 예외를 던져 파이프라인을 중단시키지는 않는다
        # (그러면 이미 저장된 리포트/CSV/가중치를 확인하기 번거로워진다). 결과를 리포트 파일에
        # 섹션으로 남기고 콘솔에 경고만 출력한다.
        _overlap = check_part_overlap(_csv_path, tol_mm=FLANGE_ASSERT_TOL_MM)
        # report_design_comparison() 호출부(바로 위)가 save_path 인자를 생략했으므로 그 함수의
        # 기본값("reports/AI_design_v1_15_4_report.md")과 동일한 경로를 그대로 쓴다.
        append_overlap_section_to_report("reports/AI_design_v1_15_4_report.md", _overlap)

        print(f">>> Stage 4 진입 여부: {enter_stage4}")
        print("[done] weights/bpillar_17sec_v1_15_4.pt, reports/figures/AI_design_v1_15_4_result.png, "
              "reports/figures/AI_design_v1_15_4_3d.html, reports/AI_design_v1_15_4_report.md, "
              "reports/final_section_v1_15_4.csv 저장 완료")
