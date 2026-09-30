<critique>
### Validated Claims
1. Claude-(a)  
   - 근거: idea_v4.md §8 첫 번째 체크리스트에서 “`l_sparse`를 지우면  z_gate 의 clamp 로 gradient 가 log_alpha 로 흐르지 않을 수도 있다”는 확인 항목이 있으나, command_v1_3.md 에는 이를 해결-점검하는 절차가 전혀 명시되지 않았다. (§2 는 상수만 바꾸고, §2.4 는 포화 비율을 print 만 할 뿐 back-prop 검사 없음).  
   → 실제 누락 맞음.

2. Claude-(c)  
   - idea_v4.md 정정 ③ “Patch1 생존 못하면 T_MIN 을 재검토” 요구.  
   - command_v1_3.md §3.5-4에서 “Patch1 미생존 → `T_MIN` 재검토”라고 언급되지만 **구체 fallback 값이나 액션이 없다**.  
   → “dead-end” 지적 타당.

3. Claude-(d)  
   - command_v1_3.md §8 번호: 1,2,3,4,5,9,6,7,8 로 뒤섞임.  
   → 사실관계 일치.

4. Gemini-#1 (USE_AREA_WEIGHTED_SPARSE True + W_SPARSE_V13 0.0 시 no-op)  
   - §2.2 에 “B안 적용 시 W_SPARSE_V13 를 0.02~0.05 로 바꿀 것”이란 주석만 있고 **코드 수준 guard 없음**.  
   → 잠재 버그 존재.

5. Gemini-#3 (w_mass=10 이 Mp penalty 와 충돌 위험)  
   - §2.6.3 파생 기울기 계산에서 동일 order(4.4e-5)임을 인정하면서도 **모니터링 규칙은 §7 ‘완화 순서’라는 수동 절차뿐**. 자동 abort 나 실시간 로깅 없음.  
   → 위험 표시는 정당.

6. Gemini-#4 (`_per_candidate_area()` 사양 부재)  
   - §2.2 단락이 “신규 헬퍼”라고만 쓰고 내부 로직, 반환 텐서 shape 검증 부재.  
   → 설계 미비 확인.

7. OpenAI-#1 (`final_epoch` 미정의)  
   - §5.1 에 `stage4_start = final_epoch` 로 기술되지만 파일 내 어디에서도 `final_epoch` 선언/갱신 없다.  
   → 컴파일 에러 유발, 옳은 지적.

### Disputed Claims (with reasons)
1. Claude-(b) “weights dict 에 w_mass 가 없다”  
   - §2.6.2 에 “weights 기본값 블록에 `'w_mass': W_MASS_V13` 추가(§2.1의 TO-BE와 함께 적용)” 명시.  
   - 즉 같은 블록을 두 번 갱신하도록 지시했을 뿐 완전히 누락된 것은 아님.  
   → 부분적으로 과도 해석.

2. Gemini-#2 “thickness_decoder / thick_params 혼용 오류”  
   - 실제 스니펫(§5.5)  
     ```
     {'params': thick_params, 'name': 'thickness_decoder', ...}
     ```  
     기존 v1_2 코드에서도 변수 `thick_params` 는 param group ‘thickness_decoder’ 로 패킹된다. 이름·변수 일관성 확보되어 있음.  
   → 오류 주장 불인정.

3. OpenAI-#2 “main_params / thick_params 수집 방법 미기재”  
   - v1_2 원본에서 이미 `main_params, thick_params, gate_params = split_param_groups(model)` 함수가 존재. 문서가 재설명하지 않았을 뿐 결손은 아님.  
   → blocker 로 보기 어렵다.

4. OpenAI-#3 “train_step_multi 시그니처 변경이 기존 호출 깨뜨린다”  
   - §2.3 에 기본값 `alpha_ent=ALPHA_ENT_V13, stage4_active=False` 로 추가. 기존 모든 호출은 인자 없이 그대로 통과 → **TypeError 발생하지 않음**.  
   → 오독.

5. OpenAI-#5 “ALPHA_ENT 0.05 고정이 idea 범위(0.05~0.1)와 불일치”  
   - idea_v4.md §4 C.3 이 제시한 범위 안쪽 값(0.05) 선택에 불과, 불일치는 아님.  

6. OpenAI-#7 “정정 ④ 코드 위치 없음”  
   - §2.1 상단 (⚠️ 근거 정정) 과 §2.1 표 형식에서 `A_k / target_area` 로 수정 예시 제공.  
   → 반영되어 있다.

### Missing Considerations
1. Scope-creep 리스크  
   - command_v1_3.md 는 idea_v4.md 에 없던 **“단면적 5% 추가 감량(AREA_TARGET_RATIO=0.95) + l_mass 항을 새로 loss 에 편입”**(§2.6)을 도입.  
   - 이는 원 설계 근거를 넘어선 요구 확장으로, 현실적으로 Mp 제약과 동시 달성 가능성이 불투명. 세 솔버 모두 “A ← B 일치 여부”만 검증하고 **요구 추가로 인한 파라미터 불안정·프로젝트 범위 증대** 위험을 지적하지 않았다.

2. `stage4_active` 플래그 전달 누락 가능성  
   - train 루프 예시는 Stage-4 전용 호출부만 `stage4_active=True` 를 넘기고, 기존 Epoch 루프 쪽 수정 예시가 없다.  
   - 기본값 False 로 안전하지만 **누락 시 detach 해제·커리큘럼 freeze 가 작동 안 하는 논리 버그**가 생길 수 있음. 어느 솔버도 노출하지 않았다.

3. Gradient dead-zone 문제 심화 가능성  
   - `z_gate` 가 0/1 로 포화되면 면적 항의 ∂f/∂z_k = 0 이 되어 **Stage 1-3 중반 이후 log_alpha 쪽 gradient 를 사실상 잃는다.**  
   - l_sparse 제거 + ALPHA_ENT 0.05 조합이 “포화→학습 중단”을 일으킬 경우 gate 재배치 능력이 사라지는데, command_v1_3.md 는 이를 모니터링(print)만 하고 대책(soft clipping, temperature back-off 등)을 두지 않는다. 세 솔버 모두 “gradient path 상실” 후반 위험을 언급하지 않았다.

4. 새로 정의한 `_t_init_logit()` 헬퍼의 **import 순서 의존**  
   - §3.3 에 forward 와 report 양쪽에서 호출한다고 하나, 실제 구현 시 report 함수가 다른 파일(리포트 스크립트)일 경우 import 싸이클로 인해 접근 불가 위험이 있음. 아무 솔버도 circular-import 가능성을 거론하지 않았다.

5. **final_epoch NULL 에러**는 OpenAI 만 짚었으나, 그 변수는 Stage-4 뿐 아니라 §7 Go/No-Go 로직에서도 사용될 것으로 보인다. 초기 학습 종료 시점 계산법(예: last_epoch) 제시가 문서에 없다는 점까지는 세 솔버 모두 상세 분석 못 했다.
</critique>

<confidence score="67">
  <evidence>문서 비교에서 실제 코드 스니펫 및 지시 문구 대조. 결정적 텍스트(§2.6.2, §5.1 등) 인용. 솔버 주장별 일치/불일치 판정.</evidence>
  <logic>자료 대조, 변수 정의·참조 경로 추적, 설계/구현 일관성 평가.</logic>
  <expertise>SW 설계-사양 리뷰, ML 코드베이스 리뷰 경험.</expertise>
  <can_exit>true</can_exit>
</confidence>

<semantic_focus>
1. l_sparse 제거 후 gradient 흐름·dead-zone 문제
2. AREA_TARGET_RATIO 0.95 도입으로 인한 scope-creep 및 위험
3. Stage-4 플래그/변수 정의 누락(​final_epoch, stage4_active 전달)과 그 영향
</semantic_focus>
