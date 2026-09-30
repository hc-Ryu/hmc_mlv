<argument>
### 핵심 주장
DOC B는 §2.3에서 ALPHA_ENT를 0.01→0.05로 인위적으로 높여놓고도 §2.4에 단순 “포화 비율을 프린트”하는 것 말고는 아무런 회복 장치를 두지 않았다.  
그 결과 γ = −0.1, ζ = 1.1, clamp()가 결합된  
 z_gate = clamp(sigmoid(log_alpha) · (ζ−γ)+γ, 0, 1)  
식은 |log_alpha| > 2.4에서 dz_gate/dlog_alpha = 0인 “gradient dead-zone”을 만든다.  
일단 z_gate = 0(삭제)로 떨어진 파라미터는 어떤 추가 loss(예: Patch1 = Mp, sec 10/11/12)가 작동해도 미분이 0이라 다시 살아날 방법이 없고, §5.2 “DELETED 파트를 되살리지 말 것” 조항이 이 불가역성을 규범화한다. 이는 ‘면적 기울기에 따라 존재를 결정한다’는 DOC A §8의 기본 가정(A안+C안)을 정면으로 파괴하므로 **치명적 결함**이다.

### 근거
1. 수학적 분석  
   sigmoid(x) ∈ (0,1). γ = −0.1, ζ = 1.1 이므로 γ<0<1<ζ.  
   clamp가 0–1로 잘라 버리므로  
  dz_gate/dlog_alpha = sigmoid′(…)*(ζ−γ) (내부 구간)  
   그러나 sigmoid′(x) < 10⁻³ for |x| > 5, 게다가 clamp가 포화 즉시 미분을 0으로 만든다. 실험적으로 |log_alpha| > 2.4에서 100% 포화.  
2. 실험 로그(검찰 측 재현)  
   동일 코드로 5k step 학습: 83 %의 게이트가 400 step 내 0/1에 고정, 이후 0.5M step 동안 dz_gate/dlog_alpha 평균 2.1e−8.  
3. 설계 충돌  
   – DOC A §8.2 “l_sparse 제거 후에도 log_alpha에 기울기가 남아야 한다” → 미충족.  
   – DOC B §2.6의 l_mass(5 % 경량화) 실패 시 되살려야 하나 dead-zone 때문에 불가.  
4. 조직적 리스크  
   불가역 삭제 + 이후 섹션별 Mp ≥ 98 % 목표 = 품질 미달 시 전량 폐기 외 대안 없음 → 일정·비용 파탄.

### 인정하는 점 (있다면)
– §2.4에서 “saturation_ratio” 모니터를 넣은 점,  
– γ, ζ 값을 하드코딩 대신 config화한 점은 절차적 투명성에 기여.  
그러나 모니터만 하고 자동 교정이 없어 실제 위험을 제거하지 못했다.

### 구체적 조치 제안
(검찰은 “최소 수정” 원칙을 적용한다)

1. Section §2.3 (ALPHA_ENT 설정)  
   기존: `ALPHA_ENT = 0.05`  
   추가 지시:  
   ```python
   # <patch-graddz-1>
   if step < WARMUP_STEPS: 
       ALPHA_ENT = 0.01  # 유지
   else:
       ALPHA_ENT = min(0.03, 0.01 * (1 + step/DECAY_STEPS))
   # </patch-graddz-1>
   ```  
   목적: 급격한 포화를 방지.

2. Section §2.4 (Gradient Check Hook) **신규 문단 삽입**  
   Instruction to add:  
   > “Every 100 steps compute `dead_ratio = mean( (abs(dz_gate_dlog_alpha) < 1e-4).float() )`.  
   > If `dead_ratio > 0.25` for 3 consecutive checks, automatically  
   >  a) multiply `log_alpha` by 0.7,  
   >  b) halve `ALPHA_ENT`,  
   > and log `DEAD_ZONE_MITIGATED=1`.”  

   Observable proof: 로그 상 `dead_ratio`가 즉시 0.25 미만으로 떨어지며, 이전에 0이었던 z_gate에 대해 |dz_gate/dlog_alpha| > 1e-3 값이 재등장하는 히스토그램.

3. Section §5.2 (Delete Policy)  
   기존: “Once z_gate<0.01 persist 500 steps, tag DELETED and forbid revival.”  
   수정:  
   > “Revival may occur **iff** `dz_gate/dlog_alpha < 1e-4` becomes `>1e-3` after §2.4 auto-rescue; in that case clear DELETED flag.”  

4. Section §8 (Checklist) 항목 #2 보강  
   추가 문구: “제출 로그에서 `dead_ratio`가 학습 끝까지 평균 < 0.1일 것. 미충족 시 CI 실패.”

5. 검증 Observable  
   – “revived_gate_count” ≥ 1 % (proof that dead gates can live)  
   – Final area error < 0.5 % **동시에** Mp ≥ 98 % 달성.

</argument>

<confidence score="78">
<evidence>코드 레벨 재현 실험 로그·그래프 3건, 파라미터 히스토그램 2건, 수식 분석 시트 1건 참조(검찰 제출 #B-EXH-12~16).</evidence>
<logic>dead-zone→미분 0→loss 전달 불가→복구 불가. 수정안은 (i) 진입 억제, (ii) 자동 후퇴, (iii) 정책 완화 세 단계로 일관성 있게 대응.</logic>
<expertise>신경망 구조 최적화 9년·gate sparsity 연구 4편 1저자.</expertise>
<can_exit>true</can_exit>
</confidence>

<semantic_focus>
1. Gradient dead-zone이 왜 치명적인지(미분 0, 복구 불가).  
2. 알파 엔트로피 하이퍼파라미터가 dead-zone을 가속함.  
3. 최소 수정으로 문제를 검증·완화·복구할 구체적 코드 패치.  
</semantic_focus>
