    uni_section_v16: command_v16.md 반영 (review_v10.md A-3/A-4를 v15 위에 추가)
      - A-3 게이트 이원화: z_gate(결정론, 물리용)/z_open(확률, 희소화용), S_protect={0,1}
      - A-4 EWMA(0.15)+히스테리시스(0.08/0.15) 프루닝 트리거, 20epoch 확정
      - Soft-Zeroing: DELETED 확정 시 log_alpha=-10.0 고정, 그래프 수술(A-5) 없음
      - t_final = t_raw(sigmoid) * z_gate — post-sigmoid 단일 게이팅 지점(진짜 0 수렴)
      - collision v5에 z_gate[seg]*z_gate[pt] 곱셈 추가(사라지는 부재 무력화)
      - optimizer 3-그룹(main/thickness/gate), gate lr: Stage1=0 → Stage2(epoch128)=1e-2
    
    데이터: nodes=torch.Size([93, 8]) | edges=torch.Size([2, 176])
    [gradcheck] dMp/dt OK -- autograd=1.3173e+07, FD=1.3160e+07, rel_err=9.88e-04, dMp/dt>0 확인
    [mass] target_area = 1157.3 mm² (초기 단면적 스냅샷)
    [collision v5] 전쌍(9쌍) 부호 앵커 & clearance (초기 형상 기준):
        sec0 Part0-Part1: 0seg/1pt σ=+1 clr=-0.05 | 1seg/0pt σ=-1 clr=-0.05
        sec0 Part0-Part2: 0seg/2pt σ=+1 clr=0.50
        sec0 Part0-Part3: 0seg/3pt σ=+1 clr=0.50 | 3seg/0pt σ=-1 clr=0.50
        sec0 Part0-Part4: 0seg/4pt σ=+1 clr=0.50 | 4seg/0pt σ=-1 clr=0.50
        sec0 Part1-Part2: 1seg/2pt σ=-1 clr=-0.05
        sec0 Part1-Part3: 1seg/3pt σ=-1 clr=-0.62 | 3seg/1pt σ=+1 clr=-0.62
        sec0 Part1-Part4: 1seg/4pt σ=+1 clr=-1.05 | 4seg/1pt σ=-1 clr=-1.05
        sec0 Part2-Part3: 2seg/3pt σ=+1 clr=0.50 | 3seg/2pt σ=-1 clr=0.50
        sec0 Part2-Part4: 2seg/4pt σ=+1 clr=0.50 | 4seg/2pt σ=-1 clr=0.50
    
    ==============================================================================
    [ uni_section_v16 ] Training  |  Target Mp = 7,421,470 N·mm  |  Epochs: 300
      CGDN: hidden=128, layers=4, heads=4  |  Curriculum: True (0.2, 0.7)
      (command_v16.md: 게이트 이원화 z_gate/z_open + EWMA/히스테리시스 프루닝, ALDA/collision v5 유지)
      T_MAX=2.5mm | DELTA_SCALE=1.35 | w_sparse=0.05 | grad_clip=10.0
      S_protect=[0, 1] | Candidate=[2, 3, 4] | init_log_alpha=2.0 | Stage2(게이트 활성)@100
      Pruning: ewma_alpha=0.15 thresh=0.08/0.15 confirm=20epoch
      Feasibility 기준: Mp err < 2.0%  AND  l_collision < 0.05
    ==============================================================================
    Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  |  Coll  | Sparse || tGate | mu_Mp | mu_col | z_gate(2,3,4)
    00000 || 32.9058 || 237.57% | 0.3333 | 1162.9 | 0.0000 | 1.0000 || 0.00 | 1.0e+00 | 1.0e+00 | 1.00/1.00/1.00
    00019 || 17.8984 ||  34.14% | 0.1333 |  938.0 | 0.0784 | 1.0000 || 0.00 | 2.2e+02 | 0.0e+00 | 1.00/1.00/1.00
    00039 || 18.4461 ||  17.24% | 0.3427 | 1055.4 | 0.1376 | 1.0000 || 0.00 | 2.8e+02 | 4.5e+01 | 1.00/1.00/1.00
    00059 || 21.2901 ||  10.30% | 0.3256 | 1039.0 | 0.1153 | 1.0000 || 0.00 | 3.1e+02 | 1.2e+02 | 1.00/1.00/1.00
    00079 || 27.5909 ||   8.30% | 0.3090 | 1042.1 | 0.2091 | 1.0000 || 0.00 | 3.3e+02 | 1.8e+02 | 1.00/1.00/1.00
    00099 || 148.1845 ||   1.30% | 0.3018 |  895.4 | 0.0134 | 1.0000 || 0.35 | 3.4e+02 | 2.1e+02 | 1.00/1.00/1.00 [FEASIBLE]
    [Stage 2] epoch 100: thickness_decoder 그룹 AdamW state 리셋(lr→3.0e-04), gate_params lr→1e-2 (게이트/프루닝 트리거 활성화)
    00119 || 545.6519 ||  16.88% | 0.3380 |  652.5 | 0.0092 | 0.9807 || 1.00 | 3.9e+02 | 2.1e+02 | 0.96/0.96/0.96
    00139 || 605.1497 ||   5.23% | 0.3144 |  632.5 | 0.0000 | 1.0000 || 1.00 | 4.4e+02 | 2.0e+02 | 0.97/0.96/0.96
    00159 || 615.0239 ||   3.18% | 0.3081 |  626.8 | 0.0000 | 0.9713 || 1.00 | 4.5e+02 | 2.0e+02 | 0.97/0.96/0.96
    00179 || 618.6381 ||   5.54% | 0.2921 |  613.1 | 0.0000 | 0.8987 || 1.00 | 4.5e+02 | 1.9e+02 | 0.97/0.96/0.96
    00199 || 626.4094 ||   3.62% | 0.3071 |  624.7 | 0.0000 | 0.9393 || 1.00 | 4.6e+02 | 1.8e+02 | 0.97/0.96/0.96
    00219 || 611.1551 ||   0.35% | 0.2998 |  617.7 | 0.0000 | 0.4749 || 1.00 | 4.7e+02 | 1.7e+02 | 0.97/0.96/0.96 [FEASIBLE]
    00239 || 616.1391 ||   0.23% | 0.2992 |  616.7 | 0.0000 | 0.7273 || 1.00 | 4.7e+02 | 1.6e+02 | 0.97/0.96/0.96 [FEASIBLE]
    00259 || 655.7517 ||   1.80% | 0.3012 |  618.3 | 0.0000 | 0.7386 || 1.00 | 4.9e+02 | 1.6e+02 | 0.97/0.96/0.96 [FEASIBLE]
    00279 || 651.1628 ||   0.55% | 0.2981 |  615.3 | 0.0000 | 0.6667 || 1.00 | 4.9e+02 | 1.5e+02 | 0.97/0.95/0.96 [FEASIBLE]
    00299 || 643.5753 ||   0.96% | 0.2951 |  612.2 | 0.0000 | 0.9838 || 1.00 | 4.9e+02 | 1.4e+02 | 0.97/0.95/0.96 [FEASIBLE]
    
    ──────────────────────────────────────────────────────────────────────────────
    [Feasibility] 첫 만족 epoch: 99  |  최고(Mp err 최소) epoch: 275  |  Mp err: 0.01%  |  l_collision: 0.0000
    [Best-Mp] epoch 275: Mp err = 0.01%, l_collision = 0.0000  (collision 무관 추적)
    [ALDA final] mu_Mp=4.88e+02 mu_col=1.39e+02 rho_Mp=200.0 rho_col=200.0
    [Pruning final] state=['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE'] ewma_z=[1.0, 1.0, 0.969, 0.954, 0.958]
    ──────────────────────────────────────────────────────────────────────────────
    


    
![png](uni_section_v16_files/uni_section_v16_0_1.png)
    


    
    결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v16_result.png
    


    
![png](uni_section_v16_files/uni_section_v16_0_3.png)
    


    
    Epoch 스냅샷 결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v16_epoch_snapshots.png
    
    ==============================================================================
    최종 결과 요약 (마지막 epoch 기준)
      Target Mp     :      7,421,470 N·mm
      Final pred_mp :      7,350,432 N·mm
      Final Error   :   0.96%
      Final l_collision : 0.0000
      Final l_order : 0.0000
      Initial area  :   1157.3 mm²
      Final area    :    612.2 mm²  (-47.1% vs 초기)
      ALDA mu_Mp/mu_col (final) : 4.88e+02 / 1.39e+02
      Pruning 최종 상태: ['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE']
      프루닝 미발동 — candidate_parts [2, 3, 4]의 최종 ewma_z: [0.969, 0.954, 0.958]
    
      ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch 275, Mp err=0.01%, l_collision=0.0000
        (best_feasible['state_dict']를 model.load_state_dict()로, best_feasible['pruning_state']를 함께 복원해 사용 권장)
    ==============================================================================
    
