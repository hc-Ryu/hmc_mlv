    uni_section_v17: command_v17.md 반영 (idea_v17.md 5개 판정 구현)
      - 후보군 축소: CANDIDATE_PARTS=[3,4](Patch1/2), S_PROTECT=[0,1,2](Inner Hat도 보호)
      - Mp err 게이팅된 희소화(w_sparse_effective) + 엔트로피 정규화(gate_active 구간)
      - 더블쇼크 분리: 두께언락@10 vs 게이트활성@40
      - State-Only Reset: DELETED 확정 시 optimizer 재인스턴스화 + ALDA mu 리셋 + 20epoch 쿨다운(완전 리셋은 기각됨)
    
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
    [ uni_section_v17 ] Training  |  Target Mp = 7,421,470 N·mm  |  Epochs: 300
      CGDN: hidden=128, layers=4, heads=4  |  Curriculum: True (0.2, 0.7)
      (command_v17.md: 후보축소[3,4] + Mp err 게이팅 희소화/엔트로피 + 더블쇼크분리 + State-Only Reset)
      T_MAX=2.5mm | DELTA_SCALE=1.35 | w_sparse=0.1 | grad_clip=10.0
      S_protect=[0, 1, 2] | Candidate=[3, 4] | init_log_alpha=2.0
      Stage2(두께언락)@10 | GATE_ACTIVE_EPOCH(게이트/프루닝 활성)@40
      Pruning: ewma_alpha=0.15 thresh=0.08/0.15 confirm=20epoch cooldown=20epoch
      Sparse gating: tau_gate=0.05 k=50.0 | entropy alpha=0.01
      Feasibility 기준: Mp err < 2.0%  AND  l_collision < 0.05
    ==============================================================================
    Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  |  Coll  | Sparse || tGate | gAct | mu_Mp | z_gate(3,4)
    00000 || 32.8611 || 237.59% | 0.3333 | 1163.0 | 0.0000 | 1.0000 || 0.00 |    N | 1.0e+00 | 1.00/1.00
    [Stage 2] epoch 10: thickness_decoder 그룹 AdamW state 리셋(lr→3.0e-04) (좌표 헤드 모멘텀 보존)
    00019 || 75.8220 ||  10.18% | 0.2108 |  567.2 | 0.0052 | 1.0000 || 1.00 |    N | 1.4e+02 | 1.00/1.00
    00039 || 74.9908 ||   3.96% | 0.2436 |  594.3 | 0.0134 | 1.0000 || 1.00 |    N | 1.5e+02 | 1.00/1.00
    [Gate Active] epoch 40: gate_params lr → 1e-2 (게이트/프루닝 트리거 활성화, 두께 언락과 30epoch 분리)
    00059 || 68.9631 ||   0.34% | 0.2357 |  582.7 | 0.0000 | 0.9198 || 1.00 |    Y | 1.6e+02 | 0.95/0.95 [FEASIBLE]
    00079 || 66.2652 ||   0.93% | 0.2163 |  580.2 | 0.0000 | 1.0000 || 1.00 |    Y | 1.6e+02 | 0.95/0.94 [FEASIBLE]
    00099 || 68.1690 ||   2.17% | 0.2096 |  583.8 | 0.0000 | 0.9583 || 1.00 |    Y | 1.6e+02 | 0.94/0.94
    00119 || 74.8986 ||   5.56% | 0.2127 |  587.2 | 0.0000 | 1.0000 || 1.00 |    Y | 1.6e+02 | 0.94/0.94
    00139 || 80.0298 ||   2.70% | 0.2137 |  579.6 | 0.0000 | 0.5000 || 1.00 |    Y | 1.7e+02 | 0.94/0.94
    00159 || 80.7192 ||   2.09% | 0.2075 |  570.5 | 0.0000 | 0.9792 || 1.00 |    Y | 1.7e+02 | 0.94/0.93
    00179 || 81.2478 ||   1.47% | 0.2100 |  574.6 | 0.0000 | 0.7893 || 1.00 |    Y | 1.7e+02 | 0.94/0.93 [FEASIBLE]
    00199 || 82.6709 ||   0.45% | 0.2087 |  570.0 | 0.0000 | 0.8381 || 1.00 |    Y | 1.8e+02 | 0.94/0.93 [FEASIBLE]
    00219 || 91.0636 ||   1.28% | 0.2124 |  570.9 | 0.0000 | 1.0000 || 1.00 |    Y | 1.9e+02 | 0.94/0.93 [FEASIBLE]
    00239 || 96.8236 ||   1.50% | 0.2100 |  564.8 | 0.0000 | 0.5443 || 1.00 |    Y | 1.9e+02 | 0.94/0.93 [FEASIBLE]
    00259 || 96.7417 ||   0.08% | 0.2127 |  565.6 | 0.0000 | 1.0000 || 1.00 |    Y | 1.9e+02 | 0.94/0.92 [FEASIBLE]
    00279 || 106.3259 ||   3.50% | 0.2159 |  569.7 | 0.0000 | 0.9729 || 1.00 |    Y | 2.0e+02 | 0.94/0.92
    00299 || 103.8697 ||   0.89% | 0.2123 |  564.3 | 0.0000 | 0.9685 || 1.00 |    Y | 2.0e+02 | 0.93/0.92 [FEASIBLE]
    
    ──────────────────────────────────────────────────────────────────────────────
    [Feasibility] 첫 만족 epoch: 14  |  최고(Mp err 최소) epoch: 78  |  Mp err: 0.00%  |  l_collision: 0.0000
    [Best-Mp] epoch 78: Mp err = 0.00%, l_collision = 0.0000  (collision 무관 추적)
    [ALDA final] mu_Mp=2.00e+02 mu_col=0.00e+00 rho_Mp=200.0 rho_col=200.0
    [Pruning final] state=['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE'] ewma_z=[1.0, 1.0, 1.0, 0.935, 0.921]
    ──────────────────────────────────────────────────────────────────────────────
    


    
![png](uni_section_v17_files/uni_section_v17_0_1.png)
    


    
    결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v17_result.png
    


    
![png](uni_section_v17_files/uni_section_v17_0_3.png)
    


    
    Epoch 스냅샷 결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v17_epoch_snapshots.png
    
    ==============================================================================
    최종 결과 요약 (마지막 epoch 기준)
      Target Mp     :      7,421,470 N·mm
      Final pred_mp :      7,487,494 N·mm
      Final Error   :   0.89%
      Final l_collision : 0.0000
      Final l_order : 0.0000
      Initial area  :   1157.3 mm²
      Final area    :    564.3 mm²  (-51.2% vs 초기)
      ALDA mu_Mp/mu_col (final) : 2.00e+02 / 0.00e+00
      Pruning 최종 상태: ['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE']
      프루닝 미발동 — candidate_parts [3, 4]의 최종 ewma_z: [0.935, 0.921]
    
      ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch 78, Mp err=0.00%, l_collision=0.0000
        (best_feasible['state_dict']를 model.load_state_dict()로, best_feasible['pruning_state']를 함께 복원해 사용 권장)
    ==============================================================================
    
