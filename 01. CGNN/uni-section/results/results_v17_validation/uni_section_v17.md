    uni_section_v17: command_v17.md 반영 (idea_v17.md 5개 판정 구현)
      - 후보군 축소: CANDIDATE_PARTS=[3,4](Patch1/2), S_PROTECT=[0,1,2](Inner Hat도 보호)
      - Mp err 게이팅된 희소화(w_sparse_effective) + 엔트로피 정규화(gate_active 구간)
      - 더블쇼크 분리: 두께언락@10 vs 게이트활성@11
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
      T_MAX=2.5mm | DELTA_SCALE=1.35 | w_sparse=2.0 | grad_clip=10.0
      S_protect=[0, 1, 2] | Candidate=[3, 4] | init_log_alpha=2.0
      Stage2(두께언락)@10 | GATE_ACTIVE_EPOCH(게이트/프루닝 활성)@11
      Pruning: ewma_alpha=0.15 thresh=0.08/0.15 confirm=20epoch cooldown=20epoch
      Sparse gating: tau_gate=0.05 k=50.0 | entropy alpha=0.01
      Feasibility 기준: Mp err < 2.0%  AND  l_collision < 0.05
    ==============================================================================
    Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  |  Coll  | Sparse || tGate | gAct | mu_Mp | z_gate(3,4)
    00000 || 32.8611 || 237.59% | 0.3333 | 1163.0 | 0.0000 | 1.0000 || 0.00 |    N | 1.0e+00 | 1.00/1.00
    [Stage 2] epoch 10: thickness_decoder 그룹 AdamW state 리셋(lr→3.0e-04) (좌표 헤드 모멘텀 보존)
    [Gate Active] epoch 11: gate_params lr → 1e-2 (게이트/프루닝 트리거 활성화, 두께 언락과 1epoch 분리)
    00019 || 92.6774 ||  19.03% | 0.2183 |  543.7 | 0.0082 | 0.3596 || 1.00 |    Y | 1.4e+02 | 0.49/0.49
    00039 || 115.5770 ||   0.88% | 0.2286 |  597.6 | 0.0223 | 0.0930 || 1.00 |    Y | 2.0e+02 | 0.52/0.52 [FEASIBLE]
    00059 || 110.5709 ||   1.77% | 0.2497 |  591.2 | 0.0067 | 1.0000 || 1.00 |    Y | 2.0e+02 | 0.53/0.51 [FEASIBLE]
    00079 || 101.2217 ||   0.27% | 0.2506 |  582.4 | 0.0000 | 0.5000 || 1.00 |    Y | 1.9e+02 | 0.53/0.49 [FEASIBLE]
    00099 || 104.3343 ||   0.98% | 0.2522 |  578.5 | 0.0000 | 0.5000 || 1.00 |    Y | 2.0e+02 | 0.52/0.48 [FEASIBLE]
    00119 || 101.5811 ||   1.58% | 0.2524 |  579.8 | 0.0000 | 1.0000 || 1.00 |    Y | 1.9e+02 | 0.52/0.47 [FEASIBLE]
    00139 || 100.0487 ||   0.32% | 0.2411 |  577.2 | 0.0000 | 0.5000 || 1.00 |    Y | 2.0e+02 | 0.52/0.46 [FEASIBLE]
    00159 || 105.0307 ||   2.21% | 0.2287 |  573.4 | 0.0000 | 0.6090 || 1.00 |    Y | 2.0e+02 | 0.52/0.45
    00179 || 113.4774 ||   2.67% | 0.2292 |  579.9 | 0.0000 | 0.0000 || 1.00 |    Y | 2.1e+02 | 0.51/0.44
    00199 || 120.4196 ||   4.46% | 0.2274 |  582.4 | 0.0000 | 0.5663 || 1.00 |    Y | 2.1e+02 | 0.51/0.43
    00219 || 126.5625 ||   2.76% | 0.2267 |  579.8 | 0.0000 | 0.3849 || 1.00 |    Y | 2.2e+02 | 0.51/0.43
    00239 || 123.5296 ||   1.24% | 0.2242 |  573.8 | 0.0000 | 0.4592 || 1.00 |    Y | 2.2e+02 | 0.51/0.41 [FEASIBLE]
    00259 || 120.4930 ||   1.05% | 0.2207 |  576.9 | 0.0000 | 0.3511 || 1.00 |    Y | 2.2e+02 | 0.51/0.40 [FEASIBLE]
    00279 || 125.9686 ||   3.42% | 0.2096 |  570.4 | 0.0000 | 0.3400 || 1.00 |    Y | 2.2e+02 | 0.51/0.39
    00299 || 128.0474 ||   0.91% | 0.2122 |  573.7 | 0.0000 | 0.0993 || 1.00 |    Y | 2.2e+02 | 0.51/0.38 [FEASIBLE]
    
    ──────────────────────────────────────────────────────────────────────────────
    [Feasibility] 첫 만족 epoch: 25  |  최고(Mp err 최소) epoch: 145  |  Mp err: 0.01%  |  l_collision: 0.0000
    [Best-Mp] epoch 145: Mp err = 0.01%, l_collision = 0.0000  (collision 무관 추적)
    [ALDA final] mu_Mp=2.23e+02 mu_col=0.00e+00 rho_Mp=200.0 rho_col=200.0
    [Pruning final] state=['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE'] ewma_z=[1.0, 1.0, 1.0, 0.507, 0.383]
    ──────────────────────────────────────────────────────────────────────────────
    


    
![png](uni_section_v17_files/uni_section_v17_0_1.png)
    


    
    결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v17_result.png
    


    
![png](uni_section_v17_files/uni_section_v17_0_3.png)
    


    
    Epoch 스냅샷 결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\01. CGNN\uni-section\code\uni_section_v17_epoch_snapshots.png
    
    ==============================================================================
    최종 결과 요약 (마지막 epoch 기준)
      Target Mp     :      7,421,470 N·mm
      Final pred_mp :      7,354,112 N·mm
      Final Error   :   0.91%
      Final l_collision : 0.0000
      Final l_order : 0.0000
      Initial area  :   1157.3 mm²
      Final area    :    573.7 mm²  (-50.4% vs 초기)
      ALDA mu_Mp/mu_col (final) : 2.23e+02 / 0.00e+00
      Pruning 최종 상태: ['ALIVE', 'ALIVE', 'ALIVE', 'ALIVE', 'ALIVE']
      프루닝 미발동 — candidate_parts [3, 4]의 최종 ewma_z: [0.507, 0.383]
    
      ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch 145, Mp err=0.01%, l_collision=0.0000
        (best_feasible['state_dict']를 model.load_state_dict()로, best_feasible['pruning_state']를 함께 복원해 사용 권장)
    ==============================================================================
    
