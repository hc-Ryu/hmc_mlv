    uni_section_v10: command_v10.md 반영 (synod-20260710-103702-cee330 review / synod-20260710-105249-9ab67a design)
      - §1 T_MAX 2.5mm, DELTA_SCALE 1.35, bias 0.03 (두께 인플레이션 차단 + 재보정)
      - §2 collision v5: surface-gap 제곱 소프트컨택, t detach 금지, alpha 삭제, 전쌍 검사, 부호 앵커
      - §3 2D mesh order loss (w_order=1.0 상시) — v8 붕괴(criss-crossing) 근본 대책
      - §4 2단계 두께 학습: sigmoid gate@128 + thickness 그룹 국소 optimizer 리셋 + lr 0.3x
      - §5 비대칭 Huber phys loss (delta=0.05, undershoot 2.0x)
    
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
    [ uni_section_v10 ] Training  |  Target Mp = 7,421,470 N·mm  |  Epochs: 300
      CGDN: hidden=128, layers=4, heads=4  |  Curriculum: True (0.2, 0.7)
      (§2 collision v5 surface-gap² / §3 mesh order / §4 2-stage gate@128 / §5 asym Huber)
      T_MAX=2.5mm | DELTA_SCALE=1.35 | w_order=1.0 | grad_clip=5.0
      Feasibility 기준: Mp err < 2.0%  AND  l_collision < 0.05
    ==============================================================================
    Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  | Mass(gate) |  Coll  | Order  || tGate | dT | satParts
    00000 || 4.7040 || 237.57% | 0.3333 | 1162.9 | 0.0000(0.00) | 0.0000 | 0.0000 || 0.00 | +0.00 | 0/5
    00019 || 3.3624 || 109.68% | 0.3043 | 1058.0 | 0.0074(0.00) | 0.0000 | 0.0000 || 0.00 | -0.00 | 0/5
    00039 || 3.3033 || 110.83% | 0.3214 | 1073.8 | 0.0052(0.00) | 0.0005 | 0.0000 || 0.00 | -0.00 | 0/5
    00059 || 3.1887 || 106.19% | 0.2738 | 1050.7 | 0.0085(0.00) | 0.0002 | 0.0000 || 0.00 | -0.00 | 0/5
    00079 || 3.5054 || 100.36% | 0.2511 | 1040.9 | 0.0101(0.00) | 0.0002 | 0.0000 || 0.00 | -0.00 | 0/5
    00099 || 4.2473 ||  80.25% | 0.2455 | 1035.3 | 0.0111(0.00) | 0.0003 | 0.0000 || 0.00 | -0.00 | 0/5
    00119 || 5.0114 ||  54.03% | 0.2568 | 1040.9 | 0.0101(0.00) | 0.0090 | 0.0000 || 0.00 | -0.01 | 0/5
    [Stage 2] epoch 128: thickness_decoder 그룹 AdamW state 리셋, lr → 3.0e-04 (좌표 헤드 모멘텀 보존)
    00139 || 1.2197 ||  11.08% | 0.2776 |  442.4 | 0.3816(0.35) | 0.0048 | 0.0000 || 1.00 | -2.20 | 5/5
    00159 || 0.8957 ||   3.23% | 0.2963 |  349.9 | 0.4868(0.54) | 0.0000 | 0.0000 || 1.00 | -2.69 | 5/5
    00179 || 0.7769 ||   0.75% | 0.3081 |  351.7 | 0.4846(0.60) | 0.0000 | 0.0000 || 1.00 | -2.72 | 5/5 [FEASIBLE]
    00199 || 0.7694 ||   1.10% | 0.3075 |  355.2 | 0.4804(0.60) | 0.0000 | 0.0000 || 1.00 | -2.72 | 5/5 [FEASIBLE]
    00219 || 0.8047 ||   3.25% | 0.3091 |  359.7 | 0.4750(0.54) | 0.0000 | 0.0000 || 1.00 | -2.72 | 5/5
    00239 || 0.7950 ||   1.08% | 0.2977 |  351.4 | 0.4850(0.60) | 0.0000 | 0.0000 || 1.00 | -2.74 | 5/5 [FEASIBLE]
    00259 || 0.9359 ||   3.15% | 0.2919 |  347.3 | 0.4899(0.55) | 0.0000 | 0.0000 || 1.00 | -2.75 | 5/5
    00279 || 0.7835 ||   0.32% | 0.2959 |  354.2 | 0.4815(0.61) | 0.0000 | 0.0000 || 1.00 | -2.73 | 5/5 [FEASIBLE]
    00299 || 0.7799 ||   0.03% | 0.2869 |  355.7 | 0.4798(0.62) | 0.0000 | 0.0000 || 1.00 | -2.74 | 5/5 [FEASIBLE]
    
    ──────────────────────────────────────────────────────────────────────────────
    [Feasibility] 첫 만족 epoch: 129  |  최고(Mp err 최소) epoch: 296  |  Mp err: 0.01%  |  l_collision: 0.0000
    [Best-Mp] epoch 296: Mp err = 0.01%, l_collision = 0.0000  (collision 무관 추적)
    ──────────────────────────────────────────────────────────────────────────────
    


    
![png](uni_section_v10_files/uni_section_v10_0_1.png)
    


    
    결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\code\uni_section_v10_result.png
    


    
![png](uni_section_v10_files/uni_section_v10_0_3.png)
    


    
    Epoch 스냅샷 결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\code\uni_section_v10_epoch_snapshots.png
    
    ==============================================================================
    최종 결과 요약 (마지막 epoch 기준)
      Target Mp     :      7,421,470 N·mm
      Final pred_mp :      7,419,494 N·mm
      Final Error   :   0.03%
      Final l_collision : 0.0000
      Final l_order : 0.0000
      Final delta_t : mean=-2.745 mm
    
      ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch 296, Mp err=0.01%, l_collision=0.0000
        (best_feasible['state_dict']를 model.load_state_dict()로 복원해 사용 권장)
    ==============================================================================
    
