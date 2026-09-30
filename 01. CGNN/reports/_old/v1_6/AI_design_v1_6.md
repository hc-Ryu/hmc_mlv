    [data] 17-section 그래프(사전 필터링 없음): 노드 1581개, 엣지 2992개
    [target_mp] ['26,444,055', '29,211,881', '31,397,729', '33,087,380', '33,704,378', '32,835,017', '30,467,844', '27,734,809', '26,413,902', '27,985,091', '32,922,019', '39,716,721', '45,526,897', '47,458,156', '44,876,148', '39,505,207', '35,005,895']
    [Stage 0] Static shape/index dry-run 시작...
    [Stage 0] 분할 주입 테스트 통과 - part3이 run 2개로 독립 pooling됨 (그룹 수: 6)
    [Stage 0] GPU 메모리 사용량: 9.2 MB
    [Stage 0] 통과 - 노드 1581개, 엣지 2992개, z_gate shape (17, 5), 5파트 글로벌 두께 불변 + 분할 주입 검증 완료.
    [gradcheck] dMp/dt OK -- autograd=3.4990e+08, FD=3.4976e+08, rel_err=4.09e-04, dMp/dt>0 확인
    [mass] area_init = 24056.9 mm^2 | target_area = 22854.0 mm^2 (감량 목표 95%)
    [collision v5] 153쌍(섹션x파트쌍) 부호 앵커 산정 완료
    
    ==============================================================================
    [ AI_design_v1_1 ] ASWC Training + Dynamic Splitting | 17 sections | Section-aware gating (17x2) | Stage1<100 Stage2<400 Stage3<500
    ==============================================================================
    Epoch    0 || loss=1.1583 | MpErr=8.08% | l_continuity=0.0000(w=1.000) | area=24010.8 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50) | w_continuity=1.000
    Epoch   10 || loss=15.1170 | MpErr=47.86% | l_continuity=0.0000(w=1.000) | area=11439.9 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50) | w_continuity=1.000
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    [R4] epoch 15: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)
    Epoch   20 || loss=9.5808 | MpErr=25.92% | l_continuity=0.0000(w=1.000) | area=28685.4 | g3=0.488(17/17) g4=0.488(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.69) | w_continuity=1.000
    Epoch   30 || loss=9.3431 | MpErr=25.53% | l_continuity=0.0000(w=1.000) | area=28559.1 | g3=0.458(17/17) g4=0.458(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=1.29) | w_continuity=1.000
    Epoch   40 || loss=9.1960 | MpErr=25.27% | l_continuity=0.0000(w=1.000) | area=28475.1 | g3=0.435(17/17) g4=0.435(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=2.19) | w_continuity=1.000
    Epoch   50 || loss=9.0104 | MpErr=24.95% | l_continuity=0.0000(w=1.000) | area=28370.4 | g3=0.409(17/17) g4=0.409(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=3.21) | w_continuity=1.000
    Epoch   60 || loss=4.9598 | MpErr=16.42% | l_continuity=0.0000(w=1.000) | area=25857.3 | g3=0.382(17/17) g4=0.382(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.13) | w_continuity=1.000
    Epoch   70 || loss=1.9241 | MpErr=0.96% | l_continuity=0.0000(w=1.000) | area=21179.5 | g3=0.368(17/17) g4=0.369(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.77) | w_continuity=1.000
    Epoch   80 || loss=9.8678 | MpErr=27.72% | l_continuity=0.0000(w=1.000) | area=14957.1 | g3=0.363(17/17) g4=0.363(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch   90 || loss=2.0814 | MpErr=5.24% | l_continuity=0.0000(w=1.000) | area=20201.1 | g3=0.360(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  100 || loss=16.2772 | MpErr=23.29% | l_continuity=0.0000(w=1.000) | area=27803.3 | g3=0.360(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.5937(w=5.00) | w_continuity=1.000
    Epoch  110 || loss=1.9669 | MpErr=1.20% | l_continuity=0.0000(w=1.000) | area=20741.8 | g3=0.360(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  120 || loss=1.9335 | MpErr=1.83% | l_continuity=0.0000(w=1.000) | area=20452.0 | g3=0.360(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  130 || loss=1.9806 | MpErr=1.34% | l_continuity=0.0001(w=1.000) | area=20420.4 | g3=0.360(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  140 || loss=2.0736 | MpErr=1.77% | l_continuity=0.0014(w=1.000) | area=20441.6 | g3=0.361(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  150 || loss=2.1868 | MpErr=1.12% | l_continuity=0.0029(w=1.000) | area=20451.3 | g3=0.361(17/17) g4=0.361(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  160 || loss=2.3261 | MpErr=1.48% | l_continuity=0.0030(w=1.000) | area=20425.4 | g3=0.361(17/17) g4=0.362(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  170 || loss=2.4847 | MpErr=1.81% | l_continuity=0.0040(w=1.000) | area=20406.1 | g3=0.362(17/17) g4=0.362(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  180 || loss=2.6543 | MpErr=0.65% | l_continuity=0.0145(w=1.000) | area=20473.5 | g3=0.362(17/17) g4=0.362(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  190 || loss=2.8708 | MpErr=3.06% | l_continuity=0.0041(w=1.000) | area=20338.7 | g3=0.363(17/17) g4=0.362(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  200 || loss=3.0566 | MpErr=3.14% | l_continuity=0.0081(w=1.000) | area=20346.7 | g3=0.364(17/17) g4=0.363(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  210 || loss=3.1967 | MpErr=1.27% | l_continuity=0.0311(w=1.000) | area=20446.8 | g3=0.347(17/17) g4=0.346(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  220 || loss=3.3797 | MpErr=1.53% | l_continuity=0.0456(w=1.000) | area=20436.1 | g3=0.332(17/17) g4=0.329(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  230 || loss=3.8010 | MpErr=5.43% | l_continuity=0.0109(w=1.000) | area=20194.5 | g3=0.317(17/17) g4=0.313(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  240 || loss=3.8701 | MpErr=4.44% | l_continuity=0.0487(w=1.000) | area=20262.7 | g3=0.304(17/17) g4=0.298(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  250 || loss=4.0859 | MpErr=5.02% | l_continuity=0.1083(w=1.000) | area=20256.1 | g3=0.291(17/17) g4=0.284(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  260 || loss=4.1237 | MpErr=4.04% | l_continuity=0.0538(w=1.000) | area=20303.5 | g3=0.281(17/17) g4=0.272(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  270 || loss=6.2090 | MpErr=0.73% | l_continuity=0.1765(w=1.000) | area=20655.9 | g3=0.273(17/17) g4=0.261(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4191(w=5.00) | w_continuity=1.000
    Epoch  280 || loss=4.8028 | MpErr=3.27% | l_continuity=0.1694(w=1.000) | area=20480.1 | g3=0.268(17/17) g4=0.252(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.1293(w=5.00) | w_continuity=1.000
    Epoch  290 || loss=6.3484 | MpErr=0.20% | l_continuity=0.1839(w=1.000) | area=20726.3 | g3=0.265(17/17) g4=0.245(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4452(w=5.00) | w_continuity=1.000
    Epoch  300 || loss=6.1938 | MpErr=2.12% | l_continuity=0.2197(w=1.000) | area=20726.2 | g3=0.264(17/17) g4=0.241(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4268(w=5.00) | w_continuity=1.000
    Epoch  310 || loss=6.0482 | MpErr=2.11% | l_continuity=0.0175(w=1.000) | area=20568.2 | g3=0.265(17/17) g4=0.246(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.3520(w=5.00) | w_continuity=1.000
    Epoch  320 || loss=4.5769 | MpErr=1.39% | l_continuity=0.0729(w=1.000) | area=20655.2 | g3=0.266(17/17) g4=0.255(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0827(w=5.00) | w_continuity=1.000
    Epoch  330 || loss=4.1034 | MpErr=3.58% | l_continuity=0.1221(w=1.000) | area=20654.7 | g3=0.271(17/17) g4=0.273(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000
    Epoch  340 || loss=7.5668 | MpErr=2.91% | l_continuity=0.2120(w=1.000) | area=21097.1 | g3=0.277(17/17) g4=0.294(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.6957(w=5.00) | w_continuity=1.000
    Epoch  350 || loss=7.3443 | MpErr=6.41% | l_continuity=0.0833(w=1.000) | area=20406.7 | g3=0.284(17/17) g4=0.311(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.6163(w=5.00) | w_continuity=1.000
    Epoch  360 || loss=6.3426 | MpErr=3.59% | l_continuity=0.1962(w=1.000) | area=20689.7 | g3=0.293(17/17) g4=0.325(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.5134(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 361: DELETED 확정 [[1, 1]] -> candidate run 수 [1, 2] (두께 그룹 6개로 재구성)
    [Dynamic Split] epoch 366: DELETED 확정 [[2, 1]] -> candidate run 수 [1, 2] (두께 그룹 6개로 재구성)
    Epoch  370 || loss=6.0331 | MpErr=2.00% | l_continuity=0.4089(w=1.000) | area=21149.3 | g3=0.303(17/17) g4=0.384(15/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/32 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4708(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 373: DELETED 확정 [[7, 0]] -> candidate run 수 [2, 2] (두께 그룹 7개로 재구성)
    Epoch  380 || loss=7.5288 | MpErr=3.56% | l_continuity=0.4585(w=1.000) | area=21230.1 | g3=0.330(16/17) g4=0.397(15/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/31 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7841(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 385: DELETED 확정 [[5, 1]] -> candidate run 수 [2, 3] (두께 그룹 8개로 재구성)
    [Dynamic Split] epoch 386: DELETED 확정 [[3, 0]] -> candidate run 수 [3, 3] (두께 그룹 9개로 재구성)
    [Dynamic Split] epoch 387: DELETED 확정 [[1, 0]] -> candidate run 수 [4, 3] (두께 그룹 10개로 재구성)
    Epoch  390 || loss=6.5482 | MpErr=0.30% | l_continuity=0.2658(w=1.000) | area=21136.8 | g3=0.379(14/17) g4=0.433(14/17) | groups=10 | log_alpha_sat=0.00% | dead0=0/28 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.6549(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 395: DELETED 확정 [[4, 1]] -> candidate run 수 [4, 3] (두께 그룹 10개로 재구성)
    [Dynamic Split] epoch 396: DELETED 확정 [[8, 1]] -> candidate run 수 [4, 4] (두께 그룹 11개로 재구성)
    [Dynamic Split] epoch 400: DELETED 확정 [[0, 1]] -> candidate run 수 [4, 3] (두께 그룹 10개로 재구성)
    Epoch  400 || loss=7.5371 | MpErr=1.13% | l_continuity=0.3313(w=1.000) | area=21441.9 | g3=0.387(14/17) g4=0.560(11/17) | groups=10 | log_alpha_sat=0.00% | dead0=0/25 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.9423(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 404: DELETED 확정 [[7, 1]] -> candidate run 수 [4, 3] (두께 그룹 10개로 재구성)
    [Dynamic Split] epoch 406: DELETED 확정 [[9, 1]] -> candidate run 수 [4, 3] (두께 그룹 10개로 재구성)
    Epoch  410 || loss=7.7212 | MpErr=0.97% | l_continuity=0.3394(w=1.000) | area=21522.0 | g3=0.392(14/17) g4=0.691(9/17) | groups=10 | log_alpha_sat=0.00% | dead0=0/23 | suspect=0 | alpha_ent=0.0100 | l_col_aux=0.9092(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 417: DELETED 확정 [[0, 0]] -> candidate run 수 [3, 3] (두께 그룹 9개로 재구성)
    Epoch  420 || loss=6.2866 | MpErr=3.08% | l_continuity=0.4356(w=1.000) | area=21418.7 | g3=0.424(13/17) g4=0.696(9/17) | groups=9 | log_alpha_sat=0.00% | dead0=0/22 | suspect=0 | alpha_ent=0.0200 | l_col_aux=0.6448(w=5.00) | w_continuity=1.000
    Epoch  430 || loss=6.8753 | MpErr=0.38% | l_continuity=0.1990(w=1.000) | area=21328.2 | g3=0.426(13/17) g4=0.700(9/17) | groups=9 | log_alpha_sat=0.00% | dead0=0/22 | suspect=0 | alpha_ent=0.0300 | l_col_aux=0.6172(w=5.00) | w_continuity=1.000
    Epoch  440 || loss=7.2522 | MpErr=0.80% | l_continuity=0.3628(w=1.000) | area=21541.7 | g3=0.429(13/17) g4=0.704(9/17) | groups=9 | log_alpha_sat=0.00% | dead0=0/22 | suspect=0 | alpha_ent=0.0400 | l_col_aux=0.7645(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 441: DELETED 확정 [[6, 1]] -> candidate run 수 [3, 2] (두께 그룹 8개로 재구성)
    [Adaptive Extension] loop_max -> 501 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  450 || loss=6.9376 | MpErr=2.81% | l_continuity=0.2863(w=1.000) | area=21783.6 | g3=0.432(13/17) g4=0.797(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/21 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.7844(w=5.00) | w_continuity=1.000
    Epoch  460 || loss=9.7650 | MpErr=7.21% | l_continuity=0.2602(w=1.000) | area=21787.0 | g3=0.435(13/17) g4=0.802(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/21 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.1888(w=5.00) | w_continuity=1.000
    Epoch  470 || loss=6.6446 | MpErr=3.98% | l_continuity=0.2671(w=1.000) | area=21704.7 | g3=0.437(13/17) g4=0.805(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/21 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.7797(w=5.00) | w_continuity=1.000
    Epoch  480 || loss=7.7223 | MpErr=3.98% | l_continuity=0.1978(w=1.000) | area=21642.9 | g3=0.439(13/17) g4=0.808(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/21 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.0044(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 487: DELETED 확정 [[8, 0]] -> candidate run 수 [3, 2] (두께 그룹 8개로 재구성)
    [Adaptive Extension] loop_max -> 547 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  490 || loss=5.1929 | MpErr=0.65% | l_continuity=0.1829(w=1.000) | area=21417.3 | g3=0.477(12/17) g4=0.811(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/20 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.4969(w=5.00) | w_continuity=1.000
    Epoch  500 || loss=6.1363 | MpErr=1.27% | l_continuity=0.1826(w=1.000) | area=21644.9 | g3=0.479(12/17) g4=0.813(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/20 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.7327(w=5.00) | w_continuity=1.000
    Epoch  510 || loss=4.7137 | MpErr=3.33% | l_continuity=0.1878(w=1.000) | area=21167.1 | g3=0.480(12/17) g4=0.814(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/20 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.2685(w=5.00) | w_continuity=1.000
    Epoch  520 || loss=4.8543 | MpErr=1.06% | l_continuity=0.1503(w=1.000) | area=21608.6 | g3=0.482(12/17) g4=0.816(8/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/20 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.4836(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 521: DELETED 확정 [[3, 1]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    [Adaptive Extension] loop_max -> 581 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  530 || loss=5.2434 | MpErr=5.60% | l_continuity=0.2045(w=1.000) | area=21751.7 | g3=0.483(12/17) g4=0.934(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/19 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.4915(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 538: DELETED 확정 [[9, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    [Adaptive Extension] loop_max -> 598 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  540 || loss=4.9918 | MpErr=0.60% | l_continuity=0.1259(w=1.000) | area=21431.9 | g3=0.524(11/17) g4=0.935(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/18 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.4912(w=5.00) | w_continuity=1.000
    Epoch  550 || loss=5.3297 | MpErr=1.26% | l_continuity=0.1515(w=1.000) | area=21605.6 | g3=0.525(11/17) g4=0.937(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/18 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.5963(w=5.00) | w_continuity=1.000
    Epoch  560 || loss=4.4103 | MpErr=1.46% | l_continuity=0.1047(w=1.000) | area=21432.0 | g3=0.526(11/17) g4=0.938(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/18 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.3817(w=5.00) | w_continuity=1.000
    Epoch  570 || loss=4.7690 | MpErr=0.48% | l_continuity=0.1888(w=1.000) | area=21560.1 | g3=0.527(11/17) g4=0.939(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/18 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.4654(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 575: DELETED 확정 [[4, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    [Adaptive Extension] loop_max -> 620 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  580 || loss=3.6964 | MpErr=0.17% | l_continuity=0.1143(w=1.000) | area=21194.7 | g3=0.576(10/17) g4=0.941(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/17 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.2564(w=5.00) | w_continuity=1.000
    Epoch  590 || loss=6.5369 | MpErr=0.11% | l_continuity=0.1877(w=1.000) | area=21307.4 | g3=0.578(10/17) g4=0.944(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/17 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.8315(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 598: DELETED 확정 [[6, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    Epoch  600 || loss=3.4508 | MpErr=2.53% | l_continuity=0.1201(w=1.000) | area=21498.3 | g3=0.641(9/17) g4=0.946(7/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/16 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.2480(w=5.00) | w_continuity=1.000
    [Dynamic Split] epoch 609: DELETED 확정 [[2, 0]] -> candidate run 수 [2, 1] (두께 그룹 6개로 재구성)
    Epoch  610 || loss=3.4823 | MpErr=1.71% | l_continuity=0.1142(w=1.000) | area=21318.9 | g3=0.721(8/17) g4=0.947(7/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/15 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.2198(w=5.00) | w_continuity=1.000
    Epoch  619 || loss=4.4852 | MpErr=4.96% | l_continuity=0.1128(w=1.000) | area=21391.8 | g3=0.722(8/17) g4=0.948(7/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/15 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.3495(w=5.00) | w_continuity=1.000
    
    [Dynamic Split] 분할 이벤트 epoch: [361, 366, 373, 385, 386, 387, 395, 396, 400, 404, 406, 417, 441, 487, 521, 538, 575, 598, 609], 최종 두께 그룹 6개
    >>> Stage 4: 존재 확정 + 형상/두께 재수렴 (620 -> 820), 애매 게이트 15개
    [Stage4] epoch  620 || loss=3.6708 | MpErr=3.23% | area=21293.8
    [Stage4] epoch  640 || loss=3.0016 | MpErr=1.13% | area=21515.2
    [Stage4] epoch  660 || loss=2.7386 | MpErr=0.79% | area=20314.8
    [Stage4] epoch  680 || loss=3.6631 | MpErr=1.85% | area=20330.7
    [Stage4] epoch  700 || loss=3.6399 | MpErr=0.71% | area=20031.4
    [Stage4] epoch  720 || loss=3.6168 | MpErr=0.69% | area=19962.8
    [Stage4] epoch  740 || loss=3.7983 | MpErr=1.39% | area=19492.1
    [Stage4] epoch  760 || loss=3.5724 | MpErr=1.05% | area=19918.2
    [Stage4] epoch  780 || loss=3.4754 | MpErr=1.00% | area=19921.1
    [Stage4] epoch  800 || loss=3.8535 | MpErr=1.41% | area=19371.5
    [Stage4] epoch  819 || loss=3.5702 | MpErr=0.49% | area=19516.6
    [viz] 학습 결과 dashboard 저장: reports/figures/AI_design_v1_6_result.png
    [viz] 인터랙티브 3D 저장: reports/figures/AI_design_v1_6_3d.html
    [report] hard existence 판정용 temperature=0.3000 (epoch 619 기준)
    
    # AI_design_v1_2 — Initial vs Final Design Property Report
    
    (hard existence 판정 temperature=0.3000 — 학습 종료 시점 어닐링 온도와 일치)
    
    ## 1. Section-wise Mp (N*mm)
    
    | sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 55,297,616 | 26,444,055 | 28,310,868 | 28,310,868 | 107.1 | target miss |
    | 1 | 52,739,376 | 29,211,881 | 30,140,644 | 30,140,644 | 103.2 | target miss |
    | 2 | 50,241,736 | 31,397,729 | 31,972,146 | 31,972,146 | 101.8 |  |
    | 3 | 47,804,696 | 33,087,380 | 33,728,632 | 33,728,632 | 101.9 |  |
    | 4 | 45,428,256 | 33,704,378 | 33,755,916 | 33,755,916 | 100.2 |  |
    | 5 | 43,112,400 | 32,835,017 | 32,843,620 | 32,843,620 | 100.0 |  |
    | 6 | 40,857,144 | 30,467,844 | 30,898,954 | 30,898,954 | 101.4 |  |
    | 7 | 38,662,480 | 27,734,809 | 28,025,024 | 28,025,024 | 101.0 |  |
    | 8 | 36,528,412 | 26,413,902 | 26,536,828 | 26,536,828 | 100.5 |  |
    | 9 | 34,454,944 | 27,985,091 | 28,289,588 | 28,289,588 | 101.1 |  |
    | 10 | 32,442,064 | 32,922,019 | 32,876,182 | 32,876,182 | 99.9 |  |
    | 11 | 30,489,774 | 39,716,721 | 40,690,680 | 40,690,680 | 102.5 | target miss |
    | 12 | 28,598,076 | 45,526,897 | 45,563,424 | 45,563,424 | 100.1 |  |
    | 13 | 26,766,974 | 47,458,156 | 47,163,200 | 47,163,200 | 99.4 |  |
    | 14 | 24,996,464 | 44,876,148 | 44,504,392 | 44,504,392 | 99.2 |  |
    | 15 | 23,286,536 | 39,505,207 | 39,511,908 | 39,511,908 | 100.0 |  |
    | 16 | 21,637,200 | 35,005,895 | 35,178,840 | 35,178,840 | 100.5 |  |
    | **sum** | 633,344,148 | 584,293,129 | 589,990,846 | 589,990,846 | 101.0 | |
    
    ## 2. Total cross-section area (mm^2, 17-section sum)
    
    | initial | final(soft) | final(hard) | delta(hard-init) |
    |--------:|------------:|------------:|-----------------:|
    | 24,056.9 | 19,789.5 | 19,789.5 | -4,267.4 (-17.7%) |
    
    | 감량 목표 | 22,854.0 mm² (초기 대비 95.0%) |
    | 달성률 | 86.6% (100% 이하면 목표 달성) |
    
    - 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **0개** (임계 0.3 적용 시 구제: 0개)
    
    ## 3. Part thickness (mm, 제조 기준 = pooled t_raw)
    
    | part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |
    |:-----|----------:|:------------------|:--------------------------------|
    | #00(Outer) | 2.300 | 2.290 (전 섹션 공통) | 항상 존재 |
    | #03(Plate) | 1.600 | 0.700 (전 섹션 공통) | 항상 존재 |
    | #06(Inner) | 1.600 | 1.509 (전 섹션 공통) | 항상 존재 |
    | #07(Patch1) | 1.400 | 0.700 (run 1개) | XXXXXXXXXXOOOOOOO (7/17 alive) |
    | #08(Patch2) | 1.600 | 2.290 (run 1개) | XXXXXXXXXXOOOOOOO (7/17 alive) |
    
    [report] 비교 리포트 저장: reports/AI_design_v1_6_report.md
    >>> Stage 4 진입 여부: True
    [done] weights/bpillar_17sec_v1_6.pt, reports/figures/AI_design_v1_6_result.png, reports/figures/AI_design_v1_6_3d.html, reports/AI_design_v1_6_report.md 저장 완료
    
