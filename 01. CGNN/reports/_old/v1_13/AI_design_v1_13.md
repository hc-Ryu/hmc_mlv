    [data] 17-section 그래프(사전 필터링 없음): 노드 1581개, 엣지 2992개
    [target_mp] ['333,000,000', '192,000,000', '128,000,000', '97,700,000', '88,300,000', '89,600,000', '84,100,000', '71,000,000', '71,600,000', '67,700,000', '57,700,000', '64,400,000', '53,500,000', '47,400,000', '41,300,000', '35,600,000', '31,300,000']
    [target_my] ['171,000,000', '92,100,000', '49,000,000', '38,200,000', '34,700,000', '34,500,000', '34,800,000', '31,100,000', '30,300,000', '28,800,000', '24,900,000', '25,000,000', '14,700,000', '14,500,000', '17,300,000', '14,900,000', '13,100,000'] (리포트 전용, 학습에 미반영)
    [Stage 0] Static shape/index dry-run 시작...
    [Stage 0] 분할 주입 테스트 통과 - part3이 run 2개로 독립 pooling됨 (그룹 수: 6)
    [Stage 0] GPU 메모리 사용량: 19.0 MB
    [Stage 0] 통과 - 노드 1581개, 엣지 2992개, z_gate shape (17, 5), 5파트 글로벌 두께 불변 + 분할 주입 검증 완료.
    [gradcheck] dMp/dt OK -- autograd=3.4926e+08, FD=3.4944e+08, rel_err=5.18e-04, dMp/dt>0 확인
    [mass] area_init = 24056.9 mm^2 | target_area = 22854.0 mm^2 (감량 목표 95%)
    [collision v5] 153쌍(섹션x파트쌍) 부호 앵커 산정 완료
    [smooth-lse] 정적 이웃 맵 1411개 노드(전체 1581개 중) 구축 완료
    
    ==============================================================================
    [ AI_design_v1_1 ] ASWC Training + Dynamic Splitting | 17 sections | Section-aware gating (17x2) | Stage1<100 Stage2<400 Stage3<500
    ==============================================================================
    Epoch    0 || loss=32.8470 | MpErr=59.53% | l_continuity=0.0000(w=1.000) | area=23934.4 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   10 || loss=45.3292 | MpErr=50.47% | l_continuity=0.0000(w=1.000) | area=30713.7 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50) | w_continuity=1.000 | l_smooth_lse=57.1018
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    Epoch   20 || loss=59.7619 | MpErr=52.54% | l_continuity=0.0000(w=1.000) | area=28837.7 | g3=0.526(17/17) g4=0.526(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.69) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   30 || loss=59.7079 | MpErr=52.41% | l_continuity=0.0000(w=1.000) | area=28954.7 | g3=0.556(17/17) g4=0.556(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=1.29) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   40 || loss=59.5973 | MpErr=52.27% | l_continuity=0.0000(w=1.000) | area=29077.0 | g3=0.587(17/17) g4=0.586(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=2.19) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   50 || loss=59.4929 | MpErr=52.14% | l_continuity=0.0000(w=1.000) | area=29194.5 | g3=0.616(17/17) g4=0.615(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=3.21) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   60 || loss=59.3944 | MpErr=52.01% | l_continuity=0.0000(w=1.000) | area=29307.3 | g3=0.645(17/17) g4=0.641(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.13) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   70 || loss=59.3034 | MpErr=51.90% | l_continuity=0.0000(w=1.000) | area=29413.6 | g3=0.672(17/17) g4=0.664(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.77) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   80 || loss=59.2197 | MpErr=51.78% | l_continuity=0.0000(w=1.000) | area=29512.9 | g3=0.698(17/17) g4=0.683(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch   90 || loss=59.1431 | MpErr=51.68% | l_continuity=0.0000(w=1.000) | area=29604.6 | g3=0.722(17/17) g4=0.697(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=57.1018
    Epoch  100 || loss=91.1150 | MpErr=48.72% | l_continuity=1.5153(w=1.000) | area=30384.9 | g3=0.745(17/17) g4=0.708(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=5.0618(w=5.00) | w_continuity=1.000 | l_smooth_lse=71.1583
    Epoch  110 || loss=57.7565 | MpErr=52.85% | l_continuity=0.0147(w=1.000) | area=29283.5 | g3=0.758(17/17) g4=0.712(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=51.7084
    Epoch  120 || loss=55.0854 | MpErr=49.20% | l_continuity=0.2475(w=1.000) | area=29308.0 | g3=0.764(17/17) g4=0.714(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0001(w=5.00) | w_continuity=1.000 | l_smooth_lse=52.6582
    Epoch  130 || loss=57.4438 | MpErr=46.46% | l_continuity=0.0861(w=1.000) | area=29495.9 | g3=0.768(17/17) g4=0.715(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.8253(w=5.00) | w_continuity=1.000 | l_smooth_lse=52.9017
    Epoch  140 || loss=52.0201 | MpErr=38.16% | l_continuity=0.2558(w=1.000) | area=30292.5 | g3=0.771(17/17) g4=0.715(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=63.7163
    Epoch  150 || loss=57.3262 | MpErr=40.67% | l_continuity=0.2980(w=1.000) | area=30119.8 | g3=0.774(17/17) g4=0.714(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.9057(w=5.00) | w_continuity=1.000 | l_smooth_lse=49.7315
    Epoch  160 || loss=60.1869 | MpErr=23.74% | l_continuity=0.6461(w=1.000) | area=32133.4 | g3=0.776(17/17) g4=0.714(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.5515(w=5.00) | w_continuity=1.000 | l_smooth_lse=75.5307
    Epoch  170 || loss=47.6225 | MpErr=24.52% | l_continuity=0.2830(w=1.000) | area=31764.2 | g3=0.777(17/17) g4=0.713(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.5515(w=5.00) | w_continuity=1.000 | l_smooth_lse=50.8054
    Epoch  180 || loss=38.6903 | MpErr=21.98% | l_continuity=0.5914(w=1.000) | area=31891.2 | g3=0.778(17/17) g4=0.712(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.2718(w=5.00) | w_continuity=1.000 | l_smooth_lse=46.4775
    Epoch  190 || loss=55.7163 | MpErr=13.66% | l_continuity=0.2430(w=1.000) | area=32900.2 | g3=0.779(17/17) g4=0.710(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=3.4640(w=5.00) | w_continuity=1.000 | l_smooth_lse=50.5560
    Epoch  200 || loss=37.5927 | MpErr=23.36% | l_continuity=0.0937(w=1.000) | area=31824.6 | g3=0.779(17/17) g4=0.708(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0023(w=5.00) | w_continuity=1.000 | l_smooth_lse=45.5595
    Epoch  210 || loss=34.2033 | MpErr=16.71% | l_continuity=0.0564(w=1.000) | area=32509.1 | g3=0.810(17/17) g4=0.730(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=42.0243
    Epoch  220 || loss=34.4530 | MpErr=20.69% | l_continuity=0.0324(w=1.000) | area=32131.9 | g3=0.839(17/17) g4=0.751(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=40.4691
    Epoch  230 || loss=39.2437 | MpErr=11.32% | l_continuity=0.1096(w=1.000) | area=33320.0 | g3=0.866(17/17) g4=0.768(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7111(w=5.00) | w_continuity=1.000 | l_smooth_lse=43.6611
    Epoch  240 || loss=34.9556 | MpErr=18.21% | l_continuity=0.0575(w=1.000) | area=32520.7 | g3=0.891(17/17) g4=0.783(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=41.3462
    Epoch  250 || loss=35.8564 | MpErr=16.42% | l_continuity=0.1298(w=1.000) | area=32802.7 | g3=0.914(17/17) g4=0.795(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.3004(w=5.00) | w_continuity=1.000 | l_smooth_lse=40.3112
    Epoch  260 || loss=40.0809 | MpErr=12.04% | l_continuity=0.0754(w=1.000) | area=33316.9 | g3=0.935(17/17) g4=0.800(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.0566(w=5.00) | w_continuity=1.000 | l_smooth_lse=41.1209
    Epoch  270 || loss=35.8742 | MpErr=16.85% | l_continuity=0.0841(w=1.000) | area=32827.8 | g3=0.954(17/17) g4=0.800(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.3420(w=5.00) | w_continuity=1.000 | l_smooth_lse=39.1520
    Epoch  280 || loss=35.0494 | MpErr=16.68% | l_continuity=0.0318(w=1.000) | area=32890.8 | g3=0.971(17/17) g4=0.798(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.1232(w=5.00) | w_continuity=1.000 | l_smooth_lse=39.2646
    Epoch  290 || loss=36.1750 | MpErr=14.76% | l_continuity=0.0263(w=1.000) | area=33127.5 | g3=0.986(17/17) g4=0.794(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4140(w=5.00) | w_continuity=1.000 | l_smooth_lse=38.7884
    [R4] epoch 293: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)
    Epoch  300 || loss=35.2203 | MpErr=19.03% | l_continuity=0.0166(w=1.000) | area=32728.0 | g3=0.997(17/17) g4=0.785(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=1.000 | l_smooth_lse=38.5232
    Epoch  310 || loss=40.2726 | MpErr=15.32% | l_continuity=0.0196(w=1.000) | area=33123.7 | g3=1.000(17/17) g4=0.772(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.1303(w=5.00) | w_continuity=1.000 | l_smooth_lse=37.8359
    Epoch  320 || loss=36.6855 | MpErr=16.86% | l_continuity=0.0691(w=1.000) | area=32971.6 | g3=1.000(17/17) g4=0.745(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.2767(w=5.00) | w_continuity=1.000 | l_smooth_lse=37.0750
    Epoch  330 || loss=43.0995 | MpErr=16.55% | l_continuity=0.0954(w=0.999) | area=32985.2 | g3=1.000(17/17) g4=0.714(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.2177(w=5.00) | w_continuity=0.999 | l_smooth_lse=38.3372
    Epoch  340 || loss=46.5457 | MpErr=16.83% | l_continuity=0.0593(w=0.998) | area=32963.7 | g3=1.000(17/17) g4=0.686(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.4993(w=5.00) | w_continuity=0.998 | l_smooth_lse=38.8584
    Epoch  350 || loss=45.9976 | MpErr=16.78% | l_continuity=0.0569(w=0.994) | area=32903.1 | g3=1.000(17/17) g4=0.661(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.0641(w=5.00) | w_continuity=0.994 | l_smooth_lse=37.1334
    Epoch  360 || loss=53.7639 | MpErr=13.57% | l_continuity=0.0575(w=0.983) | area=33229.6 | g3=1.000(17/17) g4=0.635(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.5920(w=5.00) | w_continuity=0.983 | l_smooth_lse=39.0215
    Epoch  370 || loss=54.9447 | MpErr=15.15% | l_continuity=0.0362(w=0.955) | area=32985.5 | g3=1.000(17/17) g4=0.617(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7102(w=5.00) | w_continuity=0.955 | l_smooth_lse=38.1497
    Epoch  380 || loss=59.3513 | MpErr=18.38% | l_continuity=0.0776(w=0.887) | area=32666.5 | g3=1.000(17/17) g4=0.603(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.2783(w=5.00) | w_continuity=0.887 | l_smooth_lse=36.2659
    Epoch  390 || loss=81.3842 | MpErr=13.44% | l_continuity=0.0667(w=0.745) | area=33228.2 | g3=1.000(17/17) g4=0.591(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.8877(w=5.00) | w_continuity=0.745 | l_smooth_lse=39.5457
    Epoch  400 || loss=79.5770 | MpErr=24.55% | l_continuity=0.0166(w=0.525) | area=32044.7 | g3=1.000(17/17) g4=0.583(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00) | w_continuity=0.525 | l_smooth_lse=35.3194
    Epoch  410 || loss=89.9687 | MpErr=24.92% | l_continuity=0.0206(w=0.305) | area=31958.3 | g3=1.000(17/17) g4=0.577(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0100 | l_col_aux=0.0000(w=5.00) | w_continuity=0.305 | l_smooth_lse=35.1048
    Epoch  420 || loss=99.4512 | MpErr=21.84% | l_continuity=0.0402(w=0.300) | area=32265.6 | g3=1.000(17/17) g4=0.571(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0200 | l_col_aux=0.1334(w=5.00) | w_continuity=0.300 | l_smooth_lse=35.3085
    Epoch  430 || loss=106.2859 | MpErr=23.55% | l_continuity=0.0188(w=0.300) | area=32125.2 | g3=1.000(17/17) g4=0.565(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0300 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=34.9587
    Epoch  440 || loss=112.9787 | MpErr=20.95% | l_continuity=0.0981(w=0.300) | area=32386.8 | g3=1.000(17/17) g4=0.561(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0400 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=35.7888
    Epoch  450 || loss=115.7474 | MpErr=19.80% | l_continuity=0.0901(w=0.300) | area=32447.7 | g3=1.000(17/17) g4=0.558(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=35.5740
    Epoch  460 || loss=118.6675 | MpErr=25.71% | l_continuity=0.1358(w=0.300) | area=31929.1 | g3=1.000(17/17) g4=0.555(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=34.7500
    Epoch  470 || loss=117.9496 | MpErr=22.46% | l_continuity=0.0895(w=0.300) | area=32176.5 | g3=1.000(17/17) g4=0.553(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=34.5473
    Epoch  480 || loss=120.4456 | MpErr=18.69% | l_continuity=0.0907(w=0.300) | area=32550.1 | g3=1.000(17/17) g4=0.551(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=35.4806
    Epoch  490 || loss=121.2959 | MpErr=23.79% | l_continuity=0.0423(w=0.300) | area=32116.1 | g3=1.000(17/17) g4=0.549(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=34.8848
    Epoch  499 || loss=119.9947 | MpErr=24.78% | l_continuity=0.0985(w=0.300) | area=31935.8 | g3=1.000(17/17) g4=0.546(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.0000(w=5.00) | w_continuity=0.300 | l_smooth_lse=34.2314
    
    [Dynamic Split] 분할 이벤트 없음 - candidate는 끝까지 파트당 1 두께 유지 (Phase A/B)
    >>> Stage 4: 존재 확정 + 형상/두께 재수렴 (500 -> 700), 애매 게이트 34개
    [Stage4] optimizer 모멘텀 이전 완료: 64개 파라미터
    [Stage4] epoch  500 || loss=119.5567 | MpErr=23.05% | area=32089.8
    [Stage4] epoch  520 || loss=118.5100 | MpErr=20.93% | area=32406.5
    [Stage4] epoch  540 || loss=118.3450 | MpErr=20.80% | area=32407.4
    [Stage4] epoch  560 || loss=118.1787 | MpErr=21.12% | area=32364.6
    [Stage4] epoch  580 || loss=118.1354 | MpErr=21.60% | area=32309.1
    [Stage4] epoch  600 || loss=117.9199 | MpErr=20.86% | area=32371.8
    [Stage4] epoch  620 || loss=117.9278 | MpErr=20.25% | area=32421.4
    [Stage4] epoch  640 || loss=117.7197 | MpErr=20.84% | area=32362.7
    [Stage4] epoch  660 || loss=117.5538 | MpErr=20.92% | area=32341.1
    [Stage4] epoch  680 || loss=117.4753 | MpErr=20.41% | area=32384.7
    [Stage4] epoch  699 || loss=117.3856 | MpErr=20.59% | area=32361.3
    [viz] 학습 결과 dashboard 저장: reports/figures/AI_design_v1_8_result.png
    [viz] 인터랙티브 3D 저장: reports/figures/AI_design_v1_8_3d.html
    [report] hard existence 판정용 temperature=0.3000 (epoch 499 기준)
    
    # AI_design_v1_2 — Initial vs Final Design Property Report
    
    (hard existence 판정 temperature=0.3000 — 학습 종료 시점 어닐링 온도와 일치)
    
    ## 1. Section-wise Mp (N*mm)
    
    | sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 55,292,436 | 333,000,000 | 119,377,664 | 119,377,664 | 35.8 | target miss |
    | 1 | 52,726,464 | 192,000,000 | 109,968,208 | 109,968,208 | 57.3 | target miss |
    | 2 | 50,221,472 | 128,000,000 | 106,708,992 | 106,708,992 | 83.4 | target miss |
    | 3 | 47,777,448 | 97,700,000 | 99,052,832 | 99,052,832 | 101.4 |  |
    | 4 | 45,394,400 | 88,300,000 | 91,901,392 | 91,901,392 | 104.1 | target miss |
    | 5 | 43,072,324 | 89,600,000 | 86,880,008 | 86,880,008 | 97.0 | target miss |
    | 6 | 40,811,220 | 84,100,000 | 81,263,136 | 81,263,136 | 96.6 | target miss |
    | 7 | 38,611,088 | 71,000,000 | 74,886,816 | 74,886,816 | 105.5 | target miss |
    | 8 | 36,471,928 | 71,600,000 | 70,566,656 | 70,566,656 | 98.6 |  |
    | 9 | 34,393,740 | 67,700,000 | 65,910,648 | 65,910,648 | 97.4 | target miss |
    | 10 | 32,376,516 | 57,700,000 | 60,283,876 | 60,283,876 | 104.5 | target miss |
    | 11 | 30,420,268 | 64,400,000 | 57,344,848 | 57,344,848 | 89.0 | target miss |
    | 12 | 28,524,984 | 53,500,000 | 51,611,440 | 51,611,440 | 96.5 | target miss |
    | 13 | 26,690,672 | 47,400,000 | 46,404,560 | 46,404,560 | 97.9 | target miss |
    | 14 | 24,917,324 | 41,300,000 | 41,474,088 | 41,474,088 | 100.4 |  |
    | 15 | 23,204,946 | 35,600,000 | 36,891,916 | 36,891,916 | 103.6 | target miss |
    | 16 | 21,553,532 | 31,300,000 | 33,001,664 | 33,001,664 | 105.4 | target miss |
    | **sum** | 632,460,762 | 1,554,200,000 | 1,233,528,744 | 1,233,528,744 | 79.4 | |
    
    ## 2. Section-wise My (N*mm, 탄성 first-yield 기준, 리포트 전용 — 단면 생성에 미반영)
    
    | sec | My initial | My target | My final(soft) | My final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 17,980,672 | 171,000,000 | 54,006,212 | 54,006,212 | 31.6 | target miss |
    | 1 | 17,116,638 | 92,100,000 | 48,733,780 | 48,733,780 | 52.9 | target miss |
    | 2 | 16,273,970 | 49,000,000 | 46,913,824 | 46,913,824 | 95.7 | target miss |
    | 3 | 15,452,663 | 38,200,000 | 43,393,664 | 43,393,664 | 113.6 | target miss |
    | 4 | 14,652,726 | 34,700,000 | 40,140,968 | 40,140,968 | 115.7 | target miss |
    | 5 | 13,874,148 | 34,500,000 | 38,215,264 | 38,215,264 | 110.8 | target miss |
    | 6 | 13,116,937 | 34,800,000 | 35,621,696 | 35,621,696 | 102.4 | target miss |
    | 7 | 12,381,085 | 31,100,000 | 32,244,192 | 32,244,192 | 103.7 | target miss |
    | 8 | 11,666,593 | 30,300,000 | 30,586,926 | 30,586,926 | 100.9 |  |
    | 9 | 10,973,465 | 28,800,000 | 28,497,212 | 28,497,212 | 98.9 |  |
    | 10 | 10,301,690 | 24,900,000 | 25,580,576 | 25,580,576 | 102.7 | target miss |
    | 11 | 9,651,272 | 25,000,000 | 24,907,078 | 24,907,078 | 99.6 |  |
    | 12 | 9,022,210 | 14,700,000 | 21,961,468 | 21,961,468 | 149.4 | target miss |
    | 13 | 8,414,504 | 14,500,000 | 19,731,336 | 19,731,336 | 136.1 | target miss |
    | 14 | 7,828,146 | 17,300,000 | 17,301,364 | 17,301,364 | 100.0 |  |
    | 15 | 7,263,140 | 14,900,000 | 15,036,501 | 15,036,501 | 100.9 |  |
    | 16 | 6,719,478 | 13,100,000 | 13,163,010 | 13,163,010 | 100.5 |  |
    | **sum** | 202,689,336 | 668,900,000 | 536,035,071 | 536,035,071 | 80.1 | |
    
    ## 3. Total cross-section area (mm^2, 17-section sum)
    
    | initial | final(soft) | final(hard) | delta(hard-init) |
    |--------:|------------:|------------:|-----------------:|
    | 24,056.9 | 32,360.3 | 32,360.3 | +8,303.5 (+34.5%) |
    
    | 감량 목표 | 22,854.0 mm² (초기 대비 95.0%) |
    | 달성률 | 141.6% (100% 이하면 목표 달성) |
    
    - 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **0개** (임계 0.3 적용 시 구제: 0개)
    
    ## 4. Part thickness (mm, 제조 기준 = pooled t_raw)
    
    | part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |
    |:-----|----------:|:------------------|:--------------------------------|
    | #00(Outer) | 2.300 | 2.300 (전 섹션 공통) | 항상 존재 |
    | #03(Plate) | 1.600 | 2.300 (전 섹션 공통) | 항상 존재 |
    | #06(Inner) | 1.600 | 2.300 (전 섹션 공통) | 항상 존재 |
    | #07(Patch1) | 1.400 | 2.300 (run 1개) | OOOOOOOOOOOOOOOOO (17/17 alive) |
    | #08(Patch2) | 1.600 | 2.300 (run 1개) | OOOOOOOOOOOOOXXXX (13/17 alive) |
    
    [report] 비교 리포트 저장: reports/AI_design_v1_8_report.md
    >>> Stage 4 진입 여부: True
    [done] weights/bpillar_17sec_v1_8.pt, reports/figures/AI_design_v1_8_result.png, reports/figures/AI_design_v1_8_3d.html, reports/AI_design_v1_8_report.md 저장 완료
    
