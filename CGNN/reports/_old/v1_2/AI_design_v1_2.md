    [data] 17-section 그래프(사전 필터링 없음): 노드 1581개, 엣지 2992개
    [target_mp] ['26,444,055', '29,211,881', '31,397,729', '33,087,380', '33,704,378', '32,835,017', '30,467,844', '27,734,809', '26,413,902', '27,985,091', '32,922,019', '39,716,721', '45,526,897', '47,458,156', '44,876,148', '39,505,207', '35,005,895']
    [Stage 0] Static shape/index dry-run 시작...
    [Stage 0] 분할 주입 테스트 통과 - part3이 run 2개로 독립 pooling됨 (그룹 수: 6)
    [Stage 0] GPU 메모리 사용량: 22.4 MB
    [Stage 0] 통과 - 노드 1581개, 엣지 2992개, z_gate shape (17, 5), 5파트 글로벌 두께 불변 + 분할 주입 검증 완료.
    [gradcheck] dMp/dt OK -- autograd=3.4990e+08, FD=3.4976e+08, rel_err=4.09e-04, dMp/dt>0 확인
    [mass] target_area = 24056.9 mm^2 (17-section 합산 초기 단면적)
    [collision v5] 153쌍(섹션x파트쌍) 부호 앵커 산정 완료
    
    ==============================================================================
    [ AI_design_v1_1 ] ASWC Training + Dynamic Splitting | 17 sections | Section-aware gating (17x2) | Stage1<100 Stage2<400 Stage3<1500
    ==============================================================================
    Epoch    0 || loss=1.3225 | MpErr=8.43% | l_continuity=0.0000(w=1.000) | area=24065.6 | g3=1.000(17/17) g4=1.000(17/17) | groups=5
    Epoch   10 || loss=2.5139 | MpErr=15.54% | l_continuity=0.0000(w=1.000) | area=18133.1 | g3=1.000(17/17) g4=1.000(17/17) | groups=5
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    Epoch   20 || loss=9.7131 | MpErr=27.49% | l_continuity=0.0000(w=1.000) | area=28997.1 | g3=0.965(17/17) g4=0.964(17/17) | groups=5
    Epoch   30 || loss=2.4619 | MpErr=7.54% | l_continuity=0.0000(w=1.000) | area=20108.3 | g3=0.963(17/17) g4=0.960(17/17) | groups=5
    Epoch   40 || loss=2.2280 | MpErr=1.25% | l_continuity=0.0000(w=1.000) | area=22277.5 | g3=0.962(17/17) g4=0.958(17/17) | groups=5
    Epoch   50 || loss=2.2053 | MpErr=0.18% | l_continuity=0.0000(w=1.000) | area=21917.5 | g3=0.962(17/17) g4=0.955(17/17) | groups=5
    Epoch   60 || loss=2.2231 | MpErr=4.41% | l_continuity=0.0000(w=1.000) | area=20860.7 | g3=0.961(17/17) g4=0.951(17/17) | groups=5
    Epoch   70 || loss=2.2062 | MpErr=3.77% | l_continuity=0.0000(w=1.000) | area=21007.7 | g3=0.959(17/17) g4=0.946(17/17) | groups=5
    Epoch   80 || loss=2.2363 | MpErr=1.21% | l_continuity=0.0000(w=1.000) | area=22228.9 | g3=0.957(17/17) g4=0.941(17/17) | groups=5
    Epoch   90 || loss=2.1820 | MpErr=2.86% | l_continuity=0.0000(w=1.000) | area=21215.2 | g3=0.955(17/17) g4=0.935(17/17) | groups=5
    Epoch  100 || loss=2.3697 | MpErr=4.69% | l_continuity=0.0000(w=1.000) | area=22429.2 | g3=0.952(17/17) g4=0.927(17/17) | groups=5
    Epoch  110 || loss=2.1367 | MpErr=2.55% | l_continuity=0.0000(w=1.000) | area=20610.7 | g3=0.950(17/17) g4=0.921(17/17) | groups=5
    Epoch  120 || loss=2.1089 | MpErr=2.70% | l_continuity=0.0000(w=1.000) | area=20905.3 | g3=0.947(17/17) g4=0.917(17/17) | groups=5
    Epoch  130 || loss=2.2531 | MpErr=3.78% | l_continuity=0.0000(w=1.000) | area=22093.2 | g3=0.945(17/17) g4=0.912(17/17) | groups=5
    Epoch  140 || loss=2.0998 | MpErr=2.46% | l_continuity=0.0000(w=1.000) | area=20819.2 | g3=0.943(17/17) g4=0.906(17/17) | groups=5
    Epoch  150 || loss=2.1028 | MpErr=2.58% | l_continuity=0.0000(w=1.000) | area=20764.4 | g3=0.939(17/17) g4=0.899(17/17) | groups=5
    Epoch  160 || loss=2.1192 | MpErr=0.21% | l_continuity=0.0006(w=1.000) | area=21273.7 | g3=0.933(17/17) g4=0.889(17/17) | groups=5
    Epoch  170 || loss=2.0942 | MpErr=2.01% | l_continuity=0.0000(w=1.000) | area=20836.4 | g3=0.927(17/17) g4=0.877(17/17) | groups=5
    Epoch  180 || loss=2.1211 | MpErr=0.88% | l_continuity=0.0016(w=1.000) | area=21146.6 | g3=0.919(17/17) g4=0.864(17/17) | groups=5
    Epoch  190 || loss=2.0736 | MpErr=2.33% | l_continuity=0.0003(w=1.000) | area=20570.4 | g3=0.910(17/17) g4=0.848(17/17) | groups=5
    Epoch  200 || loss=2.0966 | MpErr=0.72% | l_continuity=0.0003(w=1.000) | area=20847.7 | g3=0.901(17/17) g4=0.833(17/17) | groups=5
    Epoch  210 || loss=2.0722 | MpErr=1.04% | l_continuity=0.0007(w=1.000) | area=20694.3 | g3=0.891(17/17) g4=0.817(17/17) | groups=5
    Epoch  220 || loss=2.0282 | MpErr=1.41% | l_continuity=0.0007(w=1.000) | area=20657.7 | g3=0.881(17/17) g4=0.800(17/17) | groups=5
    Epoch  230 || loss=2.0373 | MpErr=1.18% | l_continuity=0.0007(w=1.000) | area=20674.8 | g3=0.869(17/17) g4=0.784(17/17) | groups=5
    Epoch  240 || loss=1.9875 | MpErr=1.31% | l_continuity=0.0005(w=1.000) | area=20697.6 | g3=0.855(17/17) g4=0.767(17/17) | groups=5
    Epoch  250 || loss=2.0395 | MpErr=3.18% | l_continuity=0.0009(w=1.000) | area=20544.5 | g3=0.843(17/17) g4=0.751(17/17) | groups=5
    Epoch  260 || loss=2.0091 | MpErr=3.26% | l_continuity=0.0002(w=1.000) | area=20426.3 | g3=0.833(17/17) g4=0.739(17/17) | groups=5
    Epoch  270 || loss=2.0001 | MpErr=0.74% | l_continuity=0.0012(w=1.000) | area=21157.6 | g3=0.822(17/17) g4=0.726(17/17) | groups=5
    Epoch  280 || loss=2.0241 | MpErr=0.74% | l_continuity=0.0005(w=1.000) | area=21133.1 | g3=0.809(17/17) g4=0.710(17/17) | groups=5
    Epoch  290 || loss=1.9946 | MpErr=0.30% | l_continuity=0.0015(w=1.000) | area=20859.6 | g3=0.794(17/17) g4=0.691(17/17) | groups=5
    Epoch  300 || loss=1.9790 | MpErr=0.63% | l_continuity=0.0002(w=1.000) | area=20936.3 | g3=0.776(17/17) g4=0.671(17/17) | groups=5
    Epoch  310 || loss=1.9634 | MpErr=1.36% | l_continuity=0.0004(w=1.000) | area=20741.8 | g3=0.757(17/17) g4=0.649(17/17) | groups=5
    Epoch  320 || loss=1.9394 | MpErr=1.03% | l_continuity=0.0015(w=1.000) | area=20686.7 | g3=0.739(17/17) g4=0.625(17/17) | groups=5
    Epoch  330 || loss=1.9890 | MpErr=1.19% | l_continuity=0.0007(w=0.999) | area=20790.2 | g3=0.720(17/17) g4=0.602(17/17) | groups=5
    Epoch  340 || loss=1.9641 | MpErr=2.02% | l_continuity=0.0009(w=0.998) | area=20371.6 | g3=0.700(17/17) g4=0.581(17/17) | groups=5
    Epoch  350 || loss=1.9357 | MpErr=1.12% | l_continuity=0.0012(w=0.994) | area=20625.3 | g3=0.681(17/17) g4=0.562(17/17) | groups=5
    Epoch  360 || loss=1.9475 | MpErr=1.25% | l_continuity=0.0027(w=0.983) | area=20571.9 | g3=0.663(17/17) g4=0.542(17/17) | groups=5
    Epoch  370 || loss=1.9666 | MpErr=1.15% | l_continuity=0.0024(w=0.955) | area=20504.5 | g3=0.645(17/17) g4=0.520(17/17) | groups=5
    Epoch  380 || loss=1.9544 | MpErr=1.13% | l_continuity=0.0038(w=0.887) | area=20530.2 | g3=0.626(17/17) g4=0.496(17/17) | groups=5
    Epoch  390 || loss=2.0086 | MpErr=1.44% | l_continuity=0.0034(w=0.745) | area=20544.4 | g3=0.608(17/17) g4=0.475(17/17) | groups=5
    Epoch  400 || loss=2.0425 | MpErr=1.27% | l_continuity=0.0196(w=0.525) | area=21132.9 | g3=0.592(17/17) g4=0.454(17/17) | groups=5
    Epoch  410 || loss=2.0754 | MpErr=0.58% | l_continuity=0.0114(w=0.305) | area=21733.1 | g3=0.573(17/17) g4=0.434(17/17) | groups=5
    Epoch  420 || loss=2.0585 | MpErr=0.44% | l_continuity=0.0377(w=0.163) | area=21784.9 | g3=0.552(17/17) g4=0.411(17/17) | groups=5
    Epoch  430 || loss=2.0733 | MpErr=0.84% | l_continuity=0.0536(w=0.095) | area=22049.7 | g3=0.534(17/17) g4=0.389(17/17) | groups=5
    Epoch  440 || loss=2.1789 | MpErr=3.14% | l_continuity=0.0724(w=0.067) | area=22464.1 | g3=0.516(17/17) g4=0.367(17/17) | groups=5
    Epoch  450 || loss=2.2525 | MpErr=2.47% | l_continuity=0.0044(w=0.056) | area=21937.8 | g3=0.502(17/17) g4=0.349(17/17) | groups=5
    Epoch  460 || loss=2.2185 | MpErr=0.79% | l_continuity=0.1250(w=0.052) | area=22503.8 | g3=0.487(17/17) g4=0.329(17/17) | groups=5
    Epoch  470 || loss=2.3007 | MpErr=1.47% | l_continuity=0.1477(w=0.051) | area=23453.5 | g3=0.473(17/17) g4=0.309(17/17) | groups=5
    Epoch  480 || loss=2.3262 | MpErr=1.73% | l_continuity=0.2240(w=0.050) | area=24107.6 | g3=0.460(17/17) g4=0.291(17/17) | groups=5
    [Dynamic Split] epoch 488: DELETED 확정 [[8, 0]] -> candidate run 수 [2, 1] (두께 그룹 6개로 재구성)
    Epoch  490 || loss=2.4870 | MpErr=1.48% | l_continuity=0.4088(w=0.050) | area=24398.5 | g3=0.471(16/17) g4=0.273(17/17) | groups=6
    Epoch  500 || loss=2.5974 | MpErr=1.05% | l_continuity=0.3791(w=0.050) | area=24689.2 | g3=0.461(16/17) g4=0.257(17/17) | groups=6
    Epoch  510 || loss=2.4908 | MpErr=2.50% | l_continuity=0.0679(w=0.050) | area=24527.0 | g3=0.452(16/17) g4=0.241(17/17) | groups=6
    [Dynamic Split] epoch 516: DELETED 확정 [[3, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    Epoch  520 || loss=2.3963 | MpErr=1.10% | l_continuity=0.2678(w=0.050) | area=24246.4 | g3=0.471(15/17) g4=0.224(17/17) | groups=7
    [Dynamic Split] epoch 530: DELETED 확정 [[6, 1]] -> candidate run 수 [3, 2] (두께 그룹 8개로 재구성)
    Epoch  530 || loss=2.4705 | MpErr=1.36% | l_continuity=0.3273(w=0.050) | area=24857.5 | g3=0.465(15/17) g4=0.212(16/17) | groups=8
    Epoch  540 || loss=2.5608 | MpErr=0.21% | l_continuity=0.3749(w=0.050) | area=25490.2 | g3=0.459(15/17) g4=0.200(16/17) | groups=8
    [Dynamic Split] epoch 542: DELETED 확정 [[5, 0]] -> candidate run 수 [4, 2] (두께 그룹 9개로 재구성)
    [Dynamic Split] epoch 544: DELETED 확정 [[2, 0]] -> candidate run 수 [4, 2] (두께 그룹 9개로 재구성)
    [Dynamic Split] epoch 546: DELETED 확정 [[6, 0]] -> candidate run 수 [4, 2] (두께 그룹 9개로 재구성)
    Epoch  550 || loss=2.6463 | MpErr=0.12% | l_continuity=0.4170(w=0.050) | area=25694.9 | g3=0.569(12/17) g4=0.187(16/17) | groups=9
    [Dynamic Split] epoch 560: DELETED 확정 [[4, 0]] -> candidate run 수 [3, 2] (두께 그룹 8개로 재구성)
    Epoch  560 || loss=2.6360 | MpErr=2.93% | l_continuity=0.3042(w=0.050) | area=25910.2 | g3=0.619(11/17) g4=0.175(16/17) | groups=8
    Epoch  570 || loss=2.8063 | MpErr=3.37% | l_continuity=0.5492(w=0.050) | area=25713.7 | g3=0.617(11/17) g4=0.166(16/17) | groups=8
    [Dynamic Split] epoch 576: DELETED 확정 [[7, 0]] -> candidate run 수 [2, 2] (두께 그룹 7개로 재구성)
    Epoch  580 || loss=2.6978 | MpErr=1.01% | l_continuity=0.5293(w=0.050) | area=25904.7 | g3=0.678(10/17) g4=0.157(16/17) | groups=7
    [Dynamic Split] epoch 586: DELETED 확정 [[1, 0], [8, 1]] -> candidate run 수 [2, 3] (두께 그룹 8개로 재구성)
    Epoch  590 || loss=2.6798 | MpErr=0.58% | l_continuity=0.5357(w=0.050) | area=25661.4 | g3=0.751(9/17) g4=0.151(15/17) | groups=8
    Epoch  600 || loss=2.7994 | MpErr=3.41% | l_continuity=0.4388(w=0.050) | area=26340.4 | g3=0.749(9/17) g4=0.142(15/17) | groups=8
    Epoch  610 || loss=3.2709 | MpErr=3.30% | l_continuity=0.7297(w=0.050) | area=26163.5 | g3=0.746(9/17) g4=0.133(15/17) | groups=8
    [Dynamic Split] epoch 619: DELETED 확정 [[0, 1]] -> candidate run 수 [2, 3] (두께 그룹 8개로 재구성)
    Epoch  620 || loss=3.0584 | MpErr=6.25% | l_continuity=0.5727(w=0.050) | area=26571.4 | g3=0.744(9/17) g4=0.134(14/17) | groups=8
    Epoch  630 || loss=2.8684 | MpErr=1.67% | l_continuity=0.4730(w=0.050) | area=26522.7 | g3=0.741(9/17) g4=0.129(14/17) | groups=8
    Epoch  640 || loss=2.8689 | MpErr=0.18% | l_continuity=0.5258(w=0.050) | area=26340.8 | g3=0.736(9/17) g4=0.123(14/17) | groups=8
    [Dynamic Split] epoch 641: DELETED 확정 [[9, 0]] -> candidate run 수 [2, 3] (두께 그룹 8개로 재구성)
    Epoch  650 || loss=2.8724 | MpErr=3.53% | l_continuity=0.4842(w=0.050) | area=26507.3 | g3=0.823(8/17) g4=0.116(14/17) | groups=8
    Epoch  660 || loss=3.1247 | MpErr=1.48% | l_continuity=0.6653(w=0.050) | area=26317.4 | g3=0.817(8/17) g4=0.110(14/17) | groups=8
    Epoch  670 || loss=2.8978 | MpErr=3.45% | l_continuity=0.4279(w=0.050) | area=26567.9 | g3=0.813(8/17) g4=0.106(14/17) | groups=8
    [Dynamic Split] epoch 677: DELETED 확정 [[0, 0]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    Epoch  680 || loss=3.0501 | MpErr=5.11% | l_continuity=0.4814(w=0.050) | area=26749.6 | g3=0.926(7/17) g4=0.102(14/17) | groups=7
    [Dynamic Split] epoch 685: DELETED 확정 [[5, 1]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    Epoch  690 || loss=3.0841 | MpErr=0.68% | l_continuity=0.6360(w=0.050) | area=26529.4 | g3=0.923(7/17) g4=0.104(13/17) | groups=7
    Epoch  700 || loss=2.9837 | MpErr=3.29% | l_continuity=0.4881(w=0.050) | area=26682.8 | g3=0.920(7/17) g4=0.102(13/17) | groups=7
    [Dynamic Split] epoch 703: DELETED 확정 [[1, 1], [2, 1]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    [Dynamic Split] epoch 706: DELETED 확정 [[4, 1]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    Epoch  710 || loss=2.9602 | MpErr=4.20% | l_continuity=0.5698(w=0.050) | area=26711.1 | g3=0.918(7/17) g4=0.118(10/17) | groups=7
    Epoch  720 || loss=2.8987 | MpErr=2.70% | l_continuity=0.6098(w=0.050) | area=26662.7 | g3=0.914(7/17) g4=0.114(10/17) | groups=7
    [Dynamic Split] epoch 727: DELETED 확정 [[7, 1]] -> candidate run 수 [1, 2] (두께 그룹 6개로 재구성)
    Epoch  730 || loss=2.9514 | MpErr=2.03% | l_continuity=0.3612(w=0.050) | area=26645.8 | g3=0.908(7/17) g4=0.119(9/17) | groups=6
    Epoch  740 || loss=2.9620 | MpErr=0.75% | l_continuity=0.5835(w=0.050) | area=26583.5 | g3=0.903(7/17) g4=0.115(9/17) | groups=6
    Epoch  750 || loss=2.8673 | MpErr=1.94% | l_continuity=0.4871(w=0.050) | area=26920.0 | g3=0.897(7/17) g4=0.111(9/17) | groups=6
    Epoch  760 || loss=2.9802 | MpErr=4.81% | l_continuity=0.4200(w=0.050) | area=27135.4 | g3=0.892(7/17) g4=0.107(9/17) | groups=6
    Epoch  770 || loss=2.8862 | MpErr=2.24% | l_continuity=0.4219(w=0.050) | area=27054.9 | g3=0.887(7/17) g4=0.103(9/17) | groups=6
    Epoch  780 || loss=2.9968 | MpErr=2.73% | l_continuity=0.4744(w=0.050) | area=26906.3 | g3=0.881(7/17) g4=0.100(9/17) | groups=6
    Epoch  790 || loss=2.9681 | MpErr=0.73% | l_continuity=0.6164(w=0.050) | area=26679.7 | g3=0.875(7/17) g4=0.098(9/17) | groups=6
    Epoch  800 || loss=2.9610 | MpErr=5.07% | l_continuity=0.3514(w=0.050) | area=27162.6 | g3=0.871(7/17) g4=0.095(9/17) | groups=6
    [Dynamic Split] epoch 803: DELETED 확정 [[3, 1]] -> candidate run 수 [1, 1] (두께 그룹 5개로 재구성)
    Epoch  810 || loss=2.9093 | MpErr=0.98% | l_continuity=0.3629(w=0.050) | area=26929.1 | g3=0.867(7/17) g4=0.103(8/17) | groups=5
    [Dynamic Split] epoch 818: DELETED 확정 [[9, 1]] -> candidate run 수 [1, 1] (두께 그룹 5개로 재구성)
    Epoch  820 || loss=2.9762 | MpErr=3.76% | l_continuity=0.3017(w=0.050) | area=26962.3 | g3=0.865(7/17) g4=0.106(7/17) | groups=5
    Epoch  830 || loss=2.8457 | MpErr=2.85% | l_continuity=0.3348(w=0.050) | area=26973.5 | g3=0.862(7/17) g4=0.103(7/17) | groups=5
    Epoch  840 || loss=2.9781 | MpErr=3.87% | l_continuity=0.2743(w=0.050) | area=27056.1 | g3=0.857(7/17) g4=0.099(7/17) | groups=5
    Epoch  850 || loss=3.0965 | MpErr=3.66% | l_continuity=0.5220(w=0.050) | area=26944.6 | g3=0.853(7/17) g4=0.096(7/17) | groups=5
    Epoch  860 || loss=2.9437 | MpErr=2.28% | l_continuity=0.2768(w=0.050) | area=26994.6 | g3=0.849(7/17) g4=0.093(7/17) | groups=5
    Epoch  870 || loss=3.0387 | MpErr=1.47% | l_continuity=0.2409(w=0.050) | area=26913.0 | g3=0.844(7/17) g4=0.089(7/17) | groups=5
    Epoch  880 || loss=2.8687 | MpErr=1.38% | l_continuity=0.2503(w=0.050) | area=26923.5 | g3=0.840(7/17) g4=0.086(7/17) | groups=5
    Epoch  890 || loss=2.8596 | MpErr=2.87% | l_continuity=0.2435(w=0.050) | area=26936.9 | g3=0.835(7/17) g4=0.082(7/17) | groups=5
    Epoch  900 || loss=2.8840 | MpErr=2.94% | l_continuity=0.2329(w=0.050) | area=26978.1 | g3=0.827(7/17) g4=0.077(7/17) | groups=5
    Epoch  910 || loss=2.9443 | MpErr=0.43% | l_continuity=0.2577(w=0.050) | area=26715.8 | g3=0.823(7/17) g4=0.073(7/17) | groups=5
    Epoch  920 || loss=2.9132 | MpErr=3.74% | l_continuity=0.3157(w=0.050) | area=26929.7 | g3=0.819(7/17) g4=0.070(7/17) | groups=5
    [Dynamic Split] epoch 922: DELETED 확정 [[11, 1]] -> candidate run 수 [1, 2] (두께 그룹 6개로 재구성)
    Epoch  930 || loss=2.8871 | MpErr=0.62% | l_continuity=0.3029(w=0.050) | area=27028.4 | g3=0.815(7/17) g4=0.072(6/17) | groups=6
    Epoch  940 || loss=2.8325 | MpErr=3.45% | l_continuity=0.2727(w=0.050) | area=27031.1 | g3=0.811(7/17) g4=0.069(6/17) | groups=6
    Epoch  950 || loss=2.8602 | MpErr=0.68% | l_continuity=0.2453(w=0.050) | area=26818.7 | g3=0.806(7/17) g4=0.067(6/17) | groups=6
    Epoch  960 || loss=2.9061 | MpErr=3.99% | l_continuity=0.2311(w=0.050) | area=26974.0 | g3=0.803(7/17) g4=0.065(6/17) | groups=6
    [Dynamic Split] epoch 969: DELETED 확정 [[13, 1]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    Epoch  970 || loss=3.2544 | MpErr=6.44% | l_continuity=0.3138(w=0.050) | area=27081.8 | g3=0.800(7/17) g4=0.063(5/17) | groups=7
    Epoch  980 || loss=2.8027 | MpErr=1.84% | l_continuity=0.2418(w=0.050) | area=26943.1 | g3=0.795(7/17) g4=0.062(5/17) | groups=7
    Epoch  990 || loss=2.9164 | MpErr=2.95% | l_continuity=0.2794(w=0.050) | area=27108.3 | g3=0.790(7/17) g4=0.059(5/17) | groups=7
    Epoch 1000 || loss=2.8819 | MpErr=4.16% | l_continuity=0.2564(w=0.050) | area=26784.8 | g3=0.786(7/17) g4=0.057(5/17) | groups=7
    Epoch 1010 || loss=2.8474 | MpErr=3.84% | l_continuity=0.3073(w=0.050) | area=26758.2 | g3=0.782(7/17) g4=0.055(5/17) | groups=7
    [Dynamic Split] epoch 1017: DELETED 확정 [[15, 1]] -> candidate run 수 [1, 4] (두께 그룹 8개로 재구성)
    Epoch 1020 || loss=2.7863 | MpErr=2.13% | l_continuity=0.2465(w=0.050) | area=26979.6 | g3=0.779(7/17) g4=0.051(4/17) | groups=8
    Epoch 1030 || loss=2.8735 | MpErr=3.57% | l_continuity=0.2454(w=0.050) | area=27046.5 | g3=0.775(7/17) g4=0.049(4/17) | groups=8
    Epoch 1040 || loss=2.8884 | MpErr=0.13% | l_continuity=0.2330(w=0.050) | area=26852.1 | g3=0.770(7/17) g4=0.048(4/17) | groups=8
    Epoch 1050 || loss=2.7476 | MpErr=1.91% | l_continuity=0.2382(w=0.050) | area=26810.9 | g3=0.767(7/17) g4=0.046(4/17) | groups=8
    Epoch 1060 || loss=2.8020 | MpErr=1.15% | l_continuity=0.2589(w=0.050) | area=26943.0 | g3=0.761(7/17) g4=0.044(4/17) | groups=8
    Epoch 1070 || loss=2.8968 | MpErr=0.11% | l_continuity=0.2718(w=0.050) | area=26860.6 | g3=0.754(7/17) g4=0.040(4/17) | groups=8
    Epoch 1080 || loss=2.7665 | MpErr=2.34% | l_continuity=0.2653(w=0.050) | area=26966.3 | g3=0.748(7/17) g4=0.037(4/17) | groups=8
    Epoch 1090 || loss=2.7972 | MpErr=0.20% | l_continuity=0.2429(w=0.050) | area=26700.1 | g3=0.744(7/17) g4=0.034(4/17) | groups=8
    Epoch 1100 || loss=2.8563 | MpErr=0.22% | l_continuity=0.2569(w=0.050) | area=26807.6 | g3=0.739(7/17) g4=0.031(4/17) | groups=8
    Epoch 1110 || loss=2.7405 | MpErr=2.17% | l_continuity=0.2609(w=0.050) | area=26785.5 | g3=0.737(7/17) g4=0.029(4/17) | groups=8
    [Dynamic Split] epoch 1112: DELETED 확정 [[14, 1]] -> candidate run 수 [1, 3] (두께 그룹 7개로 재구성)
    Epoch 1120 || loss=2.7581 | MpErr=0.43% | l_continuity=0.2639(w=0.050) | area=26648.2 | g3=0.735(7/17) g4=0.025(3/17) | groups=7
    [Dynamic Split] epoch 1125: DELETED 확정 [[10, 1]] -> candidate run 수 [1, 2] (두께 그룹 6개로 재구성)
    Epoch 1130 || loss=2.7377 | MpErr=2.84% | l_continuity=0.2392(w=0.050) | area=26782.4 | g3=0.732(7/17) g4=0.023(2/17) | groups=6
    Epoch 1140 || loss=2.6904 | MpErr=2.05% | l_continuity=0.2428(w=0.050) | area=26696.7 | g3=0.727(7/17) g4=0.022(2/17) | groups=6
    [Dynamic Split] epoch 1147: DELETED 확정 [[16, 1]] -> candidate run 수 [1, 1] (두께 그룹 5개로 재구성)
    Epoch 1150 || loss=2.8314 | MpErr=1.17% | l_continuity=0.2418(w=0.050) | area=26591.8 | g3=0.724(7/17) g4=0.022(1/17) | groups=5
    Epoch 1160 || loss=3.0616 | MpErr=0.77% | l_continuity=0.2754(w=0.050) | area=26790.7 | g3=0.719(7/17) g4=0.019(1/17) | groups=5
    Epoch 1170 || loss=2.7099 | MpErr=1.34% | l_continuity=0.2631(w=0.050) | area=26602.8 | g3=0.715(7/17) g4=0.017(1/17) | groups=5
    Epoch 1180 || loss=2.7031 | MpErr=1.82% | l_continuity=0.2986(w=0.050) | area=26664.2 | g3=0.712(7/17) g4=0.015(1/17) | groups=5
    Epoch 1190 || loss=2.7663 | MpErr=2.58% | l_continuity=0.3192(w=0.050) | area=26561.8 | g3=0.708(7/17) g4=0.013(1/17) | groups=5
    Epoch 1200 || loss=2.6901 | MpErr=1.27% | l_continuity=0.2979(w=0.050) | area=26315.3 | g3=0.705(7/17) g4=0.011(1/17) | groups=5
    Epoch 1210 || loss=2.6911 | MpErr=3.48% | l_continuity=0.2918(w=0.050) | area=26237.2 | g3=0.700(7/17) g4=0.010(1/17) | groups=5
    Epoch 1220 || loss=2.6506 | MpErr=2.41% | l_continuity=0.2678(w=0.050) | area=26374.2 | g3=0.696(7/17) g4=0.009(1/17) | groups=5
    Epoch 1230 || loss=2.7307 | MpErr=3.01% | l_continuity=0.2680(w=0.050) | area=26639.6 | g3=0.692(7/17) g4=0.006(1/17) | groups=5
    Epoch 1240 || loss=2.7117 | MpErr=2.88% | l_continuity=0.2752(w=0.050) | area=26616.9 | g3=0.687(7/17) g4=0.004(1/17) | groups=5
    Epoch 1250 || loss=2.6982 | MpErr=2.33% | l_continuity=0.2694(w=0.050) | area=26690.1 | g3=0.681(7/17) g4=0.003(1/17) | groups=5
    [Dynamic Split] epoch 1260: DELETED 확정 [[12, 1]] -> candidate run 수 [1, 0] (두께 그룹 4개로 재구성)
    Epoch 1260 || loss=2.6354 | MpErr=1.39% | l_continuity=0.2492(w=0.050) | area=26450.9 | g3=0.674(7/17) g4=0.000(0/17) | groups=4
    Epoch 1270 || loss=2.7555 | MpErr=0.61% | l_continuity=0.2680(w=0.050) | area=26507.4 | g3=0.666(7/17) g4=0.000(0/17) | groups=4
    Epoch 1280 || loss=2.6662 | MpErr=2.33% | l_continuity=0.2598(w=0.050) | area=26453.7 | g3=0.659(7/17) g4=0.000(0/17) | groups=4
    Epoch 1290 || loss=2.6228 | MpErr=2.04% | l_continuity=0.2793(w=0.050) | area=26377.3 | g3=0.653(7/17) g4=0.000(0/17) | groups=4
    Epoch 1300 || loss=2.6921 | MpErr=1.62% | l_continuity=0.2798(w=0.050) | area=26547.0 | g3=0.648(7/17) g4=0.000(0/17) | groups=4
    Epoch 1310 || loss=2.6909 | MpErr=0.20% | l_continuity=0.2559(w=0.050) | area=26375.1 | g3=0.643(7/17) g4=0.000(0/17) | groups=4
    Epoch 1320 || loss=2.6916 | MpErr=3.59% | l_continuity=0.2623(w=0.050) | area=26462.8 | g3=0.640(7/17) g4=0.000(0/17) | groups=4
    Epoch 1330 || loss=2.6169 | MpErr=1.71% | l_continuity=0.2552(w=0.050) | area=26501.8 | g3=0.637(7/17) g4=0.000(0/17) | groups=4
    Epoch 1340 || loss=2.6794 | MpErr=1.52% | l_continuity=0.2652(w=0.050) | area=26458.8 | g3=0.635(7/17) g4=0.000(0/17) | groups=4
    Epoch 1350 || loss=2.7367 | MpErr=2.51% | l_continuity=0.3447(w=0.050) | area=26415.0 | g3=0.633(7/17) g4=0.000(0/17) | groups=4
    Epoch 1360 || loss=2.7537 | MpErr=0.07% | l_continuity=0.2999(w=0.050) | area=26217.2 | g3=0.631(7/17) g4=0.000(0/17) | groups=4
    Epoch 1370 || loss=2.7141 | MpErr=0.08% | l_continuity=0.2818(w=0.050) | area=26262.6 | g3=0.629(7/17) g4=0.000(0/17) | groups=4
    Epoch 1380 || loss=2.6584 | MpErr=2.52% | l_continuity=0.2646(w=0.050) | area=26379.2 | g3=0.626(7/17) g4=0.000(0/17) | groups=4
    Epoch 1390 || loss=2.6497 | MpErr=1.45% | l_continuity=0.2714(w=0.050) | area=26375.8 | g3=0.623(7/17) g4=0.000(0/17) | groups=4
    Epoch 1400 || loss=2.7703 | MpErr=4.23% | l_continuity=0.2449(w=0.050) | area=26481.8 | g3=0.617(7/17) g4=0.000(0/17) | groups=4
    Epoch 1410 || loss=2.6971 | MpErr=0.75% | l_continuity=0.3127(w=0.050) | area=26307.4 | g3=0.612(7/17) g4=0.000(0/17) | groups=4
    Epoch 1420 || loss=2.9534 | MpErr=1.87% | l_continuity=0.2770(w=0.050) | area=26287.1 | g3=0.608(7/17) g4=0.000(0/17) | groups=4
    Epoch 1430 || loss=2.8083 | MpErr=3.97% | l_continuity=0.2659(w=0.050) | area=26484.3 | g3=0.604(7/17) g4=0.000(0/17) | groups=4
    Epoch 1440 || loss=2.6909 | MpErr=3.03% | l_continuity=0.2958(w=0.050) | area=26399.4 | g3=0.601(7/17) g4=0.000(0/17) | groups=4
    Epoch 1450 || loss=3.0171 | MpErr=2.05% | l_continuity=0.2696(w=0.050) | area=26582.5 | g3=0.597(7/17) g4=0.000(0/17) | groups=4
    Epoch 1460 || loss=2.7404 | MpErr=0.41% | l_continuity=0.3594(w=0.050) | area=25817.4 | g3=0.596(7/17) g4=0.000(0/17) | groups=4
    Epoch 1470 || loss=2.8657 | MpErr=3.36% | l_continuity=0.3588(w=0.050) | area=26437.8 | g3=0.594(7/17) g4=0.000(0/17) | groups=4
    Epoch 1480 || loss=2.6793 | MpErr=0.65% | l_continuity=0.2623(w=0.050) | area=26304.5 | g3=0.591(7/17) g4=0.000(0/17) | groups=4
    Epoch 1490 || loss=2.7838 | MpErr=3.85% | l_continuity=0.2697(w=0.050) | area=26649.7 | g3=0.588(7/17) g4=0.000(0/17) | groups=4
    Epoch 1499 || loss=2.7651 | MpErr=0.32% | l_continuity=0.3095(w=0.050) | area=26548.7 | g3=0.584(7/17) g4=0.000(0/17) | groups=4
    
    [Dynamic Split] 분할 이벤트 epoch: [488, 516, 530, 542, 544, 546, 560, 576, 586, 619, 641, 677, 685, 703, 706, 727, 803, 818, 922, 969, 1017, 1112, 1125, 1147, 1260], 최종 두께 그룹 4개
    [viz] 학습 결과 dashboard 저장: reports/figures/AI_design_v1_2_result.png
    [viz] 인터랙티브 3D 저장: reports/figures/AI_design_v1_2_3d.html
    [report] hard existence 판정용 temperature=0.3000 (epoch 1499 기준)
    
    # AI_design_v1_2 — Initial vs Final Design Property Report
    
    (hard existence 판정 temperature=0.3000 — 학습 종료 시점 어닐링 온도와 일치)
    
    ## 1. Section-wise Mp (N*mm)
    
    | sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 55,297,616 | 26,444,055 | 30,700,700 | 30,700,700 | 116.1 | target miss |
    | 1 | 52,739,376 | 29,211,881 | 30,583,778 | 30,583,778 | 104.7 | target miss |
    | 2 | 50,241,736 | 31,397,729 | 31,791,412 | 31,791,412 | 101.3 |  |
    | 3 | 47,804,696 | 33,087,380 | 33,142,812 | 33,142,812 | 100.2 |  |
    | 4 | 45,428,256 | 33,704,378 | 33,629,476 | 33,629,476 | 99.8 |  |
    | 5 | 43,112,400 | 32,835,017 | 32,717,508 | 32,717,508 | 99.6 |  |
    | 6 | 40,857,144 | 30,467,844 | 30,405,002 | 30,405,002 | 99.8 |  |
    | 7 | 38,662,480 | 27,734,809 | 27,902,768 | 27,902,768 | 100.6 |  |
    | 8 | 36,528,412 | 26,413,902 | 26,557,224 | 26,557,224 | 100.5 |  |
    | 9 | 34,454,944 | 27,985,091 | 27,512,224 | 27,512,224 | 98.3 |  |
    | 10 | 32,442,064 | 32,922,019 | 32,181,280 | 31,993,256 | 97.2 | target miss |
    | 11 | 30,489,774 | 39,716,721 | 39,063,688 | 37,735,384 | 95.0 | soft-hard gap 3.4% target miss |
    | 12 | 28,598,076 | 45,526,897 | 45,317,936 | 44,748,352 | 98.3 |  |
    | 13 | 26,766,974 | 47,458,156 | 48,546,000 | 50,044,544 | 105.4 | soft-hard gap 3.1% target miss |
    | 14 | 24,996,464 | 44,876,148 | 45,551,724 | 45,725,744 | 101.9 |  |
    | 15 | 23,286,536 | 39,505,207 | 39,781,040 | 39,903,232 | 101.0 |  |
    | 16 | 21,637,200 | 35,005,895 | 34,937,632 | 34,937,956 | 99.8 |  |
    | **sum** | 633,344,148 | 584,293,129 | 590,322,204 | 590,031,372 | 101.0 | |
    
    ## 2. Total cross-section area (mm^2, 17-section sum)
    
    | initial | final(soft) | final(hard) | delta(hard-init) |
    |--------:|------------:|------------:|-----------------:|
    | 24,056.9 | 26,520.0 | 26,497.8 | +2,440.9 (+10.1%) |
    
    ## 3. Part thickness (mm, 제조 기준 = pooled t_raw)
    
    | part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |
    |:-----|----------:|:------------------|:--------------------------------|
    | #00(Outer) | 2.300 | 2.394 (전 섹션 공통) | 항상 존재 |
    | #03(Plate) | 1.600 | 2.180 (전 섹션 공통) | 항상 존재 |
    | #06(Inner) | 1.600 | 2.095 (전 섹션 공통) | 항상 존재 |
    | #07(Patch1) | 1.400 | 2.131 (run 1개) | XXXXXXXXXXXXXOOOO (4/17 alive) |
    | #08(Patch2) | 1.600 | 2.152 (run 1개) | XXXXXXXXXXXXXXXXX (0/17 alive) |
    
    > [WARNING] soft-hard Mp 괴리 >2%인 섹션 2개 — 게이트가 아직 0/1로 수렴하지 않았습니다. 추가 fine-tuning epoch를 권장합니다.
    
    [report] 비교 리포트 저장: reports/AI_design_v1_2_report.md
    [done] weights/bpillar_17sec_v1_2.pt, reports/figures/AI_design_v1_2_result.png, reports/figures/AI_design_v1_2_3d.html, reports/AI_design_v1_2_report.md 저장 완료
    
