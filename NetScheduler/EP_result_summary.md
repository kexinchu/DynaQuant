## Qwen-235B-A22B EP=8
```shell
=== 测试结果分析 ===
--- Query长度测试结果 ---
Query长度 512:
  TTFT: 平均=5273.18ms, 中位数=5101.84ms, 标准差=466.12ms
  TPOT: 平均=657.99ms, 中位数=711.93ms, 标准差=100.09ms
  Overall: 平均=17577.27ms, 中位数=17006.13ms, 标准差=1553.75ms
Query长度 1024:
  TTFT: 平均=5287.25ms, 中位数=5203.42ms, 标准差=309.10ms
  TPOT: 平均=538.73ms, 中位数=500.37ms, 标准差=121.44ms
  Overall: 平均=17624.17ms, 中位数=17344.75ms, 标准差=1030.34ms
Query长度 2048:
  TTFT: 平均=5032.02ms, 中位数=5010.73ms, 标准差=43.43ms
  TPOT: 平均=454.94ms, 中位数=448.18ms, 标准差=81.52ms
  Overall: 平均=16773.39ms, 中位数=16702.44ms, 标准差=144.77ms
Query长度 4096:
  TTFT: 平均=5059.70ms, 中位数=5058.26ms, 标准差=7.09ms
  TPOT: 平均=575.48ms, 中位数=561.27ms, 标准差=144.88ms
  Overall: 平均=16865.67ms, 中位数=16860.86ms, 标准差=23.64ms
Query长度 8192:
  TTFT: 平均=5151.50ms, 中位数=5133.91ms, 标准差=46.67ms
  TPOT: 平均=522.22ms, 中位数=519.07ms, 标准差=78.73ms
  Overall: 平均=17171.67ms, 中位数=17113.04ms, 标准差=155.56ms
Query长度 16384:
  TTFT: 平均=5471.45ms, 中位数=5454.44ms, 标准差=42.45ms
  TPOT: 平均=568.73ms, 中位数=538.16ms, 标准差=105.62ms
  Overall: 平均=18238.15ms, 中位数=18181.46ms, 标准差=141.50ms

--- QPS测试结果 ---
QPS 2:
  TTFT: 平均=5696.05ms, 中位数=5833.68ms, 标准差=359.94ms
  TPOT: 平均=699.69ms, 中位数=680.05ms, 标准差=153.70ms
  Overall: 平均=18986.83ms, 中位数=19445.60ms, 标准差=1199.78ms
QPS 4:
  TTFT: 平均=6419.11ms, 中位数=6713.48ms, 标准差=740.87ms
  TPOT: 平均=692.32ms, 中位数=743.81ms, 标准差=169.58ms
  Overall: 平均=21397.03ms, 中位数=22378.26ms, 标准差=2469.57ms
QPS 8:
  TTFT: 平均=7167.27ms, 中位数=7180.68ms, 标准差=44.06ms
  TPOT: 平均=768.29ms, 中位数=761.59ms, 标准差=88.55ms
  Overall: 平均=23890.91ms, 中位数=23935.61ms, 标准差=146.87ms
QPS 16:
  TTFT: 平均=7094.25ms, 中位数=7088.24ms, 标准差=22.86ms
  TPOT: 平均=677.49ms, 中位数=665.16ms, 标准差=61.43ms
  Overall: 平均=23647.49ms, 中位数=23627.48ms, 标准差=76.21ms
QPS 32:
  TTFT: 平均=7138.71ms, 中位数=7142.30ms, 标准差=21.67ms
  TPOT: 平均=879.83ms, 中位数=795.75ms, 标准差=118.36ms
  Overall: 平均=23795.69ms, 中位数=23807.68ms, 标准差=72.23ms
QPS 64:
  TTFT: 平均=7098.55ms, 中位数=7107.07ms, 标准差=24.65ms
  TPOT: 平均=762.06ms, 中位数=754.80ms, 标准差=47.63ms
  Overall: 平均=23661.82ms, 中位数=23690.22ms, 标准差=82.18ms
QPS 128:
  TTFT: 平均=7153.05ms, 中位数=7163.73ms, 标准差=26.83ms
  TPOT: 平均=684.81ms, 中位数=696.24ms, 标准差=75.22ms
  Overall: 平均=23843.49ms, 中位数=23879.11ms, 标准差=89.44ms
```

# Qwen-235B-A22B EP=2
```shell
--- 测试组1: 不同query长度 (QPS=1) ---
测试query长度: 512
  请求 1: TTFT=4849.44ms, TPOT=235.74ms, Overall=16164.79ms
  请求 2: TTFT=4840.75ms, TPOT=289.62ms, Overall=16135.84ms
  请求 3: TTFT=4835.65ms, TPOT=304.95ms, Overall=16118.84ms
  请求 4: TTFT=4820.63ms, TPOT=340.85ms, Overall=16068.75ms
  请求 5: TTFT=4879.55ms, TPOT=291.94ms, Overall=16265.17ms
测试query长度: 1024
  请求 1: TTFT=4835.44ms, TPOT=268.64ms, Overall=16118.13ms
  请求 2: TTFT=4835.92ms, TPOT=245.30ms, Overall=16119.72ms
  请求 3: TTFT=4817.36ms, TPOT=244.36ms, Overall=16057.86ms
  请求 4: TTFT=4815.67ms, TPOT=340.50ms, Overall=16052.23ms
  请求 5: TTFT=4811.71ms, TPOT=255.17ms, Overall=16039.04ms
测试query长度: 2048
  请求 1: TTFT=4847.01ms, TPOT=314.16ms, Overall=16156.71ms
  请求 2: TTFT=4872.62ms, TPOT=232.03ms, Overall=16242.08ms
  请求 3: TTFT=4879.25ms, TPOT=258.75ms, Overall=16264.17ms
  请求 4: TTFT=4858.66ms, TPOT=251.93ms, Overall=16195.52ms
  请求 5: TTFT=4916.01ms, TPOT=318.63ms, Overall=16386.70ms
测试query长度: 4096
  请求 1: TTFT=4944.46ms, TPOT=256.38ms, Overall=16481.55ms
  请求 2: TTFT=4939.99ms, TPOT=261.97ms, Overall=16466.65ms
  请求 3: TTFT=4955.56ms, TPOT=282.02ms, Overall=16518.53ms
  请求 4: TTFT=5009.83ms, TPOT=299.73ms, Overall=16699.44ms
  请求 5: TTFT=4952.94ms, TPOT=251.24ms, Overall=16509.79ms
测试query长度: 8192
  请求 1: TTFT=5420.46ms, TPOT=351.33ms, Overall=18068.21ms
  请求 2: TTFT=5362.96ms, TPOT=305.21ms, Overall=17876.53ms
  请求 3: TTFT=5372.36ms, TPOT=321.42ms, Overall=17907.86ms
  请求 4: TTFT=5346.87ms, TPOT=277.24ms, Overall=17822.89ms
  请求 5: TTFT=5379.89ms, TPOT=380.40ms, Overall=17932.97ms
测试query长度: 16384
  请求 1: TTFT=6112.54ms, TPOT=356.56ms, Overall=20375.14ms
  请求 2: TTFT=6103.18ms, TPOT=316.46ms, Overall=20343.93ms
  请求 3: TTFT=6142.63ms, TPOT=398.13ms, Overall=20475.45ms
  请求 4: TTFT=6080.91ms, TPOT=363.82ms, Overall=20269.69ms
  请求 5: TTFT=6086.59ms, TPOT=417.71ms, Overall=20288.65ms

--- 测试组2: 不同QPS (query长度=256) ---
测试QPS: 2
  请求: TTFT=5895.68ms, TPOT=416.87ms, Overall=19652.27ms
  请求: TTFT=5850.57ms, TPOT=379.20ms, Overall=19501.91ms
  请求: TTFT=5883.16ms, TPOT=319.24ms, Overall=19610.52ms
  请求: TTFT=5827.70ms, TPOT=412.06ms, Overall=19425.68ms
  请求: TTFT=5069.71ms, TPOT=288.52ms, Overall=16899.05ms
测试QPS: 4
  请求: TTFT=6793.44ms, TPOT=396.28ms, Overall=22644.81ms
  请求: TTFT=6780.51ms, TPOT=405.67ms, Overall=22601.69ms
  请求: TTFT=6705.39ms, TPOT=355.59ms, Overall=22351.29ms
  请求: TTFT=6686.26ms, TPOT=520.04ms, Overall=22287.55ms
  请求: TTFT=5078.63ms, TPOT=269.32ms, Overall=16928.76ms
测试QPS: 8
  请求: TTFT=8462.44ms, TPOT=359.01ms, Overall=28208.14ms
  请求: TTFT=8277.89ms, TPOT=508.29ms, Overall=27592.98ms
  请求: TTFT=8353.16ms, TPOT=541.41ms, Overall=27843.86ms
  请求: TTFT=8504.54ms, TPOT=551.22ms, Overall=28348.48ms
  请求: TTFT=8315.89ms, TPOT=669.09ms, Overall=27719.63ms
  请求: TTFT=8467.06ms, TPOT=519.91ms, Overall=28223.54ms
  请求: TTFT=8391.79ms, TPOT=383.94ms, Overall=27972.63ms
  请求: TTFT=8429.57ms, TPOT=447.02ms, Overall=28098.58ms
测试QPS: 16
  请求: TTFT=11519.06ms, TPOT=625.07ms, Overall=38396.88ms
  请求: TTFT=11572.81ms, TPOT=600.07ms, Overall=38576.02ms
  请求: TTFT=11496.70ms, TPOT=687.84ms, Overall=38322.34ms
  请求: TTFT=11477.67ms, TPOT=686.70ms, Overall=38258.91ms
  请求: TTFT=11362.89ms, TPOT=646.67ms, Overall=37876.30ms
  请求: TTFT=11611.14ms, TPOT=732.23ms, Overall=38703.79ms
  请求: TTFT=11401.59ms, TPOT=682.15ms, Overall=38005.29ms
  请求: TTFT=11419.98ms, TPOT=619.69ms, Overall=38066.61ms
  请求: TTFT=11592.51ms, TPOT=676.23ms, Overall=38641.69ms
  请求: TTFT=11344.84ms, TPOT=735.31ms, Overall=37816.14ms
  请求: TTFT=11383.08ms, TPOT=885.35ms, Overall=37943.61ms
  请求: TTFT=11516.79ms, TPOT=866.86ms, Overall=38389.32ms
  请求: TTFT=11459.45ms, TPOT=685.61ms, Overall=38198.17ms
  请求: TTFT=11536.16ms, TPOT=791.70ms, Overall=38453.86ms
  请求: TTFT=11440.60ms, TPOT=667.37ms, Overall=38135.34ms
  请求: TTFT=11555.73ms, TPOT=691.37ms, Overall=38519.09ms
测试QPS: 32
  请求: TTFT=17528.38ms, TPOT=1239.38ms, Overall=58427.93ms
  请求: TTFT=17431.93ms, TPOT=884.23ms, Overall=58106.45ms
  请求: TTFT=17681.47ms, TPOT=937.65ms, Overall=58938.22ms
  请求: TTFT=17643.04ms, TPOT=1055.57ms, Overall=58810.14ms
  请求: TTFT=17402.81ms, TPOT=944.34ms, Overall=58009.36ms
  请求: TTFT=17491.10ms, TPOT=1103.04ms, Overall=58303.68ms
  请求: TTFT=17422.85ms, TPOT=1016.33ms, Overall=58076.18ms
  请求: TTFT=17692.26ms, TPOT=1058.51ms, Overall=58974.19ms
  请求: TTFT=17411.76ms, TPOT=990.91ms, Overall=58039.21ms
  请求: TTFT=17451.90ms, TPOT=798.45ms, Overall=58173.00ms
  请求: TTFT=17460.17ms, TPOT=1018.51ms, Overall=58200.56ms
  请求: TTFT=17634.73ms, TPOT=1055.07ms, Overall=58782.45ms
  请求: TTFT=17480.93ms, TPOT=1315.77ms, Overall=58269.75ms
  请求: TTFT=17567.50ms, TPOT=1138.63ms, Overall=58558.33ms
  请求: TTFT=17587.00ms, TPOT=1243.53ms, Overall=58623.33ms
  请求: TTFT=17595.20ms, TPOT=954.78ms, Overall=58650.65ms
  请求: TTFT=17673.91ms, TPOT=1057.41ms, Overall=58913.02ms
  请求: TTFT=17576.08ms, TPOT=1108.40ms, Overall=58586.94ms
  请求: TTFT=17548.51ms, TPOT=853.05ms, Overall=58495.04ms
  请求: TTFT=17654.67ms, TPOT=895.53ms, Overall=58848.89ms
  请求: TTFT=17614.87ms, TPOT=1325.85ms, Overall=58716.24ms
  请求: TTFT=17518.89ms, TPOT=929.03ms, Overall=58396.29ms
  请求: TTFT=17443.15ms, TPOT=830.63ms, Overall=58143.82ms
  请求: TTFT=17529.85ms, TPOT=1105.49ms, Overall=58432.83ms
  请求: TTFT=17500.97ms, TPOT=1047.07ms, Overall=58336.57ms
  请求: TTFT=17607.17ms, TPOT=1027.08ms, Overall=58690.57ms
  请求: TTFT=17539.97ms, TPOT=1049.40ms, Overall=58466.56ms
  请求: TTFT=17663.05ms, TPOT=1212.17ms, Overall=58876.82ms
  请求: TTFT=17472.75ms, TPOT=948.13ms, Overall=58242.49ms
  请求: TTFT=17511.51ms, TPOT=1104.33ms, Overall=58371.71ms
  请求: TTFT=17559.48ms, TPOT=1170.63ms, Overall=58531.60ms
  请求: TTFT=17627.11ms, TPOT=1468.93ms, Overall=58757.03ms
测试QPS: 64
  请求: TTFT=29753.88ms, TPOT=1826.99ms, Overall=99179.60ms
  请求: TTFT=29835.16ms, TPOT=1657.51ms, Overall=99450.52ms
  请求: TTFT=29949.64ms, TPOT=1625.17ms, Overall=99832.13ms
  请求: TTFT=30019.30ms, TPOT=1843.29ms, Overall=100064.33ms
  请求: TTFT=29959.87ms, TPOT=2255.04ms, Overall=99866.24ms
  请求: TTFT=29969.95ms, TPOT=1487.87ms, Overall=99899.84ms
  请求: TTFT=29989.84ms, TPOT=1749.41ms, Overall=99966.14ms
  请求: TTFT=29781.47ms, TPOT=1510.65ms, Overall=99271.55ms
  请求: TTFT=29940.91ms, TPOT=1663.38ms, Overall=99803.03ms
  请求: TTFT=29910.92ms, TPOT=1586.19ms, Overall=99703.06ms
  请求: TTFT=29851.31ms, TPOT=1658.41ms, Overall=99504.35ms
  请求: TTFT=29827.68ms, TPOT=1618.56ms, Overall=99425.62ms
  请求: TTFT=29876.44ms, TPOT=2403.85ms, Overall=99588.15ms
  请求: TTFT=29786.15ms, TPOT=1654.79ms, Overall=99287.18ms
  请求: TTFT=29810.94ms, TPOT=1391.18ms, Overall=99369.81ms
  请求: TTFT=29757.16ms, TPOT=2169.79ms, Overall=99190.55ms
  请求: TTFT=29821.05ms, TPOT=1932.85ms, Overall=99403.51ms
  请求: TTFT=29931.77ms, TPOT=1624.20ms, Overall=99772.58ms
  请求: TTFT=29752.41ms, TPOT=1826.90ms, Overall=99174.71ms
  请求: TTFT=29842.37ms, TPOT=1450.67ms, Overall=99474.55ms
  请求: TTFT=29867.28ms, TPOT=1620.71ms, Overall=99557.60ms
  请求: TTFT=29957.15ms, TPOT=1889.19ms, Overall=99857.16ms
  请求: TTFT=29902.80ms, TPOT=1661.27ms, Overall=99675.99ms
  请求: TTFT=30017.43ms, TPOT=1892.99ms, Overall=100058.09ms
  请求: TTFT=29897.05ms, TPOT=1245.71ms, Overall=99656.83ms
  请求: TTFT=29982.78ms, TPOT=1457.50ms, Overall=99942.60ms
  请求: TTFT=30027.72ms, TPOT=1523.15ms, Overall=100092.40ms
  请求: TTFT=29998.09ms, TPOT=1458.24ms, Overall=99993.62ms
  请求: TTFT=30032.95ms, TPOT=1844.13ms, Overall=100109.84ms
  请求: TTFT=30003.33ms, TPOT=1707.51ms, Overall=100011.09ms
  请求: TTFT=29798.08ms, TPOT=1655.45ms, Overall=99326.94ms
  请求: TTFT=29804.51ms, TPOT=1655.81ms, Overall=99348.37ms
  请求: TTFT=29967.34ms, TPOT=1997.82ms, Overall=99891.15ms
  请求: TTFT=29894.01ms, TPOT=1788.53ms, Overall=99646.71ms
  请求: TTFT=30038.88ms, TPOT=1752.27ms, Overall=100129.58ms
  请求: TTFT=29929.13ms, TPOT=1204.05ms, Overall=99763.78ms
  请求: TTFT=30013.66ms, TPOT=2000.91ms, Overall=100045.54ms
  请求: TTFT=29873.03ms, TPOT=1483.06ms, Overall=99576.78ms
  请求: TTFT=29779.95ms, TPOT=2171.45ms, Overall=99266.49ms
  请求: TTFT=29795.31ms, TPOT=1738.06ms, Overall=99317.69ms
  请求: TTFT=30047.43ms, TPOT=1669.30ms, Overall=100158.11ms
  请求: TTFT=29949.65ms, TPOT=2055.37ms, Overall=99832.16ms
  请求: TTFT=29770.38ms, TPOT=1877.41ms, Overall=99234.62ms
  请求: TTFT=29740.39ms, TPOT=1613.82ms, Overall=99134.63ms
  请求: TTFT=29890.25ms, TPOT=1788.31ms, Overall=99634.17ms
  请求: TTFT=29920.34ms, TPOT=1790.11ms, Overall=99734.46ms
  请求: TTFT=29865.74ms, TPOT=2111.72ms, Overall=99552.45ms
  请求: TTFT=29925.55ms, TPOT=1837.53ms, Overall=99751.83ms
  请求: TTFT=29776.07ms, TPOT=1543.94ms, Overall=99253.57ms
  请求: TTFT=29985.67ms, TPOT=1706.50ms, Overall=99952.24ms
  请求: TTFT=29763.20ms, TPOT=1736.19ms, Overall=99210.68ms
  请求: TTFT=29909.22ms, TPOT=1550.85ms, Overall=99697.39ms
  请求: TTFT=29811.17ms, TPOT=1879.98ms, Overall=99370.58ms
  请求: TTFT=30010.53ms, TPOT=2000.70ms, Overall=100035.10ms
  请求: TTFT=29851.39ms, TPOT=1547.85ms, Overall=99504.63ms
  请求: TTFT=29836.73ms, TPOT=1698.03ms, Overall=99455.75ms
  请求: TTFT=29861.73ms, TPOT=1741.93ms, Overall=99539.09ms
  请求: TTFT=29747.05ms, TPOT=1416.53ms, Overall=99156.82ms
  请求: TTFT=29752.19ms, TPOT=2393.85ms, Overall=99173.95ms
  请求: TTFT=29941.91ms, TPOT=1791.40ms, Overall=99806.36ms
  请求: TTFT=29822.34ms, TPOT=1312.93ms, Overall=99407.79ms
  请求: TTFT=30041.36ms, TPOT=2124.14ms, Overall=100137.86ms
  请求: TTFT=29981.94ms, TPOT=1748.95ms, Overall=99939.79ms
  请求: TTFT=29887.82ms, TPOT=1700.93ms, Overall=99626.07ms
测试QPS: 128
  请求: TTFT=53884.73ms, TPOT=2675.13ms, Overall=179615.78ms
  请求: TTFT=54290.22ms, TPOT=2815.05ms, Overall=180967.39ms
  请求: TTFT=54415.92ms, TPOT=3431.63ms, Overall=181386.41ms
  请求: TTFT=54183.91ms, TPOT=3612.26ms, Overall=180613.02ms
  请求: TTFT=54221.39ms, TPOT=3162.91ms, Overall=180737.98ms
  请求: TTFT=54218.82ms, TPOT=3243.86ms, Overall=180729.39ms
  请求: TTFT=54326.15ms, TPOT=2880.93ms, Overall=181087.16ms
  请求: TTFT=54260.29ms, TPOT=4220.24ms, Overall=180867.63ms
  请求: TTFT=54186.40ms, TPOT=3010.36ms, Overall=180621.32ms
  请求: TTFT=54225.25ms, TPOT=3329.62ms, Overall=180750.82ms
  请求: TTFT=54177.14ms, TPOT=3718.04ms, Overall=180590.46ms
  请求: TTFT=54161.80ms, TPOT=3240.45ms, Overall=180539.34ms
  请求: TTFT=54230.78ms, TPOT=3329.96ms, Overall=180769.28ms
  请求: TTFT=54215.10ms, TPOT=3720.64ms, Overall=180716.98ms
  请求: TTFT=54108.53ms, TPOT=3322.45ms, Overall=180361.78ms
  请求: TTFT=54246.59ms, TPOT=3087.20ms, Overall=180821.96ms
  请求: TTFT=54146.57ms, TPOT=2871.41ms, Overall=180488.56ms
  请求: TTFT=54106.13ms, TPOT=3825.69ms, Overall=180353.76ms
  请求: TTFT=54278.37ms, TPOT=3724.99ms, Overall=180927.90ms
  请求: TTFT=54299.54ms, TPOT=3519.41ms, Overall=180998.48ms
  请求: TTFT=54374.11ms, TPOT=2699.42ms, Overall=181247.03ms
  请求: TTFT=54335.55ms, TPOT=2641.31ms, Overall=181118.50ms
  请求: TTFT=54133.12ms, TPOT=2631.47ms, Overall=180443.73ms
  请求: TTFT=54144.83ms, TPOT=2938.09ms, Overall=180482.77ms
  请求: TTFT=54412.05ms, TPOT=3255.42ms, Overall=181373.51ms
  请求: TTFT=54395.34ms, TPOT=2820.50ms, Overall=181317.79ms
  请求: TTFT=54151.57ms, TPOT=2938.46ms, Overall=180505.25ms
  请求: TTFT=54415.44ms, TPOT=2701.48ms, Overall=181384.78ms
  请求: TTFT=54299.14ms, TPOT=2815.51ms, Overall=180997.14ms
  请求: TTFT=54345.94ms, TPOT=3251.47ms, Overall=181153.12ms
  请求: TTFT=54103.87ms, TPOT=3945.07ms, Overall=180346.23ms
  请求: TTFT=54393.90ms, TPOT=3172.98ms, Overall=181312.98ms
  请求: TTFT=54325.78ms, TPOT=2755.66ms, Overall=181085.92ms
  请求: TTFT=54135.53ms, TPOT=3157.91ms, Overall=180451.77ms
  请求: TTFT=54360.54ms, TPOT=3624.04ms, Overall=181201.79ms
  请求: TTFT=54088.57ms, TPOT=3078.21ms, Overall=180295.24ms
  请求: TTFT=54172.51ms, TPOT=2808.94ms, Overall=180575.02ms
  请求: TTFT=54413.31ms, TPOT=3734.25ms, Overall=181377.70ms
  请求: TTFT=54354.92ms, TPOT=3427.79ms, Overall=181183.08ms
  请求: TTFT=54329.05ms, TPOT=3426.16ms, Overall=181096.82ms
  请求: TTFT=54140.16ms, TPOT=3158.18ms, Overall=180467.20ms
  请求: TTFT=54094.10ms, TPOT=2474.89ms, Overall=180313.68ms
  请求: TTFT=54423.67ms, TPOT=4378.92ms, Overall=181412.23ms
  请求: TTFT=54409.31ms, TPOT=2952.44ms, Overall=181364.38ms
  请求: TTFT=54358.18ms, TPOT=3337.78ms, Overall=181193.94ms
  请求: TTFT=54401.90ms, TPOT=2820.84ms, Overall=181339.68ms
  请求: TTFT=54298.65ms, TPOT=3519.36ms, Overall=180995.49ms
  请求: TTFT=54225.94ms, TPOT=4518.83ms, Overall=180753.14ms
  请求: TTFT=54275.10ms, TPOT=3837.63ms, Overall=180917.01ms
  请求: TTFT=54369.83ms, TPOT=2699.21ms, Overall=181232.75ms
  请求: TTFT=54351.24ms, TPOT=2818.21ms, Overall=181170.81ms
  请求: TTFT=54278.66ms, TPOT=3089.03ms, Overall=180928.86ms
  请求: TTFT=54237.26ms, TPOT=3615.82ms, Overall=180790.85ms
  请求: TTFT=54342.61ms, TPOT=3427.01ms, Overall=181142.03ms
  请求: TTFT=54210.94ms, TPOT=3243.39ms, Overall=180703.15ms
  请求: TTFT=54374.97ms, TPOT=3964.84ms, Overall=181249.88ms
  请求: TTFT=54291.27ms, TPOT=3619.42ms, Overall=180970.89ms
  请求: TTFT=54294.07ms, TPOT=2815.25ms, Overall=180980.24ms
  请求: TTFT=54377.94ms, TPOT=2950.74ms, Overall=181259.79ms
  请求: TTFT=54309.34ms, TPOT=3090.78ms, Overall=181031.13ms
  请求: TTFT=54287.74ms, TPOT=3518.65ms, Overall=180959.12ms
  请求: TTFT=54266.83ms, TPOT=3517.29ms, Overall=180889.44ms
  请求: TTFT=54122.46ms, TPOT=3323.31ms, Overall=180408.21ms
  请求: TTFT=54316.30ms, TPOT=2947.40ms, Overall=181054.33ms
  请求: TTFT=54255.16ms, TPOT=3516.54ms, Overall=180850.53ms
  请求: TTFT=54151.44ms, TPOT=3158.83ms, Overall=180504.81ms
  请求: TTFT=54198.40ms, TPOT=3084.46ms, Overall=180661.32ms
  请求: TTFT=54243.78ms, TPOT=3722.61ms, Overall=180812.59ms
  请求: TTFT=54160.69ms, TPOT=4076.61ms, Overall=180535.63ms
  请求: TTFT=54134.73ms, TPOT=3508.73ms, Overall=180449.10ms
  请求: TTFT=54221.95ms, TPOT=3085.80ms, Overall=180739.84ms
  请求: TTFT=54397.66ms, TPOT=3846.30ms, Overall=181325.54ms
  请求: TTFT=54242.95ms, TPOT=3330.71ms, Overall=180809.82ms
  请求: TTFT=54206.70ms, TPOT=3011.48ms, Overall=180689.02ms
  请求: TTFT=54219.35ms, TPOT=4081.03ms, Overall=180731.15ms
  请求: TTFT=54111.87ms, TPOT=3322.66ms, Overall=180372.89ms
  请求: TTFT=54248.81ms, TPOT=3955.64ms, Overall=180829.36ms
  请求: TTFT=54392.78ms, TPOT=2951.55ms, Overall=181309.26ms
  请求: TTFT=54337.72ms, TPOT=3169.70ms, Overall=181125.75ms
  请求: TTFT=54101.75ms, TPOT=3005.65ms, Overall=180339.16ms
  请求: TTFT=54214.47ms, TPOT=2530.01ms, Overall=180714.91ms
  请求: TTFT=54168.26ms, TPOT=3009.35ms, Overall=180560.85ms
  请求: TTFT=54273.81ms, TPOT=4366.86ms, Overall=180912.71ms
  请求: TTFT=54404.66ms, TPOT=3626.98ms, Overall=181348.88ms
  请求: TTFT=54363.59ms, TPOT=3171.21ms, Overall=181211.95ms
  请求: TTFT=54203.70ms, TPOT=3328.30ms, Overall=180678.99ms
  请求: TTFT=54127.07ms, TPOT=3007.06ms, Overall=180423.58ms
  请求: TTFT=54259.58ms, TPOT=2813.46ms, Overall=180865.28ms
  请求: TTFT=54283.58ms, TPOT=3518.38ms, Overall=180945.25ms
  请求: TTFT=54276.92ms, TPOT=3247.34ms, Overall=180923.07ms
  请求: TTFT=54166.49ms, TPOT=3082.65ms, Overall=180554.95ms
  请求: TTFT=54368.27ms, TPOT=3020.46ms, Overall=181227.56ms
  请求: TTFT=54324.03ms, TPOT=3621.60ms, Overall=181080.10ms
  请求: TTFT=54329.08ms, TPOT=2485.64ms, Overall=181096.93ms
  请求: TTFT=54384.09ms, TPOT=2884.00ms, Overall=181280.29ms
  请求: TTFT=54176.12ms, TPOT=2939.79ms, Overall=180587.07ms
  请求: TTFT=54157.43ms, TPOT=2747.12ms, Overall=180524.78ms
  请求: TTFT=54121.31ms, TPOT=3946.35ms, Overall=180404.37ms
  请求: TTFT=54260.74ms, TPOT=3246.37ms, Overall=180869.12ms
  请求: TTFT=54336.68ms, TPOT=2881.49ms, Overall=181122.26ms
  请求: TTFT=54103.16ms, TPOT=2935.83ms, Overall=180343.87ms
  请求: TTFT=54215.79ms, TPOT=2750.08ms, Overall=180719.31ms
  请求: TTFT=54383.04ms, TPOT=2758.56ms, Overall=181276.79ms
  请求: TTFT=54135.37ms, TPOT=3007.52ms, Overall=180451.22ms
  请求: TTFT=54187.45ms, TPOT=2690.16ms, Overall=180624.84ms
  请求: TTFT=54213.64ms, TPOT=3243.55ms, Overall=180712.14ms
  请求: TTFT=54328.26ms, TPOT=4371.24ms, Overall=181094.22ms
  请求: TTFT=54401.33ms, TPOT=4377.12ms, Overall=181337.78ms
  请求: TTFT=54175.17ms, TPOT=3717.90ms, Overall=180583.92ms
  请求: TTFT=54388.27ms, TPOT=3525.17ms, Overall=181294.25ms
  请求: TTFT=54315.47ms, TPOT=3168.40ms, Overall=181051.58ms
  请求: TTFT=54337.02ms, TPOT=3336.48ms, Overall=181123.38ms
  请求: TTFT=54396.78ms, TPOT=3525.72ms, Overall=181322.61ms
  请求: TTFT=54125.59ms, TPOT=3946.66ms, Overall=180418.63ms
  请求: TTFT=54181.48ms, TPOT=2528.47ms, Overall=180604.95ms
  请求: TTFT=54207.64ms, TPOT=3952.64ms, Overall=180692.15ms
  请求: TTFT=54109.81ms, TPOT=2805.69ms, Overall=180366.05ms
  请求: TTFT=54201.45ms, TPOT=3832.43ms, Overall=180671.50ms
  请求: TTFT=54141.96ms, TPOT=2578.19ms, Overall=180473.21ms
  请求: TTFT=54248.44ms, TPOT=2943.71ms, Overall=180828.15ms
  请求: TTFT=54269.54ms, TPOT=3517.47ms, Overall=180898.48ms
  请求: TTFT=54152.78ms, TPOT=2578.70ms, Overall=180509.27ms
  请求: TTFT=54193.46ms, TPOT=3951.61ms, Overall=180644.86ms
  请求: TTFT=54358.68ms, TPOT=2949.70ms, Overall=181195.61ms
  请求: TTFT=54314.73ms, TPOT=3017.48ms, Overall=181049.10ms
  请求: TTFT=54414.22ms, TPOT=3627.61ms, Overall=181380.73ms
  请求: TTFT=54296.83ms, TPOT=2754.19ms, Overall=180989.44ms
  请求: TTFT=54322.74ms, TPOT=2880.75ms, Overall=181075.81ms

=== 测试结果分析 ===

--- Query长度测试结果 ---
Query长度 512:
  TTFT: 平均=4845.20ms, 中位数=4840.75ms, 标准差=21.87ms
  TPOT: 平均=292.62ms, 中位数=291.94ms, 标准差=37.83ms
  Overall: 平均=16150.68ms, 中位数=16135.84ms, 标准差=72.89ms
Query长度 1024:
  TTFT: 平均=4823.22ms, 中位数=4817.36ms, 标准差=11.56ms
  TPOT: 平均=270.79ms, 中位数=255.17ms, 标准差=40.18ms
  Overall: 平均=16077.40ms, 中位数=16057.86ms, 标准差=38.52ms
Query长度 2048:
  TTFT: 平均=4874.71ms, 中位数=4872.62ms, 标准差=26.25ms
  TPOT: 平均=275.10ms, 中位数=258.75ms, 标准差=38.99ms
  Overall: 平均=16249.03ms, 中位数=16242.08ms, 标准差=87.49ms
Query长度 4096:
  TTFT: 平均=4960.56ms, 中位数=4952.94ms, 标准差=28.25ms
  TPOT: 平均=270.27ms, 中位数=261.97ms, 标准差=20.19ms
  Overall: 平均=16535.19ms, 中位数=16509.79ms, 标准差=94.18ms
Query长度 8192:
  TTFT: 平均=5376.51ms, 中位数=5372.36ms, 标准差=27.49ms
  TPOT: 平均=327.12ms, 中位数=321.42ms, 标准差=40.08ms
  Overall: 平均=17921.69ms, 中位数=17907.86ms, 标准差=91.63ms
Query长度 16384:
  TTFT: 平均=6105.17ms, 中位数=6103.18ms, 标准差=24.47ms
  TPOT: 平均=370.54ms, 中位数=363.82ms, 标准差=39.21ms
  Overall: 平均=20350.57ms, 中位数=20343.93ms, 标准差=81.57ms

--- QPS测试结果 ---
QPS 2:
  TTFT: 平均=5705.37ms, 中位数=5850.57ms, 标准差=356.35ms
  TPOT: 平均=363.18ms, 中位数=379.20ms, 标准差=57.08ms
  Overall: 平均=19017.89ms, 中位数=19501.91ms, 标准差=1187.82ms
QPS 4:
  TTFT: 平均=6408.85ms, 中位数=6705.39ms, 标准差=745.05ms
  TPOT: 平均=389.38ms, 中位数=396.28ms, 标准差=90.75ms
  Overall: 平均=21362.82ms, 中位数=22351.29ms, 标准差=2483.51ms
QPS 8:
  TTFT: 平均=8400.29ms, 中位数=8410.68ms, 标准差=79.68ms
  TPOT: 平均=497.49ms, 中位数=514.10ms, 标准差=99.64ms
  Overall: 平均=28000.98ms, 中位数=28035.61ms, 标准差=265.61ms
QPS 16:
  TTFT: 平均=11480.69ms, 中位数=11487.19ms, 标准差=83.11ms
  TPOT: 平均=705.01ms, 中位数=686.15ms, 标准差=81.62ms
  Overall: 平均=38268.96ms, 中位数=38290.62ms, 标准差=277.04ms
QPS 32:
  TTFT: 平均=17547.65ms, 中位数=17544.24ms, 标准差=86.17ms
  TPOT: 平均=1058.99ms, 中位数=1052.23ms, 标准差=152.36ms
  Overall: 平均=58492.18ms, 中位数=58480.80ms, 标准差=287.25ms
QPS 64:
  TTFT: 平均=29891.23ms, 中位数=29892.13ms, 标准差=91.91ms
  TPOT: 平均=1739.42ms, 中位数=1707.00ms, 标准差=250.27ms
  Overall: 平均=99637.43ms, 中位数=99640.44ms, 标准差=306.36ms
QPS 128:
  TTFT: 平均=54256.82ms, 中位数=54259.94ms, 标准差=101.55ms
  TPOT: 平均=3263.92ms, 中位数=3172.09ms, 标准差=466.80ms
  Overall: 平均=180856.06ms, 中位数=180866.46ms, 标准差=338.49ms
```

## DIFF-Length-throughput
```shell
--- 测试组1: 不同query长度 (QPS=1) ---
测试query长度: 512
Throught: 316.95561620027604
测试query长度: 1024
Throught: 530.9056856315351
测试query长度: 2048
Throught: 955.9900002381011
测试query长度: 4096
Throught: 1810.9285930192777
测试query长度: 8192
Throught: 3517.086166231666
测试query长度: 16384
Throught: 6877.38995032329
Throught: 1959.1505656820673
测试query长度: 8192
```

## Output-TPS
```shell
--- 测试组1: 不同query长度 ---
测试query长度: 512
[2025-09-01 05:12:46 TP0] Decode batch. #running-req: 113, #token: 31546, token usage: 0.88, cuda graph: False, gen throughput (token/s): 45.24, #queue-req: 399
[2025-09-01 05:13:49 TP0] Decode batch. #running-req: 113, #token: 33789, token usage: 0.94, cuda graph: False, gen throughput (token/s): 72.16, #queue-req: 406
[2025-09-01 05:14:52 TP0] Decode batch. #running-req: 104, #token: 26889, token usage: 0.75, cuda graph: False, gen throughput (token/s): 65.27, #queue-req: 308
[2025-09-01 05:15:50 TP0] Decode batch. #running-req: 104, #token: 31049, token usage: 0.86, cuda graph: False, gen throughput (token/s): 70.88, #queue-req: 308
[2025-09-01 05:16:48 TP0] Decode batch. #running-req: 104, #token: 35209, token usage: 0.98, cuda graph: False, gen throughput (token/s): 71.64, #queue-req: 308
[2025-09-01 05:17:51 TP0] Decode batch. #running-req: 106, #token: 29440, token usage: 0.82, cuda graph: False, gen throughput (token/s): 65.54, #queue-req: 202
[2025-09-01 05:18:50 TP0] Decode batch. #running-req: 106, #token: 33680, token usage: 0.94, cuda graph: False, gen throughput (token/s): 71.69, #queue-req: 202
[2025-09-01 05:19:53 TP0] Decode batch. #running-req: 103, #token: 26501, token usage: 0.74, cuda graph: False, gen throughput (token/s): 65.51, #queue-req: 105
[2025-09-01 05:20:51 TP0] Decode batch. #running-req: 103, #token: 30621, token usage: 0.85, cuda graph: False, gen throughput (token/s): 70.95, #queue-req: 105
[2025-09-01 05:21:48 TP0] Decode batch. #running-req: 103, #token: 34741, token usage: 0.97, cuda graph: False, gen throughput (token/s): 71.54, #queue-req: 105
测试query长度: 1024
[2025-09-01 05:22:53 TP0] Decode batch. #running-req: 106, #token: 28088, token usage: 0.78, cuda graph: False, gen throughput (token/s): 59.56, #queue-req: 150
[2025-09-01 05:23:52 TP0] Decode batch. #running-req: 106, #token: 32328, token usage: 0.90, cuda graph: False, gen throughput (token/s): 71.59, #queue-req: 150
[2025-09-01 05:24:52 TP0] Decode batch. #running-req: 103, #token: 35219, token usage: 0.98, cuda graph: False, gen throughput (token/s): 70.49, #queue-req: 153
[2025-09-01 05:25:56 TP0] Decode batch. #running-req: 104, #token: 29542, token usage: 0.82, cuda graph: False, gen throughput (token/s): 63.76, #queue-req: 52
[2025-09-01 05:26:55 TP0] Decode batch. #running-req: 104, #token: 33702, token usage: 0.94, cuda graph: False, gen throughput (token/s): 70.57, #queue-req: 52
[2025-09-01 05:27:44 TP0] Decode batch. #running-req: 46, #token: 12184, token usage: 0.34, cuda graph: False, gen throughput (token/s): 65.38, #queue-req: 0
[2025-09-01 05:28:15 TP0] Decode batch. #running-req: 46, #token: 14024, token usage: 0.39, cuda graph: False, gen throughput (token/s): 60.20, #queue-req: 0
测试query长度: 2048
[2025-09-01 05:29:50 TP0] Decode batch. #running-req: 113, #token: 31941, token usage: 0.89, cuda graph: False, gen throughput (token/s): 62.48, #queue-req: 15
[2025-09-01 05:30:53 TP0] Decode batch. #running-req: 106, #token: 34159, token usage: 0.95, cuda graph: False, gen throughput (token/s): 72.22, #queue-req: 22
[2025-09-01 05:31:37 TP0] Decode batch. #running-req: 22, #token: 6352, token usage: 0.18, cuda graph: False, gen throughput (token/s): 67.03, #queue-req: 0
[2025-09-01 05:31:53 TP0] Decode batch. #running-req: 15, #token: 4569, token usage: 0.13, cuda graph: False, gen throughput (token/s): 44.02, #queue-req: 0
[2025-09-01 05:32:08 TP0] Decode batch. #running-req: 12, #token: 4120, token usage: 0.11, cuda graph: False, gen throughput (token/s): 41.00, #queue-req: 0
测试query长度: 4096
[2025-09-01 05:32:48 TP0] Decode batch. #running-req: 64, #token: 17983, token usage: 0.50, cuda graph: False, gen throughput (token/s): 55.47, #queue-req: 0
[2025-09-01 05:33:28 TP0] Decode batch. #running-req: 64, #token: 20543, token usage: 0.57, cuda graph: False, gen throughput (token/s): 64.55, #queue-req: 0
测试query长度: 8192
[2025-09-01 05:34:03 TP0] Decode batch. #running-req: 32, #token: 8332, token usage: 0.23, cuda graph: False, gen throughput (token/s): 58.79, #queue-req: 0
[2025-09-01 05:34:26 TP0] Decode batch. #running-req: 32, #token: 9612, token usage: 0.27, cuda graph: False, gen throughput (token/s): 54.96, #queue-req: 0
[2025-09-01 05:34:50 TP0] Decode batch. #running-req: 32, #token: 10892, token usage: 0.30, cuda graph: False, gen throughput (token/s): 54.82, #queue-req: 0
测试query长度: 16384
[2025-09-01 05:35:07 TP0] Decode batch. #running-req: 16, #token: 4492, token usage: 0.12, cuda graph: False, gen throughput (token/s): 42.44, #queue-req: 0
[2025-09-01 05:35:22 TP0] Decode batch. #running-req: 16, #token: 5132, token usage: 0.14, cuda graph: False, gen throughput (token/s): 42.16, #queue-req: 0
```