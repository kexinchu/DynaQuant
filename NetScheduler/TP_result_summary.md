## Qwen-235B-A22B TP=8
```shell
=== 测试结果分析 ===

--- Query长度测试结果 ---
Query长度 512:
  TTFT: 平均=4389.14ms, 中位数=4209.03ms, 标准差=435.37ms
  TPOT: 平均=229.60ms, 中位数=231.48ms, 标准差=90.34ms
  Overall: 平均=14630.48ms, 中位数=14030.10ms, 标准差=1451.24ms
Query长度 1024:
  TTFT: 平均=4171.62ms, 中位数=4169.63ms, 标准差=11.55ms
  TPOT: 平均=215.13ms, 中位数=138.99ms, 标准差=182.14ms
  Overall: 平均=13905.41ms, 中位数=13898.77ms, 标准差=38.52ms
Query长度 2048:
  TTFT: 平均=4254.32ms, 中位数=4215.58ms, 标准差=110.68ms
  TPOT: 平均=225.67ms, 中位数=186.99ms, 标准差=84.42ms
  Overall: 平均=14181.06ms, 中位数=14051.92ms, 标准差=368.92ms
Query长度 4096:
  TTFT: 平均=4291.45ms, 中位数=4212.54ms, 标准差=163.37ms
  TPOT: 平均=282.19ms, 中位数=223.39ms, 标准差=149.44ms
  Overall: 平均=14304.85ms, 中位数=14041.79ms, 标准差=544.56ms
Query长度 8192:
  TTFT: 平均=4287.84ms, 中位数=4285.41ms, 标准差=17.88ms
  TPOT: 平均=233.33ms, 中位数=232.54ms, 标准差=59.69ms
  Overall: 平均=14292.80ms, 中位数=14284.70ms, 标准差=59.60ms
Query长度 16384:
  TTFT: 平均=4505.95ms, 中位数=4519.97ms, 标准差=39.83ms
  TPOT: 平均=219.28ms, 中位数=203.29ms, 标准差=42.26ms
  Overall: 平均=15019.84ms, 中位数=15066.56ms, 标准差=132.75ms

--- QPS测试结果 ---
QPS 2:
  TTFT: 平均=4859.63ms, 中位数=5015.28ms, 标准差=339.11ms
  TPOT: 平均=311.02ms, 中位数=230.97ms, 标准差=118.39ms
  Overall: 平均=16198.78ms, 中位数=16717.59ms, 标准差=1130.36ms
QPS 4:
  TTFT: 平均=5632.77ms, 中位数=5947.81ms, 标准差=762.63ms
  TPOT: 平均=391.90ms, 中位数=379.90ms, 标准差=44.18ms
  Overall: 平均=18775.91ms, 中位数=19826.02ms, 标准差=2542.09ms
QPS 8:
  TTFT: 平均=6445.25ms, 中位数=6456.92ms, 标准差=46.15ms
  TPOT: 平均=297.65ms, 中位数=260.52ms, 标准差=79.67ms
  Overall: 平均=21484.18ms, 中位数=21523.06ms, 标准差=153.83ms
QPS 16:
  TTFT: 平均=6443.19ms, 中位数=6436.02ms, 标准差=22.32ms
  TPOT: 平均=501.52ms, 中位数=429.01ms, 标准差=143.29ms
  Overall: 平均=21477.31ms, 中位数=21453.39ms, 标准差=74.39ms
QPS 32:
  TTFT: 平均=6492.57ms, 中位数=6494.38ms, 标准差=18.95ms
  TPOT: 平均=424.85ms, 中位数=361.36ms, 标准差=158.51ms
  Overall: 平均=21641.90ms, 中位数=21647.93ms, 标准差=63.15ms
QPS 64:
  TTFT: 平均=6438.42ms, 中位数=6444.68ms, 标准差=20.81ms
  TPOT: 平均=430.31ms, 中位数=417.33ms, 标准差=93.90ms
  Overall: 平均=21461.41ms, 中位数=21482.25ms, 标准差=69.35ms
QPS 128:
  TTFT: 平均=6466.73ms, 中位数=6475.41ms, 标准差=22.74ms
  TPOT: 平均=301.21ms, 中位数=279.94ms, 标准差=34.03ms
  Overall: 平均=21555.77ms, 中位数=21584.70ms, 标准差=75.79ms
```

## Qwen-235B-A22B-TP=2

```shell
--- 测试组1: 不同query长度 (QPS=1) ---
测试query长度: 512
  请求 1: TTFT=4107.70ms, TPOT=184.32ms, Overall=13692.32ms
  请求 2: TTFT=4133.19ms, TPOT=132.11ms, Overall=13777.29ms
  请求 3: TTFT=4118.47ms, TPOT=157.54ms, Overall=13728.24ms
  请求 4: TTFT=4120.05ms, TPOT=174.79ms, Overall=13733.50ms
  请求 5: TTFT=4123.66ms, TPOT=188.66ms, Overall=13745.54ms
测试query长度: 1024
  请求 1: TTFT=4150.45ms, TPOT=136.40ms, Overall=13834.82ms
  请求 2: TTFT=4129.98ms, TPOT=301.14ms, Overall=13766.60ms
  请求 3: TTFT=4875.68ms, TPOT=203.15ms, Overall=16252.25ms
  请求 4: TTFT=4136.33ms, TPOT=155.67ms, Overall=13787.78ms
  请求 5: TTFT=4159.01ms, TPOT=154.04ms, Overall=13863.37ms
测试query长度: 2048
  请求 1: TTFT=4173.12ms, TPOT=173.88ms, Overall=13910.40ms
  请求 2: TTFT=4052.26ms, TPOT=242.44ms, Overall=13507.53ms
  请求 3: TTFT=4026.97ms, TPOT=187.93ms, Overall=13423.24ms
  请求 4: TTFT=4052.82ms, TPOT=189.13ms, Overall=13509.41ms
  请求 5: TTFT=4024.15ms, TPOT=234.74ms, Overall=13413.84ms
测试query长度: 4096
  请求 1: TTFT=4174.58ms, TPOT=423.51ms, Overall=13915.28ms
  请求 2: TTFT=4130.28ms, TPOT=275.35ms, Overall=13767.59ms
  请求 3: TTFT=4138.60ms, TPOT=275.91ms, Overall=13795.35ms
  请求 4: TTFT=4158.80ms, TPOT=346.57ms, Overall=13862.68ms
  请求 5: TTFT=4139.47ms, TPOT=254.18ms, Overall=13798.23ms
测试query长度: 8192
  请求 1: TTFT=4335.95ms, TPOT=119.03ms, Overall=14453.17ms
  请求 2: TTFT=4340.10ms, TPOT=198.57ms, Overall=14466.99ms
  请求 3: TTFT=4527.19ms, TPOT=293.43ms, Overall=15090.62ms
  请求 4: TTFT=4564.25ms, TPOT=140.13ms, Overall=15214.16ms
  请求 5: TTFT=4329.74ms, TPOT=194.28ms, Overall=14432.47ms
测试query长度: 16384
  请求 1: TTFT=5011.91ms, TPOT=259.88ms, Overall=16706.35ms
  请求 2: TTFT=5014.08ms, TPOT=324.99ms, Overall=16713.61ms
  请求 3: TTFT=4994.60ms, TPOT=184.99ms, Overall=16648.67ms
  请求 4: TTFT=4963.04ms, TPOT=257.34ms, Overall=16543.45ms
  请求 5: TTFT=4945.51ms, TPOT=721.22ms, Overall=16485.02ms

--- 测试组2: 不同QPS (query长度=4096) ---
测试QPS: 2
  请求: TTFT=5210.74ms, TPOT=289.49ms, Overall=17369.12ms
  请求: TTFT=5100.39ms, TPOT=258.72ms, Overall=17001.31ms
  请求: TTFT=5151.03ms, TPOT=187.80ms, Overall=17170.10ms
  请求: TTFT=5191.98ms, TPOT=346.13ms, Overall=17306.60ms
  请求: TTFT=4238.99ms, TPOT=290.91ms, Overall=14129.98ms
测试QPS: 4
  请求: TTFT=6026.19ms, TPOT=611.35ms, Overall=20087.31ms
  请求: TTFT=5925.45ms, TPOT=345.65ms, Overall=19751.50ms
  请求: TTFT=5849.40ms, TPOT=216.64ms, Overall=19498.00ms
  请求: TTFT=6001.75ms, TPOT=285.80ms, Overall=20005.83ms
  请求: TTFT=4148.72ms, TPOT=236.11ms, Overall=13829.07ms
测试QPS: 8
  请求: TTFT=8074.84ms, TPOT=588.79ms, Overall=26916.12ms
  请求: TTFT=7873.60ms, TPOT=540.35ms, Overall=26245.34ms
  请求: TTFT=8066.42ms, TPOT=508.69ms, Overall=26888.08ms
  请求: TTFT=8105.26ms, TPOT=370.83ms, Overall=27017.52ms
  请求: TTFT=7950.71ms, TPOT=363.76ms, Overall=26502.38ms
  请求: TTFT=7989.52ms, TPOT=327.06ms, Overall=26631.74ms
  请求: TTFT=7912.74ms, TPOT=429.37ms, Overall=26375.79ms
  请求: TTFT=8028.24ms, TPOT=506.29ms, Overall=26760.81ms
测试QPS: 16
  请求: TTFT=12089.51ms, TPOT=1282.22ms, Overall=40298.38ms
  请求: TTFT=12109.29ms, TPOT=743.55ms, Overall=40364.29ms
  请求: TTFT=12030.68ms, TPOT=1477.45ms, Overall=40102.25ms
  请求: TTFT=11913.15ms, TPOT=992.76ms, Overall=39710.51ms
  请求: TTFT=11991.85ms, TPOT=1165.87ms, Overall=39972.84ms
  请求: TTFT=11932.99ms, TPOT=843.75ms, Overall=39776.65ms
  请求: TTFT=12090.29ms, TPOT=522.42ms, Overall=40300.98ms
  请求: TTFT=11952.20ms, TPOT=606.27ms, Overall=39840.67ms
  请求: TTFT=12011.28ms, TPOT=475.02ms, Overall=40037.60ms
  请求: TTFT=12051.15ms, TPOT=1222.58ms, Overall=40170.51ms
  请求: TTFT=12070.88ms, TPOT=782.37ms, Overall=40236.28ms
  请求: TTFT=11972.89ms, TPOT=846.57ms, Overall=39909.62ms
  请求: TTFT=12170.12ms, TPOT=436.88ms, Overall=40567.07ms
  请求: TTFT=12130.74ms, TPOT=589.69ms, Overall=40435.79ms
  请求: TTFT=12150.09ms, TPOT=659.31ms, Overall=40500.32ms
  请求: TTFT=11894.18ms, TPOT=603.33ms, Overall=39647.27ms
测试QPS: 32
  请求: TTFT=20062.46ms, TPOT=1200.32ms, Overall=66874.86ms
  请求: TTFT=19974.08ms, TPOT=951.15ms, Overall=66580.27ms
  请求: TTFT=20129.28ms, TPOT=1021.05ms, Overall=67097.59ms
  请求: TTFT=20046.80ms, TPOT=1230.94ms, Overall=66822.66ms
  请求: TTFT=20068.02ms, TPOT=1114.89ms, Overall=66893.40ms
  请求: TTFT=20099.32ms, TPOT=977.05ms, Overall=66997.73ms
  请求: TTFT=19923.64ms, TPOT=715.21ms, Overall=66412.15ms
  请求: TTFT=20160.01ms, TPOT=2767.06ms, Overall=67200.03ms
  请求: TTFT=20087.64ms, TPOT=1673.97ms, Overall=66958.80ms
  请求: TTFT=20037.34ms, TPOT=1508.19ms, Overall=66791.15ms
  请求: TTFT=19996.02ms, TPOT=2221.78ms, Overall=66653.39ms
  请求: TTFT=20077.73ms, TPOT=956.08ms, Overall=66925.76ms
  请求: TTFT=19984.65ms, TPOT=1110.26ms, Overall=66615.48ms
  请求: TTFT=19944.73ms, TPOT=1135.07ms, Overall=66482.44ms
  请求: TTFT=20170.98ms, TPOT=960.52ms, Overall=67236.59ms
  请求: TTFT=19955.29ms, TPOT=739.08ms, Overall=66517.63ms
  请求: TTFT=20203.54ms, TPOT=924.34ms, Overall=67345.13ms
  请求: TTFT=20109.09ms, TPOT=1020.03ms, Overall=67030.30ms
  请求: TTFT=20058.45ms, TPOT=1200.08ms, Overall=66861.49ms
  请求: TTFT=20181.54ms, TPOT=1070.23ms, Overall=67271.81ms
  请求: TTFT=20027.82ms, TPOT=1263.02ms, Overall=66759.39ms
  请求: TTFT=20016.35ms, TPOT=1229.07ms, Overall=66721.18ms
  请求: TTFT=20141.74ms, TPOT=1021.68ms, Overall=67139.13ms
  请求: TTFT=20005.97ms, TPOT=1296.68ms, Overall=66686.55ms
  请求: TTFT=20214.24ms, TPOT=725.64ms, Overall=67380.80ms
  请求: TTFT=19966.17ms, TPOT=1058.81ms, Overall=66553.90ms
  请求: TTFT=20152.56ms, TPOT=1343.50ms, Overall=67175.21ms
  请求: TTFT=20193.94ms, TPOT=1570.64ms, Overall=67313.15ms
  请求: TTFT=20121.51ms, TPOT=958.17ms, Overall=67071.69ms
  请求: TTFT=19904.71ms, TPOT=876.31ms, Overall=66349.04ms
  请求: TTFT=19936.12ms, TPOT=1134.58ms, Overall=66453.74ms
  请求: TTFT=19915.53ms, TPOT=663.85ms, Overall=66385.11ms
测试QPS: 64
  请求: TTFT=35944.03ms, TPOT=2396.27ms, Overall=119813.45ms
  请求: TTFT=36086.46ms, TPOT=1718.40ms, Overall=120288.19ms
  请求: TTFT=36007.90ms, TPOT=3231.48ms, Overall=120026.35ms
  请求: TTFT=35910.80ms, TPOT=1948.65ms, Overall=119702.66ms
  请求: TTFT=36120.50ms, TPOT=1505.02ms, Overall=120401.65ms
  请求: TTFT=36211.67ms, TPOT=1836.82ms, Overall=120705.58ms
  请求: TTFT=35927.41ms, TPOT=1197.58ms, Overall=119758.05ms
  请求: TTFT=35900.47ms, TPOT=2326.88ms, Overall=119668.23ms
  请求: TTFT=35882.09ms, TPOT=2537.12ms, Overall=119606.97ms
  请求: TTFT=35962.69ms, TPOT=1498.45ms, Overall=119875.64ms
  请求: TTFT=35889.33ms, TPOT=2147.22ms, Overall=119631.09ms
  请求: TTFT=35978.61ms, TPOT=3358.00ms, Overall=119928.72ms
  请求: TTFT=36116.00ms, TPOT=1872.68ms, Overall=120386.67ms
  请求: TTFT=36093.37ms, TPOT=2276.16ms, Overall=120311.23ms
  请求: TTFT=35895.30ms, TPOT=1903.54ms, Overall=119651.01ms
  请求: TTFT=36144.73ms, TPOT=2108.44ms, Overall=120482.43ms
  请求: TTFT=35973.32ms, TPOT=2047.26ms, Overall=119911.08ms
  请求: TTFT=35905.76ms, TPOT=1232.06ms, Overall=119685.87ms
  请求: TTFT=36173.48ms, TPOT=1507.23ms, Overall=120578.27ms
  请求: TTFT=36179.15ms, TPOT=2344.94ms, Overall=120597.16ms
  请求: TTFT=36132.54ms, TPOT=4683.85ms, Overall=120441.80ms
  请求: TTFT=36003.54ms, TPOT=1400.14ms, Overall=120011.79ms
  请求: TTFT=36202.12ms, TPOT=2111.79ms, Overall=120673.75ms
  请求: TTFT=36035.62ms, TPOT=1356.18ms, Overall=120118.72ms
  请求: TTFT=36077.02ms, TPOT=3006.42ms, Overall=120256.74ms
  请求: TTFT=35999.37ms, TPOT=2048.74ms, Overall=119997.90ms
  请求: TTFT=36026.25ms, TPOT=4670.07ms, Overall=120087.48ms
  请求: TTFT=35919.44ms, TPOT=1643.37ms, Overall=119731.47ms
  请求: TTFT=35953.18ms, TPOT=2467.38ms, Overall=119843.93ms
  请求: TTFT=36129.33ms, TPOT=1532.76ms, Overall=120431.12ms
  请求: TTFT=36049.69ms, TPOT=3235.23ms, Overall=120165.62ms
  请求: TTFT=36021.57ms, TPOT=1500.90ms, Overall=120071.91ms
  请求: TTFT=36055.88ms, TPOT=3235.78ms, Overall=120186.26ms
  请求: TTFT=35937.28ms, TPOT=2704.96ms, Overall=119790.92ms
  请求: TTFT=36068.41ms, TPOT=2274.58ms, Overall=120228.02ms
  请求: TTFT=36186.94ms, TPOT=6495.09ms, Overall=120623.12ms
  请求: TTFT=35875.00ms, TPOT=1860.19ms, Overall=119583.35ms
  请求: TTFT=36033.55ms, TPOT=2402.24ms, Overall=120111.84ms
  请求: TTFT=36016.99ms, TPOT=1715.09ms, Overall=120056.64ms
  请求: TTFT=36067.78ms, TPOT=2214.69ms, Overall=120225.94ms
  请求: TTFT=35942.51ms, TPOT=2541.39ms, Overall=119808.38ms
  请求: TTFT=36199.05ms, TPOT=1656.17ms, Overall=120663.50ms
  请求: TTFT=36062.46ms, TPOT=1314.78ms, Overall=120208.22ms
  请求: TTFT=35949.40ms, TPOT=1950.74ms, Overall=119831.33ms
  请求: TTFT=36113.88ms, TPOT=1915.13ms, Overall=120379.62ms
  请求: TTFT=35961.21ms, TPOT=1553.88ms, Overall=119870.71ms
  请求: TTFT=36156.71ms, TPOT=2163.22ms, Overall=120522.37ms
  请求: TTFT=36103.07ms, TPOT=1959.08ms, Overall=120343.58ms
  请求: TTFT=35871.01ms, TPOT=3347.96ms, Overall=119570.04ms
  请求: TTFT=36046.40ms, TPOT=1152.17ms, Overall=120154.65ms
  请求: TTFT=35928.17ms, TPOT=1581.74ms, Overall=119760.56ms
  请求: TTFT=36211.80ms, TPOT=1797.75ms, Overall=120706.00ms
  请求: TTFT=36153.64ms, TPOT=4217.92ms, Overall=120512.13ms
  请求: TTFT=35882.66ms, TPOT=2325.73ms, Overall=119608.86ms
  请求: TTFT=35990.74ms, TPOT=2332.73ms, Overall=119969.12ms
  请求: TTFT=36194.05ms, TPOT=2815.09ms, Overall=120646.84ms
  请求: TTFT=36109.80ms, TPOT=1959.45ms, Overall=120366.00ms
  请求: TTFT=36172.60ms, TPOT=2221.12ms, Overall=120575.34ms
  请求: TTFT=35973.79ms, TPOT=3497.45ms, Overall=119912.62ms
  请求: TTFT=36166.88ms, TPOT=2909.98ms, Overall=120556.27ms
  请求: TTFT=36224.31ms, TPOT=3250.90ms, Overall=120747.69ms
  请求: TTFT=36144.70ms, TPOT=3012.06ms, Overall=120482.33ms
  请求: TTFT=36087.64ms, TPOT=2215.91ms, Overall=120292.14ms
  请求: TTFT=35997.42ms, TPOT=1714.16ms, Overall=119991.41ms
测试QPS: 128
  请求: TTFT=67447.25ms, TPOT=3838.46ms, Overall=224824.17ms
  请求: TTFT=67603.14ms, TPOT=3585.01ms, Overall=225343.80ms
  请求: TTFT=67864.61ms, TPOT=4524.31ms, Overall=226215.35ms
  请求: TTFT=67933.20ms, TPOT=3686.30ms, Overall=226443.98ms
  请求: TTFT=67966.56ms, TPOT=3374.23ms, Overall=226555.19ms
  请求: TTFT=67761.16ms, TPOT=3162.19ms, Overall=225870.55ms
  请求: TTFT=67834.96ms, TPOT=2728.99ms, Overall=226116.54ms
  请求: TTFT=67822.02ms, TPOT=4057.73ms, Overall=226073.42ms
  请求: TTFT=67742.66ms, TPOT=3512.58ms, Overall=225808.87ms
  请求: TTFT=67754.35ms, TPOT=3676.59ms, Overall=225847.82ms
  请求: TTFT=67564.66ms, TPOT=3941.27ms, Overall=225215.53ms
  请求: TTFT=67659.94ms, TPOT=5443.90ms, Overall=225533.12ms
  请求: TTFT=67764.53ms, TPOT=2509.80ms, Overall=225881.76ms
  请求: TTFT=67591.14ms, TPOT=4262.50ms, Overall=225303.81ms
  请求: TTFT=67745.26ms, TPOT=5269.08ms, Overall=225817.52ms
  请求: TTFT=67644.25ms, TPOT=4153.59ms, Overall=225480.84ms
  请求: TTFT=67936.17ms, TPOT=3302.45ms, Overall=226453.90ms
  请求: TTFT=67709.82ms, TPOT=3159.79ms, Overall=225699.41ms
  请求: TTFT=67647.90ms, TPOT=4642.50ms, Overall=225493.00ms
  请求: TTFT=67814.23ms, TPOT=3296.53ms, Overall=226047.45ms
  请求: TTFT=67570.95ms, TPOT=4379.60ms, Overall=225236.50ms
  请求: TTFT=67844.25ms, TPOT=3517.85ms, Overall=226147.49ms
  请求: TTFT=67600.07ms, TPOT=2464.59ms, Overall=225333.55ms
  请求: TTFT=67735.48ms, TPOT=3099.01ms, Overall=225784.92ms
  请求: TTFT=67965.23ms, TPOT=3524.12ms, Overall=226550.78ms
  请求: TTFT=67800.42ms, TPOT=4056.44ms, Overall=226001.39ms
  请求: TTFT=67632.71ms, TPOT=9282.92ms, Overall=225442.37ms
  请求: TTFT=67624.91ms, TPOT=3586.17ms, Overall=225416.38ms
  请求: TTFT=67864.15ms, TPOT=3166.99ms, Overall=226213.82ms
  请求: TTFT=67652.43ms, TPOT=4932.99ms, Overall=225508.09ms
  请求: TTFT=67559.18ms, TPOT=4042.00ms, Overall=225197.27ms
  请求: TTFT=67675.71ms, TPOT=3851.46ms, Overall=225585.71ms
  请求: TTFT=67623.43ms, TPOT=3756.86ms, Overall=225411.44ms
  请求: TTFT=67728.23ms, TPOT=2358.69ms, Overall=225760.76ms
  请求: TTFT=67920.18ms, TPOT=6603.35ms, Overall=226400.61ms
  请求: TTFT=67905.62ms, TPOT=5463.67ms, Overall=226352.07ms
  请求: TTFT=67790.11ms, TPOT=4652.26ms, Overall=225967.03ms
  请求: TTFT=67743.62ms, TPOT=6322.74ms, Overall=225812.06ms
  请求: TTFT=67699.84ms, TPOT=6075.63ms, Overall=225666.12ms
  请求: TTFT=67726.27ms, TPOT=4158.63ms, Overall=225754.22ms
  请求: TTFT=67889.17ms, TPOT=5657.43ms, Overall=226297.22ms
  请求: TTFT=67703.54ms, TPOT=2468.36ms, Overall=225678.47ms
  请求: TTFT=67872.88ms, TPOT=3770.72ms, Overall=226242.94ms
  请求: TTFT=67928.35ms, TPOT=3602.26ms, Overall=226427.82ms
  请求: TTFT=67730.31ms, TPOT=3362.50ms, Overall=225767.71ms
  请求: TTFT=67933.36ms, TPOT=3234.92ms, Overall=226444.53ms
  请求: TTFT=67571.24ms, TPOT=3354.60ms, Overall=225237.46ms
  请求: TTFT=67824.54ms, TPOT=4945.54ms, Overall=226081.78ms
  请求: TTFT=67772.53ms, TPOT=3162.72ms, Overall=225908.43ms
  请求: TTFT=67722.46ms, TPOT=2724.47ms, Overall=225741.53ms
  请求: TTFT=67814.95ms, TPOT=5274.50ms, Overall=226049.82ms
  请求: TTFT=67943.73ms, TPOT=4171.98ms, Overall=226479.10ms
  请求: TTFT=67878.83ms, TPOT=4799.51ms, Overall=226262.77ms
  请求: TTFT=67958.85ms, TPOT=3775.49ms, Overall=226529.49ms
  请求: TTFT=67698.32ms, TPOT=3852.75ms, Overall=225661.07ms
  请求: TTFT=67763.72ms, TPOT=2983.31ms, Overall=225879.08ms
  请求: TTFT=67833.46ms, TPOT=3297.46ms, Overall=226111.52ms
  请求: TTFT=67894.59ms, TPOT=4950.65ms, Overall=226315.31ms
  请求: TTFT=67920.60ms, TPOT=3521.81ms, Overall=226402.00ms
  请求: TTFT=67712.45ms, TPOT=3674.32ms, Overall=225708.15ms
  请求: TTFT=67686.40ms, TPOT=12148.84ms, Overall=225621.32ms
  请求: TTFT=67754.87ms, TPOT=4649.84ms, Overall=225849.57ms
  请求: TTFT=67823.37ms, TPOT=3859.87ms, Overall=226077.89ms
  请求: TTFT=67885.56ms, TPOT=3167.99ms, Overall=226285.21ms
  请求: TTFT=67905.11ms, TPOT=5281.51ms, Overall=226350.37ms
  请求: TTFT=67616.91ms, TPOT=2767.94ms, Overall=225389.69ms
  请求: TTFT=67797.42ms, TPOT=4056.26ms, Overall=225991.39ms
  请求: TTFT=67675.95ms, TPOT=2924.27ms, Overall=225586.49ms
  请求: TTFT=67810.72ms, TPOT=3516.11ms, Overall=226035.73ms
  请求: TTFT=67687.74ms, TPOT=2979.96ms, Overall=225625.81ms
  请求: TTFT=67966.34ms, TPOT=3374.22ms, Overall=226554.48ms
  请求: TTFT=67661.26ms, TPOT=4266.93ms, Overall=225537.53ms
  请求: TTFT=67759.22ms, TPOT=7528.80ms, Overall=225864.06ms
  请求: TTFT=67785.39ms, TPOT=3954.15ms, Overall=225951.30ms
  请求: TTFT=67801.54ms, TPOT=3042.38ms, Overall=226005.12ms
  请求: TTFT=67584.40ms, TPOT=5840.63ms, Overall=225281.33ms
  请求: TTFT=67860.79ms, TPOT=4279.51ms, Overall=226202.62ms
  请求: TTFT=67720.84ms, TPOT=4051.67ms, Overall=225736.12ms
  请求: TTFT=67779.38ms, TPOT=3041.38ms, Overall=225931.26ms
  请求: TTFT=67724.26ms, TPOT=4158.51ms, Overall=225747.53ms
  请求: TTFT=67737.62ms, TPOT=2982.16ms, Overall=225792.07ms
  请求: TTFT=67594.72ms, TPOT=5841.52ms, Overall=225315.73ms
  请求: TTFT=67562.67ms, TPOT=5436.08ms, Overall=225208.91ms
  请求: TTFT=67784.73ms, TPOT=2054.08ms, Overall=225949.12ms
  请求: TTFT=67635.00ms, TPOT=4046.54ms, Overall=225449.99ms
  请求: TTFT=67643.27ms, TPOT=3757.96ms, Overall=225477.58ms
  请求: TTFT=67930.31ms, TPOT=3170.08ms, Overall=226434.35ms
  请求: TTFT=67917.21ms, TPOT=6095.13ms, Overall=226390.70ms
  请求: TTFT=67692.65ms, TPOT=3360.63ms, Overall=225642.15ms
  请求: TTFT=67904.60ms, TPOT=3520.98ms, Overall=226348.66ms
  请求: TTFT=67663.36ms, TPOT=5444.18ms, Overall=225544.54ms
  请求: TTFT=67856.02ms, TPOT=2987.37ms, Overall=226186.74ms
  请求: TTFT=67712.93ms, TPOT=2981.07ms, Overall=225709.78ms
  请求: TTFT=67916.86ms, TPOT=4527.79ms, Overall=226389.54ms
  请求: TTFT=67592.94ms, TPOT=5632.74ms, Overall=225309.79ms
  请求: TTFT=67902.04ms, TPOT=2989.40ms, Overall=226340.12ms
  请求: TTFT=67666.45ms, TPOT=3588.37ms, Overall=225554.82ms
  请求: TTFT=67600.55ms, TPOT=3943.37ms, Overall=225335.18ms
  请求: TTFT=67958.58ms, TPOT=3109.22ms, Overall=226528.60ms
  请求: TTFT=67567.64ms, TPOT=3583.13ms, Overall=225225.45ms
  请求: TTFT=67841.87ms, TPOT=2878.14ms, Overall=226139.58ms
  请求: TTFT=67608.76ms, TPOT=4263.62ms, Overall=225362.54ms
  请求: TTFT=67629.66ms, TPOT=7172.84ms, Overall=225432.21ms
  请求: TTFT=67962.21ms, TPOT=4173.12ms, Overall=226540.69ms
  请求: TTFT=67955.25ms, TPOT=4530.35ms, Overall=226517.50ms
  请求: TTFT=67916.60ms, TPOT=2597.90ms, Overall=226388.66ms
  请求: TTFT=67840.38ms, TPOT=2435.30ms, Overall=226134.59ms
  请求: TTFT=67687.51ms, TPOT=4268.58ms, Overall=225625.03ms
  请求: TTFT=67807.71ms, TPOT=3102.31ms, Overall=226025.71ms
  请求: TTFT=67953.84ms, TPOT=2781.74ms, Overall=226512.81ms
  请求: TTFT=67795.04ms, TPOT=4943.39ms, Overall=225983.47ms
  请求: TTFT=67817.82ms, TPOT=4945.05ms, Overall=226059.40ms
  请求: TTFT=67618.98ms, TPOT=2768.03ms, Overall=225396.59ms
  请求: TTFT=67939.03ms, TPOT=6605.18ms, Overall=226463.44ms
  请求: TTFT=67584.80ms, TPOT=3428.21ms, Overall=225282.66ms
  请求: TTFT=67702.23ms, TPOT=3159.44ms, Overall=225674.09ms
  请求: TTFT=67793.56ms, TPOT=4162.76ms, Overall=225978.54ms
  请求: TTFT=67679.75ms, TPOT=3509.32ms, Overall=225599.16ms
  请求: TTFT=67872.00ms, TPOT=5278.93ms, Overall=226239.99ms
  请求: TTFT=67862.46ms, TPOT=3231.55ms, Overall=226208.19ms
  请求: TTFT=67895.13ms, TPOT=3600.50ms, Overall=226317.09ms
  请求: TTFT=67650.83ms, TPOT=3288.58ms, Overall=225502.76ms
  请求: TTFT=67869.19ms, TPOT=3599.12ms, Overall=226230.64ms
  请求: TTFT=67852.83ms, TPOT=3368.58ms, Overall=226176.11ms
  请求: TTFT=67653.17ms, TPOT=3358.67ms, Overall=225510.58ms
  请求: TTFT=67593.16ms, TPOT=2190.52ms, Overall=225310.52ms
  请求: TTFT=67885.87ms, TPOT=4525.72ms, Overall=226286.22ms
  请求: TTFT=67621.90ms, TPOT=2921.93ms, Overall=225406.34ms

--- Query长度测试结果 ---
Query长度 512:
  TTFT: 平均=4120.61ms, 中位数=4120.05ms, 标准差=8.24ms
  TPOT: 平均=167.48ms, 中位数=174.79ms, 标准差=20.67ms
  Overall: 平均=13735.38ms, 中位数=13733.50ms, 标准差=27.45ms

Query长度 1024:
  TTFT: 平均=4290.29ms, 中位数=4150.45ms, 标准差=292.87ms
  TPOT: 平均=190.08ms, 中位数=155.67ms, 标准差=59.78ms
  Overall: 平均=14300.96ms, 中位数=13834.82ms, 标准差=976.24ms

Query长度 2048:
  TTFT: 平均=4065.86ms, 中位数=4052.26ms, 标准差=54.98ms
  TPOT: 平均=205.62ms, 中位数=189.13ms, 标准差=27.55ms
  Overall: 平均=13552.88ms, 中位数=13507.53ms, 标准差=183.25ms

Query长度 4096:
  TTFT: 平均=4148.35ms, 中位数=4139.47ms, 标准差=16.11ms
  TPOT: 平均=315.10ms, 中位数=275.91ms, 标准差=62.56ms
  Overall: 平均=13827.83ms, 中位数=13798.23ms, 标准差=53.71ms

Query长度 8192:
  TTFT: 平均=4419.45ms, 中位数=4340.10ms, 标准差=103.82ms
  TPOT: 平均=189.09ms, 中位数=194.28ms, 标准差=60.51ms
  Overall: 平均=14731.48ms, 中位数=14466.99ms, 标准差=346.06ms

Query长度 16384:
  TTFT: 平均=4985.83ms, 中位数=4994.60ms, 标准差=27.20ms
  TPOT: 平均=349.68ms, 中位数=259.88ms, 标准差=190.98ms
  Overall: 平均=16619.42ms, 中位数=16648.67ms, 标准差=90.69ms

--- QPS测试结果 ---
QPS 2:
  TTFT: 平均=4978.63ms, 中位数=5151.03ms, 标准差=371.75ms
  TPOT: 平均=274.61ms, 中位数=289.49ms, 标准差=51.75ms
  Overall: 平均=16595.42ms, 中位数=17170.10ms, 标准差=1239.18ms

QPS 4:
  TTFT: 平均=5590.30ms, 中位数=5925.45ms, 标准差=723.45ms
  TPOT: 平均=339.11ms, 中位数=285.80ms, 标准差=143.25ms
  Overall: 平均=18634.34ms, 中位数=19751.50ms, 标准差=2411.50ms

QPS 8:
  TTFT: 平均=8000.17ms, 中位数=8008.88ms, 标准差=77.48ms
  TPOT: 平均=454.39ms, 中位数=467.83ms, 标准差=88.83ms
  Overall: 平均=26667.22ms, 中位数=26696.28ms, 标准差=258.25ms

QPS 16:
  TTFT: 平均=12035.08ms, 中位数=12040.92ms, 标准差=83.67ms
  TPOT: 平均=828.13ms, 中位数=762.96ms, 标准差=305.24ms
  Overall: 平均=40116.94ms, 中位数=40136.38ms, 标准差=278.89ms

QPS 32:
  TTFT: 平均=20058.35ms, 中位数=20060.46ms, 标准差=90.82ms
  TPOT: 平均=1176.23ms, 中位数=1090.24ms, 标准差=416.79ms
  Overall: 平均=66861.17ms, 中位数=66868.18ms, 标准差=302.74ms

QPS 64:
  TTFT: 平均=36042.79ms, 中位数=36041.01ms, 标准差=103.46ms
  TPOT: 平均=2327.50ms, 中位数=2129.51ms, 标准差=935.26ms
  Overall: 平均=120142.63ms, 中位数=120136.68ms, 标准差=344.88ms

QPS 128:
  TTFT: 平均=67764.51ms, 中位数=67762.44ms, 标准差=122.81ms
  TPOT: 平均=4039.38ms, 中位数=3681.45ms, 标准差=1360.75ms
  Overall: 平均=225881.71ms, 中位数=225874.82ms, 标准差=409.38ms
```

## DIFF-Length-throughtput
```shell
--- 测试组1: 不同query长度 ---
测试query长度: 512
Throught: 337.8428138867475
测试query长度: 1024
Throught: 573.4420135815944
测试query长度: 2048
Throught: 1034.665018009447
测试query长度: 4096
Throught: 1959.1505656820673
测试query长度: 8192
Throught: 3775.9916606255497
测试query长度: 16384
Throught: 7448.402323301927
```
## Output-TPS
```shell
--- 测试组1: 不同query长度 ---
测试query长度: 512
[2025-08-31 00:34:46 TP0] Decode batch. #running-req: 128, #token: 35794, token usage: 0.87, cuda graph: False, gen throughput (token/s): 50.98, #queue-req: 384
[2025-08-31 00:35:50 TP0] Decode batch. #running-req: 128, #token: 40914, token usage: 1.00, cuda graph: False, gen throughput (token/s): 80.25, #queue-req: 384
[2025-08-31 00:36:54 TP0] Decode batch. #running-req: 118, #token: 30532, token usage: 0.75, cuda graph: False, gen throughput (token/s): 72.99, #queue-req: 281
[2025-08-31 00:37:54 TP0] Decode batch. #running-req: 118, #token: 35252, token usage: 0.86, cuda graph: False, gen throughput (token/s): 78.39, #queue-req: 281
[2025-08-31 00:38:54 TP0] Decode batch. #running-req: 118, #token: 39972, token usage: 0.98, cuda graph: False, gen throughput (token/s): 78.83, #queue-req: 281
[2025-08-31 00:42:13 TP0] Decode batch. #running-req: 118, #token: 30251, token usage: 0.74, cuda graph: False, gen throughput (token/s): 66.16, #queue-req: 50
[2025-08-31 00:43:20 TP0] Decode batch. #running-req: 118, #token: 34971, token usage: 0.85, cuda graph: False, gen throughput (token/s): 71.09, #queue-req: 50
[2025-08-31 00:42:13 TP0] Decode batch. #running-req: 118, #token: 30251, token usage: 0.74, cuda graph: False, gen throughput (token/s): 66.16, #queue-req: 50
[2025-08-31 00:43:20 TP0] Decode batch. #running-req: 118, #token: 34971, token usage: 0.85, cuda graph: False, gen throughput (token/s): 71.09, #queue-req: 50
测试query长度: 1024
[2025-08-31 00:44:31 TP0] Decode batch. #running-req: 120, #token: 30057, token usage: 0.73, cuda graph: False, gen throughput (token/s): 63.34, #queue-req: 76
[2025-08-31 00:45:38 TP0] Decode batch. #running-req: 120, #token: 34857, token usage: 0.85, cuda graph: False, gen throughput (token/s): 71.96, #queue-req: 136
[2025-08-31 00:46:45 TP0] Decode batch. #running-req: 120, #token: 39657, token usage: 0.97, cuda graph: False, gen throughput (token/s): 71.84, #queue-req: 136
[2025-08-31 00:47:55 TP0] Decode batch. #running-req: 118, #token: 31914, token usage: 0.78, cuda graph: False, gen throughput (token/s): 65.94, #queue-req: 25
[2025-08-31 00:49:02 TP0] Decode batch. #running-req: 118, #token: 36634, token usage: 0.89, cuda graph: False, gen throughput (token/s): 71.06, #queue-req: 25
[2025-08-31 00:51:29 TP0] Decode batch. #running-req: 128, #token: 34290, token usage: 0.84, cuda graph: False, gen throughput (token/s): 55.87, #queue-req: 0
[2025-08-31 00:52:40 TP0] Decode batch. #running-req: 128, #token: 39410, token usage: 0.96, cuda graph: False, gen throughput (token/s): 72.23, #queue-req: 0
测试query长度: 2048
[2025-08-31 00:54:10 TP0] Decode batch. #running-req: 64, #token: 16733, token usage: 0.41, cuda graph: False, gen throughput (token/s): 45.00, #queue-req: 0
[2025-08-31 00:54:48 TP0] Decode batch. #running-req: 64, #token: 19293, token usage: 0.47, cuda graph: False, gen throughput (token/s): 66.53, #queue-req: 0
测试query长度: 4096
[2025-08-31 00:54:10 TP0] Decode batch. #running-req: 64, #token: 16733, token usage: 0.41, cuda graph: False, gen throughput (token/s): 45.00, #queue-req: 0
[2025-08-31 00:54:48 TP0] Decode batch. #running-req: 64, #token: 19293, token usage: 0.47, cuda graph: False, gen throughput (token/s): 66.53, #queue-req: 0
[2025-08-31 00:55:27 TP0] Decode batch. #running-req: 64, #token: 21853, token usage: 0.53, cuda graph: False, gen throughput (token/s): 66.55, #queue-req: 0
测试query长度: 8192
[2025-08-31 00:55:53 TP0] Decode batch. #running-req: 32, #token: 8958, token usage: 0.22, cuda graph: False, gen throughput (token/s): 55.10, #queue-req: 0
[2025-08-31 00:56:15 TP0] Decode batch. #running-req: 32, #token: 10238, token usage: 0.25, cuda graph: False, gen throughput (token/s): 57.49, #queue-req: 0
测试query长度: 16384
[2025-08-31 00:56:36 TP0] Decode batch. #running-req: 16, #token: 4180, token usage: 0.10, cuda graph: False, gen throughput (token/s): 51.35, #queue-req: 0
[2025-08-31 00:56:51 TP0] Decode batch. #running-req: 16, #token: 4820, token usage: 0.12, cuda graph: False, gen throughput (token/s): 42.15, #queue-req: 0
[2025-08-31 00:57:05 TP0] Decode batch. #running-req: 16, #token: 5460, token usage: 0.13, cuda graph: False, gen throughput (token/s): 45.36, #queue-req: 0
```