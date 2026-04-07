Place your raw store data here (TSV or CSV).
This directory is in .gitignore — real data is never committed.

Expected schema
---------------
客户编码  地市编码  销售笔数  购入  当前卖出
凌晨时段扫码稳定性指数  早上时段扫码稳定性指数
下午时段扫码稳定性指数  晚上时段扫码稳定性指数
期初库存  期末库存  开机天数  包销售比例
日扫码稳定性指数  扫码间隔时间  日均扫码品牌宽度
存销比  销订比  进销存偏移率  单笔量  库存变化比
经营匹配指数  [质量存疑]   ← optional label column

Usage
-----
python scripts/train.py --data data/raw/your_file.tsv
