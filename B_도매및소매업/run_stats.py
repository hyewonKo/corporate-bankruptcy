# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings, os
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_parquet(os.path.join(BASE, "데이터", "8,9번.파일(최종)(진)", "M19_도매_소매업(최종)(진).parquet"))

features = [
    "금융비용부담률","매출총이익률_diff","총자본영업이익률_diff","영업CF_총부채_diff",
    "ROA_ratio","ROA변화","매출원가율","현금ROA","금융비용대매출액_ratio_industry",
    "순이익률_ratio","ROE_diff","순운전자본대총자본_ratio_industry","매입채무지급기간_diff",
    "비유동장기적합률_ratio","매출액순이익률_diff_industry","영업이익률_ratio",
    "매출채권회전율_diff","판관비율","부채비율","유보율_diff","자기자본비율_diff_industry",
    "총자산회전율_ratio","순이익증가율_diff","자본잠식률","부채비율변화","장기부채비율",
    "현금ROE_ratio","유동비율변화_diff","자기자본증가율","유형자산비율_ratio",
    "총자산증가율_diff_industry","순운전자본비율_diff","FCF_총자산_ratio",
    "영업현금흐름비율_diff","ROIC_diff","유동자산회전율","유동비율_ratio",
    "순운전자본회전율_diff","장기부채의존도","유형자산회전율_diff",
    "유형자산증가율_ratio_industry","비유동비율_ratio","매출액증가율","총부채비율",
    "비유동자산회전율_ratio","차입금의존도_ratio_industry","재고자산보유기간_ratio",
    "감가상각비율","영업CF_유동부채_diff","투하자본회전율_ratio","업력","순차입금비율"
]

LABEL = "부실라벨_ICR3년"
avail = [f for f in features if f in df.columns]
missing_f = [f for f in features if f not in df.columns]
if missing_f:
    print(f"[WARNING] 없는 피처: {missing_f}")

df_sub = df[avail + [LABEL]].copy()
n0 = (df_sub[LABEL]==0).sum()
n1 = (df_sub[LABEL]==1).sum()
print(f"전체: {len(df_sub):,}  |  정상(0): {n0:,}  |  부실(1): {n1:,}")

stats_list = []
for feat in avail:
    for lbl in [0, 1]:
        s = df_sub.loc[df_sub[LABEL]==lbl, feat].dropna()
        stats_list.append({
            "피처": feat,
            "라벨": "정상(0)" if lbl==0 else "부실(1)",
            "N": len(s),
            "평균": round(s.mean(), 6),
            "중앙값": round(s.median(), 6),
            "표준편차": round(s.std(), 6),
            "Q1": round(s.quantile(0.25), 6),
            "Q3": round(s.quantile(0.75), 6),
            "최소": round(s.min(), 6),
            "최대": round(s.max(), 6),
        })

stats_df = pd.DataFrame(stats_list)
out_path = os.path.join(BASE, "label_stats_before_outlier.csv")
stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"CSV 저장: {out_path}")

# 비교 출력
pivot_mean = stats_df.pivot(index="피처", columns="라벨", values="평균").reindex(avail)
pivot_med  = stats_df.pivot(index="피처", columns="라벨", values="중앙값").reindex(avail)
pivot_std  = stats_df.pivot(index="피처", columns="라벨", values="표준편차").reindex(avail)
pivot_q1   = stats_df.pivot(index="피처", columns="라벨", values="Q1").reindex(avail)
pivot_q3   = stats_df.pivot(index="피처", columns="라벨", values="Q3").reindex(avail)

print("\n" + "="*120)
print(f"{'피처':<38} {'정상평균':>10} {'부실평균':>10} {'차이(▲▼)':>11} | {'정상중앙':>10} {'부실중앙':>10} {'차이':>9} | {'정상std':>10} {'부실std':>10}")
print("="*120)
for feat in avail:
    m0  = pivot_mean.loc[feat, "정상(0)"]
    m1  = pivot_mean.loc[feat, "부실(1)"]
    md0 = pivot_med.loc[feat,  "정상(0)"]
    md1 = pivot_med.loc[feat,  "부실(1)"]
    s0  = pivot_std.loc[feat,  "정상(0)"]
    s1  = pivot_std.loc[feat,  "부실(1)"]
    dm  = m1 - m0
    dmd = md1 - md0
    sign = "▲" if dm > 0 else "▼"
    print(f"{feat:<38} {m0:>10.4f} {m1:>10.4f} {sign}{abs(dm):>10.4f} | {md0:>10.4f} {md1:>10.4f} {dmd:>+9.4f} | {s0:>10.4f} {s1:>10.4f}")
print("="*120)
