请将数据文件放在此目录：

- cluster_aop.xlsx：历史评论证据，最低包含 sentence, Sub_Issue, sentiment
- shap+TD_result.xlsx：性能/优化结果，默认 sheet_name=IPEA-NEW

系统会从真实 sentence 中检索 sentiment 最接近 target_score 的评论，用于生成目标状态、当前差距和简约行动方案。
