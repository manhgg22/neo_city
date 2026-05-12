import json

fails = ['pricing_008', 'pricing_009', 'pricing_011', 'pricing_015', 'pricing_016', 'pricing_019', 'sales_policy_001', 'sales_policy_003', 'sales_policy_007', 'sales_policy_008', 'product_003', 'product_004', 'product_005', 'amenities_009', 'objection_008']

with open('data/eval/retrieval_eval.jsonl', 'r', encoding='utf-8') as f:
    lines = [json.loads(l) for l in f]
    for d in lines:
        if d['id'] in fails:
            print(f"{d['id']}: {d['query']}")