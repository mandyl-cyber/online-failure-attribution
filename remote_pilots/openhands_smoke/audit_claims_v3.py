"""Materialize explicit AI review decisions for the returned v3 development run.

This is a recorded review, not an automatic semantic judge. Never repairs responses.
"""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from remote_pilots.openhands_smoke import claims_v3 as pilot
from segmenter import stage_claim_summary as core

# Labels assess ALL clauses, the actual cited chunks, and epistemic source together.
DECISIONS = {
 'planning': [
  ('supported', 'low_value', '更新8条事项有原文支持，但计数对风险背景价值低。'),
  ('partial', 'source_mismatch', '所列计划内容有支持，且没有断言已执行；但按契约计划应为 agent_assertion。遗漏 ops.linspace/TensorFlow 对象。')],
 'edit_and_source_conditions': [
  ('supported', 'underspecified', '编辑及 step 计算发生变化有支持；缺少 dtype 和分支细节。'),
  ('partial', 'overclaim', 'tf.cast 语法可见，但 ensure correct consistency 超出所引源码的直接报告；应限定为将除数转成 start.dtype。'),
  ('unsupported', 'unverified_correctness', '没有运行证据证明 endpoint 已被正确处理或 step 准确；把修改升级为正确性结论。'),
  ('unsupported', 'instruction_as_completed_fact', '原文是 Review the changes ... 命令式提示，并没有已经复核且符合预期的记录。'),
  ('partial', 'inference_and_low_value', '7条更新事实有支持，indicating ongoing work 是额外概括，不能从数量推出进展。')],
 'verification_blocked': [],
 'edit_failed_then_successful_example': [
  ('partial', 'attempt_as_completed_and_source', '只引编辑动作便写 was edited；完整输入有回执但此条未引。agent_assertion 标签不能替代执行确认。'),
  ('unsupported', 'instruction_as_completed_fact', '工具要求复核，未报告实际已复核。'),
  ('supported', 'incomplete_detail', '引文包含 Node.js 运行堆栈、shell语法/命令错误和 exit 1；未保留 TypeError 细节。这里视为本次失败的概述，不视为根因诊断。'),
  ('partial', 'temporal_scope', '更新8条有支持，但发生在阶段内部，after the coding stage 的阶段完成表述不受支持。'),
  ('partial', 'missing_action_and_causal_overreach', '所引输出支持 exit 0；未引用前后动作，不能据此断言纠正了语法错误，也遗漏具体 babylon 示例及装饰器输出。'),
  ('partial', 'temporal_scope_and_low_value', '步骤32更新4条有支持；更新本身就是最后步骤的观察，不是另有证据的阶段结束后事件。')],
 'type_edit': [
  ('supported', 'useful', '旧新类型和编辑回执均被引用，核心类型修改准确。'),
  ('partial', 'missing_action', '输出只有1.22.10，不能单凭该引用确认是 yarn；配对动作提供了信息但未被引用。')],
 'commit': [
  ('partial', 'attempt_as_completed_and_source', 'git add/commit 命令只证明尝试，完成应引观察；下一条已有成功记录，此条还重复且无消息内容。'),
  ('supported', 'useful', '提交回执明确1文件、20插入、1删除和退出0；没有升级为功能验证。')]
}
# Full/partial/missing on accepted summaries, not on repaired failed output.
COVERAGE = {
 'planning': [('partial', [2], '保留流程但丢失具体目标对象，来源标签错误。')],
 'edit_and_source_conditions': [('partial', [1, 2], '概述改动，缺少NaN初始化及明确类型条件；含过度保证。'),
                                ('missing', [], '完全未提 ops源码和 axis 默认/后端限定。')],
 'verification_blocked': [('unavailable_failed_job', [], '协议失败，未生成可入账摘要。')] * 4,
 'edit_failed_then_successful_example': [('partial', [1], '缺少条件和相对 static 的顺序，引用也不完整。'),
                                      ('partial', [3], '保留失败和exit 1，未保留TypeError。'),
                                      ('partial', [5], '只有exit 0，未保留具体示例与输出范围。')],
 'type_edit': [('full', [1], '核心从PopperProps到Partial<PopperProps>的修改准确；未逐字重复未变交集不扣分。'),
               ('partial', [2], '版本内容正确，但引用缺少yarn命令以确认对象。')],
 'commit': [('partial', [2], '保留提交成功和改动计数，未保留commit hash。'),
            ('missing', [], '没有保留提交消息的具体意图。')]
}


def audit(source, out):
    rows = pilot.rows_at(source / 'development.jsonl')
    refs = pilot.read(source / 'reference_points.json')
    protocol = pilot.read(source / 'protocol.json')
    config = pilot.read(source / 'config.json')
    if rows != pilot.rows_at(pilot.ROOT / 'development.jsonl') or refs != pilot.read(pilot.ROOT / 'reference_points.json'):
        raise ValueError('returned inputs/reference pack differ')
    if protocol != pilot.read(pilot.ROOT / 'protocol.json') or config['protocol_sha256'] != core.digest(core.encoded(protocol)):
        raise ValueError('protocol drift')
    for key, path in [('core_sha256', Path(core.__file__)), ('runner_sha256', Path(pilot.__file__))]:
        if config[key] != core.digest(path.read_bytes().replace(b'\r\n', b'\n')):
            raise ValueError('code drift: ' + key)
    reviews, durations, usages, fingerprints = [], [], [], {}
    for row, reference in zip(rows, refs):
        core.verify_row(row)
        if row['job_id'] != reference['job_id']:
            raise ValueError('reference order mismatch')
        path = source / (row['job_id'] + '.json'); result = pilot.read(path)
        fingerprints[path.name] = core.digest(path.read_bytes())
        request = result['request']
        if request['messages'][0]['content'] != core.INSTRUCTION or core.strict_json(request['messages'][1]['content']) != row['request']:
            raise ValueError('actual request drift')
        if result['input_sha256'] != row['input_sha256'] or request['model'] != config['model'] or request['temperature'] != 0 or request['max_tokens'] != protocol['limits']['max_tokens']:
            raise ValueError('request configuration drift')
        reply = result['raw_response']; content = reply['choices'][0]['message']['content']
        if reply['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('unexpected truncation')
        if result['status'] == 'ready':
            if core.validate(core.strict_json(content), row) != result['claims']:
                raise ValueError('accepted claims changed')
        else:
            try:
                core.strict_json(content)
            except ValueError:
                pass
            else:
                raise ValueError('expected invalid JSON')
        category = reference['category']; claims = result.get('claims', [])
        if len(claims) != len(DECISIONS[category]):
            raise ValueError('review count mismatch')
        audited = [dict(claim_id=c['claim_id'], text=c['text'], evidence_ids=c['evidence_ids'],
                        reviewed=True, reviewer_type='AI', support=decision[0], issue=decision[1], note=decision[2])
                   for c, decision in zip(claims, DECISIONS[category])]
        coverage = [dict(point_id=p['point_id'], priority=p['priority'], text=p['text'],
                         status=d[0], matched_claim_ids=[claims[i-1]['claim_id'] for i in d[1]], note=d[2])
                    for p, d in zip(reference['reference_points'], COVERAGE[category])]
        failed_notes = []
        if category == 'verification_blocked':
            failed_notes = [
                'JSON中出现字符串 + 字符串，不是合法JSON；保留原失败，不修补计为成功。',
                '原始文本把 absl present? False 写成 confirmed the presence，是否定反转。',
                '原始文本声称 linspace imported successfully，但引用报告 tensorflow 缺失。',
                '这些仅做失败响应定性审查，不计入17条合法claim的分母。']
        reviews.append(dict(job_id=row['job_id'], category=category, status=result['status'],
                            reviewed=True, reviewer_type='AI', claims=audited, coverage=coverage,
                            failed_response_notes=failed_notes))
        durations.append(result['seconds']); usages.append(reply['usage'])
    summary = dict(status_counts=dict(Counter(r['status'] for r in reviews)),
                   accepted_claims=sum(len(r['claims']) for r in reviews),
                   support_counts=dict(Counter(c['support'] for r in reviews for c in r['claims'])),
                   key_coverage=dict(Counter(p['status'] for r in reviews for p in r['coverage'] if p['priority']=='key')),
                   auxiliary_coverage=dict(Counter(p['status'] for r in reviews for p in r['coverage'] if p['priority']!='key')),
                   seconds=dict(total=sum(durations), median=statistics.median(durations), maximum=max(durations)),
                   tokens={k: sum(u[k] for u in usages) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                   reviewer_type='AI', decision='do_not_integrate_or_expand_yet',
                   caveats=['Six selected historical development stages, not human accuracy or independent generalization.',
                            'Strict support combines all clauses, actual citations and epistemic source.',
                            'Reference points differ in granularity; coverage is diagnostic, not a population recall estimate.',
                            'v2/v21 use different inputs and success subsets: no direct improvement percentage.'],
                   returned_file_sha256=fingerprints)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [('ai_review.json', reviews), ('summary.json', summary)]:
        (out / name).write_text(__import__('json').dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(__import__('json').dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('swe_claims_v3'))
    parser.add_argument('--out', type=Path, default=Path('pilot/swe_claims_v3_audit'))
    args = parser.parse_args(); audit(args.source, args.out)
