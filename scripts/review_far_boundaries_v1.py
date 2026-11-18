"""Persist AI review of the frozen local-prefix packets with explicit uncertainty."""
import json
from collections import Counter
from pathlib import Path

# Judgments concern the displayed local prefix, not a replacement gold annotation.
REVIEWS = [
("environment_retry",False,"缺模块后检查 pip，仍在恢复同一复现环境。"),
("verification_transition",None,"从源码和计划转向首次 pytest，但工作目的是否换段取决于阶段粒度。"),
("plan_bookkeeping",False,"计划回顾枚举键问题，后续继续复现和修补同一问题。"),
("same_goal_exploration",False,"阅读 dateformat 测试为同一日期格式问题服务。"),
("insufficient_context",None,"跨 YAML 对象和模板模块编辑，局部前缀不足以确定是否独立子目标。"),
("environment_retry",False,"安装 pytest 后再尝试测试，仍在定位可运行测试入口。"),
("same_goal_exploration",False,"hygiene 失败后继续查看同一文件，属于格式修正过程。"),
("same_goal_iteration",False,"修改 Point 虚部判断，仍在迭代同一修复。"),
("setup_exploration",False,"初始化计划后列目录，局部证据未显示独立工作目标。"),
("same_goal_exploration",False,"继续阅读已定位的 URL 辅助函数。"),
("environment_retry",False,"测试入口失败后列目录查找入口。"),
("environment_retry",False,"准备复现脚本后检查 Node 版本。"),
("status_bookkeeping",False,"复现导入尝试之后查看 git 状态，后续继续处理导入。"),
("verification_transition",None,"测试之后改用 Mod 行为探针，同一修复内是否单独成段无法确定。"),
("setup_exploration",None,"开始定位 sparse_fit，可能是实质探索起点，也可能延续初始阅读。"),
("same_goal_exploration",False,"测试后查 forms 入口，后续继续测试相同表单问题。"),
("status_bookkeeping",False,"序列化复现后列测试目录，局部证据未显示新子目标。"),
("same_goal_exploration",False,"运行测试后查相同 route_target 测试字段。"),
("same_goal_iteration",None,"提交后继续修改 responsiveFontSizes，可能重开修复，也可能只是连续完善。"),
("plan_bookkeeping",False,"修改 mask 预处理后更新计划，后续继续同一文件修复。"),
("same_goal_iteration",False,"测试和源码检查后修改相同 transfo_xl 逻辑。"),
("same_goal_exploration",False,"修改 chunksizes 后查询相同函数调用位置。"),
("delivery_transition",None,"验证后检查 git 状态并提交，是否单独交付阶段取决于粒度。"),
("plan_bookkeeping",False,"恢复 gitignore 并提交后回填任务清单。"),
("setup_exploration",False,"初始计划后的目录查看缺少独立目的变化证据。"),
("same_goal_iteration",False,"测试受阻后继续同一 URL 修复。"),
("same_goal_exploration",False,"测试与解释器查找交替出现，仍在定位同一依赖问题。"),
("verification_transition",None,"复现脚本转为相关测试，目的连续但验证范围变化。"),
("verification_transition",None,"计划之后定向验证 OrderedSet，是否开始新阶段尚有粒度歧义。"),
("same_goal_iteration",False,"类型解析修改后立即复现，并继续修补同一逻辑。"),
("same_goal_exploration",False,"继续检索 vault/YAML 解密调用关系。"),
("environment_retry",False,"查测试入口与 Python 后执行测试，后续修正命令参数。"),
("setup_exploration",False,"初始任务清单之后列项目目录。"),
("same_goal_exploration",False,"描述符探针之后继续定位 autodoc 成员检查逻辑。"),
("implementation_transition",None,"阅读策略模块后编辑 play_iterator，可能进入实现，仍服务同一问题。"),
("same_goal_exploration",False,"添加 gzip 导入后查相关测试，后续继续同一实现。"),
("same_goal_exploration",False,"计划更新后查缓存清理函数，仍在同一 resolver 修复。"),
("plan_bookkeeping",False,"提交后回填任务计划。"),
("verification_transition",None,"relim 修改后运行同一复现，是验证转换，独立阶段定义待确认。"),
("plan_bookkeeping",False,"计划从探索改为分析，后续仍测试并设计同一修改。"),
]


def main():
    root = Path("pilot/far_boundary_review_v1")
    packets = json.loads((root/"packets.json").read_text(encoding="utf-8"))
    assert len(packets) == len(REVIEWS) == 40
    reviewed = []
    for row, (kind, supported, note) in zip(packets,REVIEWS):
        reviewed.append(dict(traj_id=row["traj_id"],step=row["step"],category=kind,
            reviewed=True, reviewer_type="AI",prefix_supports_boundary=supported,
            retrospective_supports_boundary=None,reason=note,
            scope="Local prefix and displayed later steps; full-trajectory adjudication pending."))
    summary = dict(reviewed=40,categories=dict(Counter(x["category"] for x in reviewed)),
        prefix_supported=sum(x["prefix_supports_boundary"] is True for x in reviewed),
        prefix_unsupported=sum(x["prefix_supports_boundary"] is False for x in reviewed),
        prefix_uncertain=sum(x["prefix_supports_boundary"] is None for x in reviewed),
        limitations="AI local-context review; retrospective gold labels unchanged.")
    (root/"ai_review.json").write_text(json.dumps(reviewed,ensure_ascii=False,indent=2),encoding="utf-8")
    (root/"review_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    main()
