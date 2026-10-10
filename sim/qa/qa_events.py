"""qa_events: every scripted event (greet, handoff, walkout, party, engagement onset) has a rule event of
the same type and subject within match_tol_s under each of the clip's eval presets; every rule
greet / walkout / handoff has a scripted counterpart."""
import sys

from sim.qa.common import clip_arg, load_qa, report
from sim import gt_events, labels


def run(clip_dir, qa):
    q = qa["events"]
    gt = labels.read_gt(clip_dir / "gt.json")
    scripted = gt["events_scripted"]
    fails, info = [], []
    types = tuple(q["types"]) + tuple(q.get("types_extra", []))
    for preset in gt.get("eval_presets", list(gt["events_by_rule"])):
        by_rule = gt["events_by_rule"][preset]
        un_s, un_r = gt_events.match_events(scripted, by_rule, q["match_tol_s"], types)
        un_r = [r for r in un_r if r["type"] in q.get("unscripted_rule_types_fail", q["types"])]
        info.append(f"[{preset}] scripted={[(e['type'], e['t']) for e in scripted]}")
        info.append(f"[{preset}] rule={[(e['type'], e['t']) for e in by_rule]}")
        for e in un_s:
            fails.append(f"[{preset}] scripted {e['type']} t={e['t']} has no rule event within ±{q['match_tol_s']} s")
        for e in un_r:
            fails.append(f"[{preset}] rule {e['type']} t={e['t']} has no scripted counterpart")
    if not scripted:
        fails.append("no scripted events in clip")
    return fails, info


def main(argv=sys.argv):
    fails, info = run(clip_arg(argv), load_qa())
    return report("qa_events", fails, info)


if __name__ == "__main__":
    sys.exit(main())
