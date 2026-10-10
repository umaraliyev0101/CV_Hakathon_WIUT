"""qa_events: every scripted greet has a rule-based greet within tolerance and vice versa."""
import sys

from sim.qa.common import clip_arg, load_qa, report
from sim import gt_events, labels


def run(clip_dir, qa):
    q = qa["events"]
    gt = labels.read_gt(clip_dir / "gt.json")
    scripted = gt["events_scripted"]
    fails, info = [], []
    for preset, by_rule in gt["events_by_rule"].items():
        un_s, un_r = gt_events.match_events(scripted, by_rule, q["match_tol_s"], tuple(q["types"]))
        info.append(f"[{preset}] scripted={[(e['type'], e['t']) for e in scripted]} "
                    f"rule={[(e['type'], e['t']) for e in by_rule]}")
        for e in un_s:
            fails.append(f"[{preset}] scripted {e['type']} t={e['t']} has no rule event within ±{q['match_tol_s']} s")
        for e in un_r:
            fails.append(f"[{preset}] rule {e['type']} t={e['t']} has no scripted counterpart")
    if not any(e["type"] == "greet" for e in scripted):
        fails.append("no scripted greet in clip")
    return fails, info


def main(argv=sys.argv):
    fails, info = run(clip_arg(argv), load_qa())
    return report("qa_events", fails, info)


if __name__ == "__main__":
    sys.exit(main())
