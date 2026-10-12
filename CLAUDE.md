# Rules for Claude in this repo

## Never act without explicit approval

- **Never push to GitHub** (any branch, any remote, including force-push) unless the user explicitly approves that specific push in their current message.
- **Never commit** unless the user explicitly approves that specific commit.
- **Never rebuild, restart, or redeploy** containers, ngrok, or anything else running, unless the user explicitly approves it.
- Approval is per action. An earlier "push to main" or "rebuild" does not authorize later pushes or rebuilds. A request to make a change ("switch the clip", "revert the last commit") is approval to edit files locally only, not to commit or push.
- When an action is needed, describe it (exact command) and wait for the user's "go".

**Why:** the repo must not change after the hackathon deadline (1 pm), and Claude pushed two commits to `main` without asking.
