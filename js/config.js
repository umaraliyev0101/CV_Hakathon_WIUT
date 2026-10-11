// Showroom Suite settings. This is the only file you need to edit to connect the real servers.
// Each section switches from demo data to its real server on its own, so you can connect one at a time.
// See ../CONNECTING.md.
window.SUITE_CONFIG = {
  showroom: "Chilonzor showroom",

  coachUrl: "http://localhost:8000",   // Showroom Coach (salespeople)
  pulseUrl: "http://localhost:8001",   // Showroom Pulse (customers)

  live: {
    floor: false,     // Live floor: positions, waits, event log
    sessions: false,  // Replays: upload + session list
    team: false,      // Team: scorecards + staff suggestions (Coach)
    visitors: false,  // Visitors: heatmap, zones, routes (Pulse)
    demand: false,    // Demand: funnel, busy hours, car demand, campaign lift (marketing)
    insights: false,  // Insights: findings that combine both
  },

  pollMs: 1000,       // how often the Live floor asks for new positions
};

// Optional overrides from the address bar, for demos on another machine:
//   index.html?coach=http://10.0.0.5:8000&pulse=http://10.0.0.5:8001&live=team,visitors
(function () {
  const q = new URLSearchParams(location.search);
  const c = window.SUITE_CONFIG;
  if (q.get("coach")) c.coachUrl = q.get("coach");
  if (q.get("pulse")) c.pulseUrl = q.get("pulse");
  if (q.get("live")) {
    const on = q.get("live") === "all" ? Object.keys(c.live) : q.get("live").split(",");
    on.forEach((k) => { if (k in c.live) c.live[k] = true; });
  }
})();
