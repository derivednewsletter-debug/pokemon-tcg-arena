/* AI Brain dashboard — renders /api/learn aggregates. */
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const esc = (s) => String(s === undefined || s === null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function pct(x) { return x === null || x === undefined ? "—" : Math.round(x * 100) + "%"; }

async function load() {
  let data;
  try {
    data = await (await fetch("/api/learn")).json();
  } catch (e) {
    $("#stat-cards").innerHTML = `<div class="empty-state">Could not reach the server: ${esc(e.message)}</div>`;
    return;
  }
  const p = data.profile || {};
  const n = p.n_games || 0;

  // stat cards
  const cards = $("#stat-cards");
  cards.innerHTML = "";
  const stat = (label, num, cls) => {
    const c = el("div", "stat-card");
    c.appendChild(el("div", "stat-label", label));
    c.appendChild(el("div", "stat-num " + (cls || ""), num));
    cards.appendChild(c);
  };
  stat("Games recorded", n, "accent");
  stat("AI win rate", pct(p.agent_win_rate), (p.agent_win_rate || 0) > 0.5 ? "good" : "bad");
  stat("Human win rate", pct(p.human_win_rate), (p.human_win_rate || 0) > 0.5 ? "good" : "");
  stat("Avg game length", (p.avg_turns === null || p.avg_turns === undefined) ? "—" : p.avg_turns + " turns", "");

  if (n === 0) {
    cards.appendChild(el("div", "empty-state",
      "No games yet — head to the arena and play a few. The dashboard fills in as people battle the AI."));
    return;
  }

  // action distribution (shares of human decisions)
  const bars = $("#action-bars");
  bars.innerHTML = "";
  const dist = p.action_dist || {};
  Object.entries(dist).forEach(([k, v]) => bars.appendChild(barRow(k, v)));

  // openers (raw counts, normalized to the most common opener)
  const open = $("#openers");
  open.innerHTML = "";
  const openers = p.openers || [];
  const maxO = Math.max(1, ...openers.map((o) => o.count || 0));
  openers.forEach((o) => open.appendChild(
    barRow(o.action, (o.count || 0) / maxO, o.count + " times")));

  // deck preferences (normalized)
  const dbar = $("#deck-bars");
  dbar.innerHTML = "";
  const decks = p.by_deck || {};
  const maxD = Math.max(1, ...Object.values(decks).map((v) => v || 0));
  Object.entries(decks).forEach(([k, v]) =>
    dbar.appendChild(barRow(k, v / maxD, v + " games")));

  // recent table
  const table = $("#recent-table");
  const tbody = el("tbody");
  const head = table.createTHead();
  head.innerHTML = `<tr><th>Result</th><th>Your deck</th><th>AI deck</th><th>Difficulty</th>
    <th>Turns</th><th>Prizes (you / AI)</th><th>When</th></tr>`;
  (data.recent || []).forEach((r) => {
    const tr = el("tr");
    const won = r.winner === 0;
    const tag = el("td", "win-tag " + (won ? "human" : "ai"), won ? "You won" : "AI won");
    tr.appendChild(tag);
    tr.appendChild(el("td", "", esc(r.human_deck || "—")));
    tr.appendChild(el("td", "", esc(r.ai_deck || "—")));
    tr.appendChild(el("td", "", esc(r.difficulty || "—")));
    tr.appendChild(el("td", "", r.turns === undefined ? "—" : String(r.turns)));
    tr.appendChild(el("td", "", `${r.human_prizes_left === undefined ? "?" : r.human_prizes_left} / ${r.ai_prizes_left === undefined ? "?" : r.ai_prizes_left}`));
    tr.appendChild(el("td", "", new Date((r.ts || 0) * 1000).toLocaleString()));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  if (!(data.recent || []).length) {
    $("#recent-empty").appendChild(el("div", "empty-state", "No recent games."));
  }
}

function barRow(label, frac, suffix) {
  const row = el("div", "bar-row");
  row.appendChild(el("span", "bar-label", esc(label)));
  const track = el("div", "bar-track");
  const fill = el("div", "bar-fill");
  fill.style.width = Math.min(100, Math.max(3, (frac || 0) * 100)) + "%";
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el("span", "bar-pct", suffix || pct(frac)));
  return row;
}

load();
