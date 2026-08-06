/* Replay JS — fetches a match record then steps through events one by one. */

const state = { seed: 42, record: null, cursor: 0, board: { p0: null, p1: null } };

async function load(seed) {
  state.seed = seed;
  const r = await fetch(`/api/replay/${seed}`, { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  state.record = data;
  state.cursor = 0;
  state.board = { p0: emptyBoard(), p1: emptyBoard() };
  document.getElementById("replay-meta").textContent =
    `Seed ${data.seed} · turns ${data.turns} · winner ${data.winner === null ? "?" : "P" + data.winner}`;
  document.getElementById("step-counter").textContent = `0 / ${data.events.length} events`;
  render();
  renderAnnotations();
}

function emptyBoard() {
  return { active: null, bench: [], hand: 7, deck: 0, prize: 6 };
}

function stepEvents(record, idx) {
  const evs = [];
  for (let i = 0; i <= idx && i < record.events.length; i++) evs.push(record.events[i]);
  return evs;
}

function reconstruct(evs, who) {
  const board = emptyBoard();
  board.hand = 6 - 1; // rough starting hand after first draw
  for (const e of evs) {
    applyEvent(board, e, who);
  }
  return board;
}

function applyEvent(board, e, playerIdx) {
  if (e.kind === "SETUP") return;
  if (e.kind === "DRAW") {
    board.hand = Math.min(10, board.hand + 1);
    board.deck = Math.max(0, board.deck - 1);
    return;
  }
  if (e.kind === "STEP_START") {
    if (e.player === playerIdx) {
      board.hand = e.hand_size;
      board.deck = e.deck_size;
    }
    return;
  }
  if (e.kind === "ACTION") {
    const a = e.action || {};
    if (e.player !== playerIdx) return;
    if (a.kind === "PLAY_POKEMON") {
      board.bench.push({ name: a.extra || "Pokemon", hp: 100 });
      board.hand -= 1;
    }
    if (a.kind === "EVOLVE") board.hand -= 1;
    if (a.kind === "ATTACH_ENERGY") board.hand -= 1;
    if (a.kind === "PLAY_TRAINER") board.hand -= 1;
    return;
  }
  if (e.kind === "ATTACK") {
    if (e.player === playerIdx) {
      board.active = { name: e.move, hp: 100 };
    }
    return;
  }
  if (e.kind === "PRIZE_LOST" && e.player === playerIdx) {
    board.prize = Math.max(0, board.prize - 1);
    return;
  }
  if (e.kind === "PRIZE_TAKEN" && e.player === playerIdx) {
    board.prize = Math.max(0, board.prize - 1);
  }
}

function renderBoard(board, label) {
  const lines = [];
  lines.push(`<div class="pokemon-line"><span class="name">${label} Active</span>` +
            (board.active ? `<span class="hp">HP ${board.active.hp ?? "?"}</span>` : '<span class="hp low">none</span>') +
            `</div>`);
  board.bench.forEach((p, i) => {
    lines.push(`<div class="pokemon-line">` +
              `<span class="name">Bench ${i + 1}</span>` +
              `<span class="hp">${p.name}</span>` +
              `</div>`);
  });
  return `
    <div class="board">
      <h3>${label}</h3>
      <div class="row"><span class="seed-input"><label>Prizes <b>${board.prize}</b></label></span></div>
      <div class="row"><span class="seed-input"><label>Hand size <b>${board.hand}</b></label></span></div>
      <div class="row"><span class="seed-input"><label>Deck size <b>${board.deck}</b></label></span></div>
      ${lines.join("")}
    </div>
  `;
}

function render() {
  if (!state.record) return;
  const events = state.record.events;
  const evsTo = stepEvents(state.record, state.cursor);
  state.board.p0 = reconstruct(evsTo, 0);
  state.board.p1 = reconstruct(evsTo, 1);

  const boards = document.getElementById("boards");
  boards.innerHTML = renderBoard(state.board.p0, "P0") + renderBoard(state.board.p1, "P1");

  const log = document.getElementById("event-log");
  log.innerHTML = "";
  events.forEach((e, idx) => {
    const li = document.createElement("li");
    let cls = "";
    if (idx < state.cursor) cls += "step-done";
    if (idx === state.cursor) cls += "step-current";
    if (["ATTACK", "RETREAT", "PRIZE_LOST", "GAME_OVER"].includes(e.kind)) cls += " headline";
    li.className = cls.trim();
    li.innerHTML = `
      <span class="kind">${e.kind || ""}</span>
      <span class="player">P${e.player ?? "?"}</span>
      <span>turn ${e.turn ?? "?"}</span>
      <span class="desc">${formatEvent(e)}</span>
    `;
    log.appendChild(li);
  });

  document.getElementById("step-counter").textContent =
    `${state.cursor} / ${events.length} events`;
}

function formatEvent(e) {
  const parts = [];
  if (e.action && e.action.kind) parts.push(e.action.kind + (e.action.extra ? `(${e.action.extra})` : ""));
  if (e.from) parts.push(`${e.from} → ${e.to}`);
  if (e.move) parts.push(`${e.move} dmg=${e.damage ?? "?"}`);
  if (e.elapsed_ms != null) parts.push(`${e.elapsed_ms}ms`);
  if (e.remaining != null) parts.push(`prizes_left=${e.remaining}`);
  return parts.join(" · ") || JSON.stringify(e).slice(0, 120);
}

function renderAnnotations() {
  const wrap = document.getElementById("annotations");
  const fail = state.record && state.record.failure;
  if (!fail) { wrap.innerHTML = "<i>loading…</i>"; return; }
  const winner = state.record.winner;
  const labels = { p0: "Player 0", p1: "Player 1" };
  const cards = [0, 1].map((pi) => {
    const cat = fail["p" + pi];
    let outcome = "WIN";
    if (winner !== null && winner !== pi) outcome = "LOSS";
    else if (winner === null) outcome = "DRAW";
    const isWin = outcome === "WIN";
    return `
      <div class="failure-card">
        <div class="badge ${isWin ? "win" : outcome === "DRAW" ? "draw" : ""}">${cat}</div>
        <div class="label"><b>${labels["p" + pi]}</b>${outcome}</div>
        <div>${failureDescription(cat)}</div>
      </div>
    `;
  });
  wrap.innerHTML = cards.join("");
}

const FAIL_DESC = {
  DECK_OUT: "Unable to draw at the start of a turn — opponent wins by deck exhaustion.",
  NO_KO: "Never took a single prize — board never developed enough offensive pressure.",
  POOR_SETUP: "Stayed on Basics without evolving, or kept an empty bench throughout.",
  PRIZE_TRADE: "Prize trade was unfavorable — opponent matched or exceeded KOs.",
  OVER_EXT: "Committed the whole bench into one attack that could be punished.",
  ENERGY_STARVED: "Attached energy but never reached a usable attack.",
  NOT_CATEGORIZED: "Loss didn't fit a known pattern — see the event log for context.",
  WIN: "Match ended in a win for this perspective — failure analysis not applicable.",
};
function failureDescription(cat) { return FAIL_DESC[cat] || ""; }

async function init() {
  document.getElementById("btn-prev").addEventListener("click", () => {
    state.cursor = Math.max(0, state.cursor - 1);
    render();
  });
  document.getElementById("btn-next").addEventListener("click", () => {
    if (!state.record) return;
    state.cursor = Math.min(state.record.events.length - 1, state.cursor + 1);
    render();
  });
  document.getElementById("btn-load").addEventListener("click", async () => {
    const v = parseInt(document.getElementById("seed-input").value, 10);
    if (Number.isFinite(v)) {
      try { await load(v); } catch (e) { alert("Load failed: " + e); }
    }
  });
  document.getElementById("seed-input").value = state.seed;
  try { await load(state.seed); } catch (e) {
    document.getElementById("replay-meta").textContent = "Failed: " + e;
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft") { state.cursor = Math.max(0, state.cursor - 1); render(); }
    if (ev.key === "ArrowRight") { if (state.record) state.cursor = Math.min(state.record.events.length - 1, state.cursor + 1); render(); }
  });
}

init();
