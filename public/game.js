/* Pokémon TCG Arena — frontend
   Renders the board from the server's view JSON, presents the human's
   action menu, and talks to /api/game/*. */
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const APP = {
  cards: {},
  energy: {},
  decks: [],
  diffs: [],
  view: null,
  gameId: null,
  busy: false,
  sel: [],
  pick: { human: null, ai: null, diff: null },
  logRendered: 0,
};

/* ---------------- type styling ---------------- */
const TYPE = {
  0: { c: "#cfd4da", bg: "#2c3240", letter: "C" }, // colorless
  1: { c: "#6fd45c", bg: "#24351f", letter: "G" }, // grass
  2: { c: "#ff6b5e", bg: "#3a211e", letter: "R" }, // fire
  3: { c: "#5aa9ff", bg: "#1d2c42", letter: "W" }, // water
  4: { c: "#ffd166", bg: "#3d3319", letter: "L" }, // lightning
  5: { c: "#c58aff", bg: "#2e223f", letter: "P" }, // psychic
  6: { c: "#ff9f45", bg: "#3c2a18", letter: "F" }, // fighting
  7: { c: "#8f8fa8", bg: "#26262e", letter: "D" }, // darkness
  8: { c: "#b7c3d4", bg: "#2c323c", letter: "M" }, // metal
  9: { c: "#e0b84e", bg: "#3d3316", letter: "Y" }, // dragon
  10:{ c: "#ff8ae0", bg: "#352039", letter: "★" }, // rainbow
  11:{ c: "#c0a080", bg: "#332a1d", letter: "T" }, // team rocket
};
const typeOf = (et) => TYPE[et] || TYPE[0];

const KIND_ICO = {
  attack: "⚔️", play: "🃏", attach: "⚡", evolve: "🔄",
  retreat: "🏃", ability: "✨", end: "⏹", discard: "🗑", card: "🎴",
  first: "🌅", second: "🌙", other: "•",
};

/* ---------------- boot ---------------- */
async function boot() {
  try {
    const meta = await (await fetch("/api/game/meta")).json();
    APP.decks = meta.decks || [];
    APP.diffs = meta.difficulties || [];
    APP.cards = (await (await fetch("cards.json")).json()).cards || {};
  } catch (e) {
    toast("Could not reach the game server: " + e.message, true);
    return;
  }
  // default selections
  APP.pick.human = APP.decks[0] && APP.decks[0].id;
  APP.pick.ai = APP.decks[0] && APP.decks[0].id;
  APP.pick.diff = (APP.diffs[1] && APP.diffs[1].id) || "medium";
  renderStart();
  wireStart();
}

/* ---------------- start screen ---------------- */
function renderStart() {
  const dl = $("#deck-list"), al = $("#ai-deck-list"), fl = $("#difficulty-list");
  dl.innerHTML = ""; al.innerHTML = ""; fl.innerHTML = "";
  APP.decks.forEach((d) => {
    const n = Object.keys(APP.cards).length;
    const cards = (d.cards || []).map((cid) => APP.cards[String(cid)]).filter(Boolean);
    const pokemon = cards.filter((c) => c.type === "pokemon").length;
    const b = el("button", "pick", "");
    b.dataset.id = d.id;
    b.innerHTML = `<span class="pick-name">${esc(d.name)}</span>
      <span class="pick-blurb">${esc(d.blurb || "")}</span>
      <span class="pick-meta">${pokemon} Pokémon · ${cards.length} cards · ${n} card pool</span>`;
    dl.appendChild(b);
    const b2 = b.cloneNode(true);
    al.appendChild(b2);
    if (d.id === APP.pick.human) b.classList.add("selected");
    if (d.id === APP.pick.ai) b2.classList.add("selected");
    b.onclick = () => { APP.pick.human = d.id; markSelected(dl, d.id); };
    b2.onclick = () => { APP.pick.ai = d.id; markSelected(al, d.id); };
  });
  APP.diffs.forEach((d) => {
    const b = el("button", "pick", "");
    b.dataset.id = d.id;
    b.innerHTML = `<span class="pick-name">${esc(d.name)}</span>
      <span class="pick-blurb">${esc(d.blurb || "")}</span>`;
    fl.appendChild(b);
    if (d.id === APP.pick.diff) b.classList.add("selected");
    b.onclick = () => { APP.pick.diff = d.id; markSelected(fl, d.id); };
  });
}

function markSelected(listEl, id) {
  listEl.querySelectorAll(".pick").forEach((p) =>
    p.classList.toggle("selected", p.dataset.id === id));
}

function wireStart() {
  $("#start-btn").onclick = startGame;
  $("#quit-btn").onclick = showStart;
  $("#modal").addEventListener("click", (e) => {
    if (e.target === $("#modal")) hideModal();
  });
}

async function startGame() {
  const btn = $("#start-btn");
  btn.disabled = true; btn.textContent = "⏳ Preparing battle…";
  try {
    const r = await fetch("/api/game/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        human_deck: APP.pick.human,
        ai_deck: APP.pick.ai,
        difficulty: APP.pick.diff,
      }),
    });
    const view = await r.json();
    if (!r.ok) throw new Error(view.error || r.statusText);
    enterGame(view);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "⚔️  Start Battle";
  }
}

function enterGame(view) {
  APP.view = view;
  APP.gameId = view.game_id;
  APP.logRendered = 0;
  $("#start").classList.add("hidden");
  $("#game").classList.remove("hidden");
  $("#my-deck-name").textContent = deckName(view.decks && view.decks.human);
  $("#ai-deck-name").textContent = deckName(view.decks && view.decks.ai);
  render(view);
}

function showStart() {
  APP.gameId = null;
  $("#game").classList.add("hidden");
  $("#start").classList.remove("hidden");
}

/* ---------------- render ---------------- */
function render(view) {
  APP.view = view;
  const human = view.human, ai = view.ai;
  renderPokemon($("#ai-active"), ai.active, false, "active");
  renderBench($("#ai-bench"), ai.bench, false);
  renderPokemon($("#my-active"), human.active, true, "active");
  renderBench($("#my-bench"), human.bench, true);
  renderHand(human.hand);
  renderPiles(view);
  renderMenu(view.menu);
  renderLog(view.log);
  renderBanner(view);
  if (view.over) showGameOver(view);
}

function deckName(id) {
  const d = APP.decks.find((x) => x.id === id);
  return d ? d.name : (id || "");
}

function renderPiles(view) {
  setPile($("#ai-prizes"), view.ai.prizeCount);
  setPile($("#ai-deck"), view.ai.deckCount);
  setPile($("#ai-discard"), view.ai.discardCount);
  setPile($("#my-prizes"), view.human.prizeCount);
  setPile($("#my-deck"), view.human.deckCount);
  setPile($("#my-discard"), view.human.discardCount);
  $("#game-head-right").textContent = view.human.prizeCount === undefined
    ? "" : `Prize ${6 - (view.human.prizeCount || 0)} / 6 taken`;
}

function setPile(node, n) {
  const capText = node.dataset.cap ||
    (node.querySelector(".pile-cap") ? node.querySelector(".pile-cap").textContent : "");
  node.innerHTML = "";
  node.appendChild(el("div", "pile-cap", capText));
  node.appendChild(el("div", "pile-num", n === undefined ? "–" : String(n)));
  node.dataset.cap = capText;
}

function renderBanner(view) {
  const b = $("#turn-banner");
  if (view.over) {
    b.textContent = `Match over — turn ${view.turn}`;
    b.className = "turn-banner";
    return;
  }
  const yours = view.yourIndex === 0;
  b.textContent = `Turn ${view.turn} — ${yours ? "Your move" : "AI is thinking…"}`;
  b.className = "turn-banner " + (yours ? "your-turn" : "ai-turn");
}

/* ---------------- cards ---------------- */
function renderPokemon(slot, pkm, mine, role) {
  slot.innerHTML = "";
  if (!pkm) {
    slot.appendChild(el("div", "pile", ""));
    return;
  }
  const cd = cardData(pkm.id);
  const card = buildPkmCard(pkm, cd, role);
  slot.appendChild(card);
}

function renderBench(row, bench, mine) {
  row.innerHTML = "";
  if (!bench || !bench.length) {
    row.appendChild(el("div", "energy-none", "— empty bench —"));
    return;
  }
  bench.forEach((p) => {
    const cd = cardData(p.id);
    row.appendChild(buildPkmCard(p, cd, "bench"));
  });
}

function buildPkmCard(p, cd, role) {
  const et = cd.energyType !== undefined ? cd.energyType : 0;
  const t = typeOf(et);
  const card = el("div", "tcg-card pkm-card" + (role === "bench" ? " bench-card" : "") +
    (p.hp <= 0 ? " knocked" : ""));
  card.style.setProperty("--card-bg", t.bg);
  card.style.setProperty("--card-line", t.c + "55");
  card.style.setProperty("--card-glow", t.c + "33");
  card.style.setProperty("--card-glow-soft", t.c + "22");

  const top = el("div", "pkm-top");
  top.appendChild(el("div", "pkm-name", esc(p.name || cd.name || "?")));
  const hpEl = el("div", "pkm-hp");
  hpEl.innerHTML = "<b>" + p.hp + "</b> / " + (p.maxHp || cd.hp || "?") + " HP";
  top.appendChild(hpEl);
  card.appendChild(top);

  if (p.maxHp) {
    const bar = el("div", "hpbar");
    const fill = el("div", "hpbar-fill" + (p.hp < p.maxHp * 0.35 ? " low" : ""));
    fill.style.width = Math.max(2, (p.hp / p.maxHp) * 100) + "%";
    bar.appendChild(fill);
    card.appendChild(bar);
  }

  const er = el("div", "energy-row");
  (p.energies || []).forEach((e) => er.appendChild(energyChip(e)));
  if (!(p.energies || []).length) er.appendChild(el("span", "energy-none", "no energy"));
  card.appendChild(er);

  if ((p.status && Object.keys(p.status).length) || (p.tools && p.tools.length)) {
    const sr = el("div", "status-row");
    (p.tools || []).forEach((t2) => sr.appendChild(el("span", "tool-tag", "🛠 " + esc(t2))));
    Object.entries(p.status || {}).forEach(([k, v]) => {
      if (v) sr.appendChild(el("span", "status-badge", esc(k)));
    });
    card.appendChild(sr);
  }

  (cd.attacks || []).slice(0, 2).forEach((a) => {
    const line = el("div", "atk-line");
    const cost = el("span", "atk-cost");
    (a.cost || []).forEach((e) => cost.appendChild(energyChip(e, true)));
    const nm = el("span", "atk-name", esc(a.name || ""));
    const dm = el("div", "atk-dmg", (a.damage || 0) + "");
    line.appendChild(cost); line.appendChild(nm); line.appendChild(dm);
    card.appendChild(line);
  });

  const rr = el("div", "retreat-row");
  rr.textContent = "↩ retreat " + (cd.retreat || 0);
  card.appendChild(rr);
  return card;
}

function energyChip(e, small) {
  const t = typeOf(Number(e));
  const chip = el("span", "energy-chip" + (small ? "" : ""));
  chip.textContent = t.letter;
  chip.style.background = t.c;
  chip.title = APP.energy[String(e)] || "energy";
  return chip;
}

function renderHand(hand) {
  const row = $("#my-hand");
  row.innerHTML = "";
  if (!hand || !hand.length) {
    row.appendChild(el("div", "energy-none", "— no cards in hand —"));
    return;
  }
  hand.forEach((c) => {
    const cd = cardData(c.id);
    const t = typeOf(cd.energyType !== undefined ? cd.energyType : 0);
    const card = el("div", "tcg-card hand-card" + (cd.type === "energy" ? " energy-card-style" : ""));
    card.style.setProperty("--card-bg", cd.type === "energy" ? "#1a2030" : t.bg);
    card.style.setProperty("--card-line", t.c + "55");
    card.style.setProperty("--card-glow", t.c + "33");
    card.dataset.id = c.id;
    if (cd.type === "energy") {
      const chip = energyChip(cd.energyType);
      chip.style.width = "34px"; chip.style.height = "34px"; chip.style.fontSize = "15px";
      card.appendChild(chip);
      card.appendChild(el("div", "pkm-name", esc(cd.name)));
    } else {
      card.appendChild(el("div", "card-type-tag", cd.type === "pokemon"
        ? (cd.stage ? cd.stage.toUpperCase() : "POKÉMON") : cd.type.toUpperCase()));
      card.appendChild(el("div", "pkm-name", esc(cd.name)));
      const atk = (cd.attacks || [])[0];
      if (atk) {
        const ma = el("div", "mini-attack");
        ma.innerHTML = esc(atk.name) +
          " <span class='mini-dmg'>" + (atk.damage || 0) + "</span>";
        card.appendChild(ma);
      }
      if (cd.hp) card.appendChild(el("div", "card-type-tag", cd.hp + " HP"));
    }
    if (APP.sel.includes(c.id)) card.classList.add("selected");
    card.onclick = () => onHandCardClick(card, c.id);
    row.appendChild(card);
  });
}

function onHandCardClick(card, cid) {
  const menu = APP.view.menu;
  if (!menu || menu.type !== "SETUP") return;
  const item = (menu.items || []).find((i) => i.cardId === cid);
  if (!item) return;
  if (menu.maxCount > 1) {
    const i = APP.sel.indexOf(cid);
    if (i >= 0) APP.sel.splice(i, 1); else APP.sel.push(cid);
    card.classList.toggle("selected", APP.sel.includes(cid));
    updateConfirm();
  } else {
    APP.sel = [cid];
    submitPicks([item.index]);
  }
}

/* ---------------- menu ---------------- */
function renderMenu(menu) {
  const panel = $("#menu-panel");
  if (!menu) {
    panel.innerHTML = `<div class="ai-thinking"><div class="spinner"></div>
      <span>AI is playing its turn…</span></div>`;
    return;
  }
  APP.sel = [];
  panel.innerHTML = "";
  panel.appendChild(el("div", "menu-title", menu.prompt || "Choose an action"));

  if (menu.type === "IS_FIRST") {
    const sub = el("div", "menu-sub", "Going first means your opponent can't attack on turn 1 — but you move first.");
    panel.appendChild(sub);
    const list = el("div", "action-list");
    menu.items.forEach((it) => {
      const b = el("button", "action-btn " + (it.kind || ""));
      b.innerHTML = `<span class="action-ico">${KIND_ICO[it.kind] || "•"}</span><span>${esc(it.label)}</span>`;
      b.onclick = () => submitPicks([it.index]);
      list.appendChild(b);
    });
    panel.appendChild(list);
    return;
  }

  if (menu.type === "SETUP") {
    const sub = el("div", "menu-sub",
      (menu.maxCount > 1 ? `Pick ${menu.minCount}–${menu.maxCount} cards from your hand, then confirm.` :
        "Tap the card you want."));
    panel.appendChild(sub);
    if (menu.maxCount > 1) {
      const footer = el("div", "menu-footer");
      const c = el("button", "btn-primary confirm-btn");
      c.id = "confirm-btn";
      c.textContent = "Confirm";
      c.disabled = true;
      c.onclick = () => {
        const picks = (menu.items || [])
          .filter((i) => APP.sel.includes(i.cardId))
          .map((i) => i.index);
        if (picks.length) submitPicks(picks);
      };
      footer.appendChild(el("div", "selection-count", `Selected: 0`));
      footer.appendChild(c);
      panel.appendChild(footer);
    }
    return;
  }

  // MAIN
  const sub = el("div", "menu-sub", "Pick one action. Playing a card keeps your turn going.");
  panel.appendChild(sub);
  const list = el("div", "action-list");
  menu.items.forEach((it) => {
    const b = el("button", "action-btn " + (it.kind || ""));
    b.innerHTML = `<span class="action-ico">${KIND_ICO[it.kind] || "•"}</span><span>${esc(it.label)}</span>`;
    b.onclick = () => submitPicks([it.index]);
    list.appendChild(b);
  });
  panel.appendChild(list);
}

function updateConfirm() {
  const c = $("#confirm-btn");
  if (!c) return;
  c.disabled = APP.sel.length === 0;
  const sc = document.querySelector(".selection-count");
  if (sc) sc.textContent = `Selected: ${APP.sel.length}`;
}

/* ---------------- actions ---------------- */
async function submitPicks(picks) {
  if (APP.busy || !APP.gameId) return;
  APP.busy = true;
  renderMenu(null); // show spinner
  const panel = $("#menu-panel");
  panel.innerHTML = `<div class="ai-thinking"><div class="spinner"></div>
    <span>Resolving…</span></div>`;
  try {
    const r = await fetch("/api/game/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: APP.gameId, picks }),
    });
    const view = await r.json();
    if (r.status === 404 || (view.error === "game_expired")) {
      showExpired();
      return;
    }
    if (!r.ok) throw new Error(view.error || r.statusText);
    APP.view = view;
    render(view);
  } catch (e) {
    toast(e.message, true);
    if (APP.view) renderMenu(APP.view.menu);
  } finally {
    APP.busy = false;
  }
}

/* ---------------- log ---------------- */
function renderLog(lines) {
  const log = $("#log-panel");
  const fresh = lines.slice(APP.logRendered);
  APP.logRendered = lines.length;
  fresh.forEach((line) => {
    let cls = "line-sys";
    if (line.startsWith("You")) cls = "line-you";
    else if (line.startsWith("AI")) cls = "line-ai";
    if (line.includes("win")) cls = line.startsWith("You") ? "line-win" : "line-lose";
    const d = el("div", cls, line);
    log.appendChild(d);
  });
  log.scrollTop = log.scrollHeight;
}

/* ---------------- game over / modals ---------------- */
function showGameOver(view) {
  const won = view.winner === 0;
  const modal = $("#modal-card");
  modal.innerHTML = `
    <div class="modal-title ${won ? "win" : "lose"}">${won ? "🎉 You win!" : "😤 The AI wins"}</div>
    <div class="modal-sub">
      Turn ${view.turn} · your deck: ${esc(deckName(view.decks.human))} vs
      ${esc(deckName(view.decks.ai))}<br>
      Prizes left — you: ${view.human_prizes_left}, AI: ${view.ai_prizes_left}<br><br>
      ${won ? "Nice. The AI just logged your playstyle and will be slightly harder next time."
            : "It learned from this game. Rematch and surprise it!"}
    </div>
    <div class="modal-actions">
      <button class="btn-primary" id="again-btn">⚔️ Rematch</button>
      <button class="btn-ghost" id="stats-btn">📊 AI Brain</button>
    </div>`;
  showModal();
  $("#again-btn").onclick = () => { hideModal(); startGame(); };
  $("#stats-btn").onclick = () => { window.location.href = "stats.html"; };
}

function showExpired() {
  const modal = $("#modal-card");
  modal.innerHTML = `
    <div class="modal-title">💤 Server woke up</div>
    <div class="modal-sub">This game lived on a server instance that went to sleep
      (serverless!). Your finished games are still recorded — start a fresh battle.</div>
    <div class="modal-actions">
      <button class="btn-primary" id="again-btn">⚔️ New Battle</button>
    </div>`;
  showModal();
  $("#again-btn").onclick = () => { hideModal(); showStart(); };
}

function showModal() { $("#modal").classList.remove("hidden"); }
function hideModal() { $("#modal").classList.add("hidden"); }

let toastTimer = null;
function toast(msg, isError) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3800);
}

function cardData(id) {
  const c = APP.cards[String(id)];
  return c || { name: "Card #" + id, type: "pokemon", energyType: 0, hp: 0, attacks: [], retreat: 0 };
}

function esc(s) {
  return String(s === undefined || s === null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

boot();
