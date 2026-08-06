/* Deck Builder — browse the full card pool, build a 60-card deck,
   enforce TCG rules, and hand the deck to the arena. */
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

const TYPE = {
  0: { c: "#cfd4da", bg: "#2c3240" }, 1: { c: "#6fd45c", bg: "#24351f" },
  2: { c: "#ff6b5e", bg: "#3a211e" }, 3: { c: "#5aa9ff", bg: "#1d2c42" },
  4: { c: "#ffd166", bg: "#3d3319" }, 5: { c: "#c58aff", bg: "#2e223f" },
  6: { c: "#ff9f45", bg: "#3c2a18" }, 7: { c: "#8f8fa8", bg: "#26262e" },
  8: { c: "#b7c3d4", bg: "#2c323c" }, 9: { c: "#e0b84e", bg: "#3d3316" },
  10: { c: "#ff8ae0", bg: "#352039" }, 11: { c: "#c0a080", bg: "#332a1d" },
};

const POOL = {
  cards: {},
  energyNames: {},   // energyType -> name lookup
  deck: [],          // ordered 60 card ids (or fewer while building)
  filter: "all",     // card kind filter
  energy: "all",     // energy-type filter ("all" or type id string)
  search: "",
  presets: [],
};

let toastTimer = null;
function toast(msg, err) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3000);
}

/* ---------------- boot ---------------- */
async function boot() {
  try {
    const [cat, meta] = await Promise.all([
      (await fetch("catalog.json")).json(),
      (await fetch("/api/game/meta")).json(),
    ]);
    POOL.cards = cat.cards;
    POOL.energyNames = cat.energy_names;
    POOL.presets = meta.decks || [];
  } catch (e) {
    toast("Could not load the card pool: " + e.message, true);
    return;
  }
  renderFilters();
  renderPool();
  loadSaved();
  wire();
}

function wire() {
  $("#pool-search").addEventListener("input", (e) => {
    POOL.search = e.target.value.toLowerCase();
    renderPool();
  });
  $("#random-btn").onclick = randomize;
  $("#clear-btn").onclick = () => { POOL.deck = []; renderDeck(); };
  $("#save-btn").onclick = saveAndBattle;
}

/* ---------------- filters ---------------- */
const FILTERS = [
  ["all", "All"], ["pokemon", "Pokémon"], ["supporter", "Supporter"],
  ["item", "Item"], ["tool", "Tool"], ["stadium", "Stadium"], ["energy", "Energy"],
];

function renderFilters() {
  const row = $("#pool-filters");
  row.innerHTML = "";
  FILTERS.forEach(([id, label]) => {
    const chip = el("button", "chip" + (id === POOL.filter ? " active" : ""), label);
    chip.onclick = () => { POOL.filter = id; renderFilters(); renderPool(); };
    row.appendChild(chip);
  });
  const erow = $("#pool-energy");
  erow.innerHTML = "";
  const energies = [["all", "Any type"]];
  Object.values(POOL.cards).forEach((c) => {
    if (c.type === "pokemon" && !energies.some(([v]) => v === String(c.energyType))) {
      energies.push([String(c.energyType), (POOL.energyNames[String(c.energyType)] || "?").toUpperCase()]);
    }
  });
  energies.forEach(([v, label]) => {
    const chip = el("button", "chip" + (v === POOL.energy ? " active" : ""), label);
    chip.onclick = () => { POOL.energy = v; renderFilters(); renderPool(); };
    erow.appendChild(chip);
  });
}

/* ---------------- pool ---------------- */
function poolCards() {
  const out = [];
  const f = POOL.filter, e = POOL.energy, q = POOL.search;
  Object.values(POOL.cards).forEach((c) => {
    if (f !== "all" && c.type !== f) return;
    if (e !== "all" && c.type === "pokemon" && String(c.energyType) !== e) return;
    if (q && !((c.name || "").toLowerCase().includes(q) ||
        (c.text || "").toLowerCase().includes(q))) return;
    out.push(c);
  });
  out.sort((a, b) => (b.rating || 0) - (a.rating || 0));
  return out;
}

function renderPool() {
  const grid = $("#pool-grid");
  grid.innerHTML = "";
  const cards = poolCards().slice(0, 200);
  cards.forEach((c) => grid.appendChild(poolTile(c)));
  $("#pool-count").textContent = `showing ${cards.length} of ${poolCards().length} cards`;
}

function poolTile(c) {
  const t = TYPE[c.energyType] || TYPE[0];
  const tile = el("div", "pool-tile");
  tile.style.setProperty("--card-bg", t.bg);
  tile.style.setProperty("--card-line", t.c + "55");
  const atk = (c.attacks || [])[0];
  tile.innerHTML =
    `<div class="pool-type">${esc(c.type === "pokemon" ? (c.stage || "POKÉMON").toUpperCase() : c.type.toUpperCase())}${c.ex ? " · EX" : ""}</div>
     <div class="pool-name">${esc(c.name)}</div>
     <div class="pool-stats">
       ${c.hp ? `<span>${c.hp} HP</span>` : ""}
       ${atk ? `<span>⚔ ${esc(atk.name)} ${atk.damage || 0}</span>` : ""}
       ${c.retreat ? `<span>↩ ${c.retreat}</span>` : ""}
       ${c.aceSpec ? `<span class="ace">ACE</span>` : ""}
     </div>
     <div class="pool-count-badge">${countInDeck(c.id)}</div>`;
  tile.onclick = () => addCard(c);
  return tile;
}

function countInDeck(cid) {
  return POOL.deck.filter((x) => x === cid).length;
}

/* ---------------- deck editing ---------------- */
function addCard(c) {
  const n = countInDeck(c.id);
  if (c.aceSpec && n >= 1) return toast("Only 1 ACE SPEC card allowed", true);
  if (c.type === "energy" && n >= 20) return toast("That's a lot of energy…", true);
  if (c.type !== "energy" && n >= 4) return toast(`Max 4 copies of ${c.name}`, true);
  if (POOL.deck.length >= 60) return toast("Deck is full (60 cards)", true);
  POOL.deck.push(c.id);
  renderPool();
  renderDeck();
}

function removeCardAt(index) {
  POOL.deck.splice(index, 1);
  renderPool();
  renderDeck();
}

/* ---------------- deck render + validation ---------------- */
function deckValidation() {
  const errs = [];
  const deck = POOL.deck;
  if (deck.length !== 60) errs.push(`Deck must have 60 cards — you have ${deck.length}.`);
  const counts = {};
  deck.forEach((id) => { counts[id] = (counts[id] || 0) + 1; });
  let basics = 0, ace = 0;
  Object.entries(counts).forEach(([id, n]) => {
    const c = POOL.cards[id];
    if (!c) return;
    if (c.type === "pokemon" && c.stage === "basic") basics += n;
    if (c.aceSpec) ace += n;
    if (c.type !== "energy" && n > 4) errs.push(`More than 4 copies of ${c.name}.`);
  });
  if (basics < 1) errs.push("You need at least 1 Basic Pokémon to start.");
  if (ace > 1) errs.push("Only 1 ACE SPEC card allowed.");
  return errs;
}

function renderDeck() {
  renderPool();
  const list = $("#deck-list");
  list.innerHTML = "";
  const counts = {};
  POOL.deck.forEach((id) => { counts[id] = (counts[id] || 0) + 1; });

  // category breakdown
  const cats = { pokemon: 0, supporter: 0, item: 0, tool: 0, stadium: 0, energy: 0 };
  POOL.deck.forEach((id) => {
    const c = POOL.cards[id];
    if (c) cats[c.type] = (cats[c.type] || 0) + 1;
  });
  const bd = $("#deck-breakdown");
  bd.innerHTML = "";
  Object.entries(cats).forEach(([k, v]) => {
    if (v) bd.appendChild(el("span", "bd-chip", `${k} ${v}`));
  });

  // sorted card rows
  const ids = Object.keys(counts).sort((a, b) => {
    const ca = POOL.cards[a], cb = POOL.cards[b];
    return (cb.rating || 0) - (ca.rating || 0);
  });
  ids.forEach((id) => {
    const c = POOL.cards[id];
    if (!c) return; // stale saved deck referencing a dropped card id
    const row = el("div", "deck-row");
    const t = TYPE[c.energyType] || TYPE[0];
    row.style.setProperty("--card-line", t.c + "55");
    row.innerHTML = `<span class="deck-row-name">${esc(c.name)}</span>
      <span class="deck-row-cat">${esc(c.type)}${c.stage ? " · " + esc(c.stage) : ""}</span>`;
    const stepper = el("div", "deck-stepper");
    const minus = el("button", "step-btn", "−");
    minus.onclick = () => removeCardAt(POOL.deck.lastIndexOf(c.id));
    const num = el("span", "step-num", String(counts[id]));
    const plus = el("button", "step-btn", "+");
    plus.onclick = () => addCard(c);
    stepper.appendChild(minus); stepper.appendChild(num); stepper.appendChild(plus);
    row.appendChild(stepper);
    list.appendChild(row);
  });

  // status
  const errs = deckValidation();
  $("#builder-status").textContent =
    `${POOL.deck.length} / 60 cards${errs.length ? " — " + errs[0] : ""}`;
  $("#builder-status").classList.toggle("invalid", errs.length > 0);
  $("#deck-errors").innerHTML = "";
  errs.forEach((e) => $("#deck-errors").appendChild(el("div", "deck-error", "⚠ " + e)));
  $("#save-btn").disabled = errs.length > 0;
  $("#save-btn").textContent = errs.length ? "Fix the deck to battle" : "⚔️ Save & Battle";
}

/* ---------------- randomize / save ---------------- */
function randomize() {
  if (!POOL.presets.length) return;
  const p = POOL.presets[Math.floor(Math.random() * POOL.presets.length)];
  POOL.deck = (p.cards || []).slice();
  renderDeck();
  toast(`Randomized: ${p.name}`);
}

function loadSaved() {
  try {
    const raw = localStorage.getItem("tcg_custom_deck");
    if (raw) {
      const d = JSON.parse(raw);
      if (Array.isArray(d.cards) && d.cards.length) {
        POOL.deck = d.cards.slice();
      }
    }
  } catch (e) { /* ignore */ }
  renderDeck();
}

function saveAndBattle() {
  const errs = deckValidation();
  if (errs.length) return toast(errs[0], true);
  localStorage.setItem("tcg_custom_deck", JSON.stringify({ cards: POOL.deck }));
  localStorage.setItem("tcg_auto_start", "1");
  window.location.href = "/";
}

boot();
