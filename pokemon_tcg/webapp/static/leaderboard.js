/* Dashboard JS — fetches /api/leaderboard and renders the Elo bars + cards. */

async function loadLeaderboard(force) {
  const url = force ? "/api/leaderboard/refresh" : "/api/leaderboard";
  const r = await fetch(url, { cache: "no-store" });
  return r.json();
}

function fmt(n) {
  return typeof n === "number" ? n.toLocaleString() : n;
}

function maxElo(rows) {
  return Math.max(...rows.map((r) => r.elo), 1500);
}

function minElo(rows) {
  return Math.min(...rows.map((r) => r.elo), 1500);
}

function render(rows, matrixMap) {
  // Stat cards
  const top = rows[0];
  document.getElementById("stat-topagent").textContent = top.name;
  document.getElementById("stat-topagent-sub").textContent =
    `Elo ${top.elo} · ${top.wins}/${top.games} wins`;
  document.getElementById("stat-count").textContent = rows.length;
  const totalGames = rows.reduce((s, r) => s + r.games, 0) / 2; // symmetric
  document.getElementById("stat-games").textContent = fmt(totalGames);
  document.getElementById("stat-spread").textContent =
    `${Math.round(maxElo(rows) - minElo(rows))}`;

  // Agent rows (Elo bars)
  const wrap = document.getElementById("agent-rows");
  wrap.innerHTML = "";
  const scaleMax = maxElo(rows);
  const scaleMin = 1450;
  rows.forEach((row, idx) => {
    const wrapPct = Math.max(0, Math.min(100, ((row.elo - scaleMin) / (scaleMax - scaleMin)) * 100));
    const el = document.createElement("div");
    el.className = "row";
    el.innerHTML = `
      <div class="name"><span class="rank">#${idx + 1}</span>${row.name}</div>
      <div class="bar"><div style="width:${wrapPct}%"></div></div>
      <div class="elo">${row.elo}</div>
      <div class="wr">${row.winrate}%</div>
    `;
    wrap.appendChild(el);
  });

  // Win rate cards (overall + per-opponent bars)
  const grid = document.getElementById("winrate-grid");
  grid.innerHTML = "";
  rows.forEach((row) => {
    const oppBars = rows
      .filter((o) => o.name !== row.name)
      .map((o) => {
        const key = `${row.name}_vs_${o.name}`;
        const [wins, games] = matrixMap[key] || [0, 0];
        const pct = games ? Math.round((wins / games) * 100) : 0;
        return `
          <div class="win-bar">
            <div class="opp-name">${o.name}</div>
            <div class="fill"><div style="width:${pct}%"></div></div>
            <div class="pct">${pct}%</div>
          </div>
        `;
      })
      .join("");
    const card = document.createElement("div");
    card.className = "win-card";
    card.innerHTML = `
      <div class="name">${row.name}</div>
      <div class="overall">${row.winrate}%</div>
      <div class="win-bars">${oppBars}</div>
    `;
    grid.appendChild(card);
  });
}

async function init() {
  try {
    const data = await loadLeaderboard(false);
    render(data.rows || [], data.matrix || {});
  } catch (e) {
    console.error(e);
    document.getElementById("agent-rows").textContent = "Failed to load: " + e;
  }

  const btn = document.getElementById("btn-refresh");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Running…";
      try {
        const data = await loadLeaderboard(true);
        render(data.rows || [], data.matrix || {});
      } catch (e) {
        console.error(e);
      } finally {
        btn.disabled = false;
        btn.textContent = "Refresh";
      }
    });
  }
}

init();
