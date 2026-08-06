/* Matchup matrix JS — fetches /api/matrix and renders win-rate heatmap. */

async function loadMatrix() {
  const r = await fetch("/api/matrix", { cache: "no-store" });
  return r.json();
}

function colorFor(pct) {
  if (pct === null || Number.isNaN(pct)) return "#2a3147";
  // Cool → mid → warm → hot
  if (pct < 30) {
    const t = pct / 30;
    return mix("#3a4d70", "#4a6b9c", t);
  }
  if (pct < 50) {
    const t = (pct - 30) / 20;
    return mix("#4a6b9c", "#7a8a9b", t);
  }
  if (pct < 70) {
    const t = (pct - 50) / 20;
    return mix("#7a8a9b", "#d99e57", t);
  }
  const t = Math.min(1, (pct - 70) / 30);
  return mix("#d99e57", "#f04e3a", t);
}

function mix(a, b, t) {
  const pa = parseHex(a); const pb = parseHex(b);
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}
function parseHex(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}

function render(data) {
  const agents = data.agents;
  const matrix = data.matrix;
  const wrap = document.getElementById("matrix-wrap");
  wrap.innerHTML = "";

  const table = document.createElement("table");
  table.className = "matrix";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(th("corner", "vs"));
  agents.forEach((a) => headRow.appendChild(th("", a)));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  agents.forEach((a) => {
    const row = document.createElement("tr");
    row.appendChild(th("corner", a));
    agents.forEach((b) => {
      const cell = document.createElement("td");
      cell.className = a === b ? "empty" : "cell";
      if (a === b) {
        cell.textContent = "—";
      } else {
        const v = matrix[a] && matrix[a][b];
        if (v && v.games > 0) {
          const pct = Math.round((v.wins / v.games) * 100);
          cell.style.background = colorFor(pct);
          cell.style.color = pct > 50 ? "#1c1100" : "#e6edf3";
          cell.textContent = pct + "%";
          cell.title = `${a} vs ${b}: ${v.wins} of ${v.games} games`;
        } else {
          cell.textContent = "—";
          cell.classList.add("empty");
        }
      }
      row.appendChild(cell);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
}

function th(cls, text) {
  const el = document.createElement("th");
  if (cls) el.className = cls;
  el.textContent = text;
  return el;
}

async function init() {
  try {
    const data = await loadMatrix();
    render(data);
  } catch (e) {
    document.getElementById("matrix-wrap").textContent = "Failed: " + e;
  }
}

init();
